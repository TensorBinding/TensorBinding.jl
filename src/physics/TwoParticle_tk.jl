# twoparticle_tk.jl — exciton/two-particle Hamiltonian construction (1-2) and
# MPS basis-state probes, real-space and momentum-space (3)

# ─────────────────────────────────────────────────────────────────
# 1.  High-level constructor (returns TBHamiltonian)
# ─────────────────────────────────────────────────────────────────

"""
    exciton_hamiltonian(geometry, params, Ufunc; L, [on_site, scale, tol_quantics,
        maxbonddim_quantics, tol, cutoff, maxdim, kwargs...]) -> TBHamiltonian

    exciton_hamiltonian(H_c, H_v, Ufunc; [on_site, scale, tol_quantics,
        maxbonddim_quantics, tol, cutoff, maxdim]) -> TBHamiltonian

Build an exciton Hamiltonian and wrap it in a `TBHamiltonian` for use with
TensorBinding's KPM, DMRG, and spectral tools.

**Site encoding** (`2L` sites total, `L = H_c.L`):
- Odd sites  (1, 3, …) : electron position qubits
- Even sites (2, 4, …) : hole position qubits (interleaved)

`TBHamiltonian.L = L` counts position qubits per sector;
`TBHamiltonian.sites` holds all `2L` interleaved MPO sites.

**Two calling modes**

1. **Geometry string** — both bands built from the same geometry and parameters
   via `get_Hamiltonian`; any extra `kwargs` are forwarded to it:
   ```julia
   H = exciton_hamiltonian("square_2d", t, Ufunc; L=Lx+Ly, Lx=Lx, Ly=Ly,
                            on_site = x -> V(x))
   ```

2. **Pre-built sectors** — pass explicit `TBHamiltonian` objects for the electron
   (`H_c`) and hole (`H_v`) bands when the two sectors differ (different hopping,
   disorder, external fields):
   ```julia
   H_c = get_Hamiltonian("chain_1d", t_c; L=L)
   H_v = get_Hamiltonian("chain_1d", t_v; L=L)
   H   = exciton_hamiltonian(H_c, H_v, Ufunc)
   ```

**Keyword arguments**
- `on_site`             : conduction band edge modulation `V(x)`, 1-indexed.
                          Applied as `+V` to electron and `−V` to valence sector
                          (type-I confinement). Compressed via QTCI.
- `scale`               : exciton spectral half-bandwidth (0.0 → lazy DMRG estimate).
- `tol_quantics`        : QTCI tolerance for `Ufunc` and `on_site`. Default `1e-8`.
- `maxbonddim_quantics` : QTCI max bond dimension. Default `100`.
- `tol`                 : MPO assembly truncation tolerance. Default `1e-8`.
- `cutoff`              : SVD cutoff for MPO arithmetic. Default `1e-8`.
- `maxdim`              : max bond dimension of the final MPO. Default `200`.
"""
function exciton_hamiltonian(geometry::String, params, Ufunc;
                              L::Int,
                              on_site              = nothing,
                              scale                = nothing,
                              tol_quantics         = 1e-8,
                              maxbonddim_quantics  = 100,
                              tol                  = 1e-8,
                              cutoff               = 1e-8,
                              maxdim               = 200,
                              kwargs...)
    H_c = get_Hamiltonian(geometry, params; L=L, tol=tol, maxdim=maxdim, kwargs...)
    H_v = get_Hamiltonian(geometry, params; L=L, tol=tol, maxdim=maxdim, kwargs...)
    return exciton_hamiltonian(H_c, H_v, Ufunc;
                               on_site             = on_site,
                               scale               = scale,
                               tol_quantics        = tol_quantics,
                               maxbonddim_quantics = maxbonddim_quantics,
                               tol=tol, cutoff=cutoff, maxdim=maxdim)
end

function exciton_hamiltonian(H_c::TBHamiltonian, H_v::TBHamiltonian, Ufunc;
                              on_site              = nothing,
                              scale                = nothing,
                              tol_quantics         = 1e-8,
                              maxbonddim_quantics  = 100,
                              tol                  = 1e-8,
                              cutoff               = 1e-8,
                              maxdim               = 200)
    H_exc_mpo = Exciton_Hamiltonian(H_c, H_v, Ufunc;
                                     on_site             = on_site,
                                     tol_quantics        = tol_quantics,
                                     maxbonddim_quantics = maxbonddim_quantics,
                                     tol=tol, cutoff=cutoff, maxdim=maxdim)
    sites_eh = collect(Iterators.flatten(zip(H_c.sites, H_v.sites)))
    sc       = something(scale, 0.0)   # 0.0 → lazy DMRG estimation on first KPM call
    # Inherit single-sector geometry from H_c (physical positions are the same).
    # L = position qubits per sector; N = physical positions; sites = 2L interleaved
    return TBHamiltonian(H_c.L, H_c.N, sites_eh, H_exc_mpo, H_c.geometry, sc, 0.0,
                         nothing, nothing, nothing, nothing, 0, nothing)
end


# ─────────────────────────────────────────────────────────────────
# 2.  Low-level MPO builder
# ─────────────────────────────────────────────────────────────────

"""
    Exciton_Hamiltonian(H_c, H_v, Ufunc; on_site, tol_quantics, maxbonddim_quantics,
                        tol, cutoff, maxdim) -> MPO

Build the exciton Hamiltonian on the interleaved 2L-site electron-hole space:

    H_exc = (H_c ⊗ I_h − I_e ⊗ H_v) + U

where `U` is the contact interaction diagonal MPO built from `Ufunc`.

`H_c` and `H_v` are `TBHamiltonian` objects for the electron and hole
single-particle sectors (any geometry: `"chain_1d"`, `"square_2d"`, etc.).
Both must have the same `L` and distinct site indices.
`Ufunc(x)` gives the interaction strength at site `x ∈ {1, …, 2^L}` (1-indexed).

**`on_site` keyword (optional):** a function `V(x)` representing the conduction
band edge modulation, compressed via QTCI. Applied as `+V` to the electron sector
and `−V` to the valence sector, so that the hole also feels `+V` after the
`H_c − H_v` subtraction (type-I semiconductor convention: both carriers confined).
`tol_quantics` and `maxbonddim_quantics` control the QTCI compression of `V`.

Examples
--------
```julia
# 1D, uniform hopping, contact interaction, Gaussian confinement
H_c = get_Hamiltonian("chain_1d", t; L=L)
H_v = get_Hamiltonian("chain_1d", t; L=L)
H_exc = Exciton_Hamiltonian(H_c, H_v, x -> -U;
                             on_site = x -> -V0 * exp(-((x - N/2)^2) / (2σ^2)))

# 2D square lattice
H_c = get_Hamiltonian("square_2d", t; L=Lx+Ly, Lx=Lx, Ly=Ly)
H_v = get_Hamiltonian("square_2d", t; L=Lx+Ly, Lx=Lx, Ly=Ly)
H_exc = Exciton_Hamiltonian(H_c, H_v, Ufunc; on_site = x -> dot_potential(x))
```
"""
function Exciton_Hamiltonian(H_c::TBHamiltonian, H_v::TBHamiltonian, Ufunc;
                              on_site              = nothing,
                              tol_quantics         = 1e-8,
                              maxbonddim_quantics  = 100,
                              tol                  = 1e-8,
                              cutoff               = 1e-8,
                              maxdim               = 200)
    @assert H_c.L == H_v.L "Electron and hole sectors must have the same system size"

    mpo_c = H_c.mpo
    mpo_v = H_v.mpo

    if on_site !== nothing
        # on_site(x) is compressed via QTCI on the 1-indexed grid {1, …, 2^L}.
        # +V on conduction band (electron), −V on valence band (hole).
        # After H_c − H_v, both carriers feel +V → type-I confinement.
        L     = H_c.L
        xvals = range(1, 2^L; length=2^L)
        V_c   = qtt_mpo(L, xvals, H_c.sites, on_site;
                        tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
        V_v   = qtt_mpo(L, xvals, H_v.sites, on_site;
                        tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
        mpo_c = +(mpo_c,       V_c; cutoff=tol)
        mpo_v = +(mpo_v, -1.0*V_v; cutoff=tol)
    end

    sites_eh    = collect(Iterators.flatten(zip(H_c.sites, H_v.sites)))
    kinetic_mpo = interleave_mpo(mpo_c, sites_eh, 0) - interleave_mpo(mpo_v, sites_eh, 1)
    # Use sites_eh directly (unprimed ket indices) — extracting sites from
    # siteinds(kinetic_mpo) risks grabbing the primed bra indices instead.
    interaction = build_interaction_op_exciton(H_c.L, sites_eh, Ufunc)
    return kinetic_mpo + interaction
end


"""
    hopping_mpo_exciton(H_c, H_v) -> MPO

Embed the electron (`H_c`) and hole (`H_v`) single-particle Hamiltonians
into the interleaved 2L-site electron-hole space:

    H_kin = H_c ⊗ I_h  -  I_e ⊗ H_v

`H_c` sits at odd sites (1, 3, 5, …) and `H_v` at even sites (2, 4, 6, …).
Both `TBHamiltonian` objects must have the same `L` and distinct site indices.
"""
function hopping_mpo_exciton(H_c::TBHamiltonian, H_v::TBHamiltonian)
    @assert H_c.L == H_v.L "Electron and hole sectors must have the same system size"
    sites_eh = collect(Iterators.flatten(zip(H_c.sites, H_v.sites)))
    return interleave_mpo(H_c.mpo, sites_eh, 0) - interleave_mpo(H_v.mpo, sites_eh, 1)
end


"""
    build_interaction_op_exciton(L, sites, Ufunc) -> MPO

Build the electron-hole interaction MPO on the 2L-site interleaved space.
`Ufunc(x)` gives the interaction strength when electron and hole are both
at position `x` (contact interaction). `sites` must be the 2L-site interleaved
site index vector.
"""
function build_interaction_op_exciton(L, sites, Ufunc)
    evals = range(1, 2^L, length=2^L)
    hvals = range(1, 2^L, length=2^L)

    o(x, y) = x == y ? Ufunc(x) : 0

    qtt     = quanticscrossinterpolate(Float64, o, [evals, hvals]; tolerance=1e-8)[1]
    tt      = TCI.tensortrain(qtt.tci)
    int_mps = MPS(tt)

    return -mps_to_diagonal_mpo(int_mps, sites)
end


# ─────────────────────────────────────────────────────────────────
# 3.  MPS probes (real-space and momentum-space)
# ─────────────────────────────────────────────────────────────────

# ------------------------------------------------------------
# Real-space (position) basis states
# ------------------------------------------------------------

# Exciton basis state |xe, xh> on the interleaved electron-hole chain.
# xe, xh are 1-indexed (in {1, ..., 2^LPhys}), consistent with get_diagonal_mpo
# and add_onsite! conventions in TensorBinding.
function mpsexciton(xe, xh, sites)
    L     = length(sites)
    LPhys = div(L, 2)
    bits_e = to_binary_vector(Int(xe) - 1, LPhys)   # shift to 0-indexed for binary encoding
    bits_h = to_binary_vector(Int(xh) - 1, LPhys)

    elechole = Vector{String}(undef, L)
    for i in 1:LPhys
        elechole[2i - 1] = bits_e[i]
        elechole[2i]     = bits_h[i]
    end

    return MPS(sites, elechole)
end

# |x, x> bound electron-hole probe (d = 0 separation).
mpsexciton(x, sites) = mpsexciton(x, x, sites)


# ------------------------------------------------------------
# Exciton momentum-basis MPS probes
# ------------------------------------------------------------

"""
    mpsexcitonQ(Q, sites) -> MPS

Normalized fixed-total-momentum exciton state on the interleaved electron-hole
chain:

    |Q> = (1 / sqrt(N)) * sum_k |k, -k + Q>

`Q` is 1-indexed (`Q in 1:2^LPhys`), matching `mpsexciton` and the rest of the
public exciton sampling API. Internally the modular momentum arithmetic is
0-indexed and uses the QFT momentum convention: site pair 1 carries bit 0
(LSB-first).

The construction encodes the constraint `k + h = Q (mod N)` as a small
finite-state MPS rather than explicitly summing over all `N = 2^LPhys` momenta.
For each electron-hole bit pair `(a_i, b_i)` it keeps a binary carry on the MPS
bond and allows only local configurations satisfying

    a_i + b_i + carry_in = Q_i + 2 * carry_out

where `Q_i` is the `i`th bit of `Q - 1`. The first bit has `carry_in = 0`, and
the final carry is not fixed, giving addition modulo `2^LPhys`.

Scaling: the probe has `2LPhys` physical sites, bond dimension at most 2 between
bit pairs (the carry), and a local intermediate link of dimension 4 inside each
electron-hole pair. Building the probe is therefore `O(LPhys)` in storage and
time, not `O(2^LPhys)`. For very large systems the expensive part is the
subsequent MPS-KPM recursion with `H_QFT`; this compact probe construction should
not be the bottleneck.
"""
function mpsexcitonQ(Q, sites)
    L = length(sites)
    iseven(L) || error("mpsexcitonQ expects an even electron-hole site count; got $L.")
    all(s -> dim(s) == 2, sites) ||
        error("mpsexcitonQ expects qubit sites with dimension 2.")

    LPhys = div(L, 2)
    N     = 1 << LPhys
    Q_int = Int(Q)
    1 <= Q_int <= N ||
        error("mpsexcitonQ expects 1 <= Q <= $N; got $Q.")

    Q0     = Q_int - 1
    q_bits = [(Q0 >> (i - 1)) & 1 for i in 1:LPhys]
    norm   = 1 / sqrt(float(N))

    carry_links = [Index(2, "Link,excitonQ_carry=$i") for i in 1:LPhys-1]
    tensors     = Vector{ITensor}(undef, L)

    midval(c_left, a_bit) = 2 * c_left + a_bit + 1

    for i in 1:LPhys
        s_e = sites[2i - 1]
        s_h = sites[2i]
        mid = Index(4, "Link,excitonQ_mid=$i")

        if i == 1
            A = ITensor(s_e, mid)
            for a_bit in 0:1
                A[s_e => a_bit + 1, mid => midval(0, a_bit)] = norm
            end
        else
            left = carry_links[i - 1]
            A = ITensor(left, s_e, mid)
            for c_left in 0:1, a_bit in 0:1
                A[left => c_left + 1, s_e => a_bit + 1,
                  mid => midval(c_left, a_bit)] = 1.0
            end
        end
        tensors[2i - 1] = A

        q_bit = q_bits[i]
        if i == LPhys
            B = ITensor(mid, s_h)
            for c_left in 0:1, a_bit in 0:1, b_bit in 0:1, c_right in 0:1
                if a_bit + b_bit + c_left == q_bit + 2 * c_right
                    B[mid => midval(c_left, a_bit), s_h => b_bit + 1] = 1.0
                end
            end
        else
            right = carry_links[i]
            B = ITensor(mid, s_h, right)
            for c_left in 0:1, a_bit in 0:1, b_bit in 0:1, c_right in 0:1
                if a_bit + b_bit + c_left == q_bit + 2 * c_right
                    B[mid => midval(c_left, a_bit), s_h => b_bit + 1,
                      right => c_right + 1] = 1.0
                end
            end
        end
        tensors[2i] = B
    end

    return MPS(tensors)
end

"""
    mpsexcitonQTrace(Q, sites; rng=Random.default_rng()) -> MPS

Random-phase fixed-total-momentum trace probe for the electron-hole continuum:

    |r_Q> = (1 / sqrt(N)) * sum_k eta_k |k, Q-k>

where the phases are Rademacher signs `eta_k = +/-1`. The signs are generated
as a product over the bits of `k`, so the state still has the same compact
finite-state structure as [`mpsexcitonQ`](@ref): it enforces
`k + h = Q (mod N)` with a carry bond of dimension at most 2, while adding
random signs to the electron-bit tensors.

This is the MPS probe used for stochastic continuum traces. For an operator
restricted to the fixed-`Q` sector,

    E[<r_Q|A|r_Q>] = (1 / N) * sum_k <k,Q-k|A|k,Q-k>

because the random signs satisfy `E[eta_k eta_l] = delta_kl`. Building one probe
is `O(LPhys)` and avoids launching one KPM recursion per explicit relative
momentum `k`.
"""
function mpsexcitonQTrace(Q, sites; rng=Random.default_rng())
    L = length(sites)
    iseven(L) || error("mpsexcitonQTrace expects an even electron-hole site count; got $L.")
    all(s -> dim(s) == 2, sites) ||
        error("mpsexcitonQTrace expects qubit sites with dimension 2.")

    LPhys = div(L, 2)
    N     = 1 << LPhys
    Q_int = Int(Q)
    1 <= Q_int <= N ||
        error("mpsexcitonQTrace expects 1 <= Q <= $N; got $Q.")

    Q0     = Q_int - 1
    q_bits = [(Q0 >> (i - 1)) & 1 for i in 1:LPhys]
    signs  = [rand(rng, Bool) ? 1.0 : -1.0 for _ in 1:LPhys]
    norm   = 1 / sqrt(float(N))

    carry_links = [Index(2, "Link,excitonQTrace_carry=$i") for i in 1:LPhys-1]
    tensors     = Vector{ITensor}(undef, L)

    midval(c_left, a_bit) = 2 * c_left + a_bit + 1
    phase(i, a_bit) = a_bit == 0 ? 1.0 : signs[i]

    for i in 1:LPhys
        s_e = sites[2i - 1]
        s_h = sites[2i]
        mid = Index(4, "Link,excitonQTrace_mid=$i")

        if i == 1
            A = ITensor(s_e, mid)
            for a_bit in 0:1
                A[s_e => a_bit + 1, mid => midval(0, a_bit)] =
                    norm * phase(i, a_bit)
            end
        else
            left = carry_links[i - 1]
            A = ITensor(left, s_e, mid)
            for c_left in 0:1, a_bit in 0:1
                A[left => c_left + 1, s_e => a_bit + 1,
                  mid => midval(c_left, a_bit)] = phase(i, a_bit)
            end
        end
        tensors[2i - 1] = A

        q_bit = q_bits[i]
        if i == LPhys
            B = ITensor(mid, s_h)
            for c_left in 0:1, a_bit in 0:1, b_bit in 0:1, c_right in 0:1
                if a_bit + b_bit + c_left == q_bit + 2 * c_right
                    B[mid => midval(c_left, a_bit), s_h => b_bit + 1] = 1.0
                end
            end
        else
            right = carry_links[i]
            B = ITensor(mid, s_h, right)
            for c_left in 0:1, a_bit in 0:1, b_bit in 0:1, c_right in 0:1
                if a_bit + b_bit + c_left == q_bit + 2 * c_right
                    B[mid => midval(c_left, a_bit), s_h => b_bit + 1,
                      right => c_right + 1] = 1.0
                end
            end
        end
        tensors[2i] = B
    end

    return MPS(tensors)
end

"""
    mpsexcitonKQ(k, Q, sites) -> MPS

Product momentum-basis electron-hole continuum state at fixed total momentum:

    |k, Q - k>

Both `k` and `Q` are 1-indexed (`1:2^LPhys`) in the public API. Internally they
are shifted to 0-indexed modular arithmetic, and the hole momentum is computed
as `h = Q - k (mod N)`.

This is the incoherent continuum probe complement to [`mpsexcitonQ`](@ref).
`mpsexcitonQ(Q, sites)` builds the coherent pair state
`(1 / sqrt(N)) * sum_k |k, Q-k>`, while `mpsexcitonKQ(k, Q, sites)` builds one
rank-1 product basis vector in that fixed-`Q` sector. Stochastic continuum
traces sample many such `k` values and average the resulting MPS-KPM spectra.

The site convention matches the QFT momentum convention: site pair 1 carries bit
0 (LSB-first), with electron bits on odd sites and hole bits on even sites.
"""
function mpsexcitonKQ(k, Q, sites)
    L = length(sites)
    iseven(L) || error("mpsexcitonKQ expects an even electron-hole site count; got $L.")
    all(s -> dim(s) == 2, sites) ||
        error("mpsexcitonKQ expects qubit sites with dimension 2.")

    LPhys = div(L, 2)
    N     = 1 << LPhys
    k_int = Int(k)
    Q_int = Int(Q)
    1 <= k_int <= N ||
        error("mpsexcitonKQ expects 1 <= k <= $N; got $k.")
    1 <= Q_int <= N ||
        error("mpsexcitonKQ expects 1 <= Q <= $N; got $Q.")

    k0 = k_int - 1
    Q0 = Q_int - 1
    h0 = mod(Q0 - k0, N)

    state = Vector{String}(undef, L)
    for i in 1:LPhys
        state[2i - 1] = string((k0 >> (i - 1)) & 1)
        state[2i]     = string((h0 >> (i - 1)) & 1)
    end

    return MPS(sites, state)
end

