# twisted_tk.jl — Twisted multilayer tight-binding Hamiltonians via MPO
#
# Encoding: the first site is a dim-n_layers "Layer" index; the remaining
# L = Lx+Ly sites are position qubits (quantics binary, row-major).
#
# Depends on: utils.jl, Hamiltonian.jl, 2D_lattice.jl, TBSystem.jl

# ─────────────────────────────────────────────────────────────────
# 1.  Real-space lattice positions
# ─────────────────────────────────────────────────────────────────

"""
    lattice_positions(lattice, Lx, Ly; angle_deg=0.0) -> Matrix{Float64}

Return an N×2 matrix of real-space positions for a 2^Lx × 2^Ly patch of
`lattice ∈ {:square, :triangular, :honeycomb}`, optionally rotated by
`angle_deg` degrees about the origin.

Site ordering matches the quantics row-major encoding used throughout
TensorBinding: site i (1-based) corresponds to linear index i-1 = ix + iy*2^Lx.

Constraint: `:honeycomb` requires `Lx + Ly` to be odd (so N/2 is a perfect square).
"""
function lattice_positions(lattice::Symbol, Lx::Int, Ly::Int;
                           angle_deg::Real = 0.0)
    Nx = 2^Lx;  Ny = 2^Ly;  N = Nx * Ny;  L = Lx + Ly
    rs = Matrix{Float64}(undef, N, 2)

    if lattice === :square
        for i in 0:N-1
            rs[i+1, 1] = Float64(i % Nx)
            rs[i+1, 2] = Float64(i ÷ Nx)
        end
    elseif lattice === :triangular
        a1 = [1.0, 0.0];  a2 = [0.5, sqrt(3) / 2]
        for i in 0:N-1
            p = (i % Nx) .* a1 .+ (i ÷ Nx) .* a2
            rs[i+1, 1] = p[1];  rs[i+1, 2] = p[2]
        end
    elseif lattice === :honeycomb
        isodd(L) || error(":honeycomb requires Lx+Ly to be odd " *
                          "(got Lx=$Lx, Ly=$Ly → L=$L).")
        rs = honeycomb_positions(L)   # defined in TBSystem.jl
    else
        error("Unknown lattice :$lattice.  Choose :square, :triangular, or :honeycomb.")
    end

    if !iszero(angle_deg)
        θ = angle_deg * π / 180
        c, s = cos(θ), sin(θ)
        rs = Matrix{Float64}(([c -s; s c] * rs')')
    end
    return rs
end


# ─────────────────────────────────────────────────────────────────
# 2.  Single-layer MPO builder
# ─────────────────────────────────────────────────────────────────

"""
    monolayer_hamiltonian(lattice, Lx, Ly, sites; t=1.0, cutoff=1e-8) -> MPO

NN tight-binding Hamiltonian MPO on `L = Lx+Ly` qubit `sites` for
`lattice ∈ {:square, :triangular, :honeycomb}` with uniform hopping `t`.
"""
function monolayer_hamiltonian(lattice::Symbol, Lx::Int, Ly::Int, sites;
                               t::Number = 1.0, cutoff::Real = 1e-8)
    Nx = 2^Lx;  L = Lx + Ly;  N = 2^L
    hops = qtt_mpo(L, 0:N-1, sites, _ -> t)

    if lattice === :square
        return +(kineticintra2DNNN(Lx, Ly, sites, hops, 1),
                 kineticNNN(L, sites, hops, Nx); cutoff=cutoff)
    elseif lattice === :triangular
        H = +(kineticintra2DNNN(Lx, Ly, sites, hops, 1),
              kineticinterNNNtriSWNE(Lx, Ly, sites, hops, Nx + 1); cutoff=cutoff)
        return +(H, kineticinterNNNtriSENW(Lx, Ly, sites, hops, Nx - 1); cutoff=cutoff)
    elseif lattice === :honeycomb
        return +(kineticintra2DNNhex(Lx, Ly, sites, hops, 1),
                 kineticNNN(L, sites, hops, Nx); cutoff=cutoff)
    else
        error("Unknown lattice :$lattice.  Choose :square, :triangular, or :honeycomb.")
    end
end


# ─────────────────────────────────────────────────────────────────
# 3.  Layer-core prepend helpers
# ─────────────────────────────────────────────────────────────────

"""
    _prepend_layer_core(H_mpo, layer_s, op_entries) -> MPO

Internal helper: prepend a single-site layer operator to `H_mpo`, extending
it from L sites to L+1 sites.  `op_entries` is a vector of `(bra, ket, val)`
triples giving the nonzero elements of the operator in the `layer_s` basis
(1-indexed).

The returned MPO has site indices `[layer_s; original_sites...]`.
"""
function _prepend_layer_core(H_mpo::MPO, layer_s::Index,
                              op_entries::AbstractVector{<:Tuple})
    Lh    = length(H_mpo)
    bond0 = Index(1, "Link,l=0")
    Op    = ITensor(layer_s', layer_s, bond0)
    for (b, k, v) in op_entries
        Op[layer_s' => b, layer_s => k, bond0 => 1] = v
    end
    delta0 = ITensor(bond0);  delta0[bond0 => 1] = 1.0
    H1_ext = H_mpo[1] * delta0
    ext    = MPO(Lh + 1)
    ext[1] = Op
    ext[2] = H1_ext
    for k in 3:Lh+1
        ext[k] = H_mpo[k-1]
    end
    return ext
end


"""
    prepend_layer_projector(H_mpo, layer_s, k) -> MPO

Extend `H_mpo` by prepending the diagonal projector P_k = |k⟩⟨k| on `layer_s`
(layer index k is 1-based).  Used to build the intralayer term P_k ⊗ H.
"""
prepend_layer_projector(H::MPO, s::Index, k::Int) =
    _prepend_layer_core(H, s, [(k, k, 1.0)])


"""
    prepend_layer_hopping(H_mpo, layer_s, k, l) -> MPO

Extend `H_mpo` by prepending the off-diagonal operator T_{kl} = |k⟩⟨l| on
`layer_s`.  Used to build the interlayer term |k⟩⟨l| ⊗ V_{kl}.
"""
prepend_layer_hopping(H::MPO, s::Index, k::Int, l::Int) =
    _prepend_layer_core(H, s, [(k, l, 1.0)])


# ─────────────────────────────────────────────────────────────────
# 4.  Twisted multilayer Hamiltonian
# ─────────────────────────────────────────────────────────────────

"""
    twisted_multilayer_hamiltonian(lattice, Lx, Ly, angles_deg;
        t_intra=1.0, t_inter=0.3, α_decay=1/16.0,
        tol=1e-6, cutoff=1e-8, maxdim=200) -> (MPO, Vector{<:Index})

Build a twisted multilayer tight-binding Hamiltonian as an MPO.

**Site encoding** (`L+1` sites, `L = Lx+Ly`):
  - Site 1      : layer index (dim = `n_layers`)  ← replaces the bilayer qubit
  - Sites 2…L+1 : `L` position qubits (quantics binary, row-major)

Each layer k is rigidly rotated to angle `angles_deg[k]` (degrees).  The
interlayer coupling is exponentially decaying:
  `V_{kl}(i,j) = t_inter * exp(−α_decay * |r_k[i] − r_l[j]|)`.

The assembled Hamiltonian is:
  `H = Σ_k P_k ⊗ H_mono  +  Σ_{k<l} (|k⟩⟨l| ⊗ V_{kl} + |l⟩⟨k| ⊗ V_{lk})`

which is Hermitian for real V_{kl} since V_{lk}(i,j) = V_{kl}(j,i) = V_{kl}^T.

**Arguments**
- `lattice`    : `:square`, `:triangular`, or `:honeycomb`
- `Lx`, `Ly`  : `2^Lx × 2^Ly` sites per layer
- `angles_deg` : twist angles (°) for each layer; `length` = `n_layers`

**Keyword arguments**
- `t_intra`  : intra-layer NN hopping amplitude (same for all layers)
- `t_inter`  : interlayer hopping amplitude
- `α_decay`  : exponential decay constant for interlayer hopping
- `tol`      : QTCI tolerance passed to `hopping2MPO`
- `cutoff`   : MPO truncation cutoff used throughout assembly
- `maxdim`   : maximum bond dimension of the final MPO

Returns `(H_total, ext_sites)` where `ext_sites[1]` is the layer index
and `ext_sites[2:end]` are the `L` position qubits.
"""
function twisted_multilayer_hamiltonian(
    lattice::Symbol, Lx::Int, Ly::Int,
    angles_deg::AbstractVector{<:Real};
    t_intra::Number = 1.0,
    t_inter::Number = 0.3,
    α_decay::Real   = 1 / 16.0,
    tol::Real       = 1e-6,
    cutoff::Real    = 1e-8,
    maxdim::Int     = 200,
)
    n_layers = length(angles_deg)
    n_layers ≥ 2 || error("Need at least 2 layers; got $n_layers.")
    L = Lx + Ly;  N = 2^L

    # Site spaces: one dim-n_layers layer site + L position qubits
    layer_s   = Index(n_layers, "Layer")
    pos_sites = siteinds("Qubit", L)
    ext_sites = [layer_s; pos_sites]

    # Real-space positions per layer (rotated by the respective angle)
    positions = [lattice_positions(lattice, Lx, Ly; angle_deg = θ)
                 for θ in angles_deg]

    # ── Intralayer: Σ_k P_k ⊗ H_mono  ────────────────────────────
    # All layers share the same geometry; twist only affects interlayer terms.
    H_mono   = monolayer_hamiltonian(lattice, Lx, Ly, pos_sites;
                                     t = t_intra, cutoff = cutoff)
    H_intra  = prepend_layer_projector(H_mono, layer_s, 1)
    for k in 2:n_layers
        H_intra = +(H_intra, prepend_layer_projector(H_mono, layer_s, k); cutoff=cutoff)
    end

    # ── Interlayer: Σ_{k<l} (|k⟩⟨l|⊗V_{kl} + |l⟩⟨k|⊗V_{lk})  ──
    H_inter = nothing
    for k in 1:n_layers, l in (k+1):n_layers
        rsk = positions[k];  rsl = positions[l]
        V_kl(i, j) = t_inter * exp(-α_decay * norm(rsk[Int(i), :] - rsl[Int(j), :]))
        V_lk(i, j) = V_kl(j, i)   # = V_{kl}^T (transpose = h.c. for real V)

        Vkl_mpo = hopping2MPO(V_kl, N, pos_sites; tol=tol, type=Float64)
        Vlk_mpo = hopping2MPO(V_lk, N, pos_sites; tol=tol, type=Float64)

        term = +(prepend_layer_hopping(Vkl_mpo, layer_s, k, l),
                 prepend_layer_hopping(Vlk_mpo, layer_s, l, k); cutoff=cutoff)
        H_inter = H_inter === nothing ? term : +(H_inter, term; cutoff=cutoff)
    end

    # ── Assembly ──────────────────────────────────────────────────
    H_total = +(H_intra, H_inter; cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)
    # scale=0.0 → lazy DMRG estimation on first KPM_Tn call
    return TBHamiltonian(L, 2^L, ext_sites, H_total, nothing, 0.0, 0.0,
                         nothing, nothing, layer_s, nothing, 0, nothing)
end


"""
    twisted_bilayer_hamiltonian(lattice, Lx, Ly, θ_deg;
        t_intra=1.0, t_inter=0.3, α_decay=1/16.0,
        tol=1e-6, cutoff=1e-8, maxdim=200) -> (MPO, Vector{<:Index})

Convenience wrapper: two layers, layer 1 at 0° and layer 2 at `θ_deg`.
Equivalent to `twisted_multilayer_hamiltonian(lattice, Lx, Ly, [0.0, θ_deg]; …)`.
"""
function twisted_bilayer_hamiltonian(
    lattice::Symbol, Lx::Int, Ly::Int, θ_deg::Real;
    t_intra::Number = 1.0,
    t_inter::Number = 0.3,
    α_decay::Real   = 1 / 16.0,
    tol::Real       = 1e-6,
    cutoff::Real    = 1e-8,
    maxdim::Int     = 200,
)
    return twisted_multilayer_hamiltonian(
        lattice, Lx, Ly, [0.0, Float64(θ_deg)];
        t_intra=t_intra, t_inter=t_inter, α_decay=α_decay,
        tol=tol, cutoff=cutoff, maxdim=maxdim,
    )
end
