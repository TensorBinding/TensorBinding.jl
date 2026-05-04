# Topology_tk.jl — topological invariants via KPM and MPO methods
#
# Implements real-space Chern markers and winding numbers for arbitrary
# tight-binding systems encoded in the quantics representation.
#
# The key observable is the real-space Chern marker C(r):
#   C(r) = 4π Im⟨r| P [x, P] [y, P] |r⟩
#
# where P is the ground-state projector and [A, B] = AB − BA.
# Integrating C(r) over the bulk gives the integer Chern number.
#
# Position operators are specified as two-argument scalar functions:
#   xfunc(i::Int, L_chain::Int) -> Float64   x-coordinate of site i (0-indexed)
#   yfunc(i::Int, L_chain::Int) -> Float64   y-coordinate of site i (0-indexed)
#
# Two modes are supported (controlled by the `quenched` kwarg):
#
#   quenched=true (default):
#       Position operators are quenched via sin/cos of xfunc/Λ and yfunc/Λ.
#       The Chern marker is computed via a 4-term trig decomposition.
#       A Λ² prefactor is applied (since sin(x/Λ) ≈ x/Λ, using sin divides
#       each coordinate by Λ; Λ² restores physical units).
#       Removes the PBC discontinuity.
#
#   quenched=false:
#       Position operators use xfunc/yfunc directly without sin/cos wrapping.
#       No additional prefactor — formula is 2πi (Q x P y Q − P x Q y P).
#
# Ground-state projector P can be obtained via:
#   method=:KPM      — KPM Chebyshev expansion (uses cached Tn if available)
#   method=:mcweeny  — McWeeny purification (uses density cache if available)
#   method=:sp2      — SP2 purification (uses density cache if available)
#
# Bond dimension and truncation are controlled uniformly through `maxdim` and
# `cutoff` kwargs, which are threaded into every apply, add, and truncate! call.


# ============================================================
# Helper: ground-state projector from TBHamiltonian
# ============================================================

"""
    _get_projector(H; method, fermi, Nchebychev, maxdim, cutoff, Nel) -> MPO

Compute or retrieve the ground-state projector P for `H`.

- `method=:KPM` (default): uses the cached `H._tn_cache` if present; otherwise
  runs `KPM_Tn(H, Nchebychev)`.  The Fermi level `fermi` (in physical units)
  is rescaled internally.
- `method=:mcweeny`: returns `H._density_cache` if set; otherwise runs McWeeny
  purification.  `fermi` is ignored.
- `method=:sp2`: same but uses SP2 purification.  `Nel` sets the target
  electron count (default: `H.N ÷ 2`).
- `maxdim`, `cutoff`: bond dimension and truncation threshold forwarded to the
  underlying method.
"""
function _get_projector(H::TBHamiltonian;
                         method::Symbol   = :KPM,
                         fermi::Real      = 0.0,
                         Nchebychev::Int  = 300,
                         maxdim::Int      = 40,
                         cutoff::Float64  = 1e-8,
                         Nel              = nothing,
                         run_on::Symbol   = :cpu)
    if method == :KPM
        if H._tn_cache !== nothing
            Tn_list = H._tn_cache
            Ncheb   = H._tn_Ncheb
        else
            Tn_list, _, _ = KPM_Tn(H, Nchebychev; maxdim=maxdim, cutoff=cutoff, run_on=run_on)
            Ncheb = Nchebychev
        end
        fermi_resc = (fermi - H.center) / H.scale
        return get_density_from_Tn(Tn_list, Ncheb; fermi=fermi_resc, maxdim=maxdim)
    elseif method == :mcweeny
        H._density_cache !== nothing && return H._density_cache
        return mcweeny_purify(H; ϵF=fermi, maxdim=maxdim, cutoff=cutoff, run_on=run_on)
    elseif method == :sp2
        H._density_cache !== nothing && return H._density_cache
        Nel_val = Nel === nothing ? H.N ÷ 2 : Int(Nel)
        return sp2_purify(H; Nel=Nel_val, maxdim=maxdim, cutoff=cutoff, run_on=run_on)
    else
        error("Unknown method: :$method. Choose :KPM, :mcweeny, or :sp2")
    end
end


# ============================================================
# 1D winding number
# ============================================================

"""
    get_W(H::TBHamiltonian, xfunc, sz_func;
          method=:KPM, fermi=0.0, Nchebychev=300,
          maxdim=15, cutoff=1e-8, Nel=nothing,
          quenched=true, l=nothing, Λ=10) -> Function

Compute the real-space winding number and return a closure
`calculate_winding(α::Int) -> ComplexF64` that evaluates the local winding
number density at any site `α` (1-indexed).

The winding number operator is

    W_op = σ(r) · (P x̂ Q + Q x̂ P)

where `P` is the occupied-band projector, `Q = I − P`, `x̂` is the position
operator centred at `α`, and `σ` is the chirality / sublattice operator.

# Coordinate function

`xfunc(i, L_chain) -> Float64` accepts a **0-indexed** site number and
returns the raw x-coordinate.  For the SSH model with two sites per unit cell
at the same position: `xfunc(i, L_chain) = Float64(div(i, 2))`.

`sz_func(i) -> Float64` receives a 1-indexed site index (as from
`get_diagonal_mpo`) and returns the sublattice / chirality sign.
For SSH: `sz_func(i) = Float64((-1)^(i+1))`.

# Quenched vs flat mode

- `quenched=true` (default): uses the trig identity
      sin((x_r − x_α)/Λ) = sinX_r cos(x_α/Λ) − cosX_r sin(x_α/Λ)
  to pre-compute two α-independent MPOs W1 and W2, then combines them with
  scalar trig factors in the closure.  Prefactor Λ restores physical units.

      W(α) = Λ · ⟨α| cos(x_α/Λ) W1 − sin(x_α/Λ) W2 |α⟩

- `quenched=false`: builds a global (uncentred) position operator from `xfunc`
  and returns a closure over the resulting W_op MPO.

# Arguments
- `method`    : `:KPM`, `:mcweeny`, or `:sp2` (see `_get_projector`).
- `fermi`     : Fermi level in physical energy units (KPM only).
- `Nchebychev`: Chebyshev order when `method=:KPM` and no cache is present.
- `maxdim`    : MPO bond dimension during all multiplications.
- `cutoff`    : truncation threshold during all multiplications.
- `Nel`       : target electron count for SP2 (default `H.N ÷ 2`).
- `l`         : qubits per direction; inferred as `H.L ÷ 2` if `nothing`.
- `Λ`         : quenching period (angle = xfunc/Λ); sets the Λ prefactor.

# Returns
`calculate_winding(α::Int) -> ComplexF64` where `α` is 1-indexed.
"""
function get_W(H::TBHamiltonian, xfunc, sz_func;
               method::Symbol   = :KPM,
               fermi::Real      = 0.0,
               Nchebychev::Int  = 300,
               maxdim::Int      = 15,
               cutoff::Float64  = 1e-8,
               Nel              = nothing,
               quenched::Bool   = true,
               l                = nothing,
               Λ::Real          = 10,
               run_on::Symbol   = :cpu)
    backend = _resolve_backend(run_on)
    P       = _get_projector(H; method=method, fermi=fermi, Nchebychev=Nchebychev,
                              maxdim=maxdim, cutoff=cutoff, Nel=Nel, run_on=run_on)
    Q_cpu   = +(MPO(H.sites, "Id"), -1.0 * P; cutoff=1e-15)
    P_dev   = to_device(P, backend)
    Q_dev   = to_device(Q_cpu, backend)
    l_bits  = l === nothing ? div(H.L, 2) : l
    L_chain = 2^l_bits
    sz_dev  = to_device(get_diagonal_mpo(H.L, H.sites, sz_func), backend)

    if quenched
        sinX_dev = to_device(get_sinx_op(H.L, H.sites, L_chain, Λ, xfunc), backend)
        cosX_dev = to_device(get_cosx_op(H.L, H.sites, L_chain, Λ, xfunc), backend)
        T1s = apply(P_dev, apply(sinX_dev, Q_dev; maxdim, cutoff); maxdim, cutoff)
        T2s = apply(Q_dev, apply(sinX_dev, P_dev; maxdim, cutoff); maxdim, cutoff)
        W1  = apply(sz_dev, +(T1s, T2s; maxdim, cutoff); maxdim, cutoff)
        T1c = apply(P_dev, apply(cosX_dev, Q_dev; maxdim, cutoff); maxdim, cutoff)
        T2c = apply(Q_dev, apply(cosX_dev, P_dev; maxdim, cutoff); maxdim, cutoff)
        W2  = apply(sz_dev, +(T1c, T2c; maxdim, cutoff); maxdim, cutoff)

        calculate_winding = alpha -> begin
            α = to_device(binary_to_MPS(alpha - 1, H.L, H.sites), backend)
            x = xfunc(alpha - 1, L_chain)
            Λ * (cos(x / Λ) * inner(α', W1, α) - sin(x / Λ) * inner(α', W2, α))
        end

    else
        x_dev = to_device(get_diagonal_mpo(H.L, H.sites, i -> xfunc(i - 1, L_chain)), backend)
        T1    = apply(P_dev, apply(x_dev, Q_dev; maxdim, cutoff); maxdim, cutoff)
        T2    = apply(Q_dev, apply(x_dev, P_dev; maxdim, cutoff); maxdim, cutoff)
        W_op  = apply(sz_dev, +(T1, T2; maxdim, cutoff); maxdim, cutoff)

        calculate_winding = alpha -> begin
            α = to_device(binary_to_MPS(alpha - 1, H.L, H.sites), backend)
            inner(α', W_op, α)
        end
    end

    return calculate_winding
end


# ============================================================
# 2D — quenched (periodic) position operator builders
# ============================================================
#
# xfunc(i::Int, L_chain::Int) -> Float64  (i is 0-indexed)
# quenching angle = xfunc(i, L_chain) / Λ
#
# Inside get_diagonal_mpo the function receives 1-indexed site values;
# these builders convert to 0-indexed before calling xfunc / yfunc.

"""
    get_sinx_op(L, sites, L_chain, Λ, xfunc) -> MPO

Diagonal MPO for `sin(xfunc(i, L_chain) / Λ)`.
`xfunc(i, L_chain)` receives a 0-indexed site number and returns the raw
x-coordinate; dividing by `Λ` gives the quenching angle.
"""
function get_sinx_op(L, sites, L_chain, Λ, xfunc)
    f(i) = sin(xfunc(i - 1, L_chain) / Λ)
    return get_diagonal_mpo(L, sites, f)
end


"""
    get_cosx_op(L, sites, L_chain, Λ, xfunc) -> MPO

Diagonal MPO for `cos(xfunc(i, L_chain) / Λ)`.  See `get_sinx_op`.
"""
function get_cosx_op(L, sites, L_chain, Λ, xfunc)
    f(i) = cos(xfunc(i - 1, L_chain) / Λ)
    return get_diagonal_mpo(L, sites, f)
end


"""
    get_siny_op(L, sites, L_chain, Λ, yfunc) -> MPO

Diagonal MPO for `sin(yfunc(i, L_chain) / Λ)`.
"""
function get_siny_op(L, sites, L_chain, Λ, yfunc)
    f(i) = sin(yfunc(i - 1, L_chain) / Λ)
    return get_diagonal_mpo(L, sites, f)
end


"""
    get_cosy_op(L, sites, L_chain, Λ, yfunc) -> MPO

Diagonal MPO for `cos(yfunc(i, L_chain) / Λ)`.  See `get_siny_op`.
"""
function get_cosy_op(L, sites, L_chain, Λ, yfunc)
    f(i) = cos(yfunc(i - 1, L_chain) / Λ)
    return get_diagonal_mpo(L, sites, f)
end


# ============================================================
# 2D Chern marker from a pre-computed projector
# ============================================================

"""
    get_C_op_MPO_from_P(P, L, sites, xfunc, yfunc;
                        l=nothing, Λ=10, maxdim=500, cutoff=1e-8,
                        quenched=true) -> Function

Build the real-space Chern marker and return a closure `calculate_chern_number(α)`
that evaluates it at any lattice site `α` (1-indexed).

# Coordinate functions

Both `xfunc(i, L_chain)` and `yfunc(i, L_chain)` must accept a **0-indexed**
site number `i ∈ 0…2^L−1` and the number of sites per row `L_chain`.
They return the raw coordinate (not yet quenched).  Examples for a square lattice:

    xfunc(i, L_chain) = Float64(mod(i, L_chain))   # x ∈ 0…L_chain-1
    yfunc(i, L_chain) = Float64(div(i, L_chain))   # y ∈ 0…L_chain-1

For the SSH model (both sublattice sites at the same unit-cell position):

    xfunc(i, L_chain) = Float64(div(i, 2))

# Quenched mode (`quenched=true`, default)

Position operators are quenched: `sin(xfunc/Λ)`, `cos(xfunc/Λ)`, and similarly
for y.  The Chern marker is computed via a **4-term trig decomposition** that
pre-computes 4 α-independent MPOs (C1–C4) and combines them in the closure:

    C(α) = 2πi Λ² [ cos_xα cos_yα ⟨α|C1|α⟩ + sin_xα sin_yα ⟨α|C2|α⟩
                   − cos_xα sin_yα ⟨α|C3|α⟩ − sin_xα cos_yα ⟨α|C4|α⟩ ]

This exploits the identity sin(θ_r − θ_α) = sinθ_r cosθ_α − cosθ_r sinθ_α to
avoid building a centred-at-α MPO for every site, reducing the cost from
O(2^L) MPO products to 4 products computed once.

The Λ² prefactor restores physical units: since sin(x/Λ) ≈ x/Λ, using sin as
the position operator implicitly divides each coordinate by Λ, so multiplying
by Λ² recovers the true Chern marker.

# Flat mode (`quenched=false`)

Position operators use xfunc/yfunc directly (no sin/cos wrapping):

    C_op = 2πi · (Q x P y Q − P x Q y P)

The Chern marker is evaluated for each site directly from `C_op`.  This mode
does **not** centre the position operator at each reference site, so it is
most accurate for OBC systems or bulk-averaged quantities.

# Arguments
- `P`         : density matrix MPO
- `L`         : total number of quantics bits (system has `2^L` sites)
- `sites`     : ITensor site index list
- `xfunc(i, L_chain)`, `yfunc(i, L_chain)` : coordinate functions (0-indexed `i`)
- `l`         : qubits per spatial direction; inferred as `L ÷ 2` if `nothing`
- `Λ`         : quenching period (angle = coord/Λ)
- `maxdim`    : MPO bond dimension during all multiplications
- `cutoff`    : truncation threshold during all multiplications and subtractions
- `quenched`  : `true` = 4-term sin/cos decomposition; `false` = flat operators

# Returns
`calculate_chern_number(α::Int) -> ComplexF64`
where `α` is 1-indexed.  Take `real(·)` for the Chern number density.

# Example — quenched square lattice
```julia
L_chain  = 2^(L ÷ 2)
xfunc(i, L_chain) = Float64(mod(i, L_chain))
yfunc(i, L_chain) = Float64(div(i, L_chain))
C_at  = get_C_op_MPO_from_P(P, L, sites, xfunc, yfunc; Λ=L_chain, maxdim=100)
chern = real(sum(C_at(α) for α in 1:2^L)) / L_chain^2
```
"""
function get_C_op_MPO_from_P(P, L, sites, xfunc, yfunc;
                              l               = nothing,
                              Λ::Real         = 10,
                              maxdim::Int     = 500,
                              cutoff::Float64 = 1e-8,
                              quenched::Bool  = true,
                              run_on::Symbol  = :cpu)
    backend = _resolve_backend(run_on)
    l_bits  = l === nothing ? div(L, 2) : l
    L_chain = 2^l_bits

    P_dev = to_device(P, backend)
    Q_dev = to_device(+(MPO(sites, "Id"), -1.0 * P; cutoff=1e-15), backend)

    if quenched
        sinX_dev = to_device(get_sinx_op(L, sites, L_chain, Λ, xfunc), backend)
        cosX_dev = to_device(get_cosx_op(L, sites, L_chain, Λ, xfunc), backend)
        sinY_dev = to_device(get_siny_op(L, sites, L_chain, Λ, yfunc), backend)
        cosY_dev = to_device(get_cosy_op(L, sites, L_chain, Λ, yfunc), backend)

        sinY_P = apply(sinY_dev, P_dev;  maxdim=maxdim, cutoff=cutoff)
        cosY_P = apply(cosY_dev, P_dev;  maxdim=maxdim, cutoff=cutoff)
        P_sinX = apply(P_dev,  sinX_dev; maxdim=maxdim, cutoff=cutoff)
        P_cosX = apply(P_dev,  cosX_dev; maxdim=maxdim, cutoff=cutoff)
        sinY_Q = apply(sinY_dev, Q_dev;  maxdim=maxdim, cutoff=cutoff)
        cosY_Q = apply(cosY_dev, Q_dev;  maxdim=maxdim, cutoff=cutoff)
        Q_sinX = apply(Q_dev,  sinX_dev; maxdim=maxdim, cutoff=cutoff)
        Q_cosX = apply(Q_dev,  cosX_dev; maxdim=maxdim, cutoff=cutoff)
        println("Quenched operator products done")

        C1 = apply(Q_sinX, P_dev;  maxdim=maxdim, cutoff=cutoff)
        C1 = apply(C1,     sinY_Q; maxdim=maxdim, cutoff=cutoff)
        c1 = apply(P_sinX, Q_dev;  maxdim=maxdim, cutoff=cutoff)
        c1 = apply(c1,     sinY_P; maxdim=maxdim, cutoff=cutoff)
        C1 = +(C1, -1.0 * c1; maxdim=maxdim, cutoff=cutoff)
        println("C1 done")

        C2 = apply(Q_cosX, P_dev;  maxdim=maxdim, cutoff=cutoff)
        C2 = apply(C2,     cosY_Q; maxdim=maxdim, cutoff=cutoff)
        c2 = apply(P_cosX, Q_dev;  maxdim=maxdim, cutoff=cutoff)
        c2 = apply(c2,     cosY_P; maxdim=maxdim, cutoff=cutoff)
        C2 = +(C2, -1.0 * c2; maxdim=maxdim, cutoff=cutoff)
        println("C2 done")

        C3 = apply(Q_sinX, P_dev;  maxdim=maxdim, cutoff=cutoff)
        C3 = apply(C3,     cosY_Q; maxdim=maxdim, cutoff=cutoff)
        c3 = apply(P_sinX, Q_dev;  maxdim=maxdim, cutoff=cutoff)
        c3 = apply(c3,     cosY_P; maxdim=maxdim, cutoff=cutoff)
        C3 = +(C3, -1.0 * c3; maxdim=maxdim, cutoff=cutoff)
        println("C3 done")

        C4 = apply(Q_cosX, P_dev;  maxdim=maxdim, cutoff=cutoff)
        C4 = apply(C4,     sinY_Q; maxdim=maxdim, cutoff=cutoff)
        c4 = apply(P_cosX, Q_dev;  maxdim=maxdim, cutoff=cutoff)
        c4 = apply(c4,     sinY_P; maxdim=maxdim, cutoff=cutoff)
        C4 = +(C4, -1.0 * c4; maxdim=maxdim, cutoff=cutoff)
        println("C4 done")

        calculate_chern_number = alpha -> begin
            α      = to_device(binary_to_MPS(alpha - 1, L, sites), backend)
            x      = xfunc(alpha - 1, L_chain)
            y      = yfunc(alpha - 1, L_chain)
            cos_x, sin_x = cos(x / Λ), sin(x / Λ)
            cos_y, sin_y = cos(y / Λ), sin(y / Λ)
            ch  =  cos_x * cos_y * inner(α', C1, α)
            ch +=  sin_x * sin_y * inner(α', C2, α)
            ch -=  cos_x * sin_y * inner(α', C3, α)
            ch -=  sin_x * cos_y * inner(α', C4, α)
            ch * 2im * π * Λ^2
        end

    else
        x_dev = to_device(get_diagonal_mpo(L, sites, i -> xfunc(i - 1, L_chain)), backend)
        y_dev = to_device(get_diagonal_mpo(L, sites, i -> yfunc(i - 1, L_chain)), backend)

        T1    = apply(Q_dev, apply(x_dev, apply(P_dev, apply(y_dev, Q_dev;
                      maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff);
                      maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff)
        T2    = apply(P_dev, apply(x_dev, apply(Q_dev, apply(y_dev, P_dev;
                      maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff);
                      maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff)
        C_op = 2im * π * +(T1, -1.0 * T2; maxdim=maxdim, cutoff=cutoff)
        ITensorMPS.truncate!(C_op; maxdim=maxdim, cutoff=cutoff)

        calculate_chern_number = alpha -> begin
            α = to_device(binary_to_MPS(alpha - 1, L, sites), backend)
            inner(α', C_op, α)
        end
    end

    return calculate_chern_number
end


# ============================================================
# 2D Chern marker from TBHamiltonian
# ============================================================

"""
    get_C(H::TBHamiltonian, xfunc, yfunc;
          method=:KPM, fermi=0.0, l=nothing, Λ=10,
          Nchebychev=300, maxdim=500, cutoff=1e-8,
          Nel=nothing, quenched=true) -> Function

High-level wrapper: compute the ground-state projector via `method` and
return the Chern marker closure from `get_C_op_MPO_from_P`.

Reuses `H._tn_cache` or `H._density_cache` when available.  `maxdim` and
`cutoff` are forwarded uniformly to the projector computation and to all
MPO multiplications in the Chern marker assembly.

See `get_C_op_MPO_from_P` for the full documentation of `xfunc`, `yfunc`,
`l`, `Λ`, `maxdim`, `cutoff`, and `quenched`.

# Returns
`calculate_chern_number(α::Int) -> ComplexF64` (α is 1-indexed)
"""
function get_C(H::TBHamiltonian, xfunc, yfunc;
               method::Symbol   = :KPM,
               fermi::Real      = 0.0,
               l                = nothing,
               Λ::Real          = 10,
               Nchebychev::Int  = 300,
               maxdim::Int      = 500,
               cutoff::Float64  = 1e-8,
               Nel              = nothing,
               quenched::Bool   = true,
               run_on::Symbol   = :cpu)
    P = _get_projector(H; method=method, fermi=fermi, Nchebychev=Nchebychev,
                       maxdim=maxdim, cutoff=cutoff, Nel=Nel, run_on=run_on)
    return get_C_op_MPO_from_P(P, H.L, H.sites, xfunc, yfunc;
                                l=l, Λ=Λ, maxdim=maxdim, cutoff=cutoff,
                                quenched=quenched, run_on=run_on)
end


# ============================================================
# Thouless charge pump — 1D adiabatic invariant
# ============================================================
#
# Computes the pumped charge per cycle (= Chern number) via
#
#   C = (i/2π) ∫₀ᵀ Tr[P(t) [Ṗ(t), x̂]] dt
#
# This avoids building the time-ordered evolution operator U(t)
# explicitly.  P(t) is supplied as an array of MPOs computed at
# Nt evenly-spaced time steps; Ṗ(t) is estimated by central
# finite differences (one-sided at the endpoints).
#
# Building blocks
# ---------------
#   get_pump_xop       — position operator MPO (flat or quenched)
#   berry_curvature_integrand — Tr[P [Ṗ, x]] at one time step
#   thouless_pump      — time-integrate over a P_array
#   get_thouless_pump  — high-level: builds P(t) then calls thouless_pump


"""
    get_pump_xop(L, sites, xfunc; quenched=false, Λ=nothing) -> MPO

Diagonal position operator MPO for the Thouless pump formula.

`xfunc(i, N)` accepts a **0-indexed** site index `i ∈ 0…N-1` and the chain
length `N = 2^L`, and returns the raw coordinate.  For a 1-indexed chain:
`xfunc(i, N) = Float64(i + 1)`.

- `quenched=false` (default): diagonal entries are `xfunc(i, N)` directly.
- `quenched=true`: entries are `Λ * sin(xfunc(i, N) / Λ)`, which smooths
  the discontinuity at PBC at the cost of a `Λ` prefactor.  `Λ` defaults to
  `N` (one full period), giving `sin(x/N) * N ≈ x` for `x ≪ N`.
"""
function get_pump_xop(L::Int, sites::Vector{<:Index}, xfunc;
                      quenched::Bool = false,
                      Λ::Real        = -1.0)
    N     = 2^L
    Λ_val = Λ < 0 ? Float64(N) : Λ
    if quenched
        return Λ_val * get_sinx_op(L, sites, N, Λ_val, xfunc)
    else
        return get_diagonal_mpo(L, sites, i -> xfunc(i - 1, N))
    end
end


"""
    thouless_pump(P_array, dt, x_op, sites; r_center, maxdim, cutoff,
                  verbose, return_trajectory) -> Float64 or (Float64, Vector{Float64})

Compute the Thouless pump invariant (Chern number) using the local M1Q marker:

    M1Q(r, t) = ⟨r| P(t) U†(t) x̂ U(t) P(t) |r⟩

where U(t) is the adiabatic evolution operator generated by h(t) = [∂P/∂t, P(t)],
propagated with a second-order Taylor step:

    U(k) = (I + h(k−1) dt + h(k−1)² dt²/2) U(k−1),  U(0) = I

The invariant is C = M1Q(r, T) − M1Q(r, 0), evaluated at site `r_center`.
Finite differences for ∂P/∂t use central differences (one-sided at endpoints).

**Arguments**
- `P_array`           : `Vector{MPO}` of length `Nt`, one instantaneous projector per step.
- `dt`                : time step (`T / Nt`).
- `x_op`              : position operator MPO from `get_pump_xop`.
- `sites`             : physical site indices (for identity MPO and `matrix_checker`).
- `r_center`          : 0-indexed bulk site at which to evaluate M1Q.
- `maxdim`            : max bond dimension for all MPO operations.
- `cutoff`            : SVD truncation threshold.
- `verbose`           : print M1Q(0), M1Q(T), and bond dimension at each step.
- `return_trajectory` : if `true`, return `(C, M1Q_traj)` where `M1Q_traj` is a
                        `Vector{Float64}` of length `Nt+1` with M1Q at each step
                        (index 1 = t=0, index k+1 = t=k·dt).  Default `false`.
"""
function thouless_pump(P_array::Vector{<:MPO}, dt::Real, x_op::MPO,
                       sites::Vector{<:Index};
                       r_center::Int,
                       maxdim::Int             = 100,
                       cutoff::Float64         = 1e-8,
                       verbose::Bool           = false,
                       return_trajectory::Bool = false,
                       run_on::Symbol          = :cpu)
    backend = _resolve_backend(run_on)
    Nt    = length(P_array)
    P_dev = [to_device(P, backend) for P in P_array]
    x_dev = to_device(x_op, backend)
    I_dev = to_device(MPO(sites, "Id"), backend)

    PxP_0 = apply(apply(P_dev[1], x_dev; maxdim, cutoff), P_dev[1]; maxdim, cutoff)
    M1Q_0 = real(matrix_checker(PxP_0, sites, r_center, r_center; run_on=run_on))
    verbose && println("M1Q(0) = $(round(M1Q_0; digits=6))")

    M1Q_traj = return_trajectory ? Float64[M1Q_0] : Float64[]

    U = deepcopy(I_dev)
    for k in 1:Nt
        if k == 1
            Pdot = (1.0 / dt) * +(P_dev[2],    -1.0 * P_dev[1];    maxdim, cutoff)
        elseif k == Nt
            Pdot = (1.0 / dt) * +(P_dev[Nt],   -1.0 * P_dev[Nt-1]; maxdim, cutoff)
        else
            Pdot = (0.5 / dt) * +(P_dev[k+1],  -1.0 * P_dev[k-1]; maxdim, cutoff)
        end
        ITensorMPS.truncate!(Pdot; maxdim, cutoff)

        h_k  = +(apply(Pdot, P_dev[k]; maxdim, cutoff),
                 -1.0 * apply(P_dev[k], Pdot; maxdim, cutoff); maxdim, cutoff)
        ITensorMPS.truncate!(h_k; maxdim, cutoff)

        h_sq = apply(h_k, h_k; maxdim, cutoff)
        dU   = +(+(I_dev, dt * h_k; maxdim, cutoff),
                 (dt^2 / 2) * h_sq; maxdim, cutoff)
        ITensorMPS.truncate!(dU; maxdim, cutoff)
        U    = apply(dU, U; maxdim, cutoff)
        ITensorMPS.truncate!(U; maxdim, cutoff)

        verbose && println("  step $k/$Nt  maxlinkdim(U) = $(ITensorMPS.maxlinkdim(U))")

        if return_trajectory
            Ud_k    = dag(swapprime(U, 0, 1))
            UxU_k   = apply(apply(Ud_k, x_dev; maxdim, cutoff), U; maxdim, cutoff)
            PUxUP_k = apply(apply(P_dev[k], UxU_k; maxdim, cutoff), P_dev[k]; maxdim, cutoff)
            push!(M1Q_traj, real(matrix_checker(PUxUP_k, sites, r_center, r_center; run_on=run_on)))
        end
    end

    if return_trajectory
        M1Q_T = M1Q_traj[end]
    else
        Ud    = dag(swapprime(U, 0, 1))
        UxU   = apply(apply(Ud, x_dev; maxdim, cutoff), U; maxdim, cutoff)
        PUxUP = apply(apply(P_dev[end], UxU; maxdim, cutoff), P_dev[end]; maxdim, cutoff)
        M1Q_T = real(matrix_checker(PUxUP, sites, r_center, r_center; run_on=run_on))
    end
    verbose && println("M1Q(T) = $(round(M1Q_T; digits=6))")

    C = M1Q_T - M1Q_0
    return return_trajectory ? (C, M1Q_traj) : C
end


"""
    get_thouless_pump(H_of_t, Nt, T, xfunc;
                      P_method=:mcweeny, Nchebychev=200,
                      maxdim=100, cutoff=1e-8,
                      quenched=false, Λ=-1.0,
                      r_center=nothing, Nel=nothing, verbose=false) -> Float64

High-level Thouless pump: build `P(t_k)` for `k = 0…Nt-1` via `P_method`,
then compute the M1Q invariant C = M1Q(T) − M1Q(0).

**Arguments**
- `H_of_t`   : `t -> TBHamiltonian` — all calls must share the same site indices
               (pass `ref_sites` to `get_Hamiltonian` inside the factory).
- `Nt`       : number of time steps.
- `T`        : period of the pump cycle.
- `xfunc`    : coordinate function `(i, N) -> Float64`, 0-indexed.
- `P_method` : `:mcweeny`, `:sp2`, or `:KPM`.
- `r_center` : 0-indexed bulk site for M1Q evaluation; defaults to `N ÷ 2`.
- `quenched` : `false` = flat x̂; `true` = sin-quenched (removes PBC discontinuity).
- `verbose`  : print progress.
"""
function get_thouless_pump(H_of_t::Function, Nt::Int, T::Real, xfunc;
                           P_method::Symbol             = :mcweeny,
                           fermi::Real                  = 0.0,
                           Nchebychev::Int              = 200,
                           maxdim::Int                  = 100,
                           cutoff::Float64              = 1e-8,
                           quenched::Bool               = false,
                           Λ::Real                      = -1.0,
                           Nel                          = nothing,
                           r_center::Union{Nothing,Int} = nothing,
                           verbose::Bool                = false,
                           run_on::Symbol               = :cpu)
    dt    = T / Nt
    H0    = H_of_t(0.0)
    sites = H0.sites
    x_op  = get_pump_xop(H0.L, H0.sites, xfunc; quenched=quenched, Λ=Λ)
    rc    = isnothing(r_center) ? (2^H0.L) ÷ 2 : r_center

    P_array = MPO[]
    for k in 0:(Nt - 1)
        t_k = k * dt
        verbose && println("Building P(t=$(round(t_k; digits=4)))  [$(k+1)/$Nt]...")
        H_k = H_of_t(t_k)
        P_k = _get_projector(H_k; method=P_method, fermi=fermi,
                              Nchebychev=Nchebychev, maxdim=maxdim,
                              cutoff=cutoff, Nel=Nel, run_on=run_on)
        push!(P_array, P_k)
    end

    verbose && println("Computing M1Q invariant (r_center=$rc)...")
    return thouless_pump(P_array, dt, x_op, sites;
                         r_center=rc, maxdim=maxdim, cutoff=cutoff,
                         verbose=verbose, run_on=run_on)
end
