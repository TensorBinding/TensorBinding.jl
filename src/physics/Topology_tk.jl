# Topology_tk.jl — topological invariants via KPM and MPO methods
#
# Implements real-space Chern markers and winding numbers for arbitrary
# tight-binding systems encoded in the quantics representation.
#
# The key observables are:
#
#   Chern marker (2D):   C(r) = 4π Im⟨r| P [x̂, P] [ŷ, P] |r⟩
#   Winding number (1D): W(r) = ⟨r| σ_z (P x̂ Q + Q x̂ P) |r⟩
#
# where P is the ground-state projector, Q = I − P, and [A,B] = AB − BA.
# Integrating C(r) or W(r) over the bulk gives the integer invariant.
#
# == Position functions ==
#
# xfunc/yfunc are two-argument scalar functions:
#   xfunc(i::Int, L_chain::Int) -> Float64
# where i is a 0-indexed PHYSICAL site number and L_chain = 2^(L÷2).
#
# For plain (non-sublattice) models, i ∈ {0, …, 2^L − 1}.
# For sublattice models (n_sub atoms per UC), i ∈ {0, …, n_sub·2^L − 1}.
# Position MPOs are built on the L position qubits only and then extended
# to the full site chain via postpend_op(⋅, sublattice_s, I).
#
# get_C and get_W both accept xfunc=nothing / yfunc=nothing, in which case
# they auto-derive from H.geometry_uc (sublattice models) or H.geometry:
#   xfunc(i, _) = geom(i+1)[1],   yfunc(i, _) = geom(i+1)[2]
# geometry_uc returns the same Bravais UC position for all sublattice atoms
# in the same unit cell, so the position operator is constant across sublattice.
#
# == Quenching ==
#
#   quenched=true (default):
#       Position operators are quenched via sin/cos(xfunc/Λ), removing PBC
#       discontinuities.  The Chern marker uses a 4-term trig decomposition
#       (4 α-independent MPO products, combined in the closure).
#       Λ² prefactor restores physical units (sin(x/Λ) ≈ x/Λ).
#
#   quenched=false:
#       Position operators use xfunc/yfunc directly (no sin/cos).
#       Formula: C = 2πi (Q x P y Q − P x Q y P).  Best for OBC or averages.
#
# == Projector methods ==
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
                         Nel              = nothing)
    if method == :KPM
        if H._tn_cache !== nothing
            Tn_list = H._tn_cache
            Ncheb   = H._tn_Ncheb
        else
            Tn_list, _, _ = KPM_Tn(H, Nchebychev; maxdim=maxdim, cutoff=cutoff)
            Ncheb = Nchebychev
        end
        fermi_resc = (fermi - H.center) / H.scale
        return get_density_from_Tn(Tn_list, Ncheb; fermi=fermi_resc, maxdim=maxdim)
    elseif method == :mcweeny
        H._density_cache !== nothing && return H._density_cache
        return mcweeny_purify(H; ϵF=fermi, maxdim=maxdim, cutoff=cutoff)
    elseif method == :sp2
        H._density_cache !== nothing && return H._density_cache
        Nel_val = Nel === nothing ? H.N ÷ 2 : Int(Nel)
        return sp2_purify(H; Nel=Nel_val, maxdim=maxdim, cutoff=cutoff)
    else
        error("Unknown method: :$method. Choose :KPM, :mcweeny, or :sp2")
    end
end


# ============================================================
# 1D winding number
# ============================================================

"""
    get_W(H::TBHamiltonian, xfunc=nothing;
          method=:KPM, fermi=0.0, Nchebychev=300,
          maxdim=15, cutoff=1e-8, Nel=nothing,
          quenched=true, l=nothing, Λ=10) -> Function

Compute the real-space winding number and return a closure
`calculate_winding(α::Int) -> ComplexF64` that evaluates the local winding
number density at any site `α` (1-indexed).

The winding number operator is

    W_op = σ_z · (P x̂ Q + Q x̂ P)

where `P` is the occupied-band projector, `Q = I − P`, `x̂` is the position
operator, and `σ_z` is the sublattice chirality (A → +1, B → -1).

`H` must have a 2-component sublattice index (`H.sublattice_s` with dim 2).
`σ_z` is built automatically as the diagonal operator `diag(+1, −1)` on that
index tensored with identity on the position qubits.

# Coordinate function

`xfunc(i, L_chain) -> Float64` accepts a **0-indexed** physical site number
and returns the raw x-coordinate.  Defaults to `nothing`, in which case it is
auto-derived from `H.geometry_uc` (preferred) or `H.geometry`:
`xfunc(i, _) = geom(i+1)[1]`.  Because `geometry_uc` returns the same
Bravais position for both sublattice atoms in a unit cell, this correctly
assigns the same x-coordinate to both A and B sites of each UC.

# Quenched vs flat mode

- `quenched=true` (default): pre-computes two α-independent MPOs W1 and W2.
      W(α) = Λ · ⟨α| cos(x_α/Λ) W1 − sin(x_α/Λ) W2 |α⟩
- `quenched=false`: builds a global position operator and returns a closure
  over the resulting W_op MPO.

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
`calculate_winding(uc::Int) -> ComplexF64` where `uc` is a 1-indexed unit
cell number (1 … 2^L).  The value is the sum of the winding marker over both
sublattice atoms (A and B) within that unit cell.
"""
function get_W(H::TBHamiltonian, xfunc=nothing;
               method::Symbol   = :KPM,
               fermi::Real      = 0.0,
               Nchebychev::Int  = 300,
               maxdim::Int      = 15,
               cutoff::Float64  = 1e-8,
               Nel              = nothing,
               quenched::Bool   = true,
               l                = nothing,
               Λ::Real          = 10)
    H.sublattice_s === nothing || dim(H.sublattice_s) == 2 ||
        error("get_W requires a 2-component sublattice index (dim=2); got dim=$(dim(H.sublattice_s)).")
    H.sublattice_s !== nothing ||
        error("get_W requires H.sublattice_s to be set (n_sub=2 sublattice model).")

    if xfunc === nothing
        geom = H.geometry_uc !== nothing ? H.geometry_uc :
               H.geometry   !== nothing ? H.geometry   :
               error("H has no geometry function; provide xfunc explicitly.")
        xfunc = (i, _) -> geom(i + 1)[1]
    end

    pos_sites = _pos_sites(H)
    sub_s     = H.sublattice_s
    I_mat     = Matrix{Float64}(LinearAlgebra.I, 2, 2)
    σ_z_mat   = Float64[1 0; 0 -1]

    # σ_z on sublattice tensored with identity on position qubits
    sz = postpend_op(MPO(pos_sites, "Id"), sub_s, σ_z_mat)

    # xfunc for position MPOs (2^L UC positions, 0-indexed)
    xfunc_pos = (i, Lc) -> xfunc(i * 2, Lc)

    P       = _get_projector(H; method=method, fermi=fermi, Nchebychev=Nchebychev,
                              maxdim=maxdim, cutoff=cutoff, Nel=Nel)
    Q       = MPO(H.sites, "Id") - P
    l_bits  = l === nothing ? div(H.L, 2) : l
    L_chain = 2^l_bits

    all_sites = collect(H.sites)
    make_alpha_mps = alpha -> begin
        n_cell   = (alpha - 1) ÷ 2
        sub      = (alpha - 1) % 2 + 1
        pos_bits = [((n_cell >> (H.L - i)) & 1) + 1 for i in 1:H.L]
        _product_state_mps(all_sites, [pos_bits; sub])
    end

    if quenched
        # W1 = σ_z (P sinX Q + Q sinX P),  W2 = σ_z (P cosX Q + Q cosX P)
        sinX_op = postpend_op(get_sinx_op(H.L, pos_sites, L_chain, Λ, xfunc_pos), sub_s, I_mat)
        cosX_op = postpend_op(get_cosx_op(H.L, pos_sites, L_chain, Λ, xfunc_pos), sub_s, I_mat)
        T1s = apply(P, apply(sinX_op, Q; maxdim, cutoff); maxdim, cutoff)
        T2s = apply(Q, apply(sinX_op, P; maxdim, cutoff); maxdim, cutoff)
        W1  = apply(sz, +(T1s, T2s; maxdim, cutoff); maxdim, cutoff)
        T1c = apply(P, apply(cosX_op, Q; maxdim, cutoff); maxdim, cutoff)
        T2c = apply(Q, apply(cosX_op, P; maxdim, cutoff); maxdim, cutoff)
        W2  = apply(sz, +(T1c, T2c; maxdim, cutoff); maxdim, cutoff)

        calculate_winding = uc -> begin
            sum(sub -> begin
                alpha = (uc - 1) * 2 + sub
                α = make_alpha_mps(alpha)
                x = xfunc(alpha - 1, L_chain)
                Λ * (cos(x / Λ) * inner(α', W1, α) - sin(x / Λ) * inner(α', W2, α))
            end, 1:2)
        end

    else
        x_op_p = get_diagonal_mpo(H.L, pos_sites, i -> xfunc_pos(i - 1, L_chain))
        x_op   = postpend_op(x_op_p, sub_s, I_mat)
        T1     = apply(P, apply(x_op, Q; maxdim, cutoff); maxdim, cutoff)
        T2     = apply(Q, apply(x_op, P; maxdim, cutoff); maxdim, cutoff)
        W_op   = apply(sz, +(T1, T2; maxdim, cutoff); maxdim, cutoff)

        calculate_winding = uc -> begin
            sum(sub -> begin
                alpha = (uc - 1) * 2 + sub
                α = make_alpha_mps(alpha)
                inner(α', W_op, α)
            end, 1:2)
        end
    end

    return calculate_winding
end


# ============================================================
# 2D — quenched (periodic) position operator builders
# ============================================================
#
# These are low-level helpers called by get_C_op_MPO_from_P.
# `sites` must be the L position-qubit indices only (not the full H.sites
# for sublattice models); callers extend the result with postpend_op.
# xfunc(i, L_chain) receives a 0-indexed UC number (0 … 2^L−1) and returns
# the raw coordinate; get_diagonal_mpo receives it 1-indexed and converts.

"""
    get_sinx_op(L, sites, L_chain, Λ, xfunc) -> MPO

Diagonal MPO for `sin(xfunc(i, L_chain) / Λ)` over the `L`-qubit position
chain given by `sites`.  `xfunc(i, L_chain)` receives a 0-indexed site/UC
number and returns the raw x-coordinate.
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

Diagonal MPO for `sin(yfunc(i, L_chain) / Λ)`.  See `get_sinx_op`.
"""
function get_siny_op(L, sites, L_chain, Λ, yfunc)
    f(i) = sin(yfunc(i - 1, L_chain) / Λ)
    return get_diagonal_mpo(L, sites, f)
end


"""
    get_cosy_op(L, sites, L_chain, Λ, yfunc) -> MPO

Diagonal MPO for `cos(yfunc(i, L_chain) / Λ)`.  See `get_sinx_op`.
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
that evaluates it at any physical site `α` (1-indexed).

# Coordinate functions

`xfunc(i, L_chain)` and `yfunc(i, L_chain)` accept a **0-indexed** physical
site number `i` and return the raw coordinate (not yet quenched).

- Plain models (`length(sites) == L`): `i ∈ 0…2^L−1`.  Examples:
      xfunc(i, L_chain) = Float64(mod(i, L_chain))   # square x
      yfunc(i, L_chain) = Float64(div(i, L_chain))   # square y
- Sublattice models (`length(sites) == L+1`, last index has dim `n_sub`):
  `i ∈ 0…n_sub·2^L−1`.  The function should return the same coordinate for
  all `n_sub` atoms within the same unit cell — typically the Bravais position.
  The auto-derived functions from `H.geometry_uc` satisfy this automatically.
  Internally, position MPOs are built on the L position qubits only and extended
  to the full chain via `postpend_op(⋅, sub_s, I)`.

# Quenched mode (`quenched=true`, default)

Position operators are quenched: `sin(xfunc/Λ)`, `cos(xfunc/Λ)`, and similarly
for y.  The Chern marker uses a **4-term trig decomposition** that pre-computes
4 α-independent MPOs (C1–C4) and combines them per site in the closure:

    C(α) = 2πi Λ² [ cos_xα cos_yα ⟨α|C1|α⟩ + sin_xα sin_yα ⟨α|C2|α⟩
                   − cos_xα sin_yα ⟨α|C3|α⟩ − sin_xα cos_yα ⟨α|C4|α⟩ ]

Exploits `sin(θ_r − θ_α) = sinθ_r cosθ_α − cosθ_r sinθ_α` to reduce cost from
O(N) MPO products to 4 products computed once.  Λ² restores physical units.

# Flat mode (`quenched=false`)

Position operators use xfunc/yfunc directly:

    C_op = 2πi · (Q x P y Q − P x Q y P)

Most accurate for OBC systems or bulk-averaged quantities (no per-site centring).

# Arguments
- `P`       : ground-state projector MPO (over `sites`)
- `L`       : number of position qubits; system has `2^L` unit cells
- `sites`   : full ITensor site index list (`length == L` or `L+1` for sublattice)
- `xfunc`, `yfunc` : coordinate functions; `i` is 0-indexed over all physical sites
- `l`       : qubits per spatial direction; inferred as `L ÷ 2` if `nothing`
- `Λ`       : quenching period (quenching angle = coord / Λ)
- `maxdim`  : MPO bond dimension during all multiplications
- `cutoff`  : truncation threshold during all multiplications and subtractions
- `quenched`: `true` = 4-term sin/cos decomposition; `false` = flat operators

# Returns
`calculate_chern_number(uc::Int) -> ComplexF64` where `uc` is a 1-indexed
unit cell number (1 … 2^L).  For sublattice models the value is the sum of
the Chern marker over all `n_sub` atoms within that unit cell.
Take `real(·)` for the Chern number density.

# Example — quenched square lattice
```julia
L_chain = 2^(L ÷ 2)
xfunc(i, _) = Float64(mod(i, L_chain))
yfunc(i, _) = Float64(div(i, L_chain))
C_at  = get_C_op_MPO_from_P(P, L, sites, xfunc, yfunc; Λ=L_chain, maxdim=100)
chern = real(sum(C_at(α) for α in 1:2^L)) / L_chain^2
```

# Example — honeycomb via get_C (auto-derived geometry)
```julia
C_at  = get_C(H)   # xfunc/yfunc from H.geometry_uc automatically
N_sub = 2 * H.N
chern = real(sum(C_at(α) for α in 1:N_sub)) / (2^(H.L ÷ 2))^2
```
"""
function get_C_op_MPO_from_P(P, L, sites, xfunc, yfunc;
                              l               = nothing,
                              Λ::Real         = 10,
                              maxdim::Int     = 500,
                              cutoff::Float64 = 1e-8,
                              quenched::Bool  = true,
                              sequential::Bool = false,
                              pk_mpo          = nothing)
    l_bits  = l === nothing ? div(L, 2) : l
    L_chain = 2^l_bits

    # Detect sublattice: when sites has L+1 entries the last one is the aux index.
    # n_sub > 1 means pos MPOs are built on pos_sites only, then extended via
    # postpend_op(⋅, sub_s, I) so their site indices match H.sites throughout.
    n_sub     = length(sites) > L ? dim(sites[L+1]) : 1
    has_sub   = n_sub > 1
    pos_sites = has_sub ? collect(sites[1:L]) : collect(sites)
    sub_s     = has_sub ? sites[L+1] : nothing
    I_mat     = has_sub ? Matrix{Float64}(LinearAlgebra.I, n_sub, n_sub) : nothing

    # For building position MPOs over 2^L unit cells, adapt xfunc/yfunc:
    # xfunc_pos(i_uc, Lc) maps 0-indexed UC number to x-coordinate.
    # For sublattice, UC i_uc has physical site index i_uc*n_sub (0-indexed).
    xfunc_pos = has_sub ? ((i, Lc) -> xfunc(i * n_sub, Lc)) : xfunc
    yfunc_pos = has_sub ? ((i, Lc) -> yfunc(i * n_sub, Lc)) : yfunc

    # Unit cell area from the cross product of the two primitive lattice vectors.
    # a1: one step in the fast (x) direction; a2: one step in the slow (y) direction.
    a1x = xfunc_pos(1, L_chain) - xfunc_pos(0, L_chain)
    a1y = yfunc_pos(1, L_chain) - yfunc_pos(0, L_chain)
    a2x = xfunc_pos(L_chain, L_chain) - xfunc_pos(0, L_chain)
    a2y = yfunc_pos(L_chain, L_chain) - yfunc_pos(0, L_chain)
    A_cell = abs(a1x * a2y - a1y * a2x)

    Q = MPO(sites, "Id") - P

    # Closure that builds the basis MPS for physical site alpha (1-indexed).
    # For sublattice: big-endian position bits + sublattice index via _product_state_mps.
    make_alpha_mps = if has_sub
        all_sites = collect(sites)
        alpha -> begin
            n_cell   = (alpha - 1) ÷ n_sub
            sub      = (alpha - 1) % n_sub + 1
            pos_bits = [((n_cell >> (L - i)) & 1) + 1 for i in 1:L]
            _product_state_mps(all_sites, [pos_bits; sub])
        end
    else
        alpha -> binary_to_MPS(alpha - 1, L, sites)
    end

    if quenched
        sinX_op_p = get_sinx_op(L, pos_sites, L_chain, Λ, xfunc_pos)
        cosX_op_p = get_cosx_op(L, pos_sites, L_chain, Λ, xfunc_pos)
        sinY_op_p = get_siny_op(L, pos_sites, L_chain, Λ, yfunc_pos)
        cosY_op_p = get_cosy_op(L, pos_sites, L_chain, Λ, yfunc_pos)

        sinX_op = has_sub ? postpend_op(sinX_op_p, sub_s, I_mat) : sinX_op_p
        cosX_op = has_sub ? postpend_op(cosX_op_p, sub_s, I_mat) : cosX_op_p
        sinY_op = has_sub ? postpend_op(sinY_op_p, sub_s, I_mat) : sinY_op_p
        cosY_op = has_sub ? postpend_op(cosY_op_p, sub_s, I_mat) : cosY_op_p

        if sequential
            # Sequential mode: skip C1–C4 MPO construction; instead apply MPOs to
            # the basis MPS |α⟩ inside the closure.  Avoids expensive MPO×MPO products
            # at the cost of more MPS-MPO applies per site.
            #
            # Uses the trig shift identity sin(A−B) = sinA cosB − cosA sinB to fold
            # the 8-term Chern formula into 2 inner products:
            #   ch = ⟨Qα|sinΔX|P sinΔY Qα⟩ − ⟨Pα|sinΔX|Q sinΔY Pα⟩
            # where sinΔX = cos_x·sinX − sin_x·cosX and sinΔY = cos_y·sinY − sin_y·cosY.
            # This reduces P applies from 5→3 and inner products from 8→4 per site.
            calculate_chern_number = uc -> begin
                sum(sub -> begin
                    alpha = (uc - 1) * n_sub + sub
                    α_raw = make_alpha_mps(alpha)
                    α     = pk_mpo === nothing ? α_raw :
                            apply(pk_mpo, α_raw; maxdim=maxdim, cutoff=cutoff)
                    x     = xfunc(alpha - 1, L_chain)
                    y     = yfunc(alpha - 1, L_chain)
                    cos_x, sin_x = cos(x / Λ), sin(x / Λ)
                    cos_y, sin_y = cos(y / Λ), sin(y / Λ)

                    Pα = apply(P, α; maxdim=maxdim, cutoff=cutoff)
                    Qα = +(α, -1.0 * Pα; maxdim=maxdim, cutoff=cutoff)

                    sinY_Qα = apply(sinY_op, Qα; maxdim=maxdim, cutoff=cutoff)
                    cosY_Qα = apply(cosY_op, Qα; maxdim=maxdim, cutoff=cutoff)
                    sinY_Pα = apply(sinY_op, Pα; maxdim=maxdim, cutoff=cutoff)
                    cosY_Pα = apply(cosY_op, Pα; maxdim=maxdim, cutoff=cutoff)

                    # sinΔY|Qα⟩ = (cos_y·sinY − sin_y·cosY)|Qα⟩  (cheap MPS combo)
                    sinΔY_Qα = +(cos_y * sinY_Qα, -sin_y * cosY_Qα; maxdim=maxdim, cutoff=cutoff)
                    sinΔY_Pα = +(cos_y * sinY_Pα, -sin_y * cosY_Pα; maxdim=maxdim, cutoff=cutoff)

                    P_sinΔY_Qα = apply(P, sinΔY_Qα; maxdim=maxdim, cutoff=cutoff)
                    P_sinΔY_Pα = apply(P, sinΔY_Pα; maxdim=maxdim, cutoff=cutoff)
                    Q_sinΔY_Pα = +(sinΔY_Pα, -1.0 * P_sinΔY_Pα; maxdim=maxdim, cutoff=cutoff)

                    # sinΔX = cos_x·sinX − sin_x·cosX; use 3-arg inner (no intermediate MPS)
                    cq = cos_x * inner(Qα', sinX_op, P_sinΔY_Qα) -
                         sin_x * inner(Qα', cosX_op, P_sinΔY_Qα)
                    cp = cos_x * inner(Pα', sinX_op, Q_sinΔY_Pα) -
                         sin_x * inner(Pα', cosX_op, Q_sinΔY_Pα)

                    (cq - cp) * 2im * π * Λ^2
                end, 1:n_sub) / A_cell
            end

        else
            # Pre-multiply the 8 P/Q × sin/cos combinations
            sinY_P = apply(sinY_op, P;  maxdim=maxdim, cutoff=cutoff)
            cosY_P = apply(cosY_op, P;  maxdim=maxdim, cutoff=cutoff)
            P_sinX = apply(P,  sinX_op; maxdim=maxdim, cutoff=cutoff)
            P_cosX = apply(P,  cosX_op; maxdim=maxdim, cutoff=cutoff)
            sinY_Q = apply(sinY_op, Q;  maxdim=maxdim, cutoff=cutoff)
            cosY_Q = apply(cosY_op, Q;  maxdim=maxdim, cutoff=cutoff)
            Q_sinX = apply(Q,  sinX_op; maxdim=maxdim, cutoff=cutoff)
            Q_cosX = apply(Q,  cosX_op; maxdim=maxdim, cutoff=cutoff)
            println("Quenched operator products done")

            # C1 = Q sinX P sinY Q − P sinX Q sinY P
            C1 = apply(Q_sinX, P;      maxdim=maxdim, cutoff=cutoff)
            C1 = apply(C1,     sinY_Q; maxdim=maxdim, cutoff=cutoff)
            c1 = apply(P_sinX, Q;      maxdim=maxdim, cutoff=cutoff)
            c1 = apply(c1,     sinY_P; maxdim=maxdim, cutoff=cutoff)
            C1 = +(C1, -1.0 * c1; maxdim=maxdim, cutoff=cutoff)
            println("C1 done")

            # C2 = Q cosX P cosY Q − P cosX Q cosY P
            C2 = apply(Q_cosX, P;      maxdim=maxdim, cutoff=cutoff)
            C2 = apply(C2,     cosY_Q; maxdim=maxdim, cutoff=cutoff)
            c2 = apply(P_cosX, Q;      maxdim=maxdim, cutoff=cutoff)
            c2 = apply(c2,     cosY_P; maxdim=maxdim, cutoff=cutoff)
            C2 = +(C2, -1.0 * c2; maxdim=maxdim, cutoff=cutoff)
            println("C2 done")

            # C3 = Q sinX P cosY Q − P sinX Q cosY P
            C3 = apply(Q_sinX, P;      maxdim=maxdim, cutoff=cutoff)
            C3 = apply(C3,     cosY_Q; maxdim=maxdim, cutoff=cutoff)
            c3 = apply(P_sinX, Q;      maxdim=maxdim, cutoff=cutoff)
            c3 = apply(c3,     cosY_P; maxdim=maxdim, cutoff=cutoff)
            C3 = +(C3, -1.0 * c3; maxdim=maxdim, cutoff=cutoff)
            println("C3 done")

            # C4 = Q cosX P sinY Q − P cosX Q sinY P
            C4 = apply(Q_cosX, P;      maxdim=maxdim, cutoff=cutoff)
            C4 = apply(C4,     sinY_Q; maxdim=maxdim, cutoff=cutoff)
            c4 = apply(P_cosX, Q;      maxdim=maxdim, cutoff=cutoff)
            c4 = apply(c4,     sinY_P; maxdim=maxdim, cutoff=cutoff)
            C4 = +(C4, -1.0 * c4; maxdim=maxdim, cutoff=cutoff)
            println("C4 done")

            calculate_chern_number = uc -> begin
                sum(sub -> begin
                    alpha  = (uc - 1) * n_sub + sub
                    α_raw  = make_alpha_mps(alpha)
                    α      = pk_mpo === nothing ? α_raw :
                             apply(pk_mpo, α_raw; maxdim=maxdim, cutoff=cutoff)
                    x      = xfunc(alpha - 1, L_chain)
                    y      = yfunc(alpha - 1, L_chain)
                    cos_x, sin_x = cos(x / Λ), sin(x / Λ)
                    cos_y, sin_y = cos(y / Λ), sin(y / Λ)
                    ch  =  cos_x * cos_y * inner(α', C1, α)
                    ch +=  sin_x * sin_y * inner(α', C2, α)
                    ch -=  cos_x * sin_y * inner(α', C3, α)
                    ch -=  sin_x * cos_y * inner(α', C4, α)
                    ch * 2im * π * Λ^2
                end, 1:n_sub) / A_cell
            end
        end

    else
        # Flat mode: build global position MPOs directly from xfunc/yfunc
        x_op_p = get_diagonal_mpo(L, pos_sites, i -> xfunc_pos(i - 1, L_chain))
        y_op_p = get_diagonal_mpo(L, pos_sites, i -> yfunc_pos(i - 1, L_chain))
        x_op   = has_sub ? postpend_op(x_op_p, sub_s, I_mat) : x_op_p
        y_op   = has_sub ? postpend_op(y_op_p, sub_s, I_mat) : y_op_p

        T1   = apply(Q, apply(x_op, apply(P, apply(y_op, Q;
                     maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff);
                     maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff)
        T2   = apply(P, apply(x_op, apply(Q, apply(y_op, P;
                     maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff);
                     maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff)
        C_op = 2im * π * +(T1, -1.0 * T2; maxdim=maxdim, cutoff=cutoff)
        ITensorMPS.truncate!(C_op; maxdim=maxdim, cutoff=cutoff)

        calculate_chern_number = uc -> begin
            sum(sub -> begin
                alpha = (uc - 1) * n_sub + sub
                α_raw = make_alpha_mps(alpha)
                α     = pk_mpo === nothing ? α_raw :
                        apply(pk_mpo, α_raw; maxdim=maxdim, cutoff=cutoff)
                inner(α', C_op, α)
            end, 1:n_sub) / A_cell
        end
    end

    return calculate_chern_number
end


# ============================================================
# 2D Chern marker from TBHamiltonian
# ============================================================

"""
    get_C(H::TBHamiltonian, xfunc=nothing, yfunc=nothing;
          method=:KPM, fermi=0.0, l=nothing, Λ=10,
          Nchebychev=300, maxdim=500, cutoff=1e-8,
          Nel=nothing, quenched=true) -> Function

High-level wrapper: compute the ground-state projector via `method` and
return the Chern marker closure from `get_C_op_MPO_from_P`.

`xfunc(i, L_chain)` and `yfunc(i, L_chain)` accept a **0-indexed** physical
site number `i` and return raw x/y coordinates.  Both default to `nothing`,
in which case they are auto-derived:

- If `H.geometry_uc` is set (sublattice models: honeycomb, kagome, lieb, dice,
  ssh_sublattice): uses `geometry_uc(i+1)[1/2]`, which returns the same
  Bravais unit-cell position for all sublattice atoms in the same UC.
- Otherwise falls back to `H.geometry(i+1)[1/2]`.

When `xfunc`/`yfunc` are auto-derived for a sublattice model, `α` in the
returned closure ranges over `1 … n_sub·2^L` (all physical sites).

Reuses `H._tn_cache` or `H._density_cache` when available.  `maxdim` and
`cutoff` are forwarded uniformly to the projector computation and to all
MPO multiplications in the Chern marker assembly.

See `get_C_op_MPO_from_P` for full documentation of the remaining arguments.

# Returns
`calculate_chern_number(uc::Int) -> ComplexF64` where `uc` is a 1-indexed
unit cell number; the closure sums the marker over all `n_sub` sublattice
atoms in that UC.  Take `real(·)` for the density.
"""
function get_C(H::TBHamiltonian, xfunc=nothing, yfunc=nothing;
               method::Symbol   = :KPM,
               fermi::Real      = 0.0,
               l                = nothing,
               Λ::Real          = 10,
               Lambda           = nothing,  # ASCII alias for Λ
               Nchebychev::Int  = 300,
               maxdim::Int      = 500,
               cutoff::Float64  = 1e-8,
               Nel              = nothing,
               quenched::Bool   = true,
               sequential::Bool = false)
    if xfunc === nothing || yfunc === nothing
        geom = H.geometry_uc !== nothing ? H.geometry_uc :
               H.geometry   !== nothing ? H.geometry   :
               error("H has no geometry function; provide xfunc and yfunc explicitly.")
        xfunc === nothing && (xfunc = (i, _) -> geom(i + 1)[1])
        yfunc === nothing && (yfunc = (i, _) -> geom(i + 1)[2])
    end
    P = _get_projector(H; method=method, fermi=fermi, Nchebychev=Nchebychev,
                       maxdim=maxdim, cutoff=cutoff, Nel=Nel)
    return get_C_op_MPO_from_P(P, H.L, H.sites, xfunc, yfunc;
                                l=l, Λ=Λ, maxdim=maxdim, cutoff=cutoff,
                                quenched=quenched, sequential=sequential)
end


# ============================================================
# Valley operator and valley Chern number (honeycomb)
# ============================================================

"""
    get_valley_operator(H::TBHamiltonian; maxdim=500, cutoff=1e-8) -> MPO

Build the valley operator V for a 2D honeycomb Hamiltonian.

V is constructed as the Haldane NNN Hamiltonian (φ = π/2, NN term zeroed)
multiplied by an additional sublattice sign η_i: +1 on sublattice A (index 1),
−1 on sublattice B (index 2).  The global prefactor is −i/(3√3):

    t(dx,dy,fs,ts) = (−i / 3√3) · ν_{dx,dy,fs,ts} · η_{fs}

where ν ∈ {±1} is the Haldane chirality (counterclockwise = +1).

`H` must be a 2D honeycomb model with `H.Lx` set and a 2-component sublattice
index.  The returned MPO shares the same site indices as `H.mpo`.
"""
function get_valley_operator(H::TBHamiltonian;
                             maxdim::Int     = 500,
                             cutoff::Float64 = 1e-8)
    H.Lx !== nothing ||
        error("get_valley_operator requires a 2D Hamiltonian (H.Lx must be set).")
    H.sublattice_s !== nothing ||
        error("get_valley_operator requires a sublattice Hamiltonian.")
    dim(H.sublattice_s) == 2 ||
        error("get_valley_operator requires 2 sublattices (honeycomb); got $(dim(H.sublattice_s)).")

    Lx = H.Lx
    Ly = H.L - Lx

    # Deepcopy H and zero out its MPO so that add_hopping_2D! builds on
    # exactly H.sites (including the sublattice index) — avoids the index
    # mismatch that arises when get_Hamiltonian creates fresh site indices.
    H_v = deepcopy(H)
    H_v.mpo            = 0.0 * MPO(collect(H.sites), "Id")
    H_v._density_cache = nothing
    H_v._tn_cache      = nothing
    H_v._tn_mps_cache  = nothing

    haldane_ν = Dict(
        (1,  0, 1, 1) =>  1,  (0,  1, 1, 1) => -1,  (1, -1, 1, 1) => -1,
        (1,  0, 2, 2) => -1,  (0,  1, 2, 2) =>  1,  (1, -1, 2, 2) =>  1,
    )
    η = Dict(1 => 1, 2 => -1)

    prefactor = -im / (3.0 * sqrt(3.0))
    add_hopping_2D!(H_v,
        (dx, dy, fs, ts) -> prefactor * get(haldane_ν, (dx, dy, fs, ts), 0) * η[fs];
        Lx=Lx, Ly=Ly, nn=2, maxdim=maxdim, tol=cutoff)

    return H_v.mpo
end


"""
    get_valley_projectors(V_mpo, sites; maxdim=500, cutoff=1e-8) -> (PK, PK_prime)

Return the two valley projectors from the valley operator `V_mpo`:

    PK       = (I + V) / 2   (K  valley)
    PK_prime = (I − V) / 2   (K′ valley)
"""
function get_valley_projectors(V_mpo::MPO, sites;
                               maxdim::Int     = 500,
                               cutoff::Float64 = 1e-8)
    I_mpo    = MPO(sites, "Id")
    PK       = 0.5 * +(I_mpo,        V_mpo; maxdim=maxdim, cutoff=cutoff)
    PK_prime = 0.5 * +(I_mpo, -1.0 * V_mpo; maxdim=maxdim, cutoff=cutoff)
    return PK, PK_prime
end


"""
    get_valley_C(H, xfunc=nothing, yfunc=nothing;
                 valley=:K, method=:mcweeny, fermi=0.0, l=nothing, Λ=10,
                 Nchebychev=300, maxdim=500, cutoff=1e-8,
                 Nel=nothing, quenched=true) -> Function

Compute the valley-resolved Chern marker and return a closure
`calculate_valley_chern(uc::Int) -> ComplexF64`.

The valley operator V is built automatically via `get_valley_operator(H)`.
The valley-projected occupied density matrix is then formed as

    P_K = PK · P · PK,   PK = (I ± V) / 2

where `P` is the ground-state projector and the sign is chosen by `valley`
(`:K` → +, `:K_prime` → −).  The standard real-space Chern marker formula
is then evaluated with `P_K` in place of `P` via `get_C_op_MPO_from_P`.

# Arguments
- `valley` : `:K` or `:K_prime`.

All remaining arguments are forwarded to `_get_projector` and
`get_C_op_MPO_from_P`; see those functions for documentation.
"""
function get_valley_C(H::TBHamiltonian,
                      xfunc=nothing, yfunc=nothing;
                      valley::Symbol  = :K,
                      method::Symbol  = :mcweeny,
                      fermi::Real     = 0.0,
                      l               = nothing,
                      Λ::Real         = 10,
                      Nchebychev::Int = 300,
                      maxdim::Int     = 500,
                      cutoff::Float64 = 1e-8,
                      Nel             = nothing,
                      quenched::Bool  = true,
                      sequential::Bool = false)
    valley in (:K, :K_prime) ||
        error("valley must be :K or :K_prime, got :$valley")

    if xfunc === nothing || yfunc === nothing
        geom = H.geometry_uc !== nothing ? H.geometry_uc :
               H.geometry   !== nothing ? H.geometry   :
               error("H has no geometry function; provide xfunc and yfunc explicitly.")
        xfunc === nothing && (xfunc = (i, _) -> geom(i + 1)[1])
        yfunc === nothing && (yfunc = (i, _) -> geom(i + 1)[2])
    end

    V_mpo = get_valley_operator(H; maxdim=maxdim, cutoff=cutoff)
    P     = _get_projector(H; method=method, fermi=fermi, Nchebychev=Nchebychev,
                           maxdim=maxdim, cutoff=cutoff, Nel=Nel)
    sign  = valley == :K ? 1.0 : -1.0
    I_mpo = MPO(H.sites, "Id")
    PK    = 0.5 * +(I_mpo, sign * V_mpo; maxdim=maxdim, cutoff=cutoff)

    return get_C_op_MPO_from_P(P, H.L, H.sites, xfunc, yfunc;
                                l=l, Λ=Λ, maxdim=maxdim, cutoff=cutoff,
                                quenched=quenched, sequential=sequential,
                                pk_mpo=PK)
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
                       maxdim::Int          = 100,
                       cutoff::Float64      = 1e-8,
                       verbose::Bool        = false,
                       return_trajectory::Bool = false)
    Nt    = length(P_array)
    I_mpo = MPO(sites, "Id")

    # M1Q at t = 0: U(0) = I  →  ⟨r|P(0) x̂ P(0)|r⟩
    PxP_0 = apply(apply(P_array[1], x_op; maxdim, cutoff), P_array[1]; maxdim, cutoff)
    M1Q_0 = real(matrix_checker(PxP_0, sites, r_center, r_center))
    verbose && println("M1Q(0) = $(round(M1Q_0; digits=6))")

    M1Q_traj = return_trajectory ? Float64[M1Q_0] : Float64[]

    # Propagate U: h(k) = [Ṗ(k), P(k)] = Ṗ P − P Ṗ
    U = deepcopy(I_mpo)
    for k in 1:Nt
        if k == 1
            Pdot = (1.0 / dt) * +(P_array[2],  -1.0 * P_array[1];    maxdim, cutoff)
        elseif k == Nt
            Pdot = (1.0 / dt) * +(P_array[Nt], -1.0 * P_array[Nt-1]; maxdim, cutoff)
        else
            Pdot = (0.5 / dt) * +(P_array[k+1], -1.0 * P_array[k-1]; maxdim, cutoff)
        end
        ITensorMPS.truncate!(Pdot; maxdim, cutoff)

        h_k  = +(apply(Pdot, P_array[k]; maxdim, cutoff),
                 -1.0 * apply(P_array[k], Pdot; maxdim, cutoff); maxdim, cutoff)
        ITensorMPS.truncate!(h_k; maxdim, cutoff)

        h_sq = apply(h_k, h_k; maxdim, cutoff)
        dU   = +(+(I_mpo, dt * h_k; maxdim, cutoff),
                 (dt^2 / 2) * h_sq; maxdim, cutoff)
        ITensorMPS.truncate!(dU; maxdim, cutoff)
        U    = apply(dU, U; maxdim, cutoff)
        ITensorMPS.truncate!(U; maxdim, cutoff)

        verbose && println("  step $k/$Nt  maxlinkdim(U) = $(ITensorMPS.maxlinkdim(U))")

        if return_trajectory
            Ud_k    = dag(swapprime(U, 0, 1))
            UxU_k   = apply(apply(Ud_k, x_op; maxdim, cutoff), U; maxdim, cutoff)
            PUxUP_k = apply(apply(P_array[k], UxU_k; maxdim, cutoff), P_array[k]; maxdim, cutoff)
            push!(M1Q_traj, real(matrix_checker(PUxUP_k, sites, r_center, r_center)))
        end
    end

    # M1Q at t = T
    if return_trajectory
        M1Q_T = M1Q_traj[end]
    else
        Ud    = dag(swapprime(U, 0, 1))
        UxU   = apply(apply(Ud, x_op; maxdim, cutoff), U;    maxdim, cutoff)
        PUxUP = apply(apply(P_array[end], UxU; maxdim, cutoff), P_array[end]; maxdim, cutoff)
        M1Q_T = real(matrix_checker(PUxUP, sites, r_center, r_center))
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
                           verbose::Bool                = false)
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
                              cutoff=cutoff, Nel=Nel)
        push!(P_array, P_k)
    end

    verbose && println("Computing M1Q invariant (r_center=$rc)...")
    return thouless_pump(P_array, dt, x_op, sites;
                         r_center=rc, maxdim=maxdim, cutoff=cutoff, verbose=verbose)
end
