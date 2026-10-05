# Topology.jl — topological invariants via KPM and MPO methods
#
# Implements real-space Chern markers and winding numbers for arbitrary
# tight-binding systems encoded in the quantics representation.
#
# The key observables are:
#
#   Chern marker (2D):   C(r) = 2π Im⟨r| P x̂ Q ŷ P − Q x̂ P ŷ Q |r⟩
#   Winding number (1D): W(r) = ⟨r| σ_z (P x̂ Q + Q x̂ P) |r⟩
#
# where P is the ground-state projector and Q = I − P. C(r) is the diagonal of the
# Hermitian part of 2πi (Q x̂ P ŷ Q − P x̂ Q ŷ P), the operator the code assembles,
# and the mean of the two Bianco–Resta markers 4π Im⟨r|P x̂ Q ŷ P|r⟩ and
# −4π Im⟨r|Q x̂ P ŷ Q|r⟩, whose traces agree (see _hermitian_diag).
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
# chern_marker and winding_marker both accept xfunc=nothing / yfunc=nothing, in which case
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
#       Formula: C = 2πi (Q x P y Q − P x Q y P), Hermitian part.  Best for OBC
#       or averages.
#
# == Projector methods ==
#   method=:kpm      — KPM Chebyshev expansion (uses cached Tn of sufficient order)
#   method=:mcweeny  — McWeeny purification (uses a McWeeny density cache)
#   method=:sp2      — SP2 purification (uses an SP2 density cache)
#
# Bond dimension and truncation are controlled uniformly through `maxdim` and
# `cutoff` kwargs, which are threaded into every apply, add, and truncate! call.
#
# Entry points: chern_marker, winding_marker, valley_chern_marker, get_valley_operator,
#   get_thouless_pump, thouless_pump, get_C_op_MPO_from_P. The 0.1 names get_C, get_W
#   and get_valley_C are deprecated aliases (src/deprecated.jl); so are the keywords
#   Nchebychev (→ Ncheb), Λ (→ Lambda) and the value method=:KPM (→ :kpm).
# Depends on: core/Utils.jl, core/TBSystem.jl, lattice/NNNeighbor.jl,
#   solvers/kpm/recursion.jl, physics/Purification.jl (the density dispatcher
#   _density_matrix behind _get_projector, _cached_density, _density_key,
#   _half_filling; see the source map in src/TensorBinding.jl).


# ============================================================
# 1. Ground-state projector from a TBHamiltonian
# ============================================================

"""
    _get_projector(H; method=:kpm, fermi=0.0, Ncheb=300, maxdim=40,
                   cutoff=1e-8, Nel=nothing) -> MPO

Compute or retrieve the ground-state projector P for `H`.

- `method=:kpm` (default): uses the cached `H._tn_cache` if it has at least
  `Ncheb` moments; otherwise runs `KPM_Tn(H, Ncheb)`.  The Fermi level
  `fermi` (in physical units) is rescaled internally.
- `method=:mcweeny`: returns `H._density_cache` if it was computed by McWeeny
  purification at the same `fermi` (or set by hand); otherwise runs McWeeny
  purification with `ϵF=fermi`.
- `method=:sp2`: same (at the same `Nel`) but uses SP2 purification.  `Nel` sets the
  target electron count (default: half the number of states, `prod(dim, H.sites) ÷ 2`).
- `maxdim`, `cutoff`: bond dimension and truncation threshold forwarded to the
  underlying method.

The projector comes from `get_density`'s dispatcher (`_density_matrix`,
physics/Purification.jl), with the rules above: `:kpm` here, unlike `get_density`'s,
never touches the density cache; `:sp2` runs `sp2_purify`'s default 40 iterations.
`method=:KPM`, the spelling up to v0.1.1, is a deprecated alias of `:kpm` (until v0.1.1
`:kpm` was an error here). Up to v0.1.1 `:KPM` also expanded a cached Chebyshev list
shorter than `Ncheb`, always with cutoff 1e-8, and `:mcweeny`/`:sp2` returned a
density matrix cached by any method, at any Fermi level or filling.
"""
function _get_projector(H::TBHamiltonian;
                         method::Symbol   = :kpm,
                         fermi::Real      = 0.0,
                         Ncheb::Int       = 300,
                         maxdim::Int      = 40,
                         cutoff::Float64  = 1e-8,
                         Nel              = nothing)
    if method === :KPM
        _depwarn_once("method=:KPM is deprecated, use method=:kpm", :method_KPM)
        method = :kpm
    end
    if method == :kpm
        # H._tn_cache when it has ≥ Ncheb moments, else a fresh KPM_Tn(H, Ncheb)
        return _density_matrix(H, :kpm; ϵF=fermi, Ncheb=Ncheb, maxdim=maxdim,
                               cutoff=cutoff, store=false)
    elseif method == :mcweeny
        cached = _cached_density(H, _density_key(:mcweeny; ϵF=fermi))
        cached === nothing || return cached
        return _density_matrix(H, :mcweeny; ϵF=fermi, maxiters=30, maxdim=maxdim,
                               cutoff=cutoff, tol=1e-5, verbose=false)
    elseif method == :sp2
        Nel_val = Nel === nothing ? _half_filling(H) : Int(Nel)
        cached = _cached_density(H, _density_key(:sp2; Nel=Nel_val))
        cached === nothing || return cached
        return _density_matrix(H, :sp2; Nel=Nel_val, maxiters=40, maxdim=maxdim,
                               cutoff=cutoff, tol=1e-5, verbose=false)
    else
        error("Unknown method: :$method. Choose :kpm, :mcweeny, or :sp2")
    end
end


# ============================================================
# 2. 1D winding number
# ============================================================

"""
    winding_marker(H::TBHamiltonian, xfunc=nothing;
                   method=:kpm, fermi=0.0, Ncheb=300,
                   maxdim=15, cutoff=1e-8, Nel=nothing,
                   quenched=true, l=nothing, Lambda=10) -> Function

Compute the real-space winding number and return a closure
`calculate_winding(uc::Int) -> ComplexF64` that evaluates the local winding
marker of unit cell `uc` (1-indexed; see Returns).

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
      W(α) = Λ · ⟨α| cos(x_α/Λ) W1 − sin(x_α/Λ) W2 |α⟩,  Λ = `Lambda`
- `quenched=false`: builds a global position operator and returns a closure
  over the resulting W_op MPO.

# Arguments
- `method`    : `:kpm`, `:mcweeny`, or `:sp2` (see `_get_projector`).
- `fermi`     : Fermi level in physical energy units (`:kpm` and `:mcweeny`).
- `Ncheb`     : Chebyshev order when `method=:kpm` and no cache is present.
- `maxdim`    : MPO bond dimension during all multiplications.
- `cutoff`    : truncation threshold during all multiplications.
- `Nel`       : target electron count for SP2 (default: half the number of states,
                `prod(dim, H.sites) ÷ 2`).
- `l`         : qubits per direction; inferred as `H.L ÷ 2` if `nothing`.
- `Lambda`    : quenching period Λ (angle = xfunc/Λ); sets the Λ prefactor.

The keywords `Nchebychev` and `Λ` and the value `method=:KPM` (the spellings up to
v0.1.1, when this function was `get_W`) are deprecated aliases of `Ncheb`, `Lambda`
and `:kpm`; passing an old and a new spelling together is an `ArgumentError`.

# Returns
`calculate_winding(uc::Int) -> ComplexF64` where `uc` is a 1-indexed unit
cell number (1 … 2^L).  The value is the sum of the winding marker over both
sublattice atoms (A and B) within that unit cell.
"""
function winding_marker(H::TBHamiltonian, xfunc=nothing;
               method::Symbol   = :kpm,
               fermi::Real      = 0.0,
               Ncheb::Union{Nothing,Int}  = nothing,
               maxdim::Int      = 15,
               cutoff::Float64  = 1e-8,
               Nel              = nothing,
               quenched::Bool   = true,
               l                = nothing,
               Lambda::Union{Nothing,Real} = nothing,
               Nchebychev::Union{Nothing,Int} = nothing,   # deprecated: Ncheb
               Λ::Union{Nothing,Real}  = nothing)          # deprecated: Lambda
    Ncheb = _renamed_kw(:winding_marker, :Ncheb, Ncheb, :Nchebychev, Nchebychev, 300)
    Λ     = _renamed_kw(:winding_marker, :Lambda, Lambda, :Λ, Λ, 10)
    _require_binary_position_space(H, "winding_marker")
    H.sublattice_s === nothing || dim(H.sublattice_s) == 2 ||
        error("winding_marker requires a 2-component sublattice index (dim=2); got dim=$(dim(H.sublattice_s)).")
    H.sublattice_s !== nothing ||
        error("winding_marker requires H.sublattice_s to be set (n_sub=2 sublattice model).")

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

    P       = _get_projector(H; method=method, fermi=fermi, Ncheb=Ncheb,
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
# 3. Quenched (periodic) position operator builders
# ============================================================
#
# These are low-level helpers called by winding_marker, get_C_op_MPO_from_P and
# get_pump_xop.
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
# 4. 2D Chern marker from a pre-computed projector
# ============================================================

"""
    get_C_op_MPO_from_P(P, L, sites, xfunc, yfunc;
                        l=nothing, Lambda=10, maxdim=500, cutoff=1e-8,
                        quenched=true, sequential=false, pk_mpo=nothing) -> Function

Build the real-space Chern marker and return a closure `calculate_chern_number(uc)`
that evaluates it for unit cell `uc` (1-indexed; see Returns).

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

Position operators are quenched with the period Λ = `Lambda`: `sin(xfunc/Λ)`,
`cos(xfunc/Λ)`, and similarly
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

In both modes the marker is divided by the unit-cell area `A_cell = |a₁ × a₂|`,
with `a₁`, `a₂` the steps of `xfunc`/`yfunc` from unit cell 0 to unit cells 1 and
`L_chain`.

In both modes each ⟨α|C|α⟩ is that of the Hermitian part (C + C†)/2 of the marker
operator, i.e. its real part: C itself is not Hermitian, and up to v0.1.1 its
traceless anti-Hermitian part gave the local markers imaginary parts of O(0.1)
(`_hermitian_diag`).

# Arguments
- `P`       : ground-state projector MPO (over `sites`)
- `L`       : number of position qubits; system has `2^L` unit cells
- `sites`   : full ITensor site index list (`length == L` or `L+1` for sublattice)
- `xfunc`, `yfunc` : coordinate functions; `i` is 0-indexed over all physical sites
- `l`       : qubits per spatial direction; inferred as `L ÷ 2` if `nothing`
- `Lambda`  : quenching period Λ (quenching angle = coord / Λ); `Λ`, the spelling up
  to v0.1.1, is a deprecated alias
- `maxdim`  : MPO bond dimension during all multiplications
- `cutoff`  : truncation threshold during all multiplications and subtractions
- `quenched`: `true` = 4-term sin/cos decomposition; `false` = flat operators
- `sequential`: quenched mode only; `true` skips the C1–C4 MPO×MPO products and
  applies `P` and the position operators to each basis state inside the closure
  (see `chern_marker`)
- `pk_mpo`  : optional MPO `PK` (e.g. a valley projector, see `valley_chern_marker`);
  when given, the marker is evaluated as `⟨α|PK C_op PK|α⟩`

# Returns
`calculate_chern_number(uc::Int) -> ComplexF64` where `uc` is a 1-indexed
unit cell number (1 … 2^L).  For sublattice models the value is the sum of
the Chern marker over all `n_sub` atoms within that unit cell.
The value is real (its imaginary part is zero); `real(·)` gives the Chern
number density as a `Float64`.

# Example — quenched square lattice
```julia
L_chain = 2^(L ÷ 2)
xfunc(i, _) = Float64(mod(i, L_chain))
yfunc(i, _) = Float64(div(i, L_chain))
C_at  = get_C_op_MPO_from_P(P, L, sites, xfunc, yfunc; Lambda=L_chain, maxdim=100)
uc_c  = (L_chain ÷ 2) * L_chain + L_chain ÷ 2 + 1   # central unit cell
C_c   = real(C_at(uc_c))
```

# Example — honeycomb via chern_marker (auto-derived geometry)
```julia
C_at  = chern_marker(H)   # xfunc/yfunc from H.geometry_uc automatically
Nx    = 2^(H.L ÷ 2)                                  # unit cells per row
uc_c  = (Nx ÷ 2) * Nx + Nx ÷ 2 + 1                   # central unit cell
C_c   = real(C_at(uc_c))   # sums the A and B atoms of that cell
```
"""
function get_C_op_MPO_from_P(P, L, sites, xfunc, yfunc;
                              l               = nothing,
                              Lambda::Union{Nothing,Real} = nothing,
                              maxdim::Int     = 500,
                              cutoff::Float64 = 1e-8,
                              quenched::Bool  = true,
                              sequential::Bool = false,
                              pk_mpo          = nothing,
                              Λ::Union{Nothing,Real} = nothing)   # deprecated: Lambda
    Λ = _renamed_kw(:get_C_op_MPO_from_P, :Lambda, Lambda, :Λ, Λ, 10)
    return _chern_marker(P, L, sites, xfunc, yfunc; l, Λ, maxdim, cutoff, quenched,
                         sequential, pk_mpo)
end

"""
    _hermitian_diag(z) -> Complex

The diagonal element ⟨α|(C + C†)/2|α⟩ of the Hermitian part of a marker operator C,
given z = ⟨α|C|α⟩ (or a real-weighted sum of them, as the unit-cell value of the
`_chern_marker` closures): since ⟨α|C†|α⟩ = conj(z), it is `real(z)`, kept complex
so that the closures keep their return type, and it needs no MPO product for C†.

The Chern-marker operator C = 2πi (Q X P Y Q − P X Q Y P) (and its quenched,
sequential and `pk_mpo` forms) is not Hermitian. Its Hermitian part
πi [(Q X P Y Q − Q Y P X Q) − (P X Q Y P − P Y Q X P)] gives the local marker,
the mean of the Bianco–Resta P- and Q-forms. The anti-Hermitian remainder is
traceless (it drops out of the Chern number) but not locally small: on a trivial
Semenoff honeycomb (2 × 2 cells), where P is real and every local marker vanishes,
it is up to 0.16i per unit cell. Up to v0.1.1 it was the imaginary part of every
`get_C` (now `chern_marker`) value.
"""
_hermitian_diag(z::Number) = complex(real(z))

"""
    _chern_marker(P, L, sites, xfunc, yfunc; l, Λ, maxdim, cutoff, quenched,
                  sequential, pk_mpo, to_device=_on_host, device_type=ComplexF64,
                  q_add=(), q_trunc=(), c_trunc=(), flat_trunc=(:maxdim, :cutoff),
                  progress=nothing, after_step=nothing) -> Function

The Chern-marker assembly of `get_C_op_MPO_from_P` (documented there), also run on
GPU MPOs by `chern_marker_gpu`. The operators the kernel builds itself (the identity of
`Q = I − P`, the four quenched or two flat position operators, the basis states of
the closure) are moved by `to_device(·, device_type)` (see `_on_host`). The keywords
name the truncation parameters (see `_trunc_kwargs`) of the steps in which the GPU
differs, with the CPU values as defaults: `q_add` for the sum `Q = +(I, −1.0·P)` and
`q_trunc` for a `truncate!` of Q after it (none on the CPU); `c_trunc` for a
`truncate!` of each of C1–C4 (none); `flat_trunc` for the one of the flat operator.
`progress(stage, M)` is called with `:positions` (quenched position operators
built), `:products` (the eight P/Q products), `:C1` … `:C4` (with that operator)
and `after_step()` after Q, the products, each of C1–C4 and the flat operator.
"""
function _chern_marker(P, L, sites, xfunc, yfunc;
                       l               = nothing,
                       Λ::Real         = 10,
                       maxdim::Int     = 500,
                       cutoff::Real    = 1e-8,
                       quenched::Bool  = true,
                       sequential::Bool = false,
                       pk_mpo          = nothing,
                       to_device       = _on_host,
                       device_type::Type = ComplexF64,
                       q_add::Tuple    = (),
                       q_trunc::Tuple  = (),
                       c_trunc::Tuple  = (),
                       flat_trunc::Tuple = (:maxdim, :cutoff),
                       progress        = nothing,
                       after_step      = nothing)
    step!() = after_step === nothing || after_step()
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

    # Q = I − P (-1.0 * P is -P: the same values on the CPU, while on a 32-bit GPU MPO
    # the Float64 factor promotes that site, as chern_marker_gpu always did).
    Q = +(to_device(MPO(sites, "Id"), device_type), -1.0 * P; _trunc_kwargs(q_add, maxdim, cutoff)...)
    isempty(q_trunc) || ITensorMPS.truncate!(Q; _trunc_kwargs(q_trunc, maxdim, cutoff)...)
    step!()

    # Closure that builds the basis MPS for physical site alpha (1-indexed).
    # For sublattice: big-endian position bits + sublattice index via _product_state_mps.
    make_alpha_mps = if has_sub
        all_sites = collect(sites)
        alpha -> begin
            n_cell   = (alpha - 1) ÷ n_sub
            sub      = (alpha - 1) % n_sub + 1
            pos_bits = [((n_cell >> (L - i)) & 1) + 1 for i in 1:L]
            to_device(_product_state_mps(all_sites, [pos_bits; sub]), device_type)
        end
    else
        alpha -> to_device(binary_to_MPS(alpha - 1, L, sites), device_type)
    end

    # Each of the three closures below returns the unit-cell sum of ⟨α|C|α⟩ / A_cell
    # through _hermitian_diag: the diagonal of the Hermitian part of the marker operator.
    if quenched
        sinX_op_p = get_sinx_op(L, pos_sites, L_chain, Λ, xfunc_pos)
        cosX_op_p = get_cosx_op(L, pos_sites, L_chain, Λ, xfunc_pos)
        sinY_op_p = get_siny_op(L, pos_sites, L_chain, Λ, yfunc_pos)
        cosY_op_p = get_cosy_op(L, pos_sites, L_chain, Λ, yfunc_pos)

        sinX_op = to_device(has_sub ? postpend_op(sinX_op_p, sub_s, I_mat) : sinX_op_p, device_type)
        cosX_op = to_device(has_sub ? postpend_op(cosX_op_p, sub_s, I_mat) : cosX_op_p, device_type)
        sinY_op = to_device(has_sub ? postpend_op(sinY_op_p, sub_s, I_mat) : sinY_op_p, device_type)
        cosY_op = to_device(has_sub ? postpend_op(cosY_op_p, sub_s, I_mat) : cosY_op_p, device_type)
        progress === nothing || progress(:positions, nothing)

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
                end, 1:n_sub) / A_cell |> _hermitian_diag
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
            @debug "get_C_op_MPO_from_P: quenched operator products done"
            progress === nothing || progress(:products, nothing)
            step!()

            # Each Ck is +(Ck, -ck) (the same values as -1.0 * ck on the CPU; on a
            # 32-bit GPU MPO the Int factor of -ck keeps the element type), then
            # truncated with `c_trunc` when it is not empty.
            c_kwargs = _trunc_kwargs(c_trunc, maxdim, cutoff)

            # C1 = Q sinX P sinY Q − P sinX Q sinY P
            C1 = apply(Q_sinX, P;      maxdim=maxdim, cutoff=cutoff)
            C1 = apply(C1,     sinY_Q; maxdim=maxdim, cutoff=cutoff)
            c1 = apply(P_sinX, Q;      maxdim=maxdim, cutoff=cutoff)
            c1 = apply(c1,     sinY_P; maxdim=maxdim, cutoff=cutoff)
            C1 = +(C1, -c1; maxdim=maxdim, cutoff=cutoff)
            isempty(c_kwargs) || ITensorMPS.truncate!(C1; c_kwargs...)
            @debug "get_C_op_MPO_from_P: C1 done"
            progress === nothing || progress(:C1, C1)
            step!()

            # C2 = Q cosX P cosY Q − P cosX Q cosY P
            C2 = apply(Q_cosX, P;      maxdim=maxdim, cutoff=cutoff)
            C2 = apply(C2,     cosY_Q; maxdim=maxdim, cutoff=cutoff)
            c2 = apply(P_cosX, Q;      maxdim=maxdim, cutoff=cutoff)
            c2 = apply(c2,     cosY_P; maxdim=maxdim, cutoff=cutoff)
            C2 = +(C2, -c2; maxdim=maxdim, cutoff=cutoff)
            isempty(c_kwargs) || ITensorMPS.truncate!(C2; c_kwargs...)
            @debug "get_C_op_MPO_from_P: C2 done"
            progress === nothing || progress(:C2, C2)
            step!()

            # C3 = Q sinX P cosY Q − P sinX Q cosY P
            C3 = apply(Q_sinX, P;      maxdim=maxdim, cutoff=cutoff)
            C3 = apply(C3,     cosY_Q; maxdim=maxdim, cutoff=cutoff)
            c3 = apply(P_sinX, Q;      maxdim=maxdim, cutoff=cutoff)
            c3 = apply(c3,     cosY_P; maxdim=maxdim, cutoff=cutoff)
            C3 = +(C3, -c3; maxdim=maxdim, cutoff=cutoff)
            isempty(c_kwargs) || ITensorMPS.truncate!(C3; c_kwargs...)
            @debug "get_C_op_MPO_from_P: C3 done"
            progress === nothing || progress(:C3, C3)
            step!()

            # C4 = Q cosX P sinY Q − P cosX Q sinY P
            C4 = apply(Q_cosX, P;      maxdim=maxdim, cutoff=cutoff)
            C4 = apply(C4,     sinY_Q; maxdim=maxdim, cutoff=cutoff)
            c4 = apply(P_cosX, Q;      maxdim=maxdim, cutoff=cutoff)
            c4 = apply(c4,     sinY_P; maxdim=maxdim, cutoff=cutoff)
            C4 = +(C4, -c4; maxdim=maxdim, cutoff=cutoff)
            isempty(c_kwargs) || ITensorMPS.truncate!(C4; c_kwargs...)
            @debug "get_C_op_MPO_from_P: C4 done"
            progress === nothing || progress(:C4, C4)
            step!()

            if pk_mpo !== nothing
                wrap(C) = apply(pk_mpo, apply(C, pk_mpo; maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff)
                C1 = wrap(C1)
                C2 = wrap(C2)
                C3 = wrap(C3)
                C4 = wrap(C4)
                @debug "get_C_op_MPO_from_P: PK wrapping done"
            end

            calculate_chern_number = uc -> begin
                sum(sub -> begin
                    alpha  = (uc - 1) * n_sub + sub
                    α      = make_alpha_mps(alpha)
                    x      = xfunc(alpha - 1, L_chain)
                    y      = yfunc(alpha - 1, L_chain)
                    cos_x, sin_x = cos(x / Λ), sin(x / Λ)
                    cos_y, sin_y = cos(y / Λ), sin(y / Λ)
                    ch  =  cos_x * cos_y * inner(α', C1, α)
                    ch +=  sin_x * sin_y * inner(α', C2, α)
                    ch -=  cos_x * sin_y * inner(α', C3, α)
                    ch -=  sin_x * cos_y * inner(α', C4, α)
                    ch * 2im * π * Λ^2
                end, 1:n_sub) / A_cell |> _hermitian_diag
            end
        end

    else
        # Flat mode: build global position MPOs directly from xfunc/yfunc
        x_op_p = get_diagonal_mpo(L, pos_sites, i -> xfunc_pos(i - 1, L_chain))
        y_op_p = get_diagonal_mpo(L, pos_sites, i -> yfunc_pos(i - 1, L_chain))
        x_op   = to_device(has_sub ? postpend_op(x_op_p, sub_s, I_mat) : x_op_p, device_type)
        y_op   = to_device(has_sub ? postpend_op(y_op_p, sub_s, I_mat) : y_op_p, device_type)

        T1   = apply(Q, apply(x_op, apply(P, apply(y_op, Q;
                     maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff);
                     maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff)
        T2   = apply(P, apply(x_op, apply(Q, apply(y_op, P;
                     maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff);
                     maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff)
        C_op = 2im * π * +(T1, -1.0 * T2; maxdim=maxdim, cutoff=cutoff)
        if pk_mpo !== nothing
            C_op = apply(pk_mpo, apply(C_op, pk_mpo; maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff)
        end
        ITensorMPS.truncate!(C_op; _trunc_kwargs(flat_trunc, maxdim, cutoff)...)
        step!()

        calculate_chern_number = uc -> begin
            sum(sub -> begin
                alpha = (uc - 1) * n_sub + sub
                α     = make_alpha_mps(alpha)
                inner(α', C_op, α)
            end, 1:n_sub) / A_cell |> _hermitian_diag
        end
    end

    return calculate_chern_number
end


# ============================================================
# 5. 2D Chern marker from a TBHamiltonian
# ============================================================

"""
    chern_marker(H::TBHamiltonian, xfunc=nothing, yfunc=nothing;
                 method=:kpm, fermi=0.0, l=nothing, Lambda=10,
                 Ncheb=300, maxdim=500, cutoff=1e-8,
                 Nel=nothing, quenched=true, sequential=false) -> Function

High-level wrapper: compute the ground-state projector via `method` and
return the Chern marker closure from `get_C_op_MPO_from_P`.

`xfunc(i, L_chain)` and `yfunc(i, L_chain)` accept a **0-indexed** physical
site number `i` and return raw x/y coordinates.  Both default to `nothing`,
in which case they are auto-derived:

- If `H.geometry_uc` is set (sublattice models: honeycomb, kagome, lieb, dice,
  ssh_sublattice): uses `geometry_uc(i+1)[1/2]`, which returns the same
  Bravais unit-cell position for all sublattice atoms in the same UC.
- Otherwise falls back to `H.geometry(i+1)[1/2]`.

For a sublattice model the functions still receive the physical site number
(`0 … n_sub·2^L − 1`), while the returned closure takes a unit-cell number
`uc ∈ 1 … 2^L` and sums over the `n_sub` atoms of that cell (see Returns).

Reuses `H._tn_cache` or `H._density_cache` when available.  `maxdim` and
`cutoff` are forwarded uniformly to the projector computation and to all
MPO multiplications in the Chern marker assembly.

`Lambda` is the quenching period Λ (see `get_C_op_MPO_from_P`). The keywords `Λ` and
`Nchebychev` and the value `method=:KPM` (the spellings up to v0.1.1, when this
function was `get_C`) are deprecated aliases of `Lambda`, `Ncheb` and `:kpm`; passing
an old and a new spelling together is an `ArgumentError` (up to v0.1.1 `Lambda`
silently won over `Λ`).

`sequential=true` (quenched mode only) skips the C1–C4 MPO×MPO products and
instead applies `P` and the position operators to each basis state inside the
closure: cheaper setup, more MPO–MPS applies per evaluated unit cell.

See `get_C_op_MPO_from_P` for full documentation of the remaining arguments.

# Returns
`calculate_chern_number(uc::Int) -> ComplexF64` where `uc` is a 1-indexed
unit cell number; the closure sums the marker over all `n_sub` sublattice
atoms in that UC.  The value is real (zero imaginary part; see
`get_C_op_MPO_from_P`); `real(·)` gives the density as a `Float64`.
"""
function chern_marker(H::TBHamiltonian, xfunc=nothing, yfunc=nothing;
               method::Symbol   = :kpm,
               fermi::Real      = 0.0,
               l                = nothing,
               Lambda::Union{Nothing,Real} = nothing,
               Ncheb::Union{Nothing,Int}   = nothing,
               maxdim::Int      = 500,
               cutoff::Float64  = 1e-8,
               Nel              = nothing,
               quenched::Bool   = true,
               sequential::Bool = false,
               Λ::Union{Nothing,Real}         = nothing,   # deprecated: Lambda
               Nchebychev::Union{Nothing,Int} = nothing)   # deprecated: Ncheb
    Λ_val = Float64(_renamed_kw(:chern_marker, :Lambda, Lambda, :Λ, Λ, 10))
    Ncheb = _renamed_kw(:chern_marker, :Ncheb, Ncheb, :Nchebychev, Nchebychev, 300)
    _require_binary_position_space(H, "chern_marker")
    if xfunc === nothing || yfunc === nothing
        geom = H.geometry_uc !== nothing ? H.geometry_uc :
               H.geometry   !== nothing ? H.geometry   :
               error("H has no geometry function; provide xfunc and yfunc explicitly.")
        xfunc === nothing && (xfunc = (i, _) -> geom(i + 1)[1])
        yfunc === nothing && (yfunc = (i, _) -> geom(i + 1)[2])
    end
    P = _get_projector(H; method=method, fermi=fermi, Ncheb=Ncheb,
                       maxdim=maxdim, cutoff=cutoff, Nel=Nel)
    return get_C_op_MPO_from_P(P, H.L, H.sites, xfunc, yfunc;
                                l=l, Lambda=Λ_val, maxdim=maxdim, cutoff=cutoff,
                                quenched=quenched, sequential=sequential)
end


# ============================================================
# 6. Valley operator and valley Chern number (honeycomb)
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
    _require_binary_position_space(H, "get_valley_operator")
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
    valley_chern_marker(H, xfunc=nothing, yfunc=nothing;
                        valley=:K, use_sign=true, method=:mcweeny, fermi=0.0, l=nothing,
                        Lambda=10, Ncheb=300, maxdim=500, cutoff=1e-8,
                        Nel=nothing, quenched=true, sequential=false) -> Function

Compute the valley-resolved Chern marker and return a closure
`calculate_valley_chern(uc::Int) -> ComplexF64`.

The valley operator V is built automatically via `get_valley_operator(H)`.
Its sign S = sign(V) is computed via `sign_mpo` (McWeeny purification),
giving eigenvalues in {−1, +1}.  The occupied K-valley projector is then

    PK = P · (I ± S) / 2 · P

where `P` is the ground-state projector and the sign is chosen by `valley`
(`:K` → +, `:K_prime` → −).  The PK is symmetrized and passed as `pk_mpo`
to `get_C_op_MPO_from_P`, which evaluates the Chern marker as
`⟨α|PK C_op PK|α⟩`.

# Arguments
- `valley`   : `:K` or `:K_prime`.
- `use_sign` : if `true` (default), sharpen V to eigenvalues {−1,+1} via
               `sign_mpo` before forming the valley projector.  Set to
               `false` to use V directly (cheaper but less accurate when
               the valley operator spectrum is not already close to ±1).

All remaining arguments are forwarded to `_get_projector` and
`get_C_op_MPO_from_P`; see those functions for documentation. The keywords `Λ` and
`Nchebychev` and the value `method=:KPM` (the spellings up to v0.1.1, when this
function was `get_valley_C`) are deprecated aliases of `Lambda`, `Ncheb` and `:kpm`.
"""
function valley_chern_marker(H::TBHamiltonian,
                      xfunc=nothing, yfunc=nothing;
                      valley::Symbol  = :K,
                      use_sign::Bool  = true,
                      method::Symbol  = :mcweeny,
                      fermi::Real     = 0.0,
                      l               = nothing,
                      Lambda::Union{Nothing,Real} = nothing,
                      Ncheb::Union{Nothing,Int}   = nothing,
                      maxdim::Int     = 500,
                      cutoff::Float64 = 1e-8,
                      Nel             = nothing,
                      quenched::Bool  = true,
                      sequential::Bool = false,
                      Λ::Union{Nothing,Real}         = nothing,   # deprecated: Lambda
                      Nchebychev::Union{Nothing,Int} = nothing)   # deprecated: Ncheb
    Λ     = _renamed_kw(:valley_chern_marker, :Lambda, Lambda, :Λ, Λ, 10)
    Ncheb = _renamed_kw(:valley_chern_marker, :Ncheb, Ncheb, :Nchebychev, Nchebychev, 300)
    _require_binary_position_space(H, "valley_chern_marker")
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
    S     = use_sign ? sign_mpo(V_mpo, collect(H.sites); maxdim=maxdim, cutoff=cutoff) : V_mpo
    P     = _get_projector(H; method=method, fermi=fermi, Ncheb=Ncheb,
                           maxdim=maxdim, cutoff=cutoff, Nel=Nel)
    vsign = valley == :K ? 1.0 : -1.0
    I_mpo = MPO(H.sites, "Id")
    PK_v  = 0.5 * +(I_mpo, vsign * S; maxdim=maxdim, cutoff=cutoff)
    PK    = apply(P, apply(PK_v, P; maxdim=maxdim, cutoff=cutoff); maxdim=maxdim, cutoff=cutoff)
    PK    = 0.5 * +(PK, dag(swapprime(PK, 0, 1)); maxdim=maxdim, cutoff=cutoff)

    return get_C_op_MPO_from_P(P, H.L, H.sites, xfunc, yfunc;
                                l=l, Lambda=Λ, maxdim=maxdim, cutoff=cutoff,
                                quenched=quenched, sequential=sequential,
                                pk_mpo=PK)
end


# ============================================================
# 7. Thouless charge pump — 1D adiabatic invariant
# ============================================================
#
# Computes the pumped charge per cycle (= Chern number) from the local marker
#
#   M1Q(r, t) = ⟨r| P(t) U†(t) x̂ U(t) P(t) |r⟩,   C = M1Q(r, T) − M1Q(r, 0)
#
# where U(t) is the adiabatic evolution generated by h(t) = [Ṗ(t), P(t)] and
# propagated with a second-order Taylor step.  P(t) is supplied as an array of
# MPOs computed at Nt evenly-spaced time steps; Ṗ(t) is estimated by central
# finite differences (one-sided at the endpoints).
#
# == Building blocks ==
#   get_pump_xop       — position operator MPO (flat or quenched)
#   thouless_pump      — propagate U over a P_array and evaluate M1Q
#   get_thouless_pump  — high-level: builds P(t) then calls thouless_pump


"""
    get_pump_xop(L, sites, xfunc; quenched=false, Lambda=-1.0) -> MPO

Diagonal position operator MPO for the Thouless pump formula.

`xfunc(i, N)` accepts a **0-indexed** site index `i ∈ 0…N-1` and the chain
length `N = 2^L`, and returns the raw coordinate.  For a 1-indexed chain:
`xfunc(i, N) = Float64(i + 1)`.

- `quenched=false` (default): diagonal entries are `xfunc(i, N)` directly.
- `quenched=true`: entries are `Λ * sin(xfunc(i, N) / Λ)` with Λ = `Lambda`, which
  smooths the discontinuity at PBC at the cost of a `Λ` prefactor.  A negative
  `Lambda` (the default `-1.0`) means `Λ = N` (one full period), giving
  `sin(x/N) * N ≈ x` for `x ≪ N`.

`Λ`, the spelling up to v0.1.1, is a deprecated alias of `Lambda`.
"""
function get_pump_xop(L::Int, sites::Vector{<:Index}, xfunc;
                      quenched::Bool = false,
                      Lambda::Union{Nothing,Real} = nothing,
                      Λ::Union{Nothing,Real}      = nothing)   # deprecated: Lambda
    Λ     = _renamed_kw(:get_pump_xop, :Lambda, Lambda, :Λ, Λ, -1.0)
    N     = 2^L
    Λ_val = Λ < 0 ? Float64(N) : Λ
    if quenched
        return Λ_val * get_sinx_op(L, sites, N, Λ_val, xfunc)
    else
        return get_diagonal_mpo(L, sites, i -> xfunc(i - 1, N))
    end
end


"""
    thouless_pump(P_array, dt, x_op, sites; r_center, maxdim=100, cutoff=1e-8,
                  verbose=false, return_trajectory=false)
        -> Float64 or (Float64, Vector{Float64})

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
- `r_center`          : 0-indexed bulk site at which to evaluate M1Q (required).
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
                      P_method=:mcweeny, fermi=0.0, Ncheb=200,
                      maxdim=100, cutoff=1e-8,
                      quenched=false, Lambda=-1.0,
                      Nel=nothing, r_center=nothing, verbose=false) -> Float64

High-level Thouless pump: build `P(t_k)` for `k = 0…Nt-1` via `P_method`,
then compute the M1Q invariant C = M1Q(T) − M1Q(0).

**Arguments**
- `H_of_t`   : `t -> TBHamiltonian` — all calls must share the same site indices
               (pass `ref_sites` to `get_Hamiltonian` inside the factory).
- `Nt`       : number of time steps.
- `T`        : period of the pump cycle.
- `xfunc`    : coordinate function `(i, N) -> Float64`, 0-indexed.
- `P_method` : `:mcweeny`, `:sp2`, or `:kpm`; `fermi`, `Ncheb` and `Nel` are
               passed to `_get_projector` with it.
- `r_center` : 0-indexed bulk site for M1Q evaluation; defaults to `N ÷ 2`.
- `quenched` : `false` = flat x̂; `true` = sin-quenched (removes PBC discontinuity).
- `Lambda`   : quenching period of the sin-quenched x̂ (see `get_pump_xop`).
- `verbose`  : print progress.

The keywords `Nchebychev` and `Λ` and the value `P_method=:KPM` (the spellings up to
v0.1.1) are deprecated aliases of `Ncheb`, `Lambda` and `:kpm`.
"""
function get_thouless_pump(H_of_t::Function, Nt::Int, T::Real, xfunc;
                           P_method::Symbol             = :mcweeny,
                           fermi::Real                  = 0.0,
                           Ncheb::Union{Nothing,Int}    = nothing,
                           maxdim::Int                  = 100,
                           cutoff::Float64              = 1e-8,
                           quenched::Bool               = false,
                           Lambda::Union{Nothing,Real}  = nothing,
                           Nel                          = nothing,
                           r_center::Union{Nothing,Int} = nothing,
                           verbose::Bool                = false,
                           Nchebychev::Union{Nothing,Int} = nothing,   # deprecated: Ncheb
                           Λ::Union{Nothing,Real}       = nothing)     # deprecated: Lambda
    Ncheb = _renamed_kw(:get_thouless_pump, :Ncheb, Ncheb, :Nchebychev, Nchebychev, 200)
    Λ     = _renamed_kw(:get_thouless_pump, :Lambda, Lambda, :Λ, Λ, -1.0)
    dt    = T / Nt
    H0    = H_of_t(0.0)
    sites = H0.sites
    x_op  = get_pump_xop(H0.L, H0.sites, xfunc; quenched=quenched, Lambda=Λ)
    rc    = isnothing(r_center) ? (2^H0.L) ÷ 2 : r_center

    P_array = MPO[]
    for k in 0:(Nt - 1)
        t_k = k * dt
        verbose && println("Building P(t=$(round(t_k; digits=4)))  [$(k+1)/$Nt]...")
        H_k = H_of_t(t_k)
        P_k = _get_projector(H_k; method=P_method, fermi=fermi,
                              Ncheb=Ncheb, maxdim=maxdim,
                              cutoff=cutoff, Nel=Nel)
        push!(P_array, P_k)
    end

    verbose && println("Computing M1Q invariant (r_center=$rc)...")
    return thouless_pump(P_array, dt, x_op, sites;
                         r_center=rc, maxdim=maxdim, cutoff=cutoff, verbose=verbose)
end
