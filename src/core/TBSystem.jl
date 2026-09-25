# TBSystem.jl — central struct and constructor for tight-binding systems
#
# Provides TBHamiltonian, which wraps the Hamiltonian MPO together with
# metadata (geometry, KPM scale) and lazy caches for Chebyshev moments
# and the density matrix.  All observable methods (get_DoS, get_density,
# get_Chern, get_bands …) dispatch on this struct.

# ============================================================
# Position-space policy types
# ============================================================

"""
    AbstractPositionSpace

Policy object describing how physical positions are embedded in the tensor-product
register. `BinaryPositionSpace` is the ordinary `N = 2^L` quantics basis. Other
position spaces (see `position_spaces/`) specialize `physical_projector`,
`physical_site_state`, `site_axis`, and `site_permutation` after `TBHamiltonian`
is defined below.
"""
abstract type AbstractPositionSpace end

"""Ordinary binary position register containing all `2^L` basis states."""
struct BinaryPositionSpace <: AbstractPositionSpace end

# ============================================================
# TBHamiltonian struct
# ============================================================

"""
    TBHamiltonian

Central object representing a tight-binding Hamiltonian in the quantics MPO framework.

Fields
------
**Core**
- `L`        : number of position qubit sites (log₂ of the physical system size)
- `N`        : number of physical sites / unit cells (`2^L` for the binary basis)
- `sites`    : ITensor site indices (position qubits + any auxiliary DOF indices)
- `mpo`      : accumulated Hamiltonian as an ITensor MPO
- `geometry` : function `i -> position_vector` (1-indexed); `nothing` for implicit 1D
- `position_space`: policy describing the physical basis inside the tensor register

**KPM spectral bounds**
- `scale`    : energy half-bandwidth; `H/scale` has spectrum in `[-1, 1]`.
               Set analytically at construction for standard geometries.
               Reset to `0.0` by `_invalidate_cache!` after any modification —
               `_ensure_scale!` then re-estimates it on demand via DMRG.
- `center`   : spectral centre; `0.0` for particle-hole symmetric Hamiltonians.

**Auxiliary DOF indices** (`nothing` until the corresponding `add_*!` is called)
- `spin_s`       : dim-2 spin index (prepended by `add_spin!`)
- `nambu_s`      : dim-2 Nambu index (prepended by `add_superconductivity!`)
- `layer_s`      : dim-k layer index (set by bilayer/multilayer constructors)
- `sublattice_s` : dim-k sublattice index (set by kagomé/Lieb/honeycomb/dice constructors)
- `aux_side`     : `:pre` or `:post` — position of the outermost aux index in `sites`

**Lazy caches** (cleared by `_invalidate_cache!` whenever `mpo` changes)
- `_tn_cache`      : MPO Chebyshev list; set by `KPM_Tn(H, N; mode=:mpo)`
- `_tn_mps_cache`  : MPS Chebyshev state list; set by `KPM_Tn(H, N; mode=:mps, psi0=…)`
- `_tn_Ncheb`      : order of the cached Chebyshev expansion
- `_density_cache` : cached density-matrix MPO

**Stored interaction MPOs** (set via [`add_interaction!`](@ref))
- `interaction_mpo` : Hartree/CDW interaction kernel (fully scaled); used by `get_scf(H, channel)`
- `fock_mpo`        : Fock/exchange interaction kernel (fully scaled)

Do not construct directly — use [`get_Hamiltonian`](@ref).
"""
mutable struct TBHamiltonian
    L        :: Int
    N        :: Int
    sites    :: Vector{<:Index}
    mpo      :: MPO
    geometry    :: Union{Nothing, Function}   # i -> pos_vec (1-indexed, i=1…n_sub·N)
    geometry_uc :: Union{Nothing, Function}   # i -> Bravais pos; same for all sublattice atoms in a UC
    scale    :: Float64    # energy half-bandwidth; 0.0 = not yet determined (triggers lazy DMRG)
    center   :: Float64    # spectral center; 0.0 for symmetric spectra
    # ---- auxiliary indices (nothing until add_spin!/add_superconductivity!) ----
    spin_s        :: Union{Nothing, Index}
    nambu_s       :: Union{Nothing, Index}
    layer_s       :: Union{Nothing, Index}    # set by bilayer/multilayer constructors
    sublattice_s  :: Union{Nothing, Index}    # set by kagomé/Lieb constructors
    aux_side :: Symbol                        # :pre (aux at front) or :post (aux at back)
    # ---- lazy caches (invalidated whenever mpo changes) ----
    _tn_cache      :: Union{Nothing, Vector{MPO}}   # MPO Chebyshev list (mode=:mpo)
    _tn_mps_cache  :: Union{Nothing, Vector{MPS}}   # MPS Chebyshev list (mode=:mps)
    _tn_Ncheb      :: Int
    _density_cache :: Union{Nothing, MPO}
    # ---- stored interaction MPOs (set via add_interaction!) ----
    interaction_mpo :: Union{Nothing, MPO}
    fock_mpo        :: Union{Nothing, MPO}
    Lx             :: Union{Nothing, Int}           # x-qubit count for 2D (Ly = L - Lx); nothing for 1D
    position_space :: AbstractPositionSpace
end

# Backward-compatible full constructor (pre-position_space callers).
TBHamiltonian(L, N, sites, mpo, geometry, geometry_uc, scale, center,
              spin_s, nambu_s, layer_s, sublattice_s, aux_side,
              _tn_cache, _tn_mps_cache, _tn_Ncheb, _density_cache,
              interaction_mpo, fock_mpo, Lx) =
    TBHamiltonian(L, N, sites, mpo, geometry, geometry_uc, scale, center,
                  spin_s, nambu_s, layer_s, sublattice_s, aux_side,
                  _tn_cache, _tn_mps_cache, _tn_Ncheb, _density_cache,
                  interaction_mpo, fock_mpo, Lx, BinaryPositionSpace())

# Backward-compatible 17-arg constructor (pre-interaction_mpo/pre-fock_mpo/pre-Lx callers);
# appends nothing, nothing, nothing.
TBHamiltonian(L, N, sites, mpo, geometry, geometry_uc, scale, center,
              spin_s, nambu_s, layer_s, sublattice_s, aux_side,
              _tn_cache, _tn_mps_cache, _tn_Ncheb, _density_cache) =
    TBHamiltonian(L, N, sites, mpo, geometry, geometry_uc, scale, center,
                  spin_s, nambu_s, layer_s, sublattice_s, aux_side,
                  _tn_cache, _tn_mps_cache, _tn_Ncheb, _density_cache,
                  nothing, nothing, nothing)

"""
    TBHamiltonian(H::TBHamiltonian; field=value, ...) -> TBHamiltonian

Copy `H` field by field, replacing the fields named as keywords. Every other field,
including `Lx`, `interaction_mpo`, `fock_mpo` and `position_space`, keeps the value
from `H`; MPOs and index vectors are shared, not deep-copied. The lazy caches
(`_tn_cache`, `_tn_mps_cache`, `_tn_Ncheb`, `_density_cache`) start empty unless
passed explicitly, since a copy usually carries a different operator.

```julia
Hbdg = TBHamiltonian(H; sites=[nambu_s; spin_s; H.sites], mpo=bdg_mpo,
                     spin_s=spin_s, nambu_s=nambu_s, aux_side=:pre)
```
"""
function TBHamiltonian(H::TBHamiltonian; kwargs...)
    names = fieldnames(TBHamiltonian)
    for k in keys(kwargs)
        k in names || throw(ArgumentError("TBHamiltonian has no field `$k`"))
    end
    empty_caches = (_tn_cache=nothing, _tn_mps_cache=nothing, _tn_Ncheb=0,
                    _density_cache=nothing)
    vals = map(names) do f
        haskey(kwargs, f)       ? kwargs[f] :
        haskey(empty_caches, f) ? empty_caches[f] : getfield(H, f)
    end
    return TBHamiltonian(vals...)
end

# ============================================================
# Position-space interface
# ============================================================

"""
    ambient_dimension(H) -> Integer

Dimension of the position tensor register before any physical-subspace projection.
This is `2^H.L` for the quantics encodings supported by TensorBinding. Projected
position spaces may return a `BigInt` when the ambient register exceeds `Int`.
"""
ambient_dimension(H::TBHamiltonian) = ambient_dimension(H.position_space, H)
ambient_dimension(::BinaryPositionSpace, H::TBHamiltonian) = 2^H.L

"""
    physical_projector(H) -> MPO

Identity operator on the physical position space. For ordinary binary systems this
is the full identity; projected encodings specialize this method and return their
valid-state projector.
"""
physical_projector(H::TBHamiltonian) = physical_projector(H.position_space, H)
physical_projector(::BinaryPositionSpace, H::TBHamiltonian) = MPO(H.sites, "Id")

"""
    physical_site_state(H, x) -> MPS

Product-state probe for 1-indexed physical position `x`. Auxiliary and two-particle
spaces use their dedicated probe constructors.
"""
physical_site_state(H::TBHamiltonian, x::Integer) =
    physical_site_state(H.position_space, H, x)

function physical_site_state(::BinaryPositionSpace, H::TBHamiltonian, x::Integer)
    1 <= x <= H.N || throw(BoundsError(1:H.N, x))
    length(H.sites) == H.L ||
        error("physical_site_state currently requires a position-only TBHamiltonian.")
    return binary_to_MPS(x - 1, H.L, H.sites)
end

"""Return the plotting axis for physical positions or an encoding-defined ordering."""
function site_axis(H::TBHamiltonian; ordering::Symbol=:physical, kwargs...)
    return site_axis(H.position_space, H; ordering, kwargs...)
end

function site_axis(::BinaryPositionSpace, H::TBHamiltonian;
                   ordering::Symbol=:physical, kwargs...)
    ordering === :physical ||
        throw(ArgumentError("ordering=:$ordering is not available for BinaryPositionSpace"))
    return collect(0:(H.N - 1))
end

"""Return the 1-based physical-site permutation associated with a plotting ordering."""
function site_permutation(H::TBHamiltonian; ordering::Symbol=:physical, kwargs...)
    return site_permutation(H.position_space, H; ordering, kwargs...)
end

function site_permutation(::BinaryPositionSpace, H::TBHamiltonian;
                          ordering::Symbol=:physical, kwargs...)
    ordering === :physical ||
        throw(ArgumentError("ordering=:$ordering is not available for BinaryPositionSpace"))
    return collect(1:H.N)
end

_is_binary_position_space(H::TBHamiltonian) = H.position_space isa BinaryPositionSpace

function _require_binary_position_space(H::TBHamiltonian, api::AbstractString)
    _is_binary_position_space(H) && return nothing
    throw(ArgumentError("$api is not yet supported for $(typeof(H.position_space)); " *
                        "the first projected-space release supports CPU KPM DOS/LDOS only."))
end

# Backward-compatible 16-arg constructor (pre-geometry_uc callers); inserts geometry_uc=nothing.
TBHamiltonian(L, N, sites, mpo, geometry, scale, center,
              spin_s, nambu_s, layer_s, sublattice_s, aux_side,
              _tn_cache, _tn_mps_cache, _tn_Ncheb, _density_cache) =
    TBHamiltonian(L, N, sites, mpo, geometry, nothing, scale, center,
                  spin_s, nambu_s, layer_s, sublattice_s, aux_side,
                  _tn_cache, _tn_mps_cache, _tn_Ncheb, _density_cache)

# Backward-compatible 15-arg constructor (pre-sublattice_s callers); inserts sublattice_s=nothing.
TBHamiltonian(L, N, sites, mpo, geometry, scale, center,
              spin_s, nambu_s, layer_s, aux_side, _tn_cache, _tn_mps_cache, _tn_Ncheb, _density_cache) =
    TBHamiltonian(L, N, sites, mpo, geometry, nothing, scale, center,
                  spin_s, nambu_s, layer_s, nothing, aux_side, _tn_cache, _tn_mps_cache, _tn_Ncheb, _density_cache)

# Backward-compatible 14-arg constructor (pre-sublattice_s, pre-_tn_mps_cache callers).
TBHamiltonian(L, N, sites, mpo, geometry, scale, center,
              spin_s, nambu_s, layer_s, aux_side, _tn_cache, _tn_Ncheb, _density_cache) =
    TBHamiltonian(L, N, sites, mpo, geometry, nothing, scale, center,
                  spin_s, nambu_s, layer_s, nothing, aux_side, _tn_cache, nothing, _tn_Ncheb, _density_cache)

# Backward-compatible 13-arg constructor (pre-sublattice_s, pre-aux_side callers); defaults to :pre.
TBHamiltonian(L, N, sites, mpo, geometry, scale, center,
              spin_s, nambu_s, layer_s, _tn_cache, _tn_Ncheb, _density_cache) =
    TBHamiltonian(L, N, sites, mpo, geometry, nothing, scale, center,
                  spin_s, nambu_s, layer_s, nothing, :pre, _tn_cache, nothing, _tn_Ncheb, _density_cache)

# ============================================================
# Cache management
# ============================================================

"""
    _invalidate_cache!(H) -> H

Clear all cached intermediate results.  Called automatically whenever `mpo` changes
(e.g. `add_hopping!`, `add_onsite!`, `add_zeeman!`, `add_superconductivity!`).

Clears: `_tn_cache`, `_tn_mps_cache`, `_tn_Ncheb`, `_density_cache`.
Resets `scale` and `center` to `0.0` so that `_ensure_scale!` re-estimates them via
DMRG on the next KPM call.
"""
function _invalidate_cache!(H::TBHamiltonian)
    H._tn_cache      = nothing
    H._tn_mps_cache  = nothing
    H._tn_Ncheb      = 0
    H._density_cache = nothing
    # Spectrum changed — force re-estimation of scale/center on next KPM call.
    H.scale  = 0.0
    H.center = 0.0
    return H
end

"""
    truncate!(H::TBHamiltonian; cutoff=1e-10, maxdim=nothing) -> H

Truncate the Hamiltonian MPO in-place using `ITensorMPS.truncate!`.
Invalidates all caches (Chebyshev list, density matrix, scale/center).

Useful after a series of `add_hopping!` / `add_onsite!` calls that may
have inflated the bond dimension.
"""
function truncate!(H::TBHamiltonian; cutoff::Real = 1e-10, maxdim = nothing)
    old_scale, old_center = H.scale, H.center
    kwargs = maxdim === nothing ? (cutoff=cutoff,) : (cutoff=cutoff, maxdim=maxdim)
    ITensorMPS.truncate!(H.mpo; kwargs...)
    _invalidate_cache!(H)
    if !_is_binary_position_space(H)
        H.scale, H.center = old_scale, old_center
    end
    return H
end

# ============================================================
# Constructor
# ============================================================

"""
    get_Hamiltonian(geometry, params; L, [scale, tol, maxdim, kwargs...])
        -> TBHamiltonian

Build a `TBHamiltonian` from a named geometry and model parameters.

Supported geometry strings
--------------------------
| `geometry`    | `params`                       | Extra kwargs                  |
|---------------|--------------------------------|-------------------------------|
| `"chain_1d"`  | hopping amplitude `t::Number`  | direct MPO, no QTCI; use `add_onsite!` for potentials |
| `"square_2d"` | hopping amplitude `t::Number`  | `Lx`, `Ly` (default `L÷2` each) |
| `"haldane"`   | `(t2, phi, M)` NamedTuple      | `rs` (N×2 positions from `honeycomb_positions`, required) |
| `"custom"`    | hopping function `f(i,j)`      | `geometry`, `scale` (required), `type` |
| `"fibonacci"` | `(A, B[, t, onsite])` NamedTuple | `model=:hopping/:onsite`, `boundary=:periodic/:open` |
| `"metallic_mean"` | `(A, B[, t, onsite])` NamedTuple | `m` (required; `m=2` silver mean), `model`, `boundary` |
| `"kbonacci"` | `(A, B, C, ...[, t, onsite])` or `(values=(a_1, ..., a_k)[, t, onsite])` NamedTuple | `k` (required; `k=3` Tribonacci), `model`, `boundary` |
| `"kagome"`    | hopping amplitude `t::Number`  | `Lx`, `Ly`; 3-atom unit cell, sublattice index postpended |
| `"lieb"`      | hopping amplitude `t::Number`  | `Lx`, `Ly`; 3-atom unit cell, sublattice index postpended |

`"haldane"` is the textbook, C3-symmetric Haldane model `⟨i|H|j⟩ = t2 exp(i phi ν_ij)`
(Dirac masses `-M ± 3√3 t2 sin(phi)`, see [`haldane_hoppingf`](@ref)); it refuses an `rs`
whose sites are not on the `honeycomb_positions` lattice.

For `"kagome"` and `"lieb"`, `L = Lx + Ly` counts only the position qubits;
the total atom count is `3 × 2^L`.  The sublattice index is stored in
`H.sublattice_s` with `H.aux_side = :post`.  `H.geometry` returns the full
real-space position of each atom (1-indexed over all `3 × 2^L` atoms).

Common keyword arguments
------------------------
- `L`      : number of qubit sites (system size = 2^L)
- `scale`  : energy half-bandwidth for KPM normalisation (estimated if `nothing`)
- `tol`    : QTCI tolerance (default `1e-8`)
- `maxdim` : maximum MPO bond dimension after construction (default `15`)

Examples
--------
```julia
H = get_Hamiltonian("chain_1d", 1.0;    L=10)
H = get_Hamiltonian("square_2d", 1.0;   L=10, Lx=32)

rs = honeycomb_positions(10)
H  = get_Hamiltonian("haldane", (t2=0.2, phi=π/2, M=0.0); L=10, rs=rs)

H  = get_Hamiltonian("custom", (i,j) -> ...; L=10, scale=5.0, geometry=rs)
Hf = get_Hamiltonian("fibonacci", (A=1.0, B=2.0); L=8, model=:hopping)
Hs = get_Hamiltonian("metallic_mean", (A=1.0, B=2.0); L=8, m=2)   # silver mean
Ht = get_Hamiltonian("kbonacci", (A=0.64, B=0.8, C=1.0); L=8, k=3)   # Tribonacci
```

After construction, add further interaction terms with
[`add_hopping!`](@ref) and [`add_onsite!`](@ref).
"""
function get_Hamiltonian(geometry::String, params;
                         L::Int,
                         scale=nothing,
                         tol=1e-8,
                         maxdim=15,
                         ref_sites::Union{Nothing,Vector{<:Index}}=nothing,
                         kwargs...)
    if geometry == "fibonacci"
        ref_sites === nothing ||
            throw(ArgumentError("ref_sites is not supported for FibonacciPositionSpace"))
        return _build_fibonacci(params, L; scale, tol, maxdim, kwargs...)
    end
    if geometry == "metallic_mean"
        ref_sites === nothing ||
            throw(ArgumentError("ref_sites is not supported for MetallicMeanPositionSpace"))
        return _build_metallic_mean(params, L; scale, tol, maxdim, kwargs...)
    end
    if geometry == "kbonacci"
        ref_sites === nothing ||
            throw(ArgumentError("ref_sites is not supported for KBonacciPositionSpace"))
        return _build_kbonacci(params, L; scale, tol, maxdim, kwargs...)
    end

    sites = siteinds("Qubit", L)
    N     = 2^L

    if geometry == "chain_1d"
        return _build_chain_1d(params, L, N, sites; scale, tol, maxdim, kwargs...)

    elseif geometry == "haldane"
        return _build_haldane(params, L, N, sites; scale, tol, maxdim, kwargs...)

    elseif geometry == "custom"
        return _build_custom(params, L, N, sites; scale, tol, maxdim, kwargs...)

    # ---- multi-atom unit-cell lattices (kagomé, Lieb, honeycomb, dice) ----
    elseif geometry in ("kagome", "lieb", "honeycomb", "honeycomb_nnn", "dice")
        return _build_sublattice(geometry, params, L; scale, tol, maxdim, kwargs...)

    # ---- SSH with explicit sublattice index ----
    elseif geometry == "ssh_sublattice"
        t  = params isa Number                                          ? params     :
             params isa NamedTuple && hasfield(typeof(params), :t)     ? params.t   :
             params isa AbstractDict && haskey(params, :t)             ? params[:t] : 1.0
        d  = params isa NamedTuple && hasfield(typeof(params), :d)     ? params.d   :
             params isa AbstractDict && haskey(params, :d)             ? params[:d] : 0.0
        H  = ssh_sublattice_hamiltonian(L, t, d; cutoff=tol, maxdim=maxdim)
        isnothing(scale) || (H.scale = Float64(scale))
        return H

    # ---- preset models routed through build_hamiltonian ----
    elseif geometry in ("ssh", "aah", "uniform",
                        "square_2d", "hex_2d", "triangular_2d", "triangular_bravais",
                        "chern8", "chernhex", "qc2dsquare")
        return _build_preset(geometry, params, L, N, sites; scale, tol, maxdim, ref_sites, kwargs...)

    else
        known = ("chain_1d", "haldane", "custom", "fibonacci", "metallic_mean", "kbonacci",
                 "uniform", "ssh", "ssh_sublattice", "aah",
                 "square_2d", "hex_2d", "triangular_2d", "triangular_bravais",
                 "chern8", "chernhex", "qc2dsquare",
                 "kagome", "lieb", "honeycomb", "honeycomb_nnn", "dice")
        error("Unknown geometry \"$geometry\". Supported: $(join(known, ", ")).")
    end
end

# ============================================================
# Per-geometry builders (internal)
# ============================================================

# ---- Geometry functions (i -> position, 1-indexed) ----

_chain_geometry() = i -> Float64[i]

function _square_geometry(Nx)
    return i -> Float64[(i-1) % Nx, (i-1) ÷ Nx]
end

function _tri_geometry(Nx)
    function pos(i)
        ix = (i-1) % Nx
        iy = (i-1) ÷ Nx
        x  = Float64(ix) + 0.5 * (iy % 2)
        y  = iy * sqrt(3) / 2
        return Float64[x, y]
    end
    return pos
end

function _tri_bravais_geometry(Nx)
    function pos(i)
        ix = (i-1) % Nx
        iy = (i-1) ÷ Nx
        x  = Float64(ix) + 0.5 * iy
        y  = iy * sqrt(3) / 2
        return Float64[x, y]
    end
    return pos
end

function _hex_geometry(Nx)
    function pos(i)
        ix = (i-1) % Nx
        iy = (i-1) ÷ Nx
        x  = 3.0*(ix÷2) + Float64(ix%2) + (iy%2) * (Float64(ix%2) - 0.5)
        y  = iy * sqrt(3)/2
        return Float64[x, y]
    end
    return pos
end

function _build_chain_1d(t, L, N, sites;
                         scale=nothing,
                         tol=1e-8,
                         maxdim=15,
                         boundary::Symbol=:open,
                         bc=nothing)
    bc === nothing || (boundary = Symbol(bc))
    mpo = t * kinetic_1d_nn(L, sites; boundary=boundary)
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    sc  = something(scale, 2.5 * abs(t))
    return TBHamiltonian(L, N, sites, mpo, _chain_geometry(), sc, 0.0, nothing, nothing, nothing, nothing, 0, nothing)
end


# Sublattice sign of a honeycomb_positions site, read from the position: -1 on sublattice A
# (the one of site 1, at x ∈ 1.5ℤ), +1 on B (x ∈ 1.5ℤ + 1), i.e. σ = (-1)^(ix+iy+1). The index
# parity (-1)^i only tracks ix in that row-major layout (2^Lx is even).
_haldane_sigma(r) = isapprox(mod(r[1], 1.5), 1.0; atol=1e-6) ? 1 : -1

"""
    chirality(r1, r2) -> Int

Textbook Haldane sign `ν = ±1` of the next-nearest-neighbour hop between the sites at `r1`
and `r2` of the [`honeycomb_positions`](@ref) lattice: `ν = sign((d1 × d2)_z)` for the path
`r1 → k → r2` through their common nearest neighbour `k` (`d1 = r_k - r1`, `d2 = r2 - r_k`),
so `+1` when the path turns left at `k`.

On that lattice the bonds of an A site (x ∈ 1.5ℤ) point at 0° and ±120° and those of a B
site at 180° and ±60°, so `ν = -σ sign(sin 3θ)`, with `θ` the angle of `r2 - r1` and `σ` the
sublattice sign of `r1` (-1 on A, +1 on B). The three hops of a C3-related triple share one
sign, and `chirality(r2, r1) == -chirality(r1, r2)`. The result is meaningless for pairs
that are not next-nearest neighbours of that lattice.
"""
function chirality(r1, r2)
    δ = r2 .- r1
    # the six next-nearest directions sit at θ = ±30°, ±90°, ±150°, where sin 3θ = ±1
    s = sin(3 * atan(δ[2], δ[1])) > 0 ? 1 : -1
    return -_haldane_sigma(r1) * s
end

"""
    haldane_hoppingf(r1, r2, i, j; t2=0.2, phi=π/2, M=0.0) -> Number

Matrix element `⟨r1|H|r2⟩` of the textbook, C3-symmetric Haldane model
`H = Σ_ij H_ij c†_i c_j` on the honeycomb with bond length 1 laid out as in
[`honeycomb_positions`](@ref):

- on site: `M σ`, with `σ = -1` on the sublattice of site 1 (x ∈ 1.5ℤ) and `+1` on the other;
- nearest neighbours: `-1`;
- next-nearest neighbours: `t2 exp(i phi ν)`, with `ν = chirality(r1, r2) = sign((d1 × d2)_z)`
  for the path `r1 → k → r2` through the common nearest neighbour `k`.

The Dirac masses are `-M ± 3√3 t2 sin(phi)`, so the model is a Chern insulator for
`|M| < 3√3 |t2 sin(phi)|`. `i`, `j` are the site indices of the `f(i, j)` call pattern
(unused).

This is the convention of the manuscript's `build_APSOS_hamiltonian`: its `haldane_phases`
table, added with [`add_hopping_2D!`](@ref) on the `"honeycomb"` preset, gives
`⟨i|H|j⟩ = t2 exp(i phi ν_ij)` with the same `ν_ij`, so the same Chern number at the same
`t2` and `phi`. It puts `+M` rather than `-M` on its sublattice 1, which leaves the Chern
number unchanged. Earlier versions gave the four vertical next-nearest bonds `-ν`, which
made the Dirac masses `-M ± √3 t2 sin(phi)`.
"""
function haldane_hoppingf(r1, r2, i, j; t2 = 0.2, phi=pi/2, M=0.0)
    d = norm(r2 .- r1)
    if isapprox(d, 0.0; atol=1e-3)
        return M * _haldane_sigma(r1)
    elseif isapprox(d, 1.0; atol=1e-8)
        return -1.0 #t1
    elseif isapprox(d, √3; atol=1e-3)
        return t2 * cis(phi * chirality(r1, r2))
    else
        return 0.0
    end
end

# haldane_hoppingf reads the sublattice from x and the chirality from the bond angle, which is
# right only for sites of the honeycomb_positions lattice: rows at y ∈ (√3/2)ℤ and, once the
# 1.5 shift of odd rows is undone, A at x ∈ 3ℤ and B at x ∈ 3ℤ + 1. One O(N) pass, no search.
function _check_haldane_layout(rs, N; atol=1e-6)
    size(rs, 1) >= N && size(rs, 2) == 2 ||
        throw(ArgumentError("get_Hamiltonian(\"haldane\"): `rs` must be an N×2 position matrix " *
                            "with at least N = $N rows, got size $(size(rs)). Generate it with " *
                            "honeycomb_positions."))
    h = √3 / 2
    for i in 1:N
        x, y = rs[i, 1], rs[i, 2]
        ok = isfinite(x) && isfinite(y)
        if ok
            r  = round(Int, y / h)
            u  = mod(x - (isodd(r) ? 1.5 : 0.0), 3.0)
            ok = abs(y - r * h) <= atol && min(u, 3.0 - u, abs(u - 1.0)) <= atol
        end
        ok || throw(ArgumentError(
            "get_Hamiltonian(\"haldane\"): site $i of `rs`, $((x, y)), is not on the " *
            "honeycomb_positions lattice (bond length 1, one bond along x, sublattice A at " *
            "x ∈ 1.5ℤ). haldane_hoppingf reads the sublattice and the Haldane chirality from " *
            "the positions, which is only right there. Pass rs from honeycomb_positions (a " *
            "translate by a lattice vector also works), or build the model with hopping2MPO."))
    end
    return nothing
end

# Structural QTCI pivots for the Haldane matrix: every pair within R (√3 < R < 2, so
# on-site, NN and NNN) of the site nearest the centre of `rs` or of one of its neighbours.
# Around that site honeycomb_positions layouts show every sublattice / bond-direction class.
function _haldane_pivots(rs; R=1.8)
    c0   = vec(sum(rs; dims=1)) ./ size(rs, 1)
    c    = argmin([norm(rs[i, :] .- c0) for i in axes(rs, 1)])
    near = [i for i in axes(rs, 1) if norm(rs[i, :] .- rs[c, :]) <= 2R]
    rows = [i for i in near if norm(rs[i, :] .- rs[c, :]) <= R]
    return [(i, j) for i in rows for j in near if norm(rs[j, :] .- rs[i, :]) <= R]
end

# Entry (i, j) (1-based) of a Qubit MPO, site 1 = most significant bit.
function _mpo_entry(mpo::MPO, sites, i::Integer, j::Integer)
    L = length(sites)
    v = ITensor(1.0)
    for k in 1:L
        b = L - k
        v *= mpo[k] * onehot(sites[k]' => (((i - 1) >> b) & 1) + 1) *
                      onehot(sites[k]  => (((j - 1) >> b) & 1) + 1)
    end
    return scalar(v)
end

# Spot-check the compressed Haldane MPO against f at every pivot offset of a spread of
# rows (lattice extremes plus an odd golden-ratio stride through the bulk), so a QTCI
# that misses a bond class or an edge errors instead of returning a wrong Hamiltonian.
function _check_haldane_mpo(mpo, sites, f, rs, piv; tol=1e-8, nbulk=16)
    N    = size(rs, 1)
    x, y = rs[:, 1], rs[:, 2]
    s    = 2 * round(Int, 0.30901699437494745 * N) + 1
    rows = unique([argmin(x), argmax(x), argmin(y), argmax(y),
                   argmin(x .+ y), argmax(x .+ y), argmin(x .- y), argmax(x .- y),
                   (1 + mod(k * s, N) for k in 0:nbulk-1)...])
    offs = unique(j - i for (i, j) in piv)
    atol = max(1e-6, 10 * tol) * maximum(abs(f(i, j)) for (i, j) in piv)
    for i in rows, d in offs
        j = i + d
        1 <= j <= N || continue
        got, want = _mpo_entry(mpo, sites, i, j), f(i, j)
        abs(got - want) <= atol ||
            error("get_Hamiltonian(\"haldane\"): the QTCI-compressed MPO is wrong at entry " *
                  "($i, $j): $got instead of $want. Pass `rs` from honeycomb_positions, or " *
                  "build the MPO with hopping2MPO and initial_positions suited to this layout.")
    end
    return nothing
end

function _build_haldane(params, L, N, sites;
                        rs=nothing, scale=nothing, tol=1e-8, maxdim=15)
    @assert !isnothing(rs) "Haldane model requires keyword `rs` (N×2 position matrix). " *
                           "Generate it with `honeycomb_positions($L)`."
    _check_haldane_layout(rs, N)
    t2  = params.t2
    phi = params.phi
    M   = params.M
    f(i, j) = haldane_hoppingf(rs[Int(i), :], rs[Int(j), :],
                                Int(i), Int(j); t2=t2, phi=phi, M=M)
    # QTCI from structural pivots only. The default all-ones pivot plus 5 random ones made
    # the build depend on the global RNG, threw "maxsamplevalue is zero!" for M = 0 and
    # often missed a bond class (a wrong MPO with a small TCI error estimate).
    rsN = view(rs, 1:N, :)   # f only reads the first N rows
    piv = _haldane_pivots(rsN)
    mpo = hopping2MPO(f, N, sites; tol=tol, type=ComplexF64, initial_positions=piv,
                      nrandominitpivot=0, nsearchglobalpivot=0)
    _check_haldane_mpo(mpo, sites, f, rsN, piv; tol=tol)
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    # Gershgorin bound (t1 = 1): a site has |M| on site, ≤ 3 NN and ≤ 6 NNN hops, so the
    # spectral radius is ≤ 3 + 6|t2| + |M| (nearly reached at phi = 0, π); pad by 10%.
    sc  = something(scale, 1.1 * (3.0 + 6.0 * abs(t2) + abs(M)))
    rs_f = let m = Float64.(rs); i -> m[i, :]; end
    return TBHamiltonian(L, N, sites, mpo, rs_f, sc, 0.0, nothing, nothing, nothing, nothing, 0, nothing)
end


function _build_custom(f, L, N, sites;
                       geometry=nothing,
                       scale=nothing,
                       tol=1e-8,
                       maxdim=15,
                       type=ComplexF64)
    @assert !isnothing(scale) "`scale` must be provided for geometry=\"custom\"."
    geom_f = geometry isa Matrix ? (let m = Float64.(geometry); i -> m[i, :]; end) : geometry
    mpo = hopping2MPO(f, N, sites; tol=tol, type=type)
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    return TBHamiltonian(L, N, sites, mpo, geom_f, Float64(scale), 0.0, nothing, nothing, nothing, nothing, 0, nothing)
end

function _build_preset(geometry, params, L, N, sites;
                       scale=nothing, tol=1e-8, maxdim=15,
                       ref_sites::Union{Nothing,Vector{<:Index}}=nothing,
                       kwargs...)
    # Route through build_hamiltonian which dispatches on MODEL_REGISTRY.
    # params can be: a scalar, a NamedTuple, or a Dict — normalise to mparam_dict.
    dim = MODEL_REGISTRY[geometry][2]
    if dim == 1
        mpo = if params isa AbstractDict
            build_hamiltonian(geometry, L; mparam_dict=Dict{Symbol,Any}(params), kwargs...)
        elseif params isa NamedTuple
            build_hamiltonian(geometry, L; mparam_dict=Dict{Symbol,Any}(pairs(params)), kwargs...)
        elseif params isa Number
            # single-param shorthand: first required param
            req = MODEL_REGISTRY[geometry][3][1]
            build_hamiltonian(geometry, L; mparam_dict=Dict{Symbol,Any}(req => params), kwargs...)
        else
            build_hamiltonian(geometry, L; mparam_dict=Dict{Symbol,Any}(:t => params), kwargs...)
        end
    else
        # 2D: expect Lx and Ly in kwargs, or factorise L equally
        Lx = get(kwargs, :Lx, L ÷ 2)
        Ly = get(kwargs, :Ly, L - Lx)
        kw_filtered = Dict(k => v for (k, v) in kwargs if k ∉ (:Lx, :Ly))
        mpo = if params isa AbstractDict
            build_hamiltonian(geometry, Lx, Ly; mparam_dict=Dict{Symbol,Any}(params), kw_filtered...)
        elseif params isa NamedTuple
            build_hamiltonian(geometry, Lx, Ly; mparam_dict=Dict{Symbol,Any}(pairs(params)), kw_filtered...)
        elseif params isa Number
            req = MODEL_REGISTRY[geometry][3][1]
            build_hamiltonian(geometry, Lx, Ly; mparam_dict=Dict{Symbol,Any}(req => params), kw_filtered...)
        else
            build_hamiltonian(geometry, Lx, Ly; mparam_dict=Dict{Symbol,Any}(:t => params), kw_filtered...)
        end
    end
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    # The model builders (HAAH, HSSH, …) create their own site indices internally,
    # so we extract the actual sites from the MPO rather than using the ones
    # created at the top of get_Hamiltonian (which would be a different set).
    mpo_sites = getindex.(siteinds(mpo), 2)
    # If caller supplied ref_sites, replace MPO indices in-place so all
    # Hamiltonians built with the same ref_sites share identical Index objects.
    if !isnothing(ref_sites)
        fix_sites(mpo, ref_sites)
        mpo_sites = ref_sites
    end
    sc   = something(scale, _estimate_scale(geometry, params; mparams=get(kwargs, :mparams, "")))
    lx_2d = dim == 2 ? get(kwargs, :Lx, L ÷ 2) : nothing
    geom = _preset_geometry(geometry, isnothing(lx_2d) ? nothing : 2^lx_2d)
    H = TBHamiltonian(L, N, mpo_sites, mpo, geom, Float64(sc), 0.0, nothing, nothing, nothing, nothing, 0, nothing)
    H.Lx = lx_2d
    return H
end

function _build_sublattice(geometry, params, L;
                            scale=nothing, tol=1e-8, maxdim=200, kwargs...)
    Lx = get(kwargs, :Lx, L ÷ 2)
    Ly = get(kwargs, :Ly, L - Lx)
    t  = params isa Number                                           ? params      :
         params isa NamedTuple && hasfield(typeof(params), :t)      ? params.t    :
         params isa AbstractDict && haskey(params, :t)              ? params[:t]  : 1.0
    t2 = params isa NamedTuple && hasfield(typeof(params), :t2)     ? params.t2   :
         params isa AbstractDict && haskey(params, :t2)             ? params[:t2] : 0.0

    H = geometry == "kagome"        ? kagome_hamiltonian(               Lx, Ly, t;     cutoff=tol, maxdim=maxdim) :
        geometry == "lieb"          ? lieb_hamiltonian(                 Lx, Ly, t;     cutoff=tol, maxdim=maxdim) :
        geometry == "dice"          ? dice_hamiltonian(                 Lx, Ly, t;     cutoff=tol, maxdim=maxdim) :
        geometry == "honeycomb_nnn" ? honeycomb_nnn_hamiltonian(        Lx, Ly, t, t2; cutoff=tol, maxdim=maxdim) :
                                      honeycomb_sublattice_hamiltonian( Lx, Ly, t;     cutoff=tol, maxdim=maxdim)

    rs         = geometry == "kagome"    ? kagome_positions(                 Lx, Ly) :
                 geometry == "lieb"      ? lieb_positions(                   Lx, Ly) :
                 geometry == "dice"      ? dice_positions(                   Lx, Ly) :
                                          honeycomb_sublattice_positions(    Lx, Ly)
    H.geometry = let m = rs; i -> m[i, :]; end

    # UC geometry: same Bravais position for every atom in the same unit cell.
    # All four lattices share a triangular Bravais basis; n_sub = atoms per UC.
    n_sub  = geometry in ("honeycomb", "honeycomb_nnn") ? 2 : 3
    Nx_uc  = 2^Lx
    sq3_2  = sqrt(3) / 2
    H.geometry_uc = let n_sub = n_sub, Nx = Nx_uc, sq3_2 = sq3_2
        i -> begin
            n_cell = (i - 1) ÷ n_sub
            ix = n_cell % Nx
            iy = n_cell ÷ Nx
            [ix + iy * 0.5, iy * sq3_2]
        end
    end

    isnothing(scale) || (H.scale = Float64(scale))
    H.Lx = Lx
    return H
end


function _preset_geometry(geometry, Nx)
    geometry in ("uniform", "ssh", "aah", "chain_1d") && return _chain_geometry()
    geometry == "square_2d"    && return _square_geometry(Nx)
    geometry == "hex_2d"       && return _hex_geometry(Nx)
    geometry == "triangular_2d"     && return _tri_geometry(Nx)
    geometry == "triangular_bravais" && return _tri_bravais_geometry(Nx)
    return nothing
end

# Rough scale estimates for known geometries (used when scale=nothing). `mparams` is the
# parameter string _build_preset forwards to build_hamiltonian, if any.
function _estimate_scale(geometry, params; mparams::AbstractString="")
    geometry == "chernhex" && return _chernhex_scale(params, mparams)
    t = params isa Number ? abs(params) :
        params isa NamedTuple && hasfield(typeof(params), :t) ? abs(params.t) :
        params isa AbstractDict && haskey(params, :t) ? abs(params[:t]) : 1.0
    geometry == "chain_1d"     && return 2.5 * t
    geometry == "ssh"          && return 2.5 * t
    geometry == "aah"          && return (t + (params isa NamedTuple ? abs(params.V) : 1.0)) * 1.2
    geometry == "uniform"      && return 2.5 * t
    geometry == "square_2d"    && return 4.4 * t
    geometry == "hex_2d"       && return 4.0 * t
    geometry == "triangular_2d"     && return 7.0 * t
    geometry == "triangular_bravais" && return 7.0 * t
    geometry in ("chern8","qc2dsquare") && return 6.0 * t
    return 5.0 * t   # conservative fallback
end

# Default "chernhex" scale: the Gershgorin bound of H2DChernhex's terms (3 NN bonds of |t|,
# 6 NNN bonds of |t2|, on-site |Ms| with Ms = ms, or ms + 3.3√3 t2 on the right half unless
# uniformsemenoff), padded by 10% and never below the former default 6|t|. The parameters
# are merged the way _build_preset and build_hamiltonian merge them: the `mparams` string,
# then `params` on top, then the registry defaults.
function _chernhex_scale(params, mparams::AbstractString)
    p = _parse_param_string(mparams)
    q = params isa AbstractDict || params isa NamedTuple ? pairs(params) : (:t => params,)
    for (k, v) in q; p[k] = v; end
    t, t2, ms = abs(p[:t]), p[:t2], p[:ms]
    uniform   = get(p, :uniformsemenoff, MODEL_REGISTRY["chernhex"][4].uniformsemenoff)
    Mmax      = uniform ? abs(ms) : max(abs(ms), abs(ms + 3.3 * sqrt(3) * t2))
    return max(6.0 * t, 1.1 * (3.0 * t + 6.0 * abs(t2) + Mmax))
end


# ============================================================
# Geometry helpers
# ============================================================

"""
    honeycomb_positions(L; Lx=L÷2) -> Matrix{Float64}

Generate `N = 2^L` physical honeycomb positions consistent with the
quantics row-major encoding `n = ix + iy * 2^Lx`, bond length = 1.

The lattice is an armchair ribbon: even rows have intra-row bonds
`(2k, 2k+1)` and odd rows have intra-row bonds `(2k+1, 2k+2)`, with
all inter-row bonds `(iy, ix) ↔ (iy+1, ix)`.

Returns an `N × 2` matrix where row `i` (1-indexed) is the 2D position
of quantics site `i-1`.
"""
function honeycomb_positions(L::Int; Lx::Int = L ÷ 2)
    N  = 2^L
    Nx = 2^Lx
    g  = _hex_geometry(Nx)
    rs = Matrix{Float64}(undef, N, 2)
    for i in 1:N; rs[i, :] = g(i); end
    return rs
end

"""
    square_positions(L; Lx=L÷2) -> Matrix{Float64}

Physical positions for the `2^L`-site square lattice in quantics row-major
encoding `n = ix + iy·2^Lx`.  Site `i` (1-indexed) maps to `(ix, iy)`.
"""
function square_positions(L::Int; Lx::Int = L ÷ 2)
    N  = 2^L
    Nx = 2^Lx
    g  = _square_geometry(Nx)
    rs = Matrix{Float64}(undef, N, 2)
    for i in 1:N; rs[i, :] = g(i); end
    return rs
end

"""
    triangular_positions(L; Lx=L÷2) -> Matrix{Float64}

Physical positions for the `2^L`-site triangular lattice in quantics row-major
encoding `n = ix + iy·2^Lx`, bond length = 1.  Odd rows are offset by 0.5 in x:
`x = ix + 0.5·(iy % 2)`,  `y = iy·√3/2`.
"""
function triangular_positions(L::Int; Lx::Int = L ÷ 2)
    N  = 2^L
    Nx = 2^Lx
    g  = _tri_geometry(Nx)
    rs = Matrix{Float64}(undef, N, 2)
    for i in 1:N; rs[i, :] = g(i); end
    return rs
end

"""
    triangular_bravais_positions(L; Lx=L÷2) -> Matrix{Float64}

Physical positions for the `2^L`-site Bravais triangular lattice in quantics
row-major encoding `n = ix + iy·2^Lx`, bond length = 1.
Bravais vectors a1=(1,0), a2=(1/2,√3/2):  `x = ix + iy/2`,  `y = iy·√3/2`.
"""
function triangular_bravais_positions(L::Int; Lx::Int = L ÷ 2)
    N  = 2^L
    Nx = 2^Lx
    g  = _tri_bravais_geometry(Nx)
    rs = Matrix{Float64}(undef, N, 2)
    for i in 1:N; rs[i, :] = g(i); end
    return rs
end

# ============================================================
# Geometry utilities
# ============================================================

"""
    central_index(geom, N) -> Int
    central_index(H)       -> Int

Return the 1-indexed site index whose position is closest to the geometric
centroid of the lattice.  Accepts either a geometry function `geom(i)` and
system size `N`, or a `TBHamiltonian` directly.

Errors if `H.geometry` is `nothing`.
"""
function central_index(geom::Function, N::Int)
    center = sum(geom(i) for i in 1:N) / N
    return argmin(LinearAlgebra.norm(geom(i) .- center) for i in 1:N)
end

function central_index(H::TBHamiltonian)
    isnothing(H.geometry) && error("central_index requires H.geometry to be set.")
    return central_index(H.geometry, H.N)
end

# ============================================================
# Additive interaction API
# ============================================================

"""
    add_hopping!(H, f; nn, sublat, sublat_from, sublat_to, maxdim, tol, ...) -> H

Add a hopping term to `H`.

**`f` modes** (selected automatically by type):

- `f::Number` — uniform `nn`-th-neighbour hopping (amplitude `f`, shell `nn`).
- `f(i)`, 1-arg `Function` — spatially varying amplitude per site, QTCI-compressed.
- `f(i,j)`, 2-arg `Function` — full N×N matrix, QTCI-compressed (`nn` ignored).

**Sublattice keywords** (`H.sublattice_s` must be set; call before `add_spin!` etc.)

- `sublat=k` — **intra-sublattice**: restrict the full kinetic term to sublattice `k`
  by postpending the projector `|k⟩⟨k|`.  Works with all three `f` modes.

  ```julia
  add_hopping!(H, -0.1; sublat=1, nn=2)           # NNN within sublattice A
  ```

- `sublat_from=k, sublat_to=l` — **inter-sublattice**: adds
  `f · K_u^nn ⊗ |l⟩⟨k| + conj(f) · K_d^nn ⊗ |k⟩⟨l|`
  (Hermitian, `f` must be scalar).  Use `nn=0` for **intra-cell** bonds
  (no position shift; useful to correct individual bond amplitudes in an existing
  sublattice Hamiltonian):

  ```julia
  add_hopping!(H, -0.3; sublat_from=1, sublat_to=2, nn=1)   # A↔B NN inter-cell
  add_hopping!(H, δt;   sublat_from=2, sublat_to=3, nn=0)   # B↔C intra-cell only
  ```

Without sublattice keywords the function errors if any auxiliary DOF is already attached.
Invalidates all caches.
"""
function add_hopping!(H::TBHamiltonian, f;
                      nn::Integer      = 1,
                      boundary::Symbol = :open,
                      bc               = nothing,
                      maxdim           = 15,
                      tol              = 1e-8,
                      type             = ComplexF64,
                      apply_kwargs     = NamedTuple(),
                      sublat           = nothing,
                      sublat_from      = nothing,
                      sublat_to        = nothing)
    _require_binary_position_space(H, "add_hopping!")
    if !isnothing(H.Lx)
        (!isnothing(sublat) || !isnothing(sublat_from) || !isnothing(sublat_to)) &&
            error("add_hopping! sublat keywords are not supported for 2D Hamiltonians; use add_hopping_2D! directly.")
        # add_hopping_2D! would ask for lattice=/geometry= keywords that add_hopping! lacks
        (H.layer_s !== nothing && H.geometry === nothing &&
         H.spin_s === nothing && H.nambu_s === nothing) &&
            error("add_hopping! cannot be used on a layered Hamiltonian without a geometry " *
                  "(the plain and twisted layered builders leave H.geometry unset). Call " *
                  "add_hopping_2D!(H, f; Lx=H.Lx, Ly=H.L - H.Lx, nn, layer, " *
                  "lattice=:square/:triangular/:honeycomb or geometry=...) instead.")
        return add_hopping_2D!(H, f; Lx=H.Lx, Ly=H.L - H.Lx, nn=Int(nn), maxdim=maxdim, tol=tol)
    end

    any_sublat_kw = !isnothing(sublat) || !isnothing(sublat_from)

    if any_sublat_kw
        isnothing(H.sublattice_s) &&
            error("add_hopping! with sublat/sublat_from requires H.sublattice_s to be set.")
        (H.spin_s !== nothing || H.nambu_s !== nothing || H.layer_s !== nothing) &&
            error("add_hopping! with sublattice keywords must be called before add_spin!/add_superconductivity!.")
    else
        H.spin_s === nothing && H.nambu_s === nothing &&
            H.layer_s === nothing && H.sublattice_s === nothing ||
            error("add_hopping! must be called before add_spin!/add_superconductivity! " *
                  "and cannot be used on layered or sublattice Hamiltonians. " *
                  "Use sublat= or sublat_from/sublat_to= for sublattice-specific hoppings.")
    end

    pos_s = _pos_sites(H)
    sl_s  = H.sublattice_s
    apkw  = isempty(apply_kwargs) ? (; cutoff=tol, maxdim=maxdim) : apply_kwargs
    bc === nothing || (boundary = Symbol(bc))

    # ── Build position-space hopping MPO (shared for all cases) ───────────────
    function pos_hop()
        if f isa Number
            kineticNNN(H.L, pos_s, f * MPO(pos_s, "Id"), nn;
                       apply_kwargs=apply_kwargs, boundary=boundary)
        elseif f isa Function && applicable(f, 1)
            kineticNNN(H.L, pos_s, get_diagonal_mpo(H.L, pos_s, f), nn;
                       apply_kwargs=apply_kwargs, boundary=boundary)
        else
            hopping2MPO(f, H.N, pos_s; tol=tol, type=type)
        end
    end

    # Helper: postpend or prepend based on sublattice position in H.sites
    sl_pos  = isnothing(sl_s) ? 0 : findfirst(==(sl_s), H.sites)
    function extend_with_sublat(mpo, mat)
        sl_pos == length(H.sites) ?
            postpend_op(mpo, sl_s, mat) : prepend_op(mpo, sl_s, mat)
    end

    new_term = if isnothing(sublat) && isnothing(sublat_from)
        # ── Original: no sublattice keywords ──────────────────────────────────
        pos_hop()

    elseif !isnothing(sublat) && isnothing(sublat_from)
        # ── Intra-sublattice: project full kinetic onto sublattice k ──────────
        n    = dim(sl_s)
        proj = zeros(Float64, n, n);  proj[sublat, sublat] = 1.0
        extend_with_sublat(pos_hop(), proj)

    else
        # ── Inter-sublattice hopping: t * K_nn ⊗ |to⟩⟨from| + h.c. ──────────
        isnothing(sublat_to) &&
            error("sublat_from requires sublat_to.")
        f isa Number ||
            error("Inter-sublattice add_hopping! (sublat_from/to) requires a scalar amplitude.")

        n       = dim(sl_s)
        mat_fwd = zeros(ComplexF64, n, n);  mat_fwd[sublat_to,   sublat_from] = f
        mat_bwd = zeros(ComplexF64, n, n);  mat_bwd[sublat_from, sublat_to  ] = conj(f)

        if nn == 0
            # Intra-cell (zero position shift): Id_pos ⊗ (|to⟩⟨from| + |from⟩⟨to|)
            extend_with_sublat(MPO(pos_s, "Id"), mat_fwd + mat_bwd)
        else
            ku_nn, kd_nn = shift_pair_mpos(pos_s, nn; cyclic=_tb_periodic_boundary(boundary))
            +(extend_with_sublat(ku_nn, mat_fwd),
              extend_with_sublat(kd_nn, mat_bwd); cutoff=tol)
        end
    end

    H.mpo = +(H.mpo, new_term; maxdim=maxdim, cutoff=tol)
    ITensorMPS.truncate!(H.mpo; maxdim=maxdim, cutoff=tol)
    _invalidate_cache!(H)
    return H
end


"""
    add_onsite!(H, f; layer=nothing, sublat=nothing, Lx=nothing,
                tol=1e-8, maxdim=nothing) -> H

Add a diagonal (on-site) potential to `H`.

**`f` argument** — same conventions as `add_hopping_2D!`

- `f::Number` — uniform constant; builds `f · Id` directly (no QTCI).
- `f(n)` — 1-arg function; `n ∈ {0, …, N-1}` is the 0-indexed unit-cell index.
- `f(ix, iy)` — 2-arg function; `ix, iy` are 0-indexed 2D coordinates
  (`ix = n % Nx`, `iy = n ÷ Nx`). Requires `Lx=` keyword so that `Nx = 2^Lx`.

**`sublat` keyword** (`H.sublattice_s` must be set)

- `sublat=nothing` (default) — applies `f` equally to all sublattices.
- `sublat=k` — restricts the potential to sublattice `k` only via `|k⟩⟨k|`.

  Canonical use-case — **Semenoff mass** on a honeycomb Hamiltonian:
  ```julia
  add_onsite!(H_hc, +M; sublat=1)   # +M on sublattice A
  add_onsite!(H_hc, -M; sublat=2)   # −M on sublattice B  → gap = 2M
  ```

  Works for any sublattice-carrying `TBHamiltonian` (kagome, Lieb, dice, …)
  and for spatially-varying `f` as well.

**`layer` keyword** (`H.layer_s` must be set)

- `layer=nothing` — apply the same onsite term to every layer.
- `layer=k` — apply only to layer `k`.
- `layer=[...]` — apply to the selected layers.

For layered sublattice Hamiltonians, the onsite term is first built on the
`position ⊗ sublattice` monolayer space, then wrapped with the selected layer
projectors.

Invalidates all caches.
"""
function add_onsite!(H::TBHamiltonian, f; layer=nothing, sublat=nothing,
                     Lx=nothing, tol=1e-8, maxdim=nothing)
    _require_binary_position_space(H, "add_onsite!")
    if H.layer_s !== nothing
        (H.spin_s === nothing && H.nambu_s === nothing) ||
            error("Layered add_onsite! currently supports layer/position/sublattice Hamiltonians only.")

        pos_s = _pos_sites(H)
        term_sites = H.sublattice_s === nothing ? pos_s : [pos_s; H.sublattice_s]
        zero_mpo = H.sublattice_s === nothing ?
            0.0 * MPO(pos_s, "Id") :
            0.0 * postpend_op(MPO(pos_s, "Id"), H.sublattice_s,
                               Matrix{Float64}(I, dim(H.sublattice_s), dim(H.sublattice_s)))

        layers = _resolve_layer_selection(H.layer_s, layer)
        H_layered_term = nothing
        for ell in layers
            H_pos = TBHamiltonian(H; sites=term_sites, mpo=copy(zero_mpo),
                                  scale=0.0, center=0.0,
                                  spin_s=nothing, nambu_s=nothing, layer_s=nothing,
                                  sublattice_s=H.sublattice_s, aux_side=:post)
            add_onsite!(H_pos, f; layer=nothing, sublat=sublat,
                        Lx=Lx, tol=tol, maxdim=maxdim)
            term = prepend_layer_projector(H_pos.mpo, H.layer_s, ell)
            H_layered_term = H_layered_term === nothing ? term :
                +(H_layered_term, term; cutoff=tol)
        end

        H.mpo = isnothing(maxdim) ?
            +(H.mpo, H_layered_term; cutoff=tol) :
            +(H.mpo, H_layered_term; cutoff=tol, maxdim=maxdim)
        _invalidate_cache!(H)
        return H
    end

    layer === nothing ||
        error("add_onsite!: `layer` was provided but H.layer_s is not set.")

    pos_s = _pos_sites(H)
    L     = H.L

    fkind = if f isa Number
        :scalar
    elseif applicable(f, 0, 0)   # f(ix, iy) — 0-indexed 2D
        :pos2d
    elseif applicable(f, 0)      # f(n) — 0-indexed 1D
        :pos1d
    else
        error("add_onsite!: unsupported f signature.\n" *
              "Supported: Number, f(n), f(ix, iy)")
    end

    Nx = if fkind === :pos2d
        Lx !== nothing ||
            error("add_onsite! with f(ix,iy) requires Lx=... keyword.")
        2^Lx
    else
        nothing
    end

    diag_mpo = if fkind === :scalar
        get_diagonal_mpo(L, pos_s, x -> f)
    elseif fkind === :pos1d
        get_diagonal_mpo(L, pos_s, i -> f(round(Int, i) - 1))
    else  # :pos2d
        get_diagonal_mpo(L, pos_s, i -> (n = round(Int, i) - 1; f(n % Nx, n ÷ Nx)))
    end

    if H.sublattice_s !== nothing
        sl_s  = H.sublattice_s
        n     = dim(sl_s)
        sl_pos = findfirst(==(sl_s), H.sites)

        mat = if !isnothing(sublat)
            proj = zeros(Float64, n, n);  proj[sublat, sublat] = 1.0
            proj
        else
            Matrix{Float64}(I, n, n)
        end
        new_term = sl_pos == length(H.sites) ?
                   postpend_op(diag_mpo, sl_s, mat) :
                   prepend_op(diag_mpo, sl_s, mat)
    elseif !isnothing(sublat)
        isnothing(H.sublattice_s) &&
            error("add_onsite! with sublat=$sublat requires H.sublattice_s to be set.")
    else
        new_term = diag_mpo
    end

    H.mpo = isnothing(maxdim) ?
        +(H.mpo, new_term; cutoff=tol) :
        +(H.mpo, new_term; cutoff=tol, maxdim=maxdim)
    _invalidate_cache!(H)
    return H
end


# ============================================================
# Position-site accessor
# ============================================================

"""
    _pos_sites(H) -> Vector{<:Index}

Return the L position-qubit indices, filtering out any auxiliary (spin, Nambu,
layer) indices regardless of whether they sit at the front or back of `H.sites`.
"""
function _pos_sites(H::TBHamiltonian)
    aux = Index[]
    isnothing(H.spin_s)       || push!(aux, H.spin_s)
    isnothing(H.nambu_s)      || push!(aux, H.nambu_s)
    isnothing(H.layer_s)      || push!(aux, H.layer_s)
    isnothing(H.sublattice_s) || push!(aux, H.sublattice_s)
    aux_set = Set(aux)
    return filter(s -> s ∉ aux_set, H.sites)
end


# ============================================================
# Interaction storage
# ============================================================

"""
    add_interaction!(H, V; channel=:hartree, type=Float64, tol=1e-8) -> H

Store a pre-built interaction MPO in `H` for later use with `get_scf(H, channel)`.

`V` can be:
- `MPO`      — stored directly
- `Number`   — scalar coupling; stored as `V · Id` on the position sites
- `Function` with 1 arg — on-site potential `V(i)`, 0-indexed; stored as diagonal MPO
- `Function` with 2 args — interaction kernel `V(i,j)`, 0-indexed; stored as full 2D MPO

`channel`:
- `:hartree` or `:default` → stored in `H.interaction_mpo` (used by `:cdw` and `:magnetic` SCF)
- `:fock` or `:exchange`   → stored in `H.fock_mpo`
"""
function add_interaction!(H::TBHamiltonian, V;
                          channel::Symbol = :hartree,
                          type::Type = Float64,
                          tol::Real = 1e-8,
                          kwargs...)
    _require_binary_position_space(H, "add_interaction!")
    pos_s = _pos_sites(H)
    mpo = if V isa MPO
        V
    elseif V isa Number
        V * MPO(collect(pos_s), "Id")
    elseif V isa Function
        get_mpo(H.L, pos_s, V; type=type, tol=tol, kwargs...)
    else
        error("add_interaction!: unsupported V type $(typeof(V)). Use Number, Function, or MPO.")
    end

    ch = lowercase(String(channel))
    if ch in ("hartree", "default")
        H.interaction_mpo = mpo
    elseif ch in ("fock", "exchange")
        H.fock_mpo = mpo
    else
        error("add_interaction!: unknown channel :$channel. Use :hartree or :fock.")
    end
    return H
end


# ============================================================
# Spin extension
# ============================================================

"""
    add_spin!(H; cutoff=1e-8, maxdim=200) -> H

Extend `H` to a spin-½ degenerate system by prepending a spin-½ index.
The resulting Hamiltonian is `I_spin ⊗ H` (both spin sectors identical).

No-op if `H` is already spinful (`H.spin_s !== nothing`).
Invalidates all caches.
"""
function add_spin!(H::TBHamiltonian; cutoff::Real=1e-8, maxdim::Int=200,
                   position::Symbol=:pre)
    _require_binary_position_space(H, "add_spin!")
    H.spin_s === nothing || return H
    spin_s = spin_index()
    if position === :pre
        H.mpo   = prepend_spin(H.mpo, spin_s, :Id)
        H.sites = [spin_s; H.sites]
    else
        H.mpo   = postpend_spin(H.mpo, spin_s, :Id)
        H.sites = [H.sites; spin_s]
    end
    ITensorMPS.truncate!(H.mpo; maxdim=maxdim, cutoff=cutoff)
    H.spin_s   = spin_s
    H.aux_side = position
    _invalidate_cache!(H)
    return H
end


# ============================================================
# Zeeman coupling
# ============================================================

"""
    add_zeeman!(H, h; direction=:z, tol=1e-8, maxdim=200) -> H

Add a Zeeman coupling `h · Sα` to `H`.  Calls `add_spin!` automatically if
`H` is not yet spinful.

`h` can be:
- a `Number`    — uniform field amplitude `h₀`
- a `Function`  — spatially varying `h(i)`, `i ∈ {1, …, N}` (1-indexed)

`direction`: `:x`, `:y`, or `:z` (default).

If `add_superconductivity!` was already called, the Zeeman term is wrapped in
`τ_z` so it enters with opposite sign in the hole sector, as required in BdG.

Examples
--------
```julia
add_zeeman!(H, 0.1)                         # uniform h = 0.1 along z
add_zeeman!(H, i -> 0.05 * sin(2π*i/H.N))  # oscillating field
add_zeeman!(H, 0.05; direction=:x)          # in-plane
```
"""
function add_zeeman!(H::TBHamiltonian, h;
                     direction::Symbol = :z,
                     tol::Real  = 1e-8,
                     maxdim::Int = 200,
                     position::Union{Nothing,Symbol} = nothing)
    _require_binary_position_space(H, "add_zeeman!")
    direction in (:x, :y, :z) ||
        error("direction must be :x, :y, or :z; got :$direction")
    pos = something(position, H.aux_side)
    add_spin!(H; cutoff=tol, maxdim=maxdim, position=pos)

    spin_op = direction == :z ? :Sz : direction == :x ? :Sx : :Sy
    pos_s   = _pos_sites(H)
    h_mpo   = h isa Number ? h * MPO(pos_s, "Id") :
                             get_diagonal_mpo(H.L, pos_s, h)

    if H.aux_side === :pre
        H_Z = prepend_spin(h_mpo, H.spin_s, spin_op)
        H.nambu_s !== nothing && (H_Z = prepend_nambu(H_Z, H.nambu_s, :tz))
    else
        H_Z = postpend_spin(h_mpo, H.spin_s, spin_op)
        H.nambu_s !== nothing && (H_Z = postpend_nambu(H_Z, H.nambu_s, :tz))
    end

    H.mpo = +(H.mpo, H_Z; maxdim=maxdim, cutoff=tol)
    ITensorMPS.truncate!(H.mpo; maxdim=maxdim, cutoff=tol)
    _invalidate_cache!(H)
    return H
end


# ============================================================
# Superconducting pairing (BdG extension)
# ============================================================

"""
    add_superconductivity!(H, Δ; type=:swave, tol=1e-8, maxdim=200) -> H

Extend `H` to a Bogoliubov–de Gennes (BdG) Hamiltonian by prepending a
Nambu (particle–hole) index.

The BdG structure is:
    H_BdG = τ_z ⊗ H_kin  +  τ_+ ⊗ H_pair  +  τ_- ⊗ H_pair†

- **Spinless + p-wave** (`type=:pwave`, or auto-selected when spinless + `:swave`):
  `H_pair = Δ·(K_forward − K_backward)`, the antisymmetric nearest-neighbour
  matrix required by Fermi statistics (`Δ(i,j) = −Δ(j,i)`).  This is the
  Kitaev chain.  `Δ` must be a `Number`.
- **Spinful + s-wave** (`add_spin!` called first, `type=:swave`):
  `H_pair = (i·σ_y)_spin ⊗ Δ(r)`, the standard BCS singlet Cooper-pair operator.
  On-site (s-wave) pairing is allowed here because the antisymmetry is carried
  by the spin singlet factor `i·σ_y`.  `Δ` can be a `Number` or 1-arg `Function`.
- **Custom** (`type=:custom`): arbitrary pairing matrix via 2-arg function `Δ(i,j)`,
  compressed with TCI.

**Note on spinless s-wave**: on-site pairing is forbidden for spinless fermions
(`Δ(i,i) = 0` by antisymmetry).  Calling with `type=:swave` on a spinless chain
automatically redirects to `:pwave` (uniform `Δ`) or errors (spatially varying `Δ`).

`type`:
- `:swave`  (default) — diagonal pairing for spinful chains; auto-redirects to
  `:pwave` for spinless chains when `Δ isa Number`
- `:pwave`  — antisymmetric NN pairing `Δ·(K_f − K_b)` for spinless chains; `Δ` must be a `Number`
- `:custom` — arbitrary `Δ(i,j)`; pass a 2-arg function

Errors if BdG has already been applied.  Invalidates all caches.

Examples
--------
```julia
add_superconductivity!(H_spinless, 0.1)              # auto p-wave (Kitaev chain)
add_superconductivity!(H_spinless, 0.1; type=:pwave) # explicit p-wave
add_superconductivity!(H_spinful,  0.1)              # singlet s-wave (spinful required)
add_superconductivity!(H_spinful,  i -> i < N÷2 ? 0.1 : 0.0)  # spatially varying s-wave
add_superconductivity!(H, (i,j) -> ...; type=:custom)          # general pairing
```
"""
function add_superconductivity!(H::TBHamiltonian, Δ;
                                type::Symbol = :swave,
                                tol::Real    = 1e-8,
                                maxdim::Int  = 200,
                                position::Union{Nothing,Symbol} = nothing)
    _require_binary_position_space(H, "add_superconductivity!")
    H.nambu_s === nothing ||
        error("BdG already applied (H.nambu_s is set). Cannot apply twice.")

    pos   = something(position, H.aux_side)
    pos_s = _pos_sites(H)

    # ── Spinless + :swave redirect ───────────────────────────────────────────
    if H.spin_s === nothing && type === :swave
        if Δ isa Number
            println("Info: on-site (s-wave) pairing is forbidden for spinless fermions ",
                    "(Δ(i,i) = 0 by Fermi antisymmetry).  ",
                    "Constructing nearest-neighbour p-wave instead.")
            type = :pwave
        else
            error("On-site (s-wave) pairing is forbidden for spinless fermions.  " *
                  "For spatially varying spinless pairing use type=:custom with a 2-arg Function Δ(i,j).")
        end
    end

    # ── Build the pairing MPO in position space ──────────────────────────────
    H_pair_pos = if type === :pwave
        H.spin_s === nothing ||
            error("type=:pwave is only for spinless chains.  " *
                  "For spinful p-wave use type=:custom with a 2-arg Function Δ(i,j).")
        Δ isa Number ||
            error("For type=:pwave, Δ must be a Number.  " *
                  "For spatially varying spinless pairing use type=:custom with a 2-arg Function Δ(i,j).")
        # H_pair = Δ·(Kf − Kb): antisymmetric NN pairing matrix.
        # The τ- ⊗ H_pair† term handles the hole-particle sector automatically.
        pairingNNN(H.L, pos_s, Δ * MPO(pos_s, "Id"), 1)
    elseif type === :swave
        Δ isa Number   ? Δ * MPO(pos_s, "Id")            :
        Δ isa Function ? get_diagonal_mpo(H.L, pos_s, Δ) :
        error("For type=:swave, Δ must be a Number or a 1-arg Function.")
    elseif type === :custom
        Δ isa Function ||
            error("For type=:custom, Δ must be a 2-arg Function Δ(i,j).")
        pairing2MPO(Δ, H.N, pos_s; tol=tol, type=ComplexF64)
    else
        error("Unknown pairing type :$type.  Use :swave, :pwave, or :custom.")
    end

    # ── Lift pairing to spin space if needed ─────────────────────────────────
    H_pair = if H.spin_s !== nothing
        pos === :pre ? prepend_spin(H_pair_pos,  H.spin_s, :iSy) :
                       postpend_spin(H_pair_pos, H.spin_s, :iSy)
    else
        H_pair_pos
    end

    H_pair_adj = swapprime(dag(H_pair), 0, 1)

    # ── BdG assembly ─────────────────────────────────────────────────────────
    nambu_s = nambu_index()
    if pos === :pre
        H_bdg = +(+(prepend_nambu(H.mpo,      nambu_s, :tz),
                    prepend_nambu(H_pair,     nambu_s, :tp); cutoff=tol),
                    prepend_nambu(H_pair_adj, nambu_s, :tm); cutoff=tol)
        H.sites = [nambu_s; H.sites]
    else
        H_bdg = +(+(postpend_nambu(H.mpo,      nambu_s, :tz),
                    postpend_nambu(H_pair,     nambu_s, :tp); cutoff=tol),
                    postpend_nambu(H_pair_adj, nambu_s, :tm); cutoff=tol)
        H.sites = [H.sites; nambu_s]
    end
    ITensorMPS.truncate!(H_bdg; maxdim=maxdim, cutoff=tol)

    Δ_scale    = Δ isa Number ? abs(Δ) : 1.0
    H.mpo      = H_bdg
    H.nambu_s  = nambu_s
    H.aux_side = pos
    H.scale    = H.scale + Δ_scale * 1.1   # rough update; user can override
    _invalidate_cache!(H)
    return H
end


# ============================================================
# Spin-orbit coupling
# ============================================================

"""
    add_soc!(H, λ; type=:rashba, direction=:z, tol=1e-8, maxdim=200) -> H

Add spin-orbit coupling to `H`.  Calls `add_spin!` automatically if needed.

`type`:
- `:rashba` — nearest-neighbour Rashba SOC on the position chain:
              `λ · (S_y ⊗ K_u − S_y ⊗ K_d)` where `K_u/K_d` are the ±1 shift
              operators.  `λ` must be a scalar.  Breaks SU(2) spin symmetry
              while preserving time-reversal.
- `:ising`  — diagonal Ising SOC `λ(i) · S_z` (equivalent to a position-dependent
              Zeeman along z; useful for Kane–Mele type models).
- `:custom` — arbitrary position-space MPO `λ_mpo` tensor-producted with the
              spin operator given by `direction` (`:x`, `:y`, or `:z`).
              `λ` may be a Number, a 1-arg `Function λ(i)`, or a 2-arg
              `Function λ(i,j)` (the last compressed via TCI).
              For the result to be Hermitian, the position-space matrix must
              itself be Hermitian: `λ(i,j) = conj(λ(j,i))`.  Diagonal and
              real-symmetric inputs satisfy this automatically.

Examples
--------
```julia
add_soc!(H, 0.05)                             # Rashba λ=0.05
add_soc!(H, i -> 0.1*cos(2π*i/H.N); type=:ising)
add_soc!(H, (i,j)->...; type=:custom, direction=:y)
```
"""
function add_soc!(H::TBHamiltonian, λ;
                  type::Symbol      = :rashba,
                  direction::Symbol = :z,
                  tol::Real         = 1e-8,
                  maxdim::Int       = 200,
                  position::Union{Nothing,Symbol} = nothing)
    _require_binary_position_space(H, "add_soc!")
    pos = something(position, H.aux_side)
    add_spin!(H; cutoff=tol, maxdim=maxdim, position=pos)
    pos_s = _pos_sites(H)

    spin_prepend = H.aux_side === :pre ? prepend_spin : postpend_spin

    H_soc = if type === :ising
        λ_mpo = λ isa Number ? λ * MPO(pos_s, "Id") :
                               get_diagonal_mpo(H.L, pos_s, λ)
        spin_prepend(λ_mpo, H.spin_s, :Sz)

    elseif type === :rashba
        λ isa Number || error("Rashba SOC requires a scalar λ; got $(typeof(λ)).")
        K_u = generate_kin_u(pos_s, H.N)
        K_d = generate_kin_d(pos_s, H.N)
        # λ·(iσ_y) ⊗ (K_u − K_d): both factors anti-Hermitian → product Hermitian.
        # :Sy (Hermitian) ⊗ anti-Hermitian would give a non-Hermitian term.
        +(spin_prepend( λ * K_u, H.spin_s, :iSy),
          spin_prepend(-λ * K_d, H.spin_s, :iSy); cutoff=tol)

    elseif type === :custom
        direction in (:x, :y, :z) ||
            error("direction must be :x, :y, or :z; got :$direction")
        spin_op = direction == :z ? :Sz : direction == :x ? :Sx : :Sy
        λ_mpo = if λ isa Number
            λ * MPO(pos_s, "Id")
        elseif λ isa Function && applicable(λ, 1)
            get_diagonal_mpo(H.L, pos_s, λ)
        elseif λ isa Function
            hopping2MPO(λ, H.N, pos_s; tol=tol, type=ComplexF64)
        else
            error("λ must be a Number or a Function.")
        end
        spin_prepend(λ_mpo, H.spin_s, spin_op)

    else
        error("Unknown SOC type :$type.  Use :rashba, :ising, or :custom.")
    end

    H.mpo = +(H.mpo, H_soc; maxdim=maxdim, cutoff=tol)
    ITensorMPS.truncate!(H.mpo; maxdim=maxdim, cutoff=tol)
    _invalidate_cache!(H)
    return H
end


# ============================================================
# Display
# ============================================================

function Base.show(io::IO, H::TBHamiltonian)
    tn_str   = H._tn_cache !== nothing ?
               "Tn cached (Ncheb = $(H._tn_Ncheb))" : "no Tn cache"
    geom_str = isnothing(H.geometry) ? "no geometry" :
               "$(H.N) sites, $(length(H.geometry(1)))D"
    aux_str  = ""
    H.layer_s       !== nothing && (aux_str *= " +$(ITensors.dim(H.layer_s))layers")
    H.sublattice_s  !== nothing && (aux_str *= " +$(ITensors.dim(H.sublattice_s))sublattices")
    H.spin_s  !== nothing && (aux_str *= " +spin")
    H.nambu_s !== nothing && (aux_str *= " +BdG")
    # Detect exciton: interleaved 2L-site chain with no auxiliary indices
    is_exc = length(H.sites) == 2 * H.L &&
             H.layer_s === nothing && H.sublattice_s === nothing &&
             H.spin_s  === nothing && H.nambu_s      === nothing
    N_str = is_exc ? "N=$(H.N) [exciton, D=$(H.N^2)]" : "N=$(H.N)$(aux_str)"
    sc_str = H.scale == 0.0 ? "scale=auto" :
             H.center == 0.0 ? "scale=$(H.scale)" :
             "scale=$(H.scale), center=$(H.center)"
    print(io, "TBHamiltonian | L=$(H.L), $N_str, $sc_str, " *
              "maxlinkdim=$(ITensorMPS.maxlinkdim(H.mpo)) | " *
              "geometry: $geom_str | $tn_str")
end
