# TBSystem.jl — central struct and constructor for tight-binding systems
#
# Provides TBHamiltonian, which wraps the Hamiltonian MPO together with
# metadata (geometry, KPM scale) and lazy caches for Chebyshev moments
# and the density matrix.  All observable methods (get_ldos_spatial,
# get_dos_stochastic, get_density, get_C, get_bands …) dispatch on this struct.
#
# Contents: the position-space policy types and the interface the projected
# spaces of position_spaces/ specialize (ambient_dimension, physical_projector,
# physical_site_state, site_axis, site_permutation); TBHamiltonian with its
# keyword and copy constructors and cache management (_invalidate_cache!,
# truncate!); get_Hamiltonian, which looks the geometry up in the model registry
# (core/ModelRegistry.jl: builders, parameters, default KPM scales), and the direct
# builders it holds (chain, Haldane, custom); central_index; the mutators
# add_hopping!, add_onsite!, add_interaction!; _pos_sites; Base.show. The spin,
# Zeeman, pairing and SOC mutators live in core/AuxDOF.jl.
#
# Main entry points: get_Hamiltonian, TBHamiltonian, add_hopping!, add_onsite!,
# add_interaction!, truncate!, central_index, haldane_hoppingf.
#
# Depends on: Utils, Hamiltonian, geometry*, ModelRegistry*, NNNeighbor* (a *
# marks a file included later; see the source map in TensorBinding.jl).

# ============================================================
# 1. Position-space policy types
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

"""
    BinaryPositionSpace()

Ordinary binary position register containing all `2^L` basis states.
"""
struct BinaryPositionSpace <: AbstractPositionSpace end

# ============================================================
# 2. TBHamiltonian struct
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

Build one with [`get_Hamiltonian`](@ref) or a lattice builder. The keyword constructor
`TBHamiltonian(; L, N, sites, mpo, …)` builds one from its fields, and
`TBHamiltonian(H; field=value, …)` copies one, replacing the named fields.
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
    sublattice_s  :: Union{Nothing, Index}    # set by the multi-atom lattice constructors
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

"""
    TBHamiltonian(; L, N, sites, mpo, geometry=nothing, geometry_uc=nothing,
                  scale=0.0, center=0.0, spin_s=nothing, nambu_s=nothing,
                  layer_s=nothing, sublattice_s=nothing, aux_side=:pre, Lx=nothing,
                  position_space=BinaryPositionSpace(), interaction_mpo=nothing,
                  fock_mpo=nothing) -> TBHamiltonian

Build a `TBHamiltonian` from its fields, by name. `L`, `N`, `sites` and `mpo` are
required. Every other field defaults to "not set": no geometry, `scale = 0.0` (the KPM
scale is then estimated by DMRG on first use), no auxiliary index, `aux_side = :pre`, no
`Lx` (1D), the binary position space and no stored interaction. The lazy caches start
empty. Values are converted to the field types, so `scale = 3` stores `3.0`.

The builders behind `get_Hamiltonian`, `lattice/` and `position_spaces/` construct their
Hamiltonians this way. To change some fields of an existing Hamiltonian, use the copy
constructor `TBHamiltonian(H; field=value, ...)`.

```julia
s = siteinds("Qubit", 3)
H = TBHamiltonian(; L=3, N=8, sites=s, mpo=kinetic_1d_nn(3, s), scale=2.5)
```
"""
function TBHamiltonian(; L, N, sites, mpo, geometry=nothing, geometry_uc=nothing,
                       scale=0.0, center=0.0, spin_s=nothing, nambu_s=nothing,
                       layer_s=nothing, sublattice_s=nothing, aux_side=:pre, Lx=nothing,
                       position_space=BinaryPositionSpace(), interaction_mpo=nothing,
                       fock_mpo=nothing)
    return TBHamiltonian(L, N, sites, mpo, geometry, geometry_uc, scale, center,
                         spin_s, nambu_s, layer_s, sublattice_s, aux_side,
                         nothing, nothing, 0, nothing,    # empty lazy caches
                         interaction_mpo, fock_mpo, Lx, position_space)
end

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
# 3. Position-space interface
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

"""
    site_axis(H; ordering=:physical, kwargs...) -> Vector{Int}

Return the plotting axis for physical positions or an encoding-defined ordering.
`kwargs` are encoding-specific (for a Fibonacci space: `orientation`, `centered`,
`origin`, `alignment`).
"""
function site_axis(H::TBHamiltonian; ordering::Symbol=:physical, kwargs...)
    return site_axis(H.position_space, H; ordering, kwargs...)
end

function site_axis(::BinaryPositionSpace, H::TBHamiltonian;
                   ordering::Symbol=:physical, kwargs...)
    ordering === :physical ||
        throw(ArgumentError("ordering=:$ordering is not available for BinaryPositionSpace"))
    return collect(0:(H.N - 1))
end

"""
    site_permutation(H; ordering=:physical, kwargs...) -> Vector{Int}

Return the 1-based physical-site permutation associated with a plotting ordering.
`kwargs` are as for [`site_axis`](@ref).
"""
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

# ============================================================
# 4. Cache management
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
# 5. get_Hamiltonian constructor
# ============================================================

"""
    get_Hamiltonian(geometry, params; L, scale=nothing, tol=1e-8, maxdim=15,
                    ref_sites=nothing, kwargs...) -> TBHamiltonian

Build a `TBHamiltonian` from a named geometry and model parameters.

`L`, and `Lx`/`Ly` for the 2D geometries, are qubit counts: a 2D system has
`2^Lx × 2^Ly` unit cells with `L = Lx + Ly`, and `Lx` defaults to `L ÷ 2`,
`Ly` to `L - Lx`. A keyword that a geometry does not take is an error, except
for the multi-atom lattices, which ignore it.

Direct builders
---------------
| `geometry`   | `params`                        | Extra kwargs |
|--------------|---------------------------------|--------------|
| `"chain_1d"` | hopping amplitude `t::Number`   | `boundary=:open` or `:periodic` (`bc` overrides it); direct MPO, no QTCI; use `add_onsite!` for potentials |
| `"haldane"`  | `(t2, phi, M)` NamedTuple       | `rs` (N×2 positions from `honeycomb_positions`, required); no other kwargs |
| `"custom"`   | hopping function `f(i,j)`       | `scale` (required: a number, `:dmrg`, or `:small` up to 1024 sites), `geometry` (`i -> position` or an N×2 matrix), `type=ComplexF64` |

Projected position spaces (quasicrystals; `H.N` counts the admissible sites only)
---------------------------------------------------------------------------------
| `geometry`        | `params`                                        | Extra kwargs |
|-------------------|-------------------------------------------------|--------------|
| `"fibonacci"`     | `(A, B[, t, onsite])` NamedTuple or Dict        | `model=:hopping` or `:onsite`, `boundary=:periodic` or `:open`, `padding` |
| `"metallic_mean"` | `(A, B[, t, onsite])` NamedTuple or Dict        | `m` (required; `m=2` silver mean), `model`, `boundary`, `padding` |
| `"kbonacci"`      | `(A, B, C, …[, t, onsite])` or `(values=(a_1, …, a_k)[, t, onsite])` | `k` (required; `k=3` Tribonacci), `model`, `boundary`, `padding` |

These call [`fibonacci_hamiltonian`](@ref), [`metallic_mean_hamiltonian`](@ref) and
[`kbonacci_hamiltonian`](@ref) with `cutoff=tol` and `maxdim=maxdim`; they reject
`ref_sites`.

Preset models (routed through [`build_hamiltonian`](@ref) and `MODEL_REGISTRY`)
-------------------------------------------------------------------------------
`params` is a `Number` (taken as the first required parameter), a NamedTuple or a
Dict; entries beyond the required ones are passed to the builder as keywords, over
the `MODEL_REGISTRY` defaults (which also set `tol_quantics`, `maxbonddim_quantics`
and, in 2D, `cutoff`). Extra kwargs: `Lx`, `Ly` (2D only), `mparams` (a
`"key=value, …"` string that `params` overrides) and `ref_sites`.

| `geometry`             | builder                 | dim | required `params` | optional `params` (default) |
|------------------------|-------------------------|-----|-------------------|-----------------------------|
| `"uniform"`            | `HUniform`              | 1D  | `t`               | `v=1e-6`, `nn=1`            |
| `"ssh"`                | `HSSH`                  | 1D  | `t`, `d`          | `nn=1`                      |
| `"aah"`                | `HAAH`                  | 1D  | `V`, `phi`, `t`   | `b=(1+√5)/2`                |
| `"square_2d"`          | `HUniform2Dsquare`      | 2D  | `t`               |                             |
| `"hex_2d"`             | `HUniform2Dhex`         | 2D  | `t`               |                             |
| `"triangular_2d"`      | `HUniform2Dtri`         | 2D  | `t`               |                             |
| `"triangular_bravais"` | `HUniform2Dtri_bravais` | 2D  | `t`               |                             |
| `"chern8"`             | `HChern8`               | 2D  | `V`, `t`          | `t2=0.2`                    |
| `"chernhex"`           | `H2DChernhex`           | 2D  | `t`, `t2`, `ms`   | `uniformhaldane=false`, `uniformsemenoff=false` |
| `"qc2dsquare"`         | `HQC2Dsquare`           | 2D  | `t`               |                             |

Multi-atom unit cells (explicit sublattice index)
-------------------------------------------------
| `geometry`         | `params`                                   | atoms per cell | Extra kwargs |
|--------------------|--------------------------------------------|----------------|--------------|
| `"kagome"`         | `t` (a Number, `(t=…,)` or a Dict)         | 3              | `Lx`, `Ly`   |
| `"lieb"`           | `t`                                        | 3              | `Lx`, `Ly`   |
| `"dice"`           | `t`                                        | 3              | `Lx`, `Ly`   |
| `"honeycomb"`      | `t`                                        | 2              | `Lx`, `Ly`   |
| `"honeycomb_nnn"`  | `(t, t2)` NamedTuple or Dict (`t2=0`)      | 2              | `Lx`, `Ly`   |
| `"ssh_sublattice"` | `t` or `(t, d)` NamedTuple or Dict (`t=1`, `d=0`) | 2       | none (1D chain) |

`"haldane"` is the textbook, C3-symmetric Haldane model `⟨i|H|j⟩ = t2 exp(i phi ν_ij)`
(Dirac masses `-M ± 3√3 t2 sin(phi)`, see [`haldane_hoppingf`](@ref)); it refuses an `rs`
whose sites are not on the `honeycomb_positions` lattice.

For the multi-atom lattices `L` (`= Lx + Ly` in 2D) counts only the position
qubits; the total atom count is `n_sub × 2^L` with `n_sub` the atoms per cell.
The sublattice index is stored in `H.sublattice_s` with `H.aux_side = :post`.
`H.geometry` returns the full real-space position of each atom (1-indexed over
all `n_sub × 2^L` atoms) and `H.geometry_uc` the Bravais position of its cell.

Common keyword arguments
------------------------
- `L`         : number of position qubits (`2^L` sites or unit cells; the projected
                spaces keep only their admissible subset)
- `scale`     : energy half-bandwidth for KPM normalisation: a number (used as
                given), `nothing` (default: see "Default KPM scale" below; `"custom"`
                requires a scale) or an [`estimate_scale`](@ref) method, `:small`
                (dense spectrum of the model at a small size), `:geometry`
                (row-sum bound) or `:dmrg` (DMRG spectral bounds of `H`, which also
                set `center`)
- `tol`       : QTCI tolerance and truncation cutoff (default `1e-8`)
- `maxdim`    : maximum MPO bond dimension after construction (default `15`)
- `ref_sites` : preset models only: replace the MPO's site indices by these, so
                that Hamiltonians built with the same `ref_sites` share `Index`
                objects (default `nothing`; the other geometries ignore it or,
                for the projected spaces, reject it)

Default KPM scale
-----------------
Without `scale`, `"chain_1d"` and the preset models (`"chernhex"` excepted) take
`max(f, estimate_scale(geometry, params; L, kwargs..., method=:auto))`, where `f` is
the geometry's former default (`2.5|t|` for `"chain_1d"`, `"uniform"`, `"ssh"`;
`1.2(|t| + |V|)` for `"aah"`; `4.4|t|`, `4.0|t|`, `7|t|`, `7|t|` for `"square_2d"`,
`"hex_2d"`, `"triangular_2d"`, `"triangular_bravais"`; `6|t|` for `"chern8"`,
`"qc2dsquare"`) and `:auto` is the padded row-sum bound (`:geometry`) for the
size-scaled `"chern8"` and `"qc2dsquare"` and the dense small-size estimate (`:small`)
otherwise. Where `f` already reaches `1.1 ×` the row-sum bound, the estimate cannot
exceed it and is not computed. Every other geometry keeps its builder's default:
the analytic bounds of `"haldane"`, `"chernhex"`, `"ssh_sublattice"` and the projected
spaces, and the fixed multiples of `t` of the other multi-atom lattices. The centre is
0 except for the projected spaces.

Examples
--------
```julia
H = get_Hamiltonian("chain_1d", 1.0;    L=10)
H = get_Hamiltonian("square_2d", 1.0;   L=10, Lx=5)   # 32 × 32 sites

rs = honeycomb_positions(10)
H  = get_Hamiltonian("haldane", (t2=0.2, phi=π/2, M=0.0); L=10, rs=rs)

H  = get_Hamiltonian("custom", (i,j) -> ...; L=10, scale=5.0, geometry=rs)
Ha = get_Hamiltonian("aah", (V=2.0, phi=0.0, t=1.0); L=8)
Hk = get_Hamiltonian("kagome", 1.0; L=6, Lx=3, Ly=3)   # 3 × 8 × 8 atoms
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
    entry  = _model_entry(geometry)   # the registry entry (core/ModelRegistry.jl)
    method = scale isa Symbol ? _check_scale_method(entry, scale, L, kwargs) : nothing
    # Every builder but the projected spaces' receives Qubit sites; the presets and the
    # multi-atom lattices make their own and ignore them (drawn all the same, so the
    # index-id RNG stream is what it always was).
    projected = entry.kind === :projected
    sites = projected ? nothing : siteinds("Qubit", L)
    # With a scale method the builder gets a provisional scale, replaced below.
    H = entry.build(params, L, projected ? nothing : 2^L, sites;
                    scale = method === nothing ? scale : 1.0, tol, maxdim, ref_sites, kwargs...)
    if method !== nothing
        _apply_scale_method!(H, entry, method, params, L; tol, maxdim, kwargs...)
    elseif scale === nothing
        H.scale = _default_scale(entry, H, params, L; tol, maxdim, kwargs...)
    end
    return H
end

# ============================================================
# 6. Per-geometry builders (internal): 1D chain
# ============================================================

function _build_chain_1d(t, L, N, sites;
                         scale=nothing,
                         tol=1e-8,
                         maxdim=15,
                         boundary::Symbol=:open,
                         bc=nothing)
    bc === nothing || (boundary = Symbol(bc))
    mpo = t * kinetic_1d_nn(L, sites; boundary=boundary)
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    sc  = something(scale, _estimate_scale("chain_1d", t))   # 2.5|t|
    return TBHamiltonian(; L, N, sites, mpo, geometry=_chain_geometry(), scale=sc)
end


# ============================================================
# 7. Haldane model (chirality, haldane_hoppingf, the "haldane" builder)
# ============================================================

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
    haldane_hoppingf(r1, r2, i, j; t2=0.2, phi=pi/2, M=0.0) -> Number

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

# Row-sum (Gershgorin) bound of the Haldane matrix (t1 = 1): a site has |M| on site,
# at most 3 NN and 6 NNN hops.
_haldane_rowsum(t2, M) = 3.0 + 6.0 * abs(t2) + abs(M)

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
    # Gershgorin bound, padded by 10% (nearly reached at phi = 0, π).
    sc  = something(scale, _SCALE_PADDING * _haldane_rowsum(t2, M))
    rs_f = let m = Float64.(rs); i -> m[i, :]; end
    return TBHamiltonian(; L, N, sites, mpo, geometry=rs_f, scale=sc)
end


# ============================================================
# 8. Custom builder (the preset and multi-atom builders are in core/ModelRegistry.jl)
# ============================================================

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
    return TBHamiltonian(; L, N, sites, mpo, geometry=geom_f, scale=Float64(scale))
end


# ============================================================
# 9. Geometry utilities
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
# 10. Additive interaction API
# ============================================================

"""
    add_hopping!(H, f; nn=1, boundary=:open, bc=nothing, maxdim=15, tol=1e-8,
                 type=ComplexF64, apply_kwargs=NamedTuple(), sublat=nothing,
                 sublat_from=nothing, sublat_to=nothing) -> H

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

**Other keywords**

- `boundary=:open` or `:periodic` (`bc` overrides it): boundary of the position
  shift for scalar and 1-arg `f` and for the inter-sublattice hops.
- `maxdim`, `tol`: truncation of the summed MPO (`tol` is also the QTCI
  tolerance of a 2-arg `f`).
- `type`: element type of the QTCI compression of a 2-arg `f`.
- `apply_kwargs`: keywords for the `apply` calls inside [`kineticNNN`](@ref)
  (scalar and 1-arg `f`).

On a 2D Hamiltonian (`H.Lx` set) the call is forwarded to
[`add_hopping_2D!`](@ref) with `Lx = H.Lx`, `Ly = H.L - H.Lx`, `nn`, `maxdim` and
`tol`; the sublattice keywords are not supported there.

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
# 11. Position-site accessor
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
# 12. Interaction storage
# ============================================================

"""
    add_interaction!(H, V; channel=:hartree, type=Float64, tol=1e-8, kwargs...) -> H

Store a pre-built interaction MPO in `H` for later use with `get_scf(H, channel)`.

`V` can be:
- `MPO`      — stored directly
- `Number`   — scalar coupling; stored as `V · Id` on the position sites
- `Function` with 1 arg — on-site potential `V(i)`, 0-indexed; stored as diagonal MPO
- `Function` with 2 args — interaction kernel `V(i,j)`, 0-indexed; stored as full 2D MPO

`channel`:
- `:hartree` or `:default` → stored in `H.interaction_mpo` (used by `:cdw` and `:magnetic` SCF)
- `:fock` or `:exchange`   → stored in `H.fock_mpo`

`type`, `tol` and `kwargs` are passed to [`get_mpo`](@ref) for a `Function` `V`
(it forwards `kwargs` to the QTCI of a 2-arg kernel).
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
# 13. Display
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
