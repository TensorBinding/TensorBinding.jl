# TBSystem.jl — central struct and constructor for tight-binding systems
#
# Provides TBHamiltonian, which wraps the Hamiltonian MPO together with
# metadata (geometry, KPM scale) and lazy caches for Chebyshev moments
# and the density matrix.  All observable methods (get_DoS, get_density,
# get_Chern, get_bands …) dispatch on this struct.

# ============================================================
# TBHamiltonian struct
# ============================================================

"""
    TBHamiltonian

Central object representing a tight-binding Hamiltonian in the
quantics MPO framework.

Fields
------
- `L`        : number of qubit sites (log₂ of the physical system size)
- `N`        : number of physical sites (2^L)
- `sites`    : ITensor `Qubit` site indices of length `L`
- `mpo`      : accumulated Hamiltonian as an ITensor MPO
- `geometry` : `N × d` matrix of real-space positions (`nothing` for implicit 1D)
- `scale`    : energy half-bandwidth such that `H/scale` has spectrum in `[-1, 1]`;
               required for KPM.  Must satisfy `scale > spectral_radius(H)`.
- `_tn_cache`      : cached MPO Chebyshev list (`nothing` if stale); set by `KPM_Tn(...; mode=:mpo)`
- `_tn_mps_cache`  : cached MPS Chebyshev list (`nothing` if stale); set by `KPM_Tn(...; mode=:mps)`
- `_tn_Ncheb`      : order of the cached Chebyshev list (shared between both caches)
- `_density_cache` : cached density matrix MPO (`nothing` if stale)

Do not construct directly — use [`get_Hamiltonian`](@ref).
"""
mutable struct TBHamiltonian
    L        :: Int
    N        :: Int
    sites    :: Vector{<:Index}
    mpo      :: MPO
    geometry :: Union{Nothing, Function}   # i -> position vector (1-indexed, i=1…N)
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
end

# Backward-compatible 15-arg constructor (pre-sublattice_s callers); inserts sublattice_s=nothing.
TBHamiltonian(L, N, sites, mpo, geometry, scale, center,
              spin_s, nambu_s, layer_s, aux_side, _tn_cache, _tn_mps_cache, _tn_Ncheb, _density_cache) =
    TBHamiltonian(L, N, sites, mpo, geometry, scale, center,
                  spin_s, nambu_s, layer_s, nothing, aux_side, _tn_cache, _tn_mps_cache, _tn_Ncheb, _density_cache)

# Backward-compatible 14-arg constructor (pre-sublattice_s, pre-_tn_mps_cache callers).
TBHamiltonian(L, N, sites, mpo, geometry, scale, center,
              spin_s, nambu_s, layer_s, aux_side, _tn_cache, _tn_Ncheb, _density_cache) =
    TBHamiltonian(L, N, sites, mpo, geometry, scale, center,
                  spin_s, nambu_s, layer_s, nothing, aux_side, _tn_cache, nothing, _tn_Ncheb, _density_cache)

# Backward-compatible 13-arg constructor (pre-sublattice_s, pre-aux_side callers); defaults to :pre.
TBHamiltonian(L, N, sites, mpo, geometry, scale, center,
              spin_s, nambu_s, layer_s, _tn_cache, _tn_Ncheb, _density_cache) =
    TBHamiltonian(L, N, sites, mpo, geometry, scale, center,
                  spin_s, nambu_s, layer_s, nothing, :pre, _tn_cache, nothing, _tn_Ncheb, _density_cache)

# ============================================================
# Cache management
# ============================================================

"""
    _invalidate_cache!(H) -> H

Clear all cached intermediate results.  Called automatically by
`add_hopping!` and `add_onsite!` whenever the Hamiltonian changes.
"""
function _invalidate_cache!(H::TBHamiltonian)
    H._tn_cache      = nothing
    H._tn_mps_cache  = nothing
    H._tn_Ncheb      = 0
    H._density_cache = nothing
    # Spectrum changed — force re-estimation of scale/center on next KPM call.
    # Analytic constructors re-set scale immediately; layered/custom start at 0.
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
    kwargs = maxdim === nothing ? (cutoff=cutoff,) : (cutoff=cutoff, maxdim=maxdim)
    ITensorMPS.truncate!(H.mpo; kwargs...)
    _invalidate_cache!(H)
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
| `"haldane"`   | `(t2, phi, M)` NamedTuple      | `rs` (N×2 Float64 position matrix, required) |
| `"custom"`    | hopping function `f(i,j)`      | `geometry`, `scale` (required), `type` |
| `"kagome"`    | hopping amplitude `t::Number`  | `Lx`, `Ly`; 3-atom unit cell, sublattice index postpended |
| `"lieb"`      | hopping amplitude `t::Number`  | `Lx`, `Ly`; 3-atom unit cell, sublattice index postpended |

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
    sites = siteinds("Qubit", L)
    N     = 2^L

    if geometry == "chain_1d"
        return _build_chain_1d(params, L, N, sites; scale, tol, maxdim)

    elseif geometry == "haldane"
        return _build_haldane(params, L, N, sites; scale, tol, maxdim, kwargs...)

    elseif geometry == "custom"
        return _build_custom(params, L, N, sites; scale, tol, maxdim, kwargs...)

    # ---- multi-atom unit-cell lattices (kagomé, Lieb, honeycomb) ----
    elseif geometry in ("kagome", "lieb", "honeycomb", "dice")
        return _build_sublattice(geometry, params, L; scale, tol, maxdim, kwargs...)

    # ---- preset models routed through build_hamiltonian ----
    elseif geometry in ("ssh", "aah", "uniform",
                        "square_2d", "hex_2d", "triangular_2d",
                        "chern8", "chernhex", "qc2dsquare")
        return _build_preset(geometry, params, L, N, sites; scale, tol, maxdim, ref_sites, kwargs...)

    else
        known = ("chain_1d", "haldane", "custom",
                 "uniform", "ssh", "aah",
                 "square_2d", "hex_2d", "triangular_2d",
                 "chern8", "chernhex", "qc2dsquare",
                 "kagome", "lieb", "honeycomb", "dice")
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

function _build_chain_1d(t, L, N, sites; scale=nothing, tol=1e-8, maxdim=15)
    mpo = t * kinetic_1d_nn(L, sites)
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    sc  = something(scale, 2.5 * abs(t))
    return TBHamiltonian(L, N, sites, mpo, _chain_geometry(), sc, 0.0, nothing, nothing, nothing, nothing, 0, nothing)
end


function _build_haldane(params, L, N, sites;
                        rs=nothing, scale=nothing, tol=1e-8, maxdim=15)
    @assert !isnothing(rs) "Haldane model requires keyword `rs` (N×2 position matrix). " *
                           "Generate it with `honeycomb_positions($L)` or from `get_G()`."
    t2  = params.t2
    phi = params.phi
    M   = params.M
    f(i, j) = haldane_hoppingf(rs[Int(i), :], rs[Int(j), :],
                                Int(i), Int(j); t2=t2, phi=phi, M=M)
    mpo = hopping2MPO(f, N, sites; tol=tol, type=ComplexF64)
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    # bandwidth ≈ 2*(3*t1 + 6*t2) + 2*M where t1=1; conservative upper bound
    sc  = something(scale, (1.0 + abs(t2) + abs(M)) * 4.0)
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
    sc   = something(scale, _estimate_scale(geometry, params))
    geom = _preset_geometry(geometry, dim == 2 ? 2^get(kwargs, :Lx, L ÷ 2) : nothing)
    return TBHamiltonian(L, N, mpo_sites, mpo, geom, Float64(sc), 0.0, nothing, nothing, nothing, nothing, 0, nothing)
end

function _build_sublattice(geometry, params, L;
                            scale=nothing, tol=1e-8, maxdim=200, kwargs...)
    Lx = get(kwargs, :Lx, L ÷ 2)
    Ly = get(kwargs, :Ly, L - Lx)
    t  = params isa Number                                           ? params      :
         params isa NamedTuple && hasfield(typeof(params), :t)      ? params.t    :
         params isa AbstractDict && haskey(params, :t)              ? params[:t]  : 1.0

    H = geometry == "kagome"    ? kagome_hamiltonian(               Lx, Ly, t; cutoff=tol, maxdim=maxdim) :
        geometry == "lieb"      ? lieb_hamiltonian(                Lx, Ly, t; cutoff=tol, maxdim=maxdim) :
        geometry == "dice"      ? dice_hamiltonian(                Lx, Ly, t; cutoff=tol, maxdim=maxdim) :
                                  honeycomb_sublattice_hamiltonian(Lx, Ly, t; cutoff=tol, maxdim=maxdim)

    rs         = geometry == "kagome"    ? kagome_positions(                 Lx, Ly) :
                 geometry == "lieb"      ? lieb_positions(                   Lx, Ly) :
                 geometry == "dice"      ? dice_positions(                   Lx, Ly) :
                                          honeycomb_sublattice_positions(    Lx, Ly)
    H.geometry = let m = rs; i -> m[i, :]; end
    isnothing(scale) || (H.scale = Float64(scale))
    return H
end


function _preset_geometry(geometry, Nx)
    geometry in ("uniform", "ssh", "aah", "chain_1d") && return _chain_geometry()
    geometry == "square_2d"    && return _square_geometry(Nx)
    geometry == "hex_2d"       && return _hex_geometry(Nx)
    geometry == "triangular_2d"&& return _tri_geometry(Nx)
    return nothing
end

# Rough scale estimates for known geometries (used when scale=nothing)
function _estimate_scale(geometry, params)
    t = params isa Number ? abs(params) :
        params isa NamedTuple && hasfield(typeof(params), :t) ? abs(params.t) :
        params isa AbstractDict && haskey(params, :t) ? abs(params[:t]) : 1.0
    geometry == "chain_1d"     && return 2.5 * t
    geometry == "ssh"          && return 2.5 * t
    geometry == "aah"          && return (t + (params isa NamedTuple ? abs(params.V) : 1.0)) * 1.2
    geometry == "uniform"      && return 2.5 * t
    geometry == "square_2d"    && return 4.4 * t
    geometry == "hex_2d"       && return 4.0 * t
    geometry == "triangular_2d"&& return 7.0 * t
    geometry in ("chern8","chernhex","qc2dsquare") && return 6.0 * t
    return 5.0 * t   # conservative fallback
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
    add_hopping!(H, f; nn=1, maxdim=15, tol=1e-8, type=ComplexF64, apply_kwargs=NamedTuple()) -> H

Add a hopping term to `H`.  Three calling modes, selected automatically by the type of `f`:

- **Constant** (`f::Number`): uniform nth-neighbour hopping with amplitude `f`.
  Builds the MPO via `kineticNNN` with a uniform diagonal hopping weight.  Use `nn`
  to select the neighbour shell (default `nn=1` = nearest neighbour).

- **Site-dependent** (`f(i)`, 1-arg `Function`): spatially varying hopping.
  A diagonal hopping MPO is built from `f` via `get_diagonal_mpo` (1-indexed,
  `i ∈ {1, …, N}`), then passed to `kineticNNN`.  Use `nn` as above.

- **Full matrix QTCI** (`f(i,j)`, 2-arg `Function`): the full N×N hopping matrix
  is compressed via Quantics Tensor Cross Interpolation.  `nn` is ignored in
  this mode; the function itself encodes the connectivity.

`apply_kwargs` (e.g. `(; cutoff=1e-8, maxdim=100)`) are forwarded to every `apply`
call inside `kineticNNN` (constant and site-dependent modes only).

Examples
--------
```julia
add_hopping!(H, -1.0)                                    # uniform NN hopping
add_hopping!(H, -0.3; nn=2)                              # uniform NNN hopping
add_hopping!(H, i -> cos(2π*i/H.N); nn=1)               # site-dependent NN hopping
add_hopping!(H, (i, j) -> abs(i-j) == 2 ? -0.3 : 0.0)  # full QTCI (any connectivity)
```
"""
function add_hopping!(H::TBHamiltonian, f;
                      nn::Integer      = 1,
                      maxdim           = 15,
                      tol              = 1e-8,
                      type             = ComplexF64,
                      apply_kwargs     = NamedTuple())
    H.spin_s === nothing && H.nambu_s === nothing &&
        H.layer_s === nothing && H.sublattice_s === nothing ||
        error("add_hopping! must be called before add_spin!/add_superconductivity! " *
              "and cannot be used on layered or sublattice Hamiltonians.")
    pos_s    = _pos_sites(H)
    new_term = if f isa Number
        kineticNNN(H.L, pos_s, f * MPO(pos_s, "Id"), nn; apply_kwargs)
    elseif f isa Function && applicable(f, 1)
        kineticNNN(H.L, pos_s, get_diagonal_mpo(H.L, pos_s, f), nn; apply_kwargs)
    else
        hopping2MPO(f, H.N, pos_s; tol=tol, type=type)
    end
    H.mpo = +(H.mpo, new_term; maxdim=maxdim, cutoff=tol)
    ITensorMPS.truncate!(H.mpo; maxdim=maxdim, cutoff=tol)
    _invalidate_cache!(H)
    return H
end


"""
    add_onsite!(H, f; tol=1e-8) -> H

Add a diagonal (on-site) term defined by `f(i)` for site `i ∈ {1, …, N}`.
Compressed via QTCI using `get_diagonal_mpo`.  Caches are invalidated.

Examples
--------
```julia
# Linear potential (electric field)
add_onsite!(H, i -> 0.01 * i)

# Quasicrystal modulation
add_onsite!(H, i -> V * cos(2π * α * i))
```
"""
function add_onsite!(H::TBHamiltonian, f; tol=1e-8)
    new_term = get_diagonal_mpo(H.L, _pos_sites(H), f)
    H.mpo    = +(H.mpo, new_term; cutoff=tol)
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

- **Spinless** (default): simple spinless BdG; `H_pair = Δ(r) · I`.
- **Spinful** (`add_spin!` called first): singlet pairing;
  `H_pair = (i·σ_y)_spin ⊗ Δ(r)`, the standard BCS Cooper-pair operator.

`Δ` can be:
- a `Number`   — uniform on-site gap (s-wave, `type=:swave`)
- a `Function` `Δ(i)` — spatially varying diagonal gap (`type=:swave`)
- a `Function` `Δ(i,j)` — general pairing matrix compressed via TCI (`type=:custom`)

`type`:
- `:swave`  (default) — diagonal pairing, `H_pair = diag(Δ(1),…,Δ(N))`
- `:custom` — off-diagonal (p-wave, d-wave …); pass a 2-arg function `Δ(i,j)`

Errors if BdG has already been applied.  Invalidates all caches.

Examples
--------
```julia
add_superconductivity!(H, 0.1)                      # uniform s-wave
add_superconductivity!(H, i -> i < N÷2 ? 0.1 : 0.0) # half-system gap
add_superconductivity!(H, (i,j) -> ...; type=:custom) # p-wave
```
"""
function add_superconductivity!(H::TBHamiltonian, Δ;
                                type::Symbol = :swave,
                                tol::Real    = 1e-8,
                                maxdim::Int  = 200,
                                position::Union{Nothing,Symbol} = nothing)
    H.nambu_s === nothing ||
        error("BdG already applied (H.nambu_s is set). Cannot apply twice.")

    pos   = something(position, H.aux_side)
    pos_s = _pos_sites(H)

    # ── Build the pairing MPO in position space ──────────────────────────────
    H_pair_pos = if type === :swave
        Δ isa Number   ? Δ * MPO(pos_s, "Id")            :
        Δ isa Function ? get_diagonal_mpo(H.L, pos_s, Δ) :
        error("For type=:swave, Δ must be a Number or a 1-arg Function.")
    elseif type === :custom
        Δ isa Function ||
            error("For type=:custom, Δ must be a 2-arg Function Δ(i,j).")
        hopping2MPO(Δ, H.N, pos_s; tol=tol, type=ComplexF64)
    else
        error("Unknown pairing type :$type.  Use :swave or :custom.")
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
        +(spin_prepend( λ * K_u, H.spin_s, :Sy),
          spin_prepend(-λ * K_d, H.spin_s, :Sy); cutoff=tol)

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
