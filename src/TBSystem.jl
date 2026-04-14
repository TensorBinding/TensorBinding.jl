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
- `_tn_cache`      : cached Chebyshev polynomial list (`nothing` if stale)
- `_tn_Ncheb`      : order of the cached Chebyshev list
- `_density_cache` : cached density matrix MPO (`nothing` if stale)

Do not construct directly — use [`get_Hamiltonian`](@ref).
"""
mutable struct TBHamiltonian
    L        :: Int
    N        :: Int
    sites    :: Vector{<:Index}
    mpo      :: MPO
    geometry :: Union{Nothing, Matrix{Float64}}
    scale    :: Float64
    # ---- lazy caches (invalidated whenever mpo changes) ----
    _tn_cache      :: Union{Nothing, Vector{MPO}}
    _tn_Ncheb      :: Int
    _density_cache :: Union{Nothing, MPO}
end

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
    H._tn_Ncheb      = 0
    H._density_cache = nothing
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
| `"chain_1d"`  | hopping amplitude `t::Number`  | —                             |
| `"square_2d"` | hopping amplitude `t::Number`  | `Lx` (sites per row; default √N) |
| `"haldane"`   | `(t2, phi, M)` NamedTuple      | `rs` (N×2 Float64 position matrix, required) |
| `"custom"`    | hopping function `f(i,j)`      | `geometry`, `scale` (required), `type` |

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
                         kwargs...)
    sites = siteinds("Qubit", L)
    N     = 2^L

    if geometry == "chain_1d"
        return _build_chain_1d(params, L, N, sites; scale, tol, maxdim)

    elseif geometry == "square_2d"
        return _build_square_2d(params, L, N, sites; scale, tol, maxdim, kwargs...)

    elseif geometry == "haldane"
        return _build_haldane(params, L, N, sites; scale, tol, maxdim, kwargs...)

    elseif geometry == "custom"
        return _build_custom(params, L, N, sites; scale, tol, maxdim, kwargs...)

    # ---- new preset models routed through build_hamiltonian ----
    elseif geometry in ("ssh", "aah", "uniform",
                        "hex_2d", "triangular_2d",
                        "chern8", "chernhex", "qc2dsquare")
        return _build_preset(geometry, params, L, N, sites; scale, tol, maxdim, kwargs...)

    else
        known = ("chain_1d", "square_2d", "haldane", "custom",
                 "ssh", "aah", "uniform",
                 "hex_2d", "triangular_2d",
                 "chern8", "chernhex", "qc2dsquare")
        error("Unknown geometry \"$geometry\". Supported: $(join(known, ", ")).")
    end
end

# ============================================================
# Per-geometry builders (internal)
# ============================================================

function _build_chain_1d(t, L, N, sites; scale=nothing, tol=1e-8, maxdim=15)
    # kinetic_1d_nn implements NN hopping in the quantics binary representation
    # (sigma_plus/minus acting as binary increment/decrement operators)
    mpo = t * kinetic_1d_nn(L, sites)
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    geom = reshape(Float64.(1:N), N, 1)
    sc   = something(scale, 2.2 * abs(t))   # 1D chain bandwidth = 4|t|, half = 2|t|
    return TBHamiltonian(L, N, sites, mpo, geom, sc, nothing, 0, nothing)
end


function _build_square_2d(t, L, N, sites;
                          Lx=nothing, scale=nothing, tol=1e-8, maxdim=15)
    Lx = something(Lx, isqrt(N))
    @assert Lx^2 == N "square_2d requires N = 2^L to be a perfect square. " *
                      "Got N = $N with Lx = $Lx."
    hop_x = intrachain_hopping(Lx, N, sites; t=t)
    hop_y = interchain_hopping_square(Lx, N, sites; t=t)
    mpo   = +(hop_x, hop_y; cutoff=tol)
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    geom  = Matrix{Float64}(undef, N, 2)
    for i in 1:N
        geom[i, 1] = Float64(mod(i - 1, Lx) + 1)
        geom[i, 2] = Float64(div(i - 1, Lx) + 1)
    end
    sc = something(scale, 4.4 * abs(t))   # 2D square bandwidth ≈ 8|t|, half = 4|t|
    return TBHamiltonian(L, N, sites, mpo, geom, sc, nothing, 0, nothing)
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
    return TBHamiltonian(L, N, sites, mpo, Float64.(rs), sc, nothing, 0, nothing)
end


function _build_custom(f, L, N, sites;
                       geometry=nothing,
                       scale=nothing,
                       tol=1e-8,
                       maxdim=15,
                       type=ComplexF64)
    @assert !isnothing(scale) "`scale` must be provided for geometry=\"custom\"."
    mpo = hopping2MPO(f, N, sites; tol=tol, type=type)
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    return TBHamiltonian(L, N, sites, mpo, geometry, Float64(scale), nothing, 0, nothing)
end

function _build_preset(geometry, params, L, N, sites;
                       scale=nothing, tol=1e-8, maxdim=15, kwargs...)
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
    sc = something(scale, _estimate_scale(geometry, params))
    return TBHamiltonian(L, N, mpo_sites, mpo, nothing, Float64(sc), nothing, 0, nothing)
end

# Rough scale estimates for known geometries (used when scale=nothing)
function _estimate_scale(geometry, params)
    t = params isa Number ? abs(params) :
        params isa NamedTuple && hasfield(typeof(params), :t) ? abs(params.t) :
        params isa AbstractDict && haskey(params, :t) ? abs(params[:t]) : 1.0
    geometry == "ssh"          && return 2.5 * t
    geometry == "aah"          && return (t + (params isa NamedTuple ? abs(params.V) : 1.0)) * 1.2
    geometry == "uniform"      && return 2.5 * t
    geometry == "hex_2d"       && return 4.0 * t
    geometry == "triangular_2d"&& return 7.0 * t
    geometry in ("chern8","chernhex","qc2dsquare") && return 6.0 * t
    return 5.0 * t   # conservative fallback
end


# ============================================================
# Geometry helpers
# ============================================================

"""
    honeycomb_positions(L) -> Matrix{Float64}

Generate `N = 2^L` positions on a honeycomb lattice arranged in a
square patch of `√(N/2) × √(N/2)` unit cells, with nearest-neighbour
bond length = 1.

Returns an `N × 2` matrix.  Sites are ordered so that site `2k-1`
is sublattice A and site `2k` is sublattice B in unit cell `k`.

Requires `N/2` to be a perfect square.
"""
function honeycomb_positions(L::Int)
    N   = 2^L
    @assert iseven(N) "N = 2^L must be even for a honeycomb (two sites per unit cell)."
    Nc  = N ÷ 2
    Lc  = isqrt(Nc)
    @assert Lc^2 == Nc "honeycomb_positions requires N/2 to be a perfect square. " *
                       "Got N/2 = $Nc.  Try L such that 2^(L-1) is a perfect square."
    # Primitive lattice vectors (NN distance = 1)
    a1 = [√3,   0.0]
    a2 = [√3/2, 3/2]
    # Sublattice offsets within the unit cell
    dA = [0.0, 0.0]
    dB = [1.0, 0.0]
    rs  = Matrix{Float64}(undef, N, 2)
    idx = 1
    for n2 in 0:Lc-1, n1 in 0:Lc-1
        origin        = n1 .* a1 .+ n2 .* a2
        rs[idx,     :] = origin .+ dA
        rs[idx + 1, :] = origin .+ dB
        idx += 2
    end
    return rs
end

# ============================================================
# Additive interaction API
# ============================================================

"""
    add_hopping!(H, f; maxdim=15, tol=1e-8, type=ComplexF64) -> H

Add an arbitrary hopping term to `H` defined by the function
`f(i, j)` over site indices `i, j ∈ {1, …, N}`.  The term is
compressed via QTCI and added to `H.mpo`.  Caches are invalidated.

Examples
--------
```julia
# Second-neighbour hopping on a 1D chain
add_hopping!(H, (i, j) -> abs(i - j) == 2 ? -t2 : 0.0)

# NNN complex hopping (Haldane-like extra term)
add_hopping!(H, (i, j) -> ...; type=ComplexF64)
```
"""
function add_hopping!(H::TBHamiltonian, f;
                      maxdim=15, tol=1e-8, type=ComplexF64)
    new_term = hopping2MPO(f, H.N, H.sites; tol=tol, type=type)
    H.mpo    = +(H.mpo, new_term; maxdim=maxdim, cutoff=tol)
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
    new_term = get_diagonal_mpo(H.L, H.sites, f)
    H.mpo    = +(H.mpo, new_term; cutoff=tol)
    _invalidate_cache!(H)
    return H
end

# ============================================================
# Display
# ============================================================

function Base.show(io::IO, H::TBHamiltonian)
    tn_str = H._tn_cache !== nothing ?
             "Tn cached (Ncheb = $(H._tn_Ncheb))" : "no Tn cache"
    geom_str = isnothing(H.geometry) ? "implicit 1D" :
               "$(size(H.geometry, 1)) sites, $(size(H.geometry, 2))D"
    print(io, "TBHamiltonian | L=$(H.L), N=$(H.N), scale=$(H.scale), " *
              "maxlinkdim=$(ITensorMPS.maxlinkdim(H.mpo)) | " *
              "geometry: $geom_str | $tn_str")
end
