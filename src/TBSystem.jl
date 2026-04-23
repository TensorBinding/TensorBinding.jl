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
    # ---- auxiliary prepended indices (nothing until add_spin!/add_superconductivity!) ----
    spin_s   :: Union{Nothing, Index}
    nambu_s  :: Union{Nothing, Index}
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
    return TBHamiltonian(L, N, sites, mpo, geom, sc, nothing, nothing, nothing, 0, nothing)
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
    return TBHamiltonian(L, N, sites, mpo, geom, sc, nothing, nothing, nothing, 0, nothing)
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
    return TBHamiltonian(L, N, sites, mpo, Float64.(rs), sc, nothing, nothing, nothing, 0, nothing)
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
    return TBHamiltonian(L, N, sites, mpo, geometry, Float64(scale), nothing, nothing, nothing, 0, nothing)
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
    return TBHamiltonian(L, N, mpo_sites, mpo, nothing, Float64(sc), nothing, nothing, nothing, 0, nothing)
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
    H.spin_s === nothing && H.nambu_s === nothing ||
        error("add_hopping! must be called before add_spin!/add_superconductivity!. " *
              "Build the full normal-state Hamiltonian first.")
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

Return the L position-qubit indices.  These are always the last L entries of
`H.sites`; spin and Nambu indices (if any) are prepended in front of them.
"""
_pos_sites(H::TBHamiltonian) = H.sites[end - H.L + 1 : end]


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
function add_spin!(H::TBHamiltonian; cutoff::Real=1e-8, maxdim::Int=200)
    H.spin_s === nothing || return H
    spin_s   = spin_index()
    H.mpo    = prepend_spin(H.mpo, spin_s, :Id)
    ITensorMPS.truncate!(H.mpo; maxdim=maxdim, cutoff=cutoff)
    H.sites  = [spin_s; H.sites]
    H.spin_s = spin_s
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
                     maxdim::Int = 200)
    direction in (:x, :y, :z) ||
        error("direction must be :x, :y, or :z; got :$direction")
    add_spin!(H; cutoff=tol, maxdim=maxdim)

    spin_op = direction == :z ? :Sz : direction == :x ? :Sx : :Sy
    pos_s   = _pos_sites(H)
    h_mpo   = h isa Number ? h * MPO(pos_s, "Id") :
                             get_diagonal_mpo(H.L, pos_s, h)

    H_Z = prepend_spin(h_mpo, H.spin_s, spin_op)
    if H.nambu_s !== nothing
        # BdG already present: Zeeman is τ_z ⊗ S_α ⊗ h(r)
        H_Z = prepend_nambu(H_Z, H.nambu_s, :tz)
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
                                maxdim::Int  = 200)
    H.nambu_s === nothing ||
        error("BdG already applied (H.nambu_s is set). Cannot apply twice.")

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

    # ── Lift pairing to full site space ──────────────────────────────────────
    H_pair = H.spin_s !== nothing ?
             prepend_spin(H_pair_pos, H.spin_s, :iSy) :   # singlet: (iσ_y) ⊗ Δ
             H_pair_pos

    H_pair_adj = swapprime(dag(H_pair), 0, 1)

    # ── BdG assembly ─────────────────────────────────────────────────────────
    nambu_s = nambu_index()
    H_bdg   = +(+(prepend_nambu(H.mpo,      nambu_s, :tz),
                  prepend_nambu(H_pair,     nambu_s, :tp); cutoff=tol),
                  prepend_nambu(H_pair_adj, nambu_s, :tm); cutoff=tol)
    ITensorMPS.truncate!(H_bdg; maxdim=maxdim, cutoff=tol)

    Δ_scale   = Δ isa Number ? abs(Δ) : 1.0
    H.mpo     = H_bdg
    H.sites   = [nambu_s; H.sites]
    H.nambu_s = nambu_s
    H.scale   = H.scale + Δ_scale * 1.1   # rough update; user can override
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
                  maxdim::Int       = 200)
    add_spin!(H; cutoff=tol, maxdim=maxdim)
    pos_s = _pos_sites(H)

    H_soc = if type === :ising
        λ_mpo = λ isa Number ? λ * MPO(pos_s, "Id") :
                               get_diagonal_mpo(H.L, pos_s, λ)
        prepend_spin(λ_mpo, H.spin_s, :Sz)

    elseif type === :rashba
        λ isa Number || error("Rashba SOC requires a scalar λ; got $(typeof(λ)).")
        K_u = generate_kin_u(pos_s, H.N)
        K_d = generate_kin_d(pos_s, H.N)
        +(prepend_spin( λ * K_u, H.spin_s, :Sy),
          prepend_spin(-λ * K_d, H.spin_s, :Sy); cutoff=tol)

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
        prepend_spin(λ_mpo, H.spin_s, spin_op)

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
    geom_str = isnothing(H.geometry) ? "implicit 1D" :
               "$(size(H.geometry, 1)) sites, $(size(H.geometry, 2))D"
    aux_str  = ""
    H.spin_s  !== nothing && (aux_str *= " +spin")
    H.nambu_s !== nothing && (aux_str *= " +BdG")
    print(io, "TBHamiltonian | L=$(H.L), N=$(H.N)$(aux_str), scale=$(H.scale), " *
              "maxlinkdim=$(ITensorMPS.maxlinkdim(H.mpo)) | " *
              "geometry: $geom_str | $tn_str")
end
