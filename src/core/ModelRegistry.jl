# ModelRegistry.jl — the model registry behind get_Hamiltonian: one entry per geometry
# name (builder, dimension, parameters and defaults, geometry, default KPM scale rule),
# the preset dispatcher build_hamiltonian and the KPM scale maker estimate_scale.
#
# Contents: the parameter helpers _param and _parse_param_string; the builders of the
# preset, multi-atom and projected entries (_build_preset, _build_sublattice,
# _build_ssh_sublattice, the ref_sites guard of the projected spaces); today's default
# scale formulas (_estimate_scale, _chernhex_scale) and the preset geometry closures
# (_preset_geometry), read from the entries; ModelEntry and MODELS, the registry, with
# MODEL_REGISTRY, its (builder, dim, required, kw_defaults) view of the presets;
# build_hamiltonian; the scale maker: the dense small-system estimate and its cache,
# estimate_scale, and the default and explicit scale rules that get_Hamiltonian applies.
#
# Main entry points: build_hamiltonian, estimate_scale; get_Hamiltonian
# (core/TBSystem.jl) looks every geometry up in MODELS.
#
# Depends on: Utils (fix_sites, _mpo_dense_matrix), TBSystem (TBHamiltonian, the chain,
# Haldane and custom builders), geometry (the i -> position closures), Fibonacci,
# MetallicMean, KBonacci (their get_Hamiltonian wrappers), presets (the H* builders,
# looked up by Symbol at call time), sublattice* (builders and position tables, looked
# up by Symbol) and DMRG* (_estimate_spectral_bounds). Split from the former
# lattice/2Dlattice_tk.jl (first as lattice/model_registry.jl); the entries replace the
# if-chain of get_Hamiltonian and the scale and geometry tables that core/TBSystem.jl
# and lattice/geometry.jl kept per geometry name.
#
# Known quirks, kept on purpose (the golden tests pin them):
#   - _estimate_scale reads `t` from `params` only (an `mparams` string is ignored, except
#     by "chernhex"), and "aah" reads `V` only from a NamedTuple (a Dict counts as V = 1);
#     geometries without a formula get 5|t| (get_Hamiltonian never asks it for them);
#   - the preset builders ignore the Qubit sites that get_Hamiltonian draws for them;
#   - the multi-atom lattices ignore every keyword but Lx, Ly (and ref_sites, which
#     replaces their position qubits); "ssh_sublattice" ignores them all but ref_sites;
#   - MODEL_REGISTRY keyword defaults that differ from the builders' own (e.g.
#     "qc2dsquare" tol_quantics=1e-9, maxbonddim_quantics=250) stay: they set the MPOs.

# ============================================================
# 1. Parameter helpers
# ============================================================

"""
    _param(params, name, default; scalar=false) -> value

Parameter `name` of a `get_Hamiltonian` `params` argument: the field of a NamedTuple,
the entry of an AbstractDict (keyed by Symbol), and `default` when it has none. A bare
Number is the value of the parameter asked for with `scalar=true` (the shorthand
`get_Hamiltonian("kagome", 1.0)` for `t`) and gives `default` otherwise; any other
`params` gives `default`.
"""
_param(params::NamedTuple, name::Symbol, default; scalar::Bool=false) =
    hasfield(typeof(params), name) ? getfield(params, name) : default
_param(params::AbstractDict, name::Symbol, default; scalar::Bool=false) =
    haskey(params, name) ? params[name] : default
_param(params::Number, name::Symbol, default; scalar::Bool=false) =
    scalar ? params : default
_param(params, name::Symbol, default; scalar::Bool=false) = default

# |t| with the single-number shorthand, 1 when `params` has no `t`.
_abs_t(params) = abs(_param(params, :t, 1.0; scalar=true))

"""
    _parse_param_string(s) -> Dict{Symbol,Any}

Parse `"key1=val1, key2=val2, …"` into a Dict. Values are auto-typed
as Bool, Int, Float64, or String.
"""
function _parse_param_string(s::AbstractString)
    d = Dict{Symbol,Any}()
    isempty(strip(s)) && return d
    for tok in split(strip(s), [',',' ','\t'])
        isempty(tok) && continue
        kv = split(tok, '='; limit=2)
        length(kv) == 2 || error("Bad param token '$tok' (expected key=value)")
        k   = Symbol(strip(kv[1]))
        v   = strip(kv[2])
        vl  = lowercase(v)
        val::Any = vl in ("true","false")                               ? (vl=="true") :
                   occursin(r"^[+-]?\d+$", v)                          ? parse(Int, v) :
                   occursin(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$", v) ? parse(Float64, v) :
                   v
        d[k] = val
    end
    return d
end


# ============================================================
# 2. The registry entry type
# ============================================================

"""
    ModelEntry

One `get_Hamiltonian` geometry in the model registry [`MODELS`](@ref).

Fields
- `name`         : the geometry name.
- `kind`         : `:direct` (`"chain_1d"`, `"haldane"`, `"custom"`), `:preset` (the H*
                   builders reached through [`build_hamiltonian`](@ref)), `:sublattice`
                   (the multi-atom unit cells) or `:projected` (the quasicrystal spaces).
- `dim`          : spatial dimension (2D models take `Lx`, `Ly`).
- `n_sub`        : atoms per unit cell (dimension of the sublattice index).
- `params`       : parameter names (for a preset, the required ones, passed positionally).
- `defaults`     : defaults of the other parameters (for a preset, the builder keywords
                   that [`MODEL_REGISTRY`](@ref) lists).
- `builder`      : name of the function that builds the operator.
- `build`        : `(params, L, N, sites; scale, tol, maxdim, ref_sites, kwargs...) ->
                   TBHamiltonian`, called by `get_Hamiltonian` with Qubit `sites` (unused
                   by the presets and multi-atom lattices) or `nothing` (projected spaces).
- `geometry`     : `Nx -> (i -> position)` for the models whose builder takes its geometry
                   closure from the registry (`"chain_1d"` and the presets that have one).
- `positions`    : name of the `*_positions` table of a multi-atom lattice.
- `legacy_scale` : `(params, mparams) -> scale`, today's default formula when it lives in
                   the registry ([`_estimate_scale`](@ref) reads it); `nothing` when the
                   builder sets its default scale itself.
- `default_rule` : `:max` (default scale = `max(today's formula, estimate_scale(:auto))`)
                   or `:formula` (the builder's analytic default, unchanged).
- `auto`         : the method `estimate_scale(...; method=:auto)` uses.
- `rowsum`       : `(params, kwargs) -> bound`, the largest absolute row sum (Gershgorin
                   bound) of the builder's terms; `nothing` when there is no such rule.
- `resizable`    : whether the builder can be run again at a smaller size with the same
                   parameters (what `estimate_scale(...; method=:small)` needs above
                   1024 states).
"""
struct ModelEntry
    name         :: String
    kind         :: Symbol
    dim          :: Int
    n_sub        :: Int
    params       :: Vector{Symbol}
    defaults     :: NamedTuple
    builder      :: Symbol
    build        :: Function
    geometry     :: Union{Nothing,Function}
    positions    :: Union{Nothing,Symbol}
    legacy_scale :: Union{Nothing,Function}
    default_rule :: Symbol
    auto         :: Symbol
    rowsum       :: Union{Nothing,Function}
    resizable    :: Bool
end

ModelEntry(name::AbstractString; kind, dim, n_sub=1, params=Symbol[], defaults=NamedTuple(),
           builder, build, geometry=nothing, positions=nothing, legacy_scale=nothing,
           default_rule=:formula, auto, rowsum=nothing, resizable) =
    ModelEntry(String(name), kind, dim, n_sub, params, defaults, builder, build, geometry,
               positions, legacy_scale, default_rule, auto, rowsum, resizable)


# ============================================================
# 3. Builders of the preset, multi-atom and projected entries
# ============================================================

# `params` of a preset as the Dict that build_hamiltonian merges over the `mparams`
# string: a Dict or NamedTuple as given, a Number as the first required parameter,
# anything else as `t`.
function _preset_param_dict(entry::ModelEntry, params)
    params isa AbstractDict && return Dict{Symbol,Any}(params)
    params isa NamedTuple   && return Dict{Symbol,Any}(pairs(params))
    params isa Number       && return Dict{Symbol,Any}(entry.params[1] => params)
    return Dict{Symbol,Any}(:t => params)
end

# Every parameter value the preset builder sees: its registry defaults, then the
# `mparams` string, then `params` (build_hamiltonian's merge). Errors as
# build_hamiltonian does when a required parameter is missing.
function _preset_params(name::AbstractString, params, kwargs)
    entry = MODELS[name]
    p = Dict{Symbol,Any}(pairs(entry.defaults))
    for (k, v) in _parse_param_string(get(kwargs, :mparams, "")); p[k] = v; end
    for (k, v) in _preset_param_dict(entry, params); p[k] = v; end
    missing_p = [k for k in entry.params if !haskey(p, k)]
    isempty(missing_p) || error("Missing required params for '$name': $missing_p")
    return p
end

function _build_preset(geometry, params, L, N, sites;
                       scale=nothing, tol=1e-8, maxdim=15,
                       ref_sites::Union{Nothing,Vector{<:Index}}=nothing,
                       kwargs...)
    # Route through build_hamiltonian, which reads MODEL_REGISTRY; params (a scalar,
    # a NamedTuple or a Dict) becomes its mparam_dict.
    entry = MODELS[geometry]
    mparam_dict = _preset_param_dict(entry, params)
    mpo = if entry.dim == 1
        build_hamiltonian(geometry, L; mparam_dict, kwargs...)
    else
        # 2D: expect Lx and Ly in kwargs, or factorise L equally
        Lx = get(kwargs, :Lx, L ÷ 2)
        Ly = get(kwargs, :Ly, L - Lx)
        kw_filtered = Dict(k => v for (k, v) in kwargs if k ∉ (:Lx, :Ly))
        build_hamiltonian(geometry, Lx, Ly; mparam_dict, kw_filtered...)
    end
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=tol)
    # The model builders (HAAH, HSSH, …) create their own site indices internally,
    # so we extract the actual sites from the MPO rather than using the ones
    # created by get_Hamiltonian (which would be a different set).
    mpo_sites = _mpo_ket_sites(mpo)
    # If caller supplied ref_sites, replace MPO indices in-place so all
    # Hamiltonians built with the same ref_sites share identical Index objects.
    if !isnothing(ref_sites)
        fix_sites(mpo, ref_sites)
        mpo_sites = ref_sites
    end
    sc   = something(scale, _estimate_scale(geometry, params; mparams=get(kwargs, :mparams, "")))
    lx_2d = entry.dim == 2 ? get(kwargs, :Lx, L ÷ 2) : nothing
    geom = _preset_geometry(geometry, isnothing(lx_2d) ? nothing : 2^lx_2d)
    return TBHamiltonian(; L, N, sites=mpo_sites, mpo, geometry=geom, scale=Float64(sc),
                         Lx=lx_2d)
end

# The 2D multi-atom lattices: the builder `entry.builder(Lx, Ly, params...)` with the
# parameters of `entry.defaults` in order, the geometry from `entry.positions`.
function _build_sublattice(entry::ModelEntry, params, L;
                           scale=nothing, tol=1e-8, maxdim=200, ref_sites=nothing, kwargs...)
    Lx = get(kwargs, :Lx, L ÷ 2)
    Ly = get(kwargs, :Ly, L - Lx)
    args = Tuple(_param(params, k, v; scalar = k === :t) for (k, v) in pairs(entry.defaults))

    H  = getfield(@__MODULE__, entry.builder)(Lx, Ly, args...; cutoff=tol, maxdim=maxdim)
    rs = getfield(@__MODULE__, entry.positions)(Lx, Ly)
    H.geometry = let m = rs; i -> m[i, :]; end

    # UC geometry: the Bravais position of the atom's unit cell, which is that of the
    # cell's atom A (basis offset 0 in every *_positions table): ix a1 + iy a2 of the
    # lattice's own Bravais vectors (triangular for kagome, honeycomb and dice, square for
    # Lieb); n_sub = atoms per UC.
    n_sub = entry.n_sub
    H.geometry_uc = let m = rs, n_sub = n_sub
        i -> m[n_sub * ((i - 1) ÷ n_sub) + 1, :]
    end

    isnothing(scale) || (H.scale = Float64(scale))
    H.Lx = Lx
    ref_sites === nothing || _replace_pos_sites!(H, ref_sites, entry.name)
    return H
end

# SSH chain with an explicit sublattice index: `t` (or the single number) and `d`.
function _build_ssh_sublattice(params, L; scale=nothing, tol=1e-8, maxdim=15,
                               ref_sites=nothing)
    t = _param(params, :t, 1.0; scalar=true)
    d = _param(params, :d, 0.0)
    H = ssh_sublattice_hamiltonian(L, t, d; cutoff=tol, maxdim=maxdim)
    isnothing(scale) || (H.scale = Float64(scale))
    ref_sites === nothing || _replace_pos_sites!(H, ref_sites, "ssh_sublattice")
    return H
end

# `ref_sites` of get_Hamiltonian must be the L position qubits of the model, `sites` (the
# Qubit indices it would otherwise use or has built): same number, dimension 2.
function _check_ref_sites(ref_sites, sites, name)
    (length(ref_sites) == length(sites) && all(s -> dim(s) == 2, ref_sites)) ||
        throw(ArgumentError("ref_sites for \"$name\" must be its $(length(sites)) position " *
                            "qubits (dimension 2); got $(length(ref_sites)) indices of " *
                            "dimensions $(dim.(ref_sites))."))
    return ref_sites
end

# The sites a direct builder ("chain_1d", "haldane", "custom") builds on: `ref_sites` when
# given (checked), otherwise the Qubit sites get_Hamiltonian drew.
_ref_or_drawn(sites, ref_sites, name) =
    ref_sites === nothing ? sites : _check_ref_sites(ref_sites, sites, name)

# A multi-atom builder makes its own position qubits: `ref_sites` replaces them in the MPO
# and in H.sites (the sublattice index stays the builder's), so Hamiltonians built with
# the same ref_sites share their position indices.
function _replace_pos_sites!(H::TBHamiltonian, ref_sites, name)
    pos = _pos_sites(H)
    _check_ref_sites(ref_sites, pos, name)
    new_sites = [(k = findfirst(==(s), pos); k === nothing ? s : ref_sites[k]) for s in H.sites]
    fix_sites(H.mpo, new_sites)
    H.sites = new_sites
    return H
end

# get_Hamiltonian build function of a projected position space: rejects ref_sites.
_projected_build(f, space::AbstractString) =
    function (params, L, N, sites; ref_sites=nothing, kwargs...)
        ref_sites === nothing ||
            throw(ArgumentError("ref_sites is not supported for $space"))
        return f(params, L; kwargs...)
    end


# ============================================================
# 4. Today's default scale formulas and preset geometries
# ============================================================

"""
    _estimate_scale(geometry, params; mparams="") -> Float64

Today's default KPM scale formula of `geometry` (the entry's `legacy_scale`), `5|t|`
for a geometry without one. `mparams` is the parameter string the preset path forwards
to build_hamiltonian; only `"chernhex"` reads it. The default of get_Hamiltonian is
`max` of this and [`estimate_scale`](@ref) for the entries with `default_rule = :max`.
"""
function _estimate_scale(geometry, params; mparams::AbstractString="")
    entry = get(MODELS, geometry, nothing)
    rule  = entry === nothing ? nothing : entry.legacy_scale
    rule === nothing && return 5.0 * _abs_t(params)   # conservative fallback
    return rule(params, mparams)
end

# `c|t|` defaults of the preset table (t from `params` only).
_t_scale(c) = (params, mparams) -> c * _abs_t(params)

# "aah": 1.2(|t| + |V|), V read from a NamedTuple only (the quirk listed at the top).
_aah_legacy_scale(params, mparams) =
    (_abs_t(params) + (params isa NamedTuple ? abs(params.V) : 1.0)) * 1.2

# Row-sum bounds of the builders' terms (lattice/presets.jl, lattice/sublattice.jl):
# number of bonds per site times the largest amplitude, plus the largest on-site term.
_bonds(name, n) = (params, kwargs) -> n * abs(_preset_params(name, params, kwargs)[:t])
_sublattice_bonds(n) = (params, kwargs) -> n * _abs_t(params)

# HSSH: bond (x, x+nn) carries t + d on even x and t - d on odd x; a site has two bonds,
# of opposite parity for odd nn.
function _ssh_rowsum(p)
    a, b = abs(p[:t] + p[:d]), abs(p[:t] - p[:d])
    return isodd(p[:nn]) ? a + b : 2 * max(a, b)
end

# H2DChernhex: 3 NN bonds of |t|, 6 NNN bonds of |t2|, on-site |Ms| with Ms = ms, or
# ms + 3.3√3 t2 on the right half unless uniformsemenoff.
function _chernhex_rowsum(p)
    t, t2, ms = abs(p[:t]), p[:t2], p[:ms]
    Mmax = p[:uniformsemenoff] ? abs(ms) : max(abs(ms), abs(ms + 3.3 * sqrt(3) * t2))
    return 3.0 * t + 6.0 * abs(t2) + Mmax
end

# HChern8: 4 NN bonds of |t| and 4 diagonal bonds of |t| |Σₖ i V t2 cos²(k·r)| ≤ 4|t V t2|.
function _chern8_rowsum(p)
    t = abs(p[:t])
    return 4t + 16t * abs(p[:V] * get(p, :t2, 0.2 * p[:t]))   # HChern8: t2 = 0.2t by default
end

# HQC2Dsquare: 4 NN bonds of |t (1 + 0.1 Σₖ (2.5 cos + cos))| ≤ 2.4|t|.
_qc2dsquare_rowsum(p) = 4 * 2.4 * abs(p[:t])

"""
    _chernhex_scale(params, mparams) -> Float64

Default `"chernhex"` scale: the Gershgorin bound of H2DChernhex's terms (see
`_chernhex_rowsum`), padded by 10% and never below the former default 6|t|. The
parameters are merged the way _build_preset and build_hamiltonian merge them: the
registry defaults, the `mparams` string, then `params` on top.
"""
function _chernhex_scale(params, mparams::AbstractString)
    p = _preset_params("chernhex", params, (; mparams))
    return max(6.0 * abs(p[:t]), _SCALE_PADDING * _chernhex_rowsum(p))
end

"""
    _preset_geometry(geometry, Nx) -> Union{Function,Nothing}

The `i -> position` closure that get_Hamiltonian stores for `geometry` (the entry's
`geometry` rule at `Nx = 2^Lx` columns), `nothing` for geometries without one.
"""
function _preset_geometry(geometry, Nx)
    entry = get(MODELS, geometry, nothing)
    (entry === nothing || entry.geometry === nothing) && return nothing
    return entry.geometry(Nx)
end


# ============================================================
# 5. The registry
# ============================================================

_chain_rule(Nx) = _chain_geometry()

_preset(name, builder, dim, required, kw_defaults; geometry, legacy_scale=_t_scale(6.0),
        default_rule=:max, auto=:small, rowsum) =
    ModelEntry(name; kind=:preset, dim, params=required, defaults=kw_defaults, builder,
               build=(params, L, N, sites; kwargs...) ->
                   _build_preset(name, params, L, N, sites; kwargs...),
               geometry, legacy_scale, default_rule, auto, rowsum, resizable=true)

_sublattice(name, builder, positions, n_sub, defaults, rowsum) =
    ModelEntry(name; kind=:sublattice, dim=2, n_sub, params=collect(keys(defaults)),
               defaults, builder, positions,
               build=(params, L, N, sites; ref_sites=nothing, kwargs...) ->
                   _build_sublattice(MODELS[name], params, L; ref_sites, kwargs...),
               default_rule=:max, auto=:small, rowsum, resizable=true)

# In the order of get_Hamiltonian's "Supported: …" message.
const _MODEL_ENTRIES = ModelEntry[
    ModelEntry("chain_1d"; kind=:direct, dim=1, params=[:t], builder=:kinetic_1d_nn,
               build=(params, L, N, sites; ref_sites=nothing, kwargs...) ->
                   _build_chain_1d(params, L, N, _ref_or_drawn(sites, ref_sites, "chain_1d");
                                   kwargs...),
               geometry=_chain_rule, legacy_scale=_t_scale(2.5), default_rule=:max,
               auto=:small, rowsum=(params, kwargs) -> 2 * _abs_t(params), resizable=true),
    ModelEntry("haldane"; kind=:direct, dim=2, params=[:t2, :phi, :M],
               builder=:haldane_hoppingf,
               build=(params, L, N, sites; ref_sites=nothing, kwargs...) ->
                   _build_haldane(params, L, N, _ref_or_drawn(sites, ref_sites, "haldane");
                                  kwargs...),
               auto=:geometry, rowsum=(params, kwargs) -> _haldane_rowsum(params.t2, params.M),
               resizable=false),
    ModelEntry("custom"; kind=:direct, dim=1, builder=:hopping2MPO,
               build=(params, L, N, sites; ref_sites=nothing, kwargs...) ->
                   _build_custom(params, L, N, _ref_or_drawn(sites, ref_sites, "custom");
                                 kwargs...),
               auto=:dmrg, resizable=false),
    ModelEntry("fibonacci"; kind=:projected, dim=1, params=[:A, :B],
               defaults=(; t=1.0, onsite=0.0), builder=:fibonacci_hamiltonian,
               build=_projected_build(_build_fibonacci, "FibonacciPositionSpace"),
               auto=:dmrg, resizable=false),
    ModelEntry("metallic_mean"; kind=:projected, dim=1, params=[:A, :B],
               defaults=(; t=1.0, onsite=0.0), builder=:metallic_mean_hamiltonian,
               build=_projected_build(_build_metallic_mean, "MetallicMeanPositionSpace"),
               auto=:dmrg, resizable=false),
    ModelEntry("kbonacci"; kind=:projected, dim=1, params=[:values],
               defaults=(; t=1.0, onsite=0.0), builder=:kbonacci_hamiltonian,
               build=_projected_build(_build_kbonacci, "KBonacciPositionSpace"),
               auto=:dmrg, resizable=false),
    _preset("uniform", :HUniform, 1, [:t],
            (; v=1e-6, tol_quantics=1e-8, maxbonddim_quantics=10, nn=1);
            geometry=_chain_rule, legacy_scale=_t_scale(2.5),
            rowsum=(params, kwargs) -> (p = _preset_params("uniform", params, kwargs);
                                        2 * abs(p[:t]) + abs(p[:v]))),
    _preset("ssh", :HSSH, 1, [:t, :d], (; tol_quantics=1e-8, maxbonddim_quantics=10, nn=1);
            geometry=_chain_rule, legacy_scale=_t_scale(2.5),
            rowsum=(params, kwargs) -> _ssh_rowsum(_preset_params("ssh", params, kwargs))),
    ModelEntry("ssh_sublattice"; kind=:sublattice, dim=1, n_sub=2, params=[:t, :d],
               defaults=(; t=1.0, d=0.0), builder=:ssh_sublattice_hamiltonian,
               build=(params, L, N, sites; ref_sites=nothing, scale=nothing, tol=1e-8,
                      maxdim=15, kwargs...) ->
                   _build_ssh_sublattice(params, L; scale, tol, maxdim, ref_sites),
               auto=:small,
               rowsum=(params, kwargs) -> (t = _param(params, :t, 1.0; scalar=true);
                                           d = _param(params, :d, 0.0);
                                           abs(t + d) + abs(t - d)),
               resizable=true),
    _preset("aah", :HAAH, 1, [:V, :phi, :t],
            (; b=(1+sqrt(5))/2, tol_quantics=1e-8, maxbonddim_quantics=50);
            geometry=_chain_rule, legacy_scale=_aah_legacy_scale,
            rowsum=(params, kwargs) -> (p = _preset_params("aah", params, kwargs);
                                        2 * abs(p[:t]) + abs(p[:V]))),
    _preset("square_2d", :HUniform2Dsquare, 2, [:t],
            (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10);
            geometry=_square_geometry, legacy_scale=_t_scale(4.4),
            rowsum=_bonds("square_2d", 4)),
    _preset("hex_2d", :HUniform2Dhex, 2, [:t],
            (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10);
            geometry=_hex_geometry, legacy_scale=_t_scale(4.0),
            rowsum=_bonds("hex_2d", 3)),
    _preset("triangular_2d", :HUniform2Dtri, 2, [:t],
            (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10);
            geometry=_tri_geometry, legacy_scale=_t_scale(7.0),
            rowsum=_bonds("triangular_2d", 6)),
    _preset("triangular_bravais", :HUniform2Dtri_bravais, 2, [:t],
            (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10);
            geometry=_tri_bravais_geometry, legacy_scale=_t_scale(7.0),
            rowsum=_bonds("triangular_bravais", 6)),
    _preset("chern8", :HChern8, 2, [:V, :t],
            (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10);
            geometry=nothing, auto=:geometry,
            rowsum=(params, kwargs) -> _chern8_rowsum(_preset_params("chern8", params, kwargs))),
    _preset("chernhex", :H2DChernhex, 2, [:t, :t2, :ms],
            (; uniformhaldane=false, uniformsemenoff=false,
               tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10);
            geometry=nothing, legacy_scale=_chernhex_scale, default_rule=:formula,
            auto=:geometry,
            rowsum=(params, kwargs) -> _chernhex_rowsum(_preset_params("chernhex", params, kwargs))),
    _preset("qc2dsquare", :HQC2Dsquare, 2, [:t],
            (; tol_quantics=1e-9, maxbonddim_quantics=250, cutoff=1e-10);
            geometry=nothing, auto=:geometry,
            rowsum=(params, kwargs) -> _qc2dsquare_rowsum(_preset_params("qc2dsquare", params, kwargs))),
    # Multi-atom cells: 4 bonds per site at most (kagome, Lieb), 6 at the dice hub,
    # 3 NN (+ 6 NNN) on the honeycomb.
    _sublattice("kagome", :kagome_hamiltonian, :kagome_positions, 3, (; t=1.0),
                _sublattice_bonds(4)),
    _sublattice("lieb", :lieb_hamiltonian, :lieb_positions, 3, (; t=1.0),
                _sublattice_bonds(4)),
    _sublattice("honeycomb", :honeycomb_sublattice_hamiltonian,
                :honeycomb_sublattice_positions, 2, (; t=1.0), _sublattice_bonds(3)),
    _sublattice("honeycomb_nnn", :honeycomb_nnn_hamiltonian,
                :honeycomb_sublattice_positions, 2, (; t=1.0, t2=0.0),
                (params, kwargs) -> 3 * _abs_t(params) + 6 * abs(_param(params, :t2, 0.0))),
    _sublattice("dice", :dice_hamiltonian, :dice_positions, 3, (; t=1.0),
                _sublattice_bonds(6)),
]

"""
    MODELS :: Dict{String,ModelEntry}

The model registry: one [`ModelEntry`](@ref) per `get_Hamiltonian` geometry name.
"""
const MODELS = Dict{String,ModelEntry}(e.name => e for e in _MODEL_ENTRIES)

const _MODEL_NAMES = Tuple(e.name for e in _MODEL_ENTRIES)

"""
    MODEL_REGISTRY :: Dict{String, Tuple{Symbol,Int,Vector{Symbol},NamedTuple}}

Preset model name → `(builder, dim, required_params, kw_defaults)`, the view of the
preset entries of [`MODELS`](@ref) that [`build_hamiltonian`](@ref) reads.
"""
const MODEL_REGISTRY = Dict{String, Tuple{Symbol,Int,Vector{Symbol},NamedTuple}}(
    e.name => (e.builder, e.dim, e.params, e.defaults) for e in _MODEL_ENTRIES if e.kind === :preset)

function _model_entry(geometry)
    entry = get(MODELS, geometry, nothing)
    entry === nothing &&
        error("Unknown geometry \"$geometry\". Supported: $(join(_MODEL_NAMES, ", ")).")
    return entry
end


# ============================================================
# 6. build_hamiltonian dispatcher
# ============================================================

"""
    build_hamiltonian(model, L; mparams="", mparam_dict=Dict{Symbol,Any}()) -> MPO       (1D)
    build_hamiltonian(model, Lx, Ly; mparams="", mparam_dict=Dict{Symbol,Any}()) -> MPO  (2D)

Build a Hamiltonian MPO by model name using the MODEL_REGISTRY. `L`, `Lx` and
`Ly` are qubit counts (`2^L` sites, or a `2^Lx × 2^Ly` grid). The parameters are
the `mparams` string merged with `mparam_dict` (which wins); the required ones
are passed positionally to the builder, the rest as keywords on top of the
registry defaults.

```julia
H = build_hamiltonian("aah",  8; mparams="V=2.0, phi=0.0, t=1.0")
H = build_hamiltonian("square_2d", 4, 4; mparams="t=1.0")
H = build_hamiltonian("chernhex",  4, 4; mparam_dict=Dict(:t=>1.0, :t2=>0.3, :ms=>0.0))
```

Known models: $(join(sort(collect(keys(MODEL_REGISTRY))), ", "))
"""
function build_hamiltonian(model::AbstractString, L::Integer;
                           mparams::AbstractString = "",
                           mparam_dict             = Dict{Symbol,Any}())
    key = lowercase(model)
    haskey(MODEL_REGISTRY, key) || error("Unknown model '$model'. Known: $(sort(collect(keys(MODEL_REGISTRY))))")
    fn_sym, dim, required, kw_defaults = MODEL_REGISTRY[key]
    dim == 1 || error("Model '$model' is 2D; call build_hamiltonian(model, Lx, Ly; -")
    fn = getfield(@__MODULE__, fn_sym)
    p  = _parse_param_string(mparams)
    for (k,v) in mparam_dict; p[k] = v; end
    missing_p = [k for k in required if !haskey(p, k)]
    isempty(missing_p) || error("Missing required params for '$model': $missing_p")
    pos   = [p[k] for k in required]
    extra = Dict(k=>v for (k,v) in p if !(k in required))
    return fn(L, pos...; kw_defaults..., extra...)
end

function build_hamiltonian(model::AbstractString, Lx::Integer, Ly::Integer;
                           mparams::AbstractString = "",
                           mparam_dict             = Dict{Symbol,Any}())
    key = lowercase(model)
    haskey(MODEL_REGISTRY, key) || error("Unknown model '$model'. Known: $(sort(collect(keys(MODEL_REGISTRY))))")
    fn_sym, dim, required, kw_defaults = MODEL_REGISTRY[key]
    dim == 2 || error("Model '$model' is 1D; call build_hamiltonian(model, L; -")
    fn = getfield(@__MODULE__, fn_sym)
    p  = _parse_param_string(mparams)
    for (k,v) in mparam_dict; p[k] = v; end
    missing_p = [k for k in required if !haskey(p, k)]
    isempty(missing_p) || error("Missing required params for '$model': $missing_p")
    pos   = [p[k] for k in required]
    extra = Dict(k=>v for (k,v) in p if !(k in required))
    return fn(Lx, Ly, pos...; kw_defaults..., extra...)
end


# ============================================================
# 7. KPM scale maker: estimate_scale and the rules of get_Hamiltonian
# ============================================================

const _SCALE_PADDING    = 1.1       # every estimate is padded by 10%
const _SMALL_MAX_STATES = 1024      # largest dense problem of the :small method
const _SMALL_BUILD_SEED = 0x5ca1e   # RNG seed of the small builds (QTCI draws pivots)
const _SCALE_METHODS    = (:small, :geometry, :dmrg)

# :small estimates of the small builds, keyed by (geometry, params, sizes, tol, maxdim,
# other keywords); see _small_scale.
const _SMALL_SCALE_CACHE = Dict{Any,Float64}()
const _SMALL_SCALE_LOCK  = ReentrantLock()

# Position-qubit sizes of a model: (L,) in 1D, (Lx, Ly) in 2D, as its builder reads them.
function _model_sizes(entry::ModelEntry, L::Integer, kwargs)
    (entry.dim == 2 && entry.resizable) || return (Int(L),)
    Lx = get(kwargs, :Lx, L ÷ 2)
    return (Int(Lx), Int(get(kwargs, :Ly, L - Lx)))
end

_n_states(entry::ModelEntry, sizes) = entry.n_sub * 2^sum(sizes)

# Sizes of the small build: L ≤ 10 in 1D, Lx, Ly ≤ 5 in 2D, shrunk (the larger side
# first) until the build has at most _SMALL_MAX_STATES states with its sublattice.
function _small_sizes(entry::ModelEntry, sizes)
    s = collect(min.(sizes, length(sizes) == 1 ? 10 : 5))
    while _n_states(entry, s) > _SMALL_MAX_STATES && maximum(s) > 0
        s[argmax(s)] -= 1
    end
    return Tuple(s)
end

# Hashable, immutable stand-in for `params` or keyword arguments in a cache key.
_freeze(x::AbstractDict) = Tuple(sort!([Symbol(k) => _freeze(v) for (k, v) in pairs(x)];
                                       by = p -> string(first(p))))
_freeze(x::AbstractVector) = Tuple(_freeze.(x))
_freeze(x) = x
_freeze_kwargs(kwargs, drop) =
    Tuple(sort!([k => _freeze(v) for (k, v) in pairs(kwargs) if !(k in drop)]; by=first))

# Run `f()` without disturbing the caller's global and ITensors index-id RNGs; with a
# `seed`, `f` runs from that seed (so a QTCI build inside it is reproducible).
function _preserving_rngs(f, seed=nothing)
    rng, idrng = Random.default_rng(), ITensors.index_id_rng()
    saved, saved_id = copy(rng), copy(idrng)
    try
        if seed !== nothing
            Random.seed!(rng, seed)
            Random.seed!(idrng, seed)
        end
        return f()
    finally
        copy!(rng, saved)
        copy!(idrng, saved_id)
    end
end

# max(|E_min|, |E_max|) of the Hermitian part of a small MPO, from its dense spectrum.
function _dense_spectral_radius(mpo::MPO)
    M = _mpo_dense_matrix(mpo)
    return maximum(abs, eigvals(Hermitian((M + M') / 2)))
end

# Build `entry` at position-qubit `sizes` for a :small estimate (provisional scale 1.0,
# so the builder computes no default of its own).
function _build_at(entry::ModelEntry, params, sizes; tol, maxdim, kwargs...)
    Ls = sum(sizes)
    kw = length(sizes) == 2 ?
        (; (k => v for (k, v) in pairs(kwargs) if k ∉ (:Lx, :Ly))..., Lx=sizes[1], Ly=sizes[2]) :
        kwargs
    return entry.build(params, Ls, 2^Ls, siteinds("Qubit", Ls);
                       scale=1.0, tol, maxdim, ref_sites=nothing, kw...)
end

# The :small estimate: 1.1 × the dense spectral radius of the model at the sizes of
# _small_sizes. When the system is already that small the radius is that of `H` itself
# (not cached, since `H` came from the caller's RNG); otherwise the small build runs from
# a fixed seed and its estimate is cached. The caller's RNGs are left untouched.
function _small_scale(entry::ModelEntry, H, params, L; tol, maxdim, kwargs...)
    sizes = _model_sizes(entry, L, kwargs)
    small = entry.resizable ? _small_sizes(entry, sizes) : sizes
    if small == sizes && H !== nothing
        return _SCALE_PADDING * _preserving_rngs(() -> _dense_spectral_radius(H.mpo))
    end
    cacheable = entry.resizable
    key = (entry.name, _freeze(params), small, Float64(tol), maxdim,
           _freeze_kwargs(kwargs, (:Lx, :Ly)))
    if cacheable
        cached = lock(() -> get(_SMALL_SCALE_CACHE, key, nothing), _SMALL_SCALE_LOCK)
        cached === nothing || return cached
    end
    r = _preserving_rngs(_SMALL_BUILD_SEED) do
        _dense_spectral_radius(_build_at(entry, params, small; tol, maxdim, kwargs...).mpo)
    end
    value = _SCALE_PADDING * r
    cacheable && lock(() -> (_SMALL_SCALE_CACHE[key] = value), _SMALL_SCALE_LOCK)
    return value
end

# Validate an explicit method for `entry`; returns the method actually run.
function _check_scale_method(entry::ModelEntry, method::Symbol, L, kwargs)
    m = method === :auto ? entry.auto : method
    m in _SCALE_METHODS || throw(ArgumentError(
        "scale=:$method is not a scale method; use a number, nothing, :small, " *
        ":geometry or :dmrg."))
    if m === :geometry && entry.rowsum === nothing
        throw(ArgumentError("scale=:geometry: \"$(entry.name)\" has no row-sum rule " *
                            "(its terms are not known in advance); use :dmrg or a number."))
    elseif m === :small && entry.kind === :projected
        throw(ArgumentError("scale=:small is not available for \"$(entry.name)\": its " *
                            "default (scale, center) is analytic; use :dmrg or a number."))
    elseif m === :small && !entry.resizable &&
           _n_states(entry, _model_sizes(entry, L, kwargs)) > _SMALL_MAX_STATES
        throw(ArgumentError("scale=:small for \"$(entry.name)\" needs at most " *
                            "$_SMALL_MAX_STATES states (its builder cannot be run at a " *
                            "smaller size); use $(entry.rowsum === nothing ? "" : ":geometry, ")" *
                            ":dmrg or a number."))
    end
    return m
end

"""
    estimate_scale(geometry, params; L, method=:auto, tol=1e-8, maxdim=15, kwargs...) -> Float64

KPM half-bandwidth estimate (spectral centre 0) for
`get_Hamiltonian(geometry, params; L, tol, maxdim, kwargs...)`; `kwargs` are that call's
other keywords (`Lx`, `Ly`, `mparams`, `boundary`, `rs`, …).

Methods
- `:small`: exact dense spectrum of the same model with the same parameters and
  boundary, built at a small size: `L_s = min(L, 10)` in 1D, `Lx_s = min(Lx, 5)`,
  `Ly_s = min(Ly, 5)` in 2D, reduced (the larger side first) until the build has at most
  1024 states including its sublattice index. Returns `1.1 max(|E_min|, |E_max|)` of the
  Hermitian part. The small build runs from a fixed RNG seed (QTCI draws random pivots)
  and leaves the caller's RNGs untouched; its estimate is cached per (geometry, params,
  small sizes, `tol`, `maxdim`, other keywords). Within `get_Hamiltonian`, a system that
  is already that small is diagonalised directly. `"haldane"` and `"custom"`, whose
  builders cannot be run at another size, support it only up to 1024 states.
- `:geometry`: `1.1 ×` the largest absolute row sum (Gershgorin bound) of the builder's
  terms, with the sup of the amplitude functions over all arguments for the size-scaled
  `"chern8"` and `"qc2dsquare"`. Builds nothing. Not available for `"custom"` and the
  projected spaces.
- `:dmrg`: the DMRG estimate `_estimate_spectral_bounds` of the full Hamiltonian (the
  lazy `scale=0` path, run now); returns its scale and drops the centre it finds
  (`get_Hamiltonian(...; scale=:dmrg)` keeps both). Uses the global RNG.
- `:auto`: the model's own choice: `:geometry` for `"chern8"`, `"qc2dsquare"`,
  `"chernhex"` and `"haldane"`, `:dmrg` for `"custom"` and the projected spaces, `:small`
  otherwise.

When `scale` is not passed, `get_Hamiltonian` uses `max(today's formula,
estimate_scale(...; method=:auto))` for `"chain_1d"`, the preset models except
`"chernhex"` and the 2D multi-atom lattices, and the builder's own default for every other
geometry (see its docstring).

```julia
estimate_scale("aah", (V=0.5, phi=0.2, t=1.0); L=12)               # :small, built at L = 10
estimate_scale("qc2dsquare", 1.0; L=12, Lx=6)                        # :geometry, 10.56
estimate_scale("square_2d", 1.0; L=12, Lx=6, method=:geometry)       # 4.4
```
"""
function estimate_scale(geometry::AbstractString, params; L::Integer, method::Symbol=:auto,
                        tol=1e-8, maxdim=15, ref_sites=nothing, kwargs...)
    entry = _model_entry(geometry)
    m = _check_scale_method(entry, method, L, kwargs)
    m === :geometry && return _SCALE_PADDING * entry.rowsum(params, kwargs)
    m === :small    && return _small_scale(entry, nothing, params, L; tol, maxdim, kwargs...)
    sites = entry.kind === :projected ? nothing : siteinds("Qubit", L)
    H = entry.build(params, L, entry.kind === :projected ? nothing : 2^L, sites;
                    scale=1.0, tol, maxdim, ref_sites=nothing, kwargs...)
    return first(_estimate_spectral_bounds(H.mpo, H.sites))
end

# Default scale of a freshly built `H` (whose `scale` holds today's formula) when
# get_Hamiltonian got no `scale`: unchanged for `default_rule = :formula`, otherwise
# max(today's formula, estimate_scale(:auto)). The :small estimate is skipped when today's
# formula already reaches the padded row-sum bound, which 1.1 × the spectral radius
# cannot exceed.
function _default_scale(entry::ModelEntry, H::TBHamiltonian, params, L; tol, maxdim, kwargs...)
    legacy = H.scale
    entry.default_rule === :max || return legacy
    bound = _SCALE_PADDING * entry.rowsum(params, kwargs)
    entry.auto === :geometry && return max(legacy, bound)
    bound <= legacy && return legacy
    return max(legacy, _small_scale(entry, H, params, L; tol, maxdim, kwargs...))
end

# Explicit `scale=method` of get_Hamiltonian on a freshly built `H`.
function _apply_scale_method!(H::TBHamiltonian, entry::ModelEntry, method::Symbol, params, L;
                              tol, maxdim, kwargs...)
    if method === :dmrg
        H.scale, H.center = _estimate_spectral_bounds(H.mpo, H.sites)
    elseif method === :geometry
        H.scale = _SCALE_PADDING * entry.rowsum(params, kwargs)
    else
        H.scale = _small_scale(entry, H, params, L; tol, maxdim, kwargs...)
    end
    return H
end
