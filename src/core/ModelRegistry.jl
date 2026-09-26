# ModelRegistry.jl — MODEL_REGISTRY and the build_hamiltonian dispatcher for
# the preset builders in lattice/presets.jl.
#
# Contents: _parse_param_string (the "key=value, …" parameter strings),
# MODEL_REGISTRY (model name → builder, dimension, required parameters, keyword
# defaults) and the 1D and 2D methods of build_hamiltonian.
#
# Main entry point: build_hamiltonian; get_Hamiltonian (core/TBSystem.jl) routes
# its preset geometries through it.
#
# Depends on: presets (the H* builders, looked up by Symbol at call time). The
# _geom_positions helpers that map a geometry name to its *_positions table live
# in lattice/geometry.jl. Split from the former lattice/2Dlattice_tk.jl (first
# as lattice/model_registry.jl).

# ============================================================
# 1. Model registry + build_hamiltonian dispatcher
# ============================================================

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


# Maps model name → (function, dim, required_params, kw_defaults)
const MODEL_REGISTRY = Dict{String, Tuple{Symbol,Int,Vector{Symbol},NamedTuple}}(
    "uniform"         => (:HUniform,         1, [:t],          (; v=1e-6, tol_quantics=1e-8, maxbonddim_quantics=10, nn=1)),
    "ssh"             => (:HSSH,             1, [:t, :d],      (; tol_quantics=1e-8, maxbonddim_quantics=10, nn=1)),
    "aah"             => (:HAAH,             1, [:V, :phi, :t],(; b=(1+sqrt(5))/2, tol_quantics=1e-8, maxbonddim_quantics=50)),
    "square_2d"       => (:HUniform2Dsquare, 2, [:t],          (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "hex_2d"          => (:HUniform2Dhex,    2, [:t],          (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "triangular_2d"        => (:HUniform2Dtri,         2, [:t], (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "triangular_bravais"   => (:HUniform2Dtri_bravais, 2, [:t], (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "chern8"          => (:HChern8,          2, [:V, :t],      (; t2=0.2, tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "chernhex"        => (:H2DChernhex,      2, [:t, :t2, :ms],(; uniformhaldane=false, uniformsemenoff=false,
                                                                   tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "qc2dsquare"      => (:HQC2Dsquare,      2, [:t],          (; tol_quantics=1e-9, maxbonddim_quantics=250, cutoff=1e-10)),
)


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
