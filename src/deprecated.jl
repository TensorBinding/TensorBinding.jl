# deprecated.jl — the 0.1 names kept through 0.2.x as deprecated aliases. Each warns once
# per session (`_depwarn_once`, a `@warn`: `Base.depwarn` is silent unless Julia runs with
# --depwarn=yes) and calls its new name with the same arguments; the aliases go in 0.3.
# The renamed keywords (Nchebychev → Ncheb, Λ → Lambda) and the value method=:KPM (→ :kpm)
# are resolved by the new functions themselves (`_renamed_kw`). The aliases are not
# exported (the old names never were) and have no docstrings.
#
# Depends on: core/Utils.jl (_depwarn_once), physics/Topology.jl, gpu/topology.jl.

function get_C(args...; kwargs...)
    _depwarn_once("get_C is deprecated, use chern_marker", :get_C)
    return chern_marker(args...; kwargs...)
end

function get_W(args...; kwargs...)
    _depwarn_once("get_W is deprecated, use winding_marker", :get_W)
    return winding_marker(args...; kwargs...)
end

function get_valley_C(args...; kwargs...)
    _depwarn_once("get_valley_C is deprecated, use valley_chern_marker", :get_valley_C)
    return valley_chern_marker(args...; kwargs...)
end

function get_C_gpu(args...; kwargs...)
    _depwarn_once("get_C_gpu is deprecated, use chern_marker_gpu", :get_C_gpu)
    return chern_marker_gpu(args...; kwargs...)
end
