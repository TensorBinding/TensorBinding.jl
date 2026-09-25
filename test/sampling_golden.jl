# Characterization ("golden") tests for the sampling planners.
#
# These tests pin what the planners return *today*, bugs included, so that a
# refactor cannot silently change which sites or momenta are sampled. The
# expected values live in `test/data/sampling_golden.jl`, which is written by
# `test/data/generate_sampling_golden.jl` (see its header for the commit it was
# generated from and how to rerun it).
#
# A failure here means a planner's output changed. If the change is a
# regression, fix the code. If it is intentional, regenerate the data file in
# the same commit as the behaviour change and record it in the changelog.
#
# Comparison rules:
#   * every field recorded in the golden data must still be returned, must be
#     `==` to the recorded value and must have the same leaf element type
#     (`Int` stays `Int`, `Bool` stays `Bool`; a range may become a `Vector`).
#     Fields that a planner adds later are allowed;
#   * a case recorded as throwing must still throw an exception of that type
#     and, when the exception carries a message, with the recorded
#     `message_prefix` (its first MESSAGE_PREFIX_CHARS characters), so that
#     reordering the validation checks is noticed. A reworded message fails
#     here too: regenerate and check that the data diff touches only
#     `message_prefix`;
#   * a field stored as a digest (`__digest__ = true`; only in cases with more
#     than 100_000 integers) is compared through the same digest of the current
#     output;
#   * the number of cases per planner must equal EXPECTED_CASE_COUNTS below, so
#     that cases dropped from the generator are noticed;
#   * the copy of get_ldos_spatial_mps_gpu's inline plan must still match the
#     source (see `inline_plan_block`).
#
# The runner below is shared with the generator, which includes this file with
# `SAMPLING_GOLDEN_GENERATOR` defined so that only the module is loaded.

using Test
using TensorBinding

module SamplingGoldenRunner

using Test
using TensorBinding
const TB = TensorBinding

# Number of cases per planner in test/data/sampling_golden.jl. Update it by
# hand, in the same commit, when cases are added or removed on purpose (the
# generator prints the new counts and warns when they differ from these).
const EXPECTED_CASE_COUNTS = Dict{Symbol,Int}(
    :fibonacci_ldos_sampling_plan => 181,
    :gpu_mps_auto                 => 278,
    :ilinspace                    => 18,
    :kpath_setup                  => 42,
    :kspace_sampling_plan         => 245,
    :spatial_sampling_plan        => 930,
    :tb_spatial_plan_gpu          => 14,
)

# A throwing case pins the first MESSAGE_PREFIX_CHARS characters of the first
# line of the exception's `msg` (ErrorException, ArgumentError, AssertionError).
const MESSAGE_PREFIX_CHARS = 60

# ── get_ldos_spatial_mps_gpu automatic plan ─────────────────────────────────────
# `get_ldos_spatial_mps_gpu` builds its groups inline, before any GPU work, so it
# cannot be reached on a CPU-only machine. The body below is a VERBATIM copy of
# the block of src/gpu/GPU_tk.jl that starts at `groups = if x_groups !== nothing`
# inside `get_ldos_spatial_mps_gpu` and ends with the validation loop
# `for group in groups ... end`, with the keyword defaults of that function
# (GPU_MPS_AUTO_DEFAULTS). `H = (; N)` stands in for the Hamiltonian so that the
# copied `H.N` is unchanged. `check_inline_plan_copy` compares the two, so an
# edit to either side fails the tests. Retire the copy (and the check) when the
# plan moves into core/Utils.jl and `run_case` can call the real function.
function gpu_mps_auto_groups(N::Int;
                             x_groups     = nothing,
                             num_x::Int   = min(N, 100),
                             num_avg::Int = 1,
                             x_start::Int = 1,
                             x_end::Int   = N)
    H = (; N)
    groups = if x_groups !== nothing
        x_groups isa AbstractVector{<:AbstractVector} ?
            [collect(Int, group) for group in x_groups] :
            [[Int(x)] for x in x_groups]
    else
        num_x > 0 || throw(ArgumentError(
            "get_ldos_spatial_mps_gpu: num_x must be positive."
        ))
        num_avg > 0 || throw(ArgumentError(
            "get_ldos_spatial_mps_gpu: num_avg must be positive."
        ))
        1 <= x_start <= x_end <= H.N || throw(ArgumentError(
            "get_ldos_spatial_mps_gpu: expected 1 <= x_start <= x_end <= H.N."
        ))
        window = x_end - x_start + 1
        num_x <= window || throw(ArgumentError(
            "get_ldos_spatial_mps_gpu: num_x=$num_x exceeds the sampling " *
            "window length $window."
        ))
        [let
             lo = x_start + fld((i - 1) * window, num_x)
             hi = x_start + fld(i * window, num_x) - 1
             nsample = min(num_avg, hi - lo + 1)
             nsample == 1 ? Int[lo] :
                 unique(round.(Int, range(lo, hi; length=nsample)))
         end for i in 1:num_x]
    end

    isempty(groups) && throw(ArgumentError(
        "get_ldos_spatial_mps_gpu: no spatial groups were selected."
    ))
    for group in groups
        isempty(group) && throw(ArgumentError(
            "get_ldos_spatial_mps_gpu: spatial groups must not be empty."
        ))
        all(x -> 1 <= x <= H.N, group) || throw(ArgumentError(
            "get_ldos_spatial_mps_gpu: every position must lie in 1:H.N."
        ))
    end
    return groups
end

# ── Drift guard for the copy above ─────────────────────────────────────────────
const RUNNER_FILE = @__FILE__
const GPU_TK_FILE = joinpath(pkgdir(TensorBinding), "src", "gpu", "GPU_tk.jl")

# Keyword defaults of get_ldos_spatial_mps_gpu that the copy reproduces, with
# all whitespace removed (the copy writes `min(N, 100)` and `N` for `H.N`).
const GPU_MPS_AUTO_DEFAULTS = ("x_groups=nothing,", "num_x::Int=min(H.N,100),",
                               "num_avg::Int=1,", "x_start::Int=1,", "x_end::Int=H.N,")

_indent(line) = length(line) - length(lstrip(line))

"""
    inline_plan_block(path, function_line) -> Vector{String}

The whitespace-stripped, non-empty lines of the automatic-plan block in `path`:
from the first line after `function_line` that starts with
`groups = if x_groups !== nothing`, through the `end` that closes the next
`for group in groups` loop. Errors if an anchor is missing.
"""
function inline_plan_block(path::AbstractString, function_line::AbstractString)
    lines = readlines(path)
    i_fun = findfirst(l -> startswith(strip(l), function_line), lines)
    i_fun === nothing && error("inline_plan_block: `$function_line` not found in $path")
    i0 = findnext(l -> startswith(strip(l), "groups = if x_groups !== nothing"), lines, i_fun)
    i0 === nothing && error("inline_plan_block: `groups = if x_groups !== nothing` not found in $path")
    i_for = findnext(l -> strip(l) == "for group in groups", lines, i0)
    i_for === nothing && error("inline_plan_block: `for group in groups` not found in $path")
    i_end = findnext(l -> strip(l) == "end" && _indent(l) == _indent(lines[i_for]), lines, i_for + 1)
    i_end === nothing && error("inline_plan_block: the end of `for group in groups` not found in $path")
    return [String(strip(l)) for l in lines[i0:i_end] if !isempty(strip(l))]
end

"""
    check_inline_plan_copy() -> Bool

`true` when `gpu_mps_auto_groups` above still reproduces the automatic plan of
`get_ldos_spatial_mps_gpu` in src/gpu/GPU_tk.jl: the same block, whitespace
aside, and the same keyword defaults. Logs the first differing line otherwise.
"""
function check_inline_plan_copy()
    src  = inline_plan_block(GPU_TK_FILE, "function get_ldos_spatial_mps_gpu(")
    copy = inline_plan_block(RUNNER_FILE, "function gpu_mps_auto_groups(")
    ok = true
    if src != copy
        i = findfirst(k -> k > length(src) || k > length(copy) || src[k] != copy[k],
                      1:max(length(src), length(copy)))
        @error "get_ldos_spatial_mps_gpu's inline plan and its copy in test/sampling_golden.jl differ" line = i src = get(src, i, "(missing)") copy = get(copy, i, "(missing)")
        ok = false
    end
    lines = readlines(GPU_TK_FILE)
    i_fun = findfirst(l -> startswith(strip(l), "function get_ldos_spatial_mps_gpu("), lines)
    i_body = findnext(l -> occursin("Ncheb >= 2", l), lines, i_fun)
    signature = replace(join(lines[i_fun:i_body]), r"\s" => "")
    for default in GPU_MPS_AUTO_DEFAULTS
        occursin(default, signature) && continue
        @error "get_ldos_spatial_mps_gpu keyword default changed; update gpu_mps_auto_groups" expected = default
        ok = false
    end
    return ok
end

"""
    run_case(planner, args, kwargs) -> NamedTuple

Call one planner the way the golden data describes it. Planners that return a
non-NamedTuple value are wrapped so that every result can be compared field by
field.
"""
function run_case(planner::Symbol, @nospecialize(args::Tuple), @nospecialize(kwargs::NamedTuple))
    # @nospecialize: compiled once rather than once per keyword-set type, which
    # keeps the whole testset to a few seconds.
    if planner === :spatial_sampling_plan
        return TB.spatial_sampling_plan(args...; kwargs...)
    elseif planner === :fibonacci_ldos_sampling_plan
        return TB.fibonacci_ldos_sampling_plan(args...; kwargs...)
    elseif planner === :kspace_sampling_plan
        return TB.kspace_sampling_plan(args...; kwargs...)
    elseif planner === :ilinspace
        return (; value = TB.ilinspace(args...))
    elseif planner === :kpath_setup
        k_groups, ticks, labels = TB.kpath_setup(args...; kwargs...)
        return (; k_groups, ticks, labels)
    elseif planner === :tb_spatial_plan_gpu
        # args = (number of position qubits,); the wrapper only reads the sites' count/dims
        return TB._tb_spatial_plan_gpu(TB.siteinds("Qubit", args[1]); kwargs...)
    elseif planner === :gpu_mps_auto
        return (; groups = gpu_mps_auto_groups(args...; kwargs...))
    else
        error("SamplingGoldenRunner: unknown planner :$planner")
    end
end

# ── Digests for very large fields ──────────────────────────────────────────────
const DIGEST_EDGE = 200

_flat(v::AbstractVector) =
    eltype(v) <: AbstractVector || any(x -> x isa AbstractVector, v) ?
        Int[x for g in v for x in g] : collect(v)

_weighted(flat) = sum((Int128(i) * Int128(x) for (i, x) in enumerate(flat)); init=Int128(0))

"""
    digest(v::AbstractVector) -> NamedTuple

Compact fingerprint of a long vector (or vector of groups): its length, the
first and last $DIGEST_EDGE entries, and the sum and position-weighted sum
`sum(i * x)` of the flattened integers. For a vector of groups the flattened
length and the weighted sum of the group lengths are included too, so that
moving an entry between groups changes the digest.
"""
function digest(v::AbstractVector)
    n    = length(v)
    head = collect(v[1:min(n, DIGEST_EDGE)])
    tail = collect(v[max(1, n - DIGEST_EDGE + 1):n])
    flat = _flat(v)
    if eltype(v) <: AbstractVector || any(x -> x isa AbstractVector, v)
        lens = length.(v)
        return (; __digest__ = true, length = n, flat_length = length(flat),
                sum = sum(Int128, flat; init=Int128(0)), weighted_sum = _weighted(flat),
                lengths_weighted_sum = _weighted(lens), head, tail)
    end
    return (; __digest__ = true, length = n,
            sum = sum(Int128, flat; init=Int128(0)), weighted_sum = _weighted(flat),
            head, tail)
end

is_digest(x) = x isa NamedTuple && haskey(x, :__digest__)

# Element type at the bottom of any nesting of arrays: Int for 1:3, [1, 2] and
# [[1], [2, 3]]; the type itself for anything else. Container kinds are not
# compared, so a range that becomes a Vector with the same entries still passes.
leaftype(x) = _leaftype(typeof(x))
_leaftype(::Type{T}) where {T<:AbstractArray} = _leaftype(eltype(T))
_leaftype(::Type{T}) where {T} = T

"""
    message_prefix(err) -> Union{String,Nothing}

The first MESSAGE_PREFIX_CHARS characters of the first line of `err.msg`, or
`nothing` for an exception without a string message (e.g. DivideError).
"""
function message_prefix(err)
    hasfield(typeof(err), :msg) || return nothing
    msg = getfield(err, :msg)
    msg isa AbstractString || return nothing
    line = first(split(msg, '\n'))
    return String(first(line, MESSAGE_PREFIX_CHARS))
end

# ── Field comparison with a failure message naming the case and field ──────────
function _mismatch_detail(actual, expected)
    if actual isa AbstractVector && expected isa AbstractVector
        length(actual) == length(expected) ||
            return "length $(length(actual)) != expected $(length(expected))"
        i = findfirst(k -> actual[k] != expected[k], eachindex(expected))
        i === nothing && return "same entries, different container"
        return "first difference at index $i: got $(repr(actual[i])), expected $(repr(expected[i]))"
    end
    if actual isa NamedTuple && expected isa NamedTuple   # a digest or a nested record
        differing = [k for k in keys(expected) if !haskey(actual, k) || actual[k] != expected[k]]
        return "differing entries: $(join(differing, ", "))"
    end
    return "got $(repr(actual)), expected $(repr(expected))"
end

"""
    field_matches(case_name, field, actual, expected) -> Bool

`true` when `actual` has `field` and its value equals the golden one (through
the digest when the golden value is a digest) with the same leaf element type.
Logs an error naming the case and field otherwise.
"""
function field_matches(case_name::AbstractString, field::Symbol,
                       @nospecialize(actual), @nospecialize(expected))
    if !(actual isa NamedTuple) || !haskey(actual, field)
        @error "Sampling plan changed: field missing" case = case_name field = field
        return false
    end
    value = getfield(actual, field)
    if is_digest(expected)
        got = value isa AbstractVector ? digest(value) : value
        got == expected && return true
        @error "Sampling plan changed" case = case_name field = field detail = _mismatch_detail(got, expected)
        return false
    end
    if value != expected
        @error "Sampling plan changed" case = case_name field = field detail = _mismatch_detail(value, expected)
        return false
    end
    leaftype(value) == leaftype(expected) && return true
    @error "Sampling plan changed: element type" case = case_name field = field got = leaftype(value) expected = leaftype(expected)
    return false
end

"""
    throws_as(case) -> Bool

`true` when the case still throws an exception of the pinned type, with the
pinned message prefix when one is recorded (`case.expected.message_prefix`).
Logs an error naming the case otherwise.
"""
function throws_as(@nospecialize(case))
    try
        run_case(case.planner, case.args, case.kwargs)
    catch err
        if !(err isa case.throws)
            @error "Sampling plan changed: different exception" case = case.name expected = case.throws got = typeof(err)
            return false
        end
        expected = case.expected === nothing ? nothing : case.expected.message_prefix
        got = message_prefix(err)
        got == expected && return true
        @error "Sampling plan changed: different error message" case = case.name expected = expected got = got
        return false
    end
    @error "Sampling plan changed: case no longer throws" case = case.name expected = case.throws
    return false
end

function check_case(@nospecialize(case))
    if case.throws !== nothing
        @test throws_as(case)
        return
    end
    actual, err = try
        (run_case(case.planner, case.args, case.kwargs), nothing)
    catch e
        (nothing, e)
    end
    if err !== nothing
        @error "Sampling plan changed: case now throws" case = case.name exception = err
        @test err === nothing
        return
    end
    for field in keys(case.expected)
        @test field_matches(case.name, field, actual, case.expected[field])
    end
end

# One testset per planner rather than per case: a failure then lists a handful
# of rows instead of ~1700, and the logged error names the case and field.
case_counts(cases) = Dict{Symbol,Int}(p => count(c -> c.planner === p, cases)
                                      for p in unique(c.planner for c in cases))

function run_tests(cases)
    @testset "Sampling plans are pinned" begin
        @test allunique(case.name for case in cases)
        counts = case_counts(cases)
        counts == EXPECTED_CASE_COUNTS ||
            @error "Golden case counts differ from EXPECTED_CASE_COUNTS" got = counts expected = EXPECTED_CASE_COUNTS
        @test counts == EXPECTED_CASE_COUNTS
        @test check_inline_plan_copy()
        for planner in unique(case.planner for case in cases)
            @testset "$planner" begin
                for case in cases
                    case.planner === planner && check_case(case)
                end
            end
        end
    end
end

end # module SamplingGoldenRunner

if !isdefined(@__MODULE__, :SAMPLING_GOLDEN_GENERATOR)
    SamplingGoldenRunner.run_tests(include(joinpath(@__DIR__, "data", "sampling_golden.jl")))
end
