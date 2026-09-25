# Generator for test/data/nh_golden.jl, the pinned ("golden") outputs of the
# non-Hermitian toolkit (src/physics/NH_tk.jl), checked by test/golden_nh.jl.
#
# Reference: the golden data was first generated from commit 1a5548b ("Run the
# third-round audit regression tests from runtests.jl", branch Anouar), before
# the Tier 1 code reorganisation (docs/dev/REORGANISATION_TODO.md). It pins
# what the NH code did at that commit, remaining bugs included.
#
# The case list lives in test/golden_nh.jl (module NHGolden); this script runs
# every case once, in one process, and writes the records as Julia literals
# (`repr` round-trips Float64 and ComplexF64 exactly). Line 4 of the data file
# records the git tree hash of src/ at HEAD and whether the working-tree src/
# differed from it, so a rerun after test- or docs-only changes reproduces the
# file byte for byte.
#
# Regenerate ONLY when a behaviour change is intentional. Do it in the same
# commit as the change and review the diff of nh_golden.jl case by case. A
# refactor that is meant to leave outputs alone must pass test/golden_nh.jl
# against the existing data unchanged.
#
# How to run, from the repository root with the package environment:
#
#     JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. test/data/generate_nh_golden.jl
#
# The script reloads the file it wrote and checks that every record round-trips
# exactly (same types, `isequal` values), then replays the test against it.

using Test
using TensorBinding, ITensors, ITensorMPS

const NH_GOLDEN_GENERATOR = true
include(joinpath(@__DIR__, "..", "golden_nh.jl"))   # loads module NHGolden only

const OUTFILE = NHGolden.DATA_FILE

# ── Literal writer ────────────────────────────────────────────────────────────

lit(x::NamedTuple) =
    isempty(x) ? "NamedTuple()" :
    "(" * join(("$k = $(lit(v))" for (k, v) in pairs(x)), ", ") * ",)"

lit(x::Tuple) = isempty(x) ? "()" :
    "(" * join(map(lit, x), ", ") * (length(x) == 1 ? ",)" : ")")

function lit(x::AbstractArray)
    if eltype(x) <: Number
        v = repr(vec(collect(x)))
        return ndims(x) == 1 ? v : "reshape($v, $(size(x)))"
    end
    ndims(x) == 1 || error("lit: only vectors of non-numbers are supported, got $(typeof(x))")
    # `(T)[a, b]` is `getindex(T, a, b)`, a typed vector: without the type, a
    # Float64 and a ComplexF64 matrix would be promoted to one element type.
    return "($(repr(eltype(x))))[" * join(map(lit, x), ", ") * "]"
end

lit(x::Union{Number,AbstractString,Symbol,Nothing}) = repr(x)
lit(x) = error("lit: no literal form for $(typeof(x))")

# ── Provenance ────────────────────────────────────────────────────────────────

function git_line(args...)
    try
        return strip(read(`git -C $(pkgdir(TensorBinding)) $args`, String))
    catch
        return "unknown"
    end
end

function provenance()
    tree  = git_line("rev-parse", "HEAD:src")
    dirty = isempty(git_line("status", "--porcelain", "--", "src")) ? "clean" : "DIRTY"
    return "src tree $tree ($dirty working tree)"
end

# ── Run ───────────────────────────────────────────────────────────────────────

function generate()
    records = Pair{String,NamedTuple}[]
    t0 = time()
    for c in NHGolden.CASES
        t = @elapsed rec = NHGolden.run_case(c)
        rec isa NamedTuple || error("case $(c.name) returned $(typeof(rec)), expected a NamedTuple")
        push!(records, c.name => rec)
        println(rpad(c.name, 56), " ", round(t; digits = 2), " s")
    end
    println("all cases: ", round(time() - t0; digits = 1), " s")

    io = IOBuffer()
    println(io, "# Golden data for test/golden_nh.jl, written by test/data/generate_nh_golden.jl.")
    println(io, "# Do not edit by hand: regenerate (see the generator header).")
    println(io, "# Julia $(VERSION), ITensors $(pkgversion(ITensors)), ITensorMPS $(pkgversion(ITensorMPS)).")
    println(io, "# ", provenance())
    println(io, "# $(length(records)) cases.")
    println(io)
    println(io, "NH_GOLDEN = Pair{String,NamedTuple}[]")
    for (name, rec) in records
        println(io)
        println(io, "push!(NH_GOLDEN, ", repr(name), " => ", lit(rec), ")")
    end
    println(io)
    println(io, "NH_GOLDEN")
    write(OUTFILE, take!(io))
    println("wrote $OUTFILE ($(filesize(OUTFILE)) bytes)")

    reloaded = NHGolden.load_golden(OUTFILE)
    length(reloaded) == length(records) || error("round trip: case count changed")
    for ((n1, r1), (n2, r2)) in zip(records, reloaded)
        n1 == n2 || error("round trip: case $n1 came back as $n2")
        (typeof(r1) == typeof(r2) && isequal(r1, r2)) ||
            error("round trip: case $n1 does not reproduce exactly")
    end
    println("round trip: all $(length(records)) records reproduce exactly")
    return reloaded
end

NHGolden.run_tests(generate())
