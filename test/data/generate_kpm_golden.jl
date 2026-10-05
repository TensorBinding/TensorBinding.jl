# Generator for test/data/kpm_golden.jl, the pinned ("golden") outputs of the
# KPM solver (src/solvers/kpm/, formerly src/solvers/KPM_tk.jl), checked by
# test/golden_kpm.jl.
#
# Reference: the golden data was generated from commit 1a5548b ("Run the
# third-round audit regression tests from runtests.jl"), before the Tier 1
# split of the former KPM_tk.jl (docs/dev/REORGANISATION_TODO.md). It pins what the KPM
# functions do today, suspected bugs included. Line 4 of the data file records
# the git tree hash of the working-tree src/, which does not move when only
# tests or docs change. The five density/* cases were regenerated in the merge
# of release-0.1.1 into Anouar, which made get_density_from_Tn the
# occupied-state projector; every other entry kept its values until the 2026-09 bug
# pass, which regenerated the unnormalised-psi0, Fibonacci LDOS and empty-group cases
# (87f6562) and the exc3 exciton cases (a64b949).
#
# The cases themselves (models, inputs, and which outputs are recorded) live in
# KPMGoldenRunner in test/golden_kpm.jl, so that the test replays exactly what
# was generated. This script runs every case once, in one Julia process, and
# writes the results as plain Julia literals (`repr` round-trips Float64 and
# ComplexF64 exactly).
#
# Regenerate ONLY when a behaviour change is intentional. Do it in the same
# commit as the change and review the diff of kpm_golden.jl case by case. A
# refactor that is meant to leave outputs alone must pass test/golden_kpm.jl
# against the existing data unchanged.
#
# How to run, from the repository root with the package environment:
#
#     JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. test/data/generate_kpm_golden.jl
#
# An optional first argument writes the data to that path instead, e.g. to
# compare a regeneration with the committed file without overwriting it.
#
# Cases whose `requires` returns false (their function has been deleted) are
# not run and are left out of the data; test/golden_kpm.jl skips them whether or
# not the data still holds them.
#
# The output is deterministic: no timestamps, cases in a fixed order, the
# global RNG reseeded before every case. The script reloads the file it wrote
# and checks that every entry round-trips exactly, element types included.

using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random

const KPM_GOLDEN_GENERATOR = true
include(joinpath(@__DIR__, "..", "golden_kpm.jl"))   # loads KPMGoldenRunner only
const R = KPMGoldenRunner

const OUTFILE = isempty(ARGS) ? joinpath(@__DIR__, "kpm_golden.jl") : abspath(ARGS[1])

# An exception of one of these types means the case itself is malformed (a
# misspelt keyword, a wrong argument type), not a behaviour worth pinning.
const GENERATOR_BUGS = (MethodError, UndefVarError, UndefKeywordError, TypeError,
                        BoundsError, DimensionMismatch, KeyError)

# Cases that do throw one of GENERATOR_BUGS today, on purpose: the behaviour is
# pinned as it is (see the case in test/golden_kpm.jl for the details).
# None today ("exciton/spatial/empty_group" was one, a BoundsError, until
# spatial_sampling_plan rejected empty groups itself).
const PINNED_BUGLIKE_THROWS = Set{String}()

function evaluate(cases)
    entries = NamedTuple[]
    notes = String[]
    skipped = String[]
    for case in cases
        if !case.requires()   # its function has been deleted: nothing left to pin
            push!(skipped, case.name)
            continue
        end
        t0 = time()
        got = R.run_case(case)
        dt = time() - t0
        if got.err === nothing
            push!(entries, (; name = case.name, throws = nothing, expected = got.value))
        else
            err = got.err
            any(T -> err isa T, GENERATOR_BUGS) && !(case.name in PINNED_BUGLIKE_THROWS) &&
                error("case $(case.name) throws $(typeof(err)); fix the case:\n" *
                      sprint(showerror, err))
            push!(notes, "throws $(typeof(err)): $(case.name) -- " *
                         first(replace(sprint(showerror, err), '\n' => ' '), 120))
            push!(entries, (; name = case.name, throws = typeof(err),
                              expected = (; message_prefix = R.message_prefix(err))))
        end
        dt > 2.0 && push!(notes, "slow ($(round(dt; digits=1)) s): $(case.name)")
    end
    return entries, notes, skipped
end

# The git tree hash of the working-tree src/, computed through a throwaway index.
function src_info()
    index = tempname()
    try
        repo = normpath(joinpath(@__DIR__, "..", ".."))
        tree = withenv("GIT_INDEX_FILE" => index) do
            run(`git -C $repo read-tree HEAD`)
            run(`git -C $repo add -A -- src`)
            readchomp(`git -C $repo write-tree --prefix=src/`)
        end
        return "src/ tree $(first(tree, 12))"
    catch
        return "an unknown src/ tree (not a git checkout)"
    finally
        rm(index; force=true)
    end
end

function write_golden(path, entries)
    source = src_info()
    open(path, "w") do io
        println(io, "# AUTO-GENERATED by test/data/generate_kpm_golden.jl -- do not edit by hand.")
        println(io, "#")
        println(io, "# Pinned outputs of the KPM solver, checked by test/golden_kpm.jl.")
        println(io, "# Generated from $source")
        println(io, "# with Julia $VERSION. Regenerate only for an intentional behaviour change;")
        println(io, "# see the generator's header.")
        println(io, "#")
        println(io, "# Each entry is (name, throws, expected): `name` selects a case in")
        println(io, "# KPMGoldenRunner.CASES, `throws` is the exception type a failing case must")
        println(io, "# still raise (otherwise `nothing`), and `expected` holds the case's outputs")
        println(io, "# or, for a throwing case, the first $(R.MESSAGE_PREFIX_CHARS) characters of its message.")
        println(io, "#")
        println(io, "# $(length(entries)) cases, $(count(e -> e.throws !== nothing, entries)) of them pinned as throwing.")
        println(io)
        println(io, "KPM_GOLDEN_CASES = Any[]")
        for e in entries
            println(io)
            println(io, "push!(KPM_GOLDEN_CASES, (name = ", repr(e.name), ", throws = ", repr(e.throws), ",")
            println(io, "    expected = ", repr(e.expected), "))")
        end
        println(io)
        println(io, "KPM_GOLDEN_CASES")
    end
end

t_start = time()
entries, notes, skipped = evaluate(R.CASES)
foreach(println, notes)
foreach(name -> println("skipped, function deleted: ", name), skipped)
write_golden(OUTFILE, entries)

# Round trip: every entry must read back exactly as it was computed, with the
# same types and element types that test/golden_kpm.jl compares.
loaded = include(OUTFILE)
length(loaded) == length(entries) ||
    error("round trip: $(length(loaded)) entries read back, $(length(entries)) written")
for (a, b) in zip(loaded, entries)
    a.name == b.name && a.throws === b.throws ||
        error("round trip changed the name or exception type of case $(b.name)")
    m = R.mismatch(a.expected, b.expected; exact=true)
    m === nothing || error("round trip failed for case $(b.name): $m")
end

println("Wrote $(length(entries)) cases to $OUTFILE ($(filesize(OUTFILE)) bytes) in ",
        round(time() - t_start; digits=1), " s", isempty(skipped) ? "" : "; skipped $(length(skipped))")
length(entries) + length(skipped) == R.EXPECTED_CASE_COUNT ||
    @warn "The case count differs from EXPECTED_CASE_COUNT in test/golden_kpm.jl. " *
          "If the change is intended, update it in the same commit." got = length(entries) + length(skipped) expected = R.EXPECTED_CASE_COUNT
