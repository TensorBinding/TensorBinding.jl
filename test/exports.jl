# The export list (src/TensorBinding.jl): every exported name is defined; none clashes
# with a name exported by the dependencies, Base, Core or the standard libraries users
# load next to the package; a module that loads ITensors, ITensorMPS and TensorBinding
# resolves each one unambiguously; only functions are exported; and the "Public API"
# section of docs/src/index.md lists exactly the exported names.
#
# Wrapped in a module so that `using TensorBinding` here does not change what Main sees.
module ExportsTests

using TensorBinding, ITensors, ITensorMPS, Test

const TB = TensorBinding

# Packages whose exports must not collide with TensorBinding's: the package's
# dependencies, then the standard libraries that its users and its example scripts load.
const CHECKED_PACKAGES = [
    ("ITensors", "9136182c-28ba-11e9-034c-db9fb085ebd5"),
    ("ITensorMPS", "0d1a4710-d33b-49a5-8f18-73bdf49b47e2"),
    ("NDTensors", "23ae76d9-e61a-49c4-8f12-3f1a16adf9cf"),
    ("Quantics", "87f76fb3-a40a-40c9-a63c-29fcfe7b7547"),
    ("QuanticsGrids", "634c7f73-3e90-4749-a1bd-001b8efc642d"),
    ("QuanticsTCI", "b11687fd-3a1c-4c41-97d0-998ab401d50e"),
    ("TensorCrossInterpolation", "b261b2ec-6378-4871-b32e-9173bb050604"),
    ("FFTW", "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"),
    ("LinearAlgebra", "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"),
    ("Random", "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"),
    ("SparseArrays", "2f01184e-e22b-5df5-ae63-d93ebab69eaf"),
    ("Statistics", "10745b16-79ce-11e8-11f9-7d13ad32a3b2"),
    ("Printf", "de0858da-6303-5e67-8744-51eddeeeb8d7"),
    ("Test", "8dfed614-e22c-5e08-85e1-65c5234f0b40"),
    ("Dates", "ade2ca70-3891-5945-98fb-dc099432e06a"),
    ("DelimitedFiles", "8bb1440f-4735-579b-a4ab-409b98df4dab"),
    ("Logging", "56ddb016-857b-54e1-b83d-db4d58db5568"),
]

# What a module exports, as name => object.
exports_of(m::Module) =
    Dict{Symbol,Any}(n => getglobal(m, n) for n in names(m)
                     if Base.isexported(m, n) && isdefined(m, n))

# The `export` statements in a package's source, as name => nothing (object unknown).
function exports_in_source(src::AbstractString)
    found = Dict{Symbol,Any}()
    function collect!(ex)
        ex isa Expr || return
        if ex.head === :export
            for a in ex.args
                a isa Symbol && (found[a] = nothing)
            end
        else
            foreach(collect!, ex.args)
        end
    end
    for (root, _, files) in walkdir(src), f in files
        endswith(f, ".jl") && collect!(Meta.parseall(read(joinpath(root, f), String)))
    end
    return found
end

# The exports of a checked package: from the loaded module when the active environment
# can load it; otherwise, for a standard library outside the environment (Statistics and
# DelimitedFiles in the `Pkg.test` sandbox), from its source, since loading it by id
# there would also try to load its package extensions without their dependencies.
function package_exports(name, uuid)
    id = Base.PkgId(Base.UUID(uuid), name)
    if haskey(Base.loaded_modules, id) || Base.identify_package(name) == id
        return exports_of(Base.require(id))
    end
    src = joinpath(Sys.STDLIB, name, "src")
    return isdir(src) ? exports_in_source(src) : nothing
end

# TensorBinding's exported names, without the module's own name.
exported_names() = filter(n -> n !== :TensorBinding && Base.isexported(TB, n), names(TB))

# The exported names that are ITensors' own objects, re-exported (MPO, inner, …).
is_reexported(n) = isdefined(ITensors, n) && getglobal(ITensors, n) === getglobal(TB, n)

# The names listed in the "Public API" section of docs/src/index.md: every `name` in
# backticks in the section's bullet list, from its first "- " line to the first blank
# line after it (the paragraphs above and below the list are not read).
function documented_names(path)
    lines = rstrip.(readlines(path))
    start = findfirst(==("## Public API"), lines)
    start === nothing && return nothing
    stop = findnext(l -> startswith(l, "## "), lines, start + 1)
    section = lines[(start + 1):(stop === nothing ? end : stop - 1)]
    first_item = findfirst(l -> startswith(l, "- "), section)
    first_item === nothing && return Set{Symbol}()
    last_item = something(findnext(isempty, section, first_item), length(section) + 1) - 1
    list = join(section[first_item:last_item], "\n")
    return Set(Symbol(m.captures[1]) for m in eachmatch(r"`([^`]+)`", list))
end

@testset "Export list" begin
    exported = exported_names()

    @testset "every exported name is defined" begin
        @test !isempty(exported)
        @test isempty([n for n in exported if !isdefined(TB, n)])
    end

    @testset "no clash with the dependencies and standard libraries" begin
        checked = Pair{String,Any}["Base" => exports_of(Base), "Core" => exports_of(Core)]
        append!(checked, [name => package_exports(name, uuid)
                          for (name, uuid) in CHECKED_PACKAGES])
        @test isempty([name for (name, ex) in checked if ex === nothing])   # all read
        # A clash is a package that exports the same name for a different object; the
        # re-exported ITensors names are the same objects, so they do not clash.
        clashes = [(n, name) for (name, ex) in checked if ex !== nothing for n in exported
                   if haskey(ex, n) && ex[n] !== getglobal(TB, n)]
        @test isempty(clashes)
    end

    @testset "a user module resolves each name unambiguously" begin
        probe = Module(:ExportsProbe)
        Core.eval(probe, :(using ITensors, ITensorMPS, TensorBinding))
        unresolved = [n for n in exported
                      if !(isdefined(probe, n) && getglobal(probe, n) === getglobal(TB, n))]
        @test isempty(unresolved)
    end

    @testset "only functions are exported" begin
        # A type visible in Main prints without its `TensorBinding.` prefix, which would
        # change the error messages the golden tests pin; constants are not exported either.
        # The re-exported ITensors names (MPO, MPS, OpSum are types) are ITensors' own.
        own = filter(!is_reexported, exported)
        @test issubset(filter(is_reexported, exported), names(ITensors))
        @test isempty([n for n in own if !(getglobal(TB, n) isa Function)])
    end

    @testset "docs/src/index.md lists the exported names" begin
        index = joinpath(@__DIR__, "..", "docs", "src", "index.md")
        if isfile(index)
            documented = documented_names(index)
            @test documented !== nothing
            if documented !== nothing
                @test isempty(setdiff(Set(exported), documented))   # exported, not listed
                @test isempty(setdiff(documented, Set(exported)))   # listed, not exported
            end
        end
    end
end

end # module ExportsTests
