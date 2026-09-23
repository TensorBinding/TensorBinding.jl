# FibonacciIO.jl -- HDF5 persistence for projected Fibonacci Hamiltonians
#
# This file is included after Fibonacci.jl. Loading HDF5 here also activates
# ITensors' HDF5 extension, which supplies serialization for IndexSet and MPO.

import HDF5

const _FIBONACCI_HAMILTONIAN_FORMAT = "TensorBinding.FibonacciHamiltonian"
const _FIBONACCI_HAMILTONIAN_FORMAT_VERSION = 1
const _FIBONACCI_HAMILTONIAN_DATASETS =
    ("mpo", "sites", "physical_projector", "model_metadata", "model_metadata_types")
const _FIBONACCI_HAMILTONIAN_HEADER_FIELDS = ("L", "N", "scale", "center")

const _FIBONACCI_METADATA_SCALAR = Union{
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    ComplexF32, ComplexF64,
    AbstractString, Symbol,
}

function _fibonacci_metadata_value(key::AbstractString, value)
    value isa _FIBONACCI_METADATA_SCALAR || throw(ArgumentError(
        "Fibonacci Hamiltonian metadata '$key' has unsupported type " *
        "$(typeof(value)); use a scalar Bool, fixed-width integer, Float32/64, " *
        "ComplexF32/64, String, or Symbol",
    ))
    if value isa Symbol
        return String(value), "Symbol"
    elseif value isa AbstractString
        return String(value), "String"
    end
    return value, string(typeof(value))
end

function _fibonacci_canonical_metadata(metadata)
    metadata isa Union{AbstractDict, NamedTuple} || throw(ArgumentError(
        "metadata must be a dictionary or NamedTuple with String or Symbol keys",
    ))
    canonical = Dict{String,Tuple{Any,String}}()
    for (raw_key, raw_value) in pairs(metadata)
        raw_key isa Union{AbstractString,Symbol} || throw(ArgumentError(
            "Fibonacci Hamiltonian metadata keys must be String or Symbol, " *
            "got $(typeof(raw_key))",
        ))
        key = String(raw_key)
        isempty(key) && throw(ArgumentError("Fibonacci Hamiltonian metadata keys cannot be empty"))
        occursin('\0', key) && throw(ArgumentError(
            "Fibonacci Hamiltonian metadata keys cannot contain a NUL character",
        ))
        haskey(canonical, key) && throw(ArgumentError(
            "duplicate Fibonacci Hamiltonian metadata key '$key' after key normalization",
        ))
        canonical[key] = _fibonacci_metadata_value(key, raw_value)
    end
    return canonical
end

function _fibonacci_canonical_expected_header(expected_header)
    expected_header isa Union{AbstractDict,NamedTuple} || throw(ArgumentError(
        "expected_header must be a dictionary or NamedTuple with String or Symbol keys",
    ))
    canonical = Dict{String,Any}()
    for (raw_key, value) in pairs(expected_header)
        raw_key isa Union{AbstractString,Symbol} || throw(ArgumentError(
            "expected_header keys must be String or Symbol, got $(typeof(raw_key))",
        ))
        key = String(raw_key)
        key in _FIBONACCI_HAMILTONIAN_HEADER_FIELDS || throw(ArgumentError(
            "unsupported expected_header key '$key'; use a subset of " *
            "$(join(_FIBONACCI_HAMILTONIAN_HEADER_FIELDS, ", "))",
        ))
        haskey(canonical, key) && throw(ArgumentError(
            "duplicate expected_header key '$key' after key normalization",
        ))
        if key in ("L", "N")
            value isa Integer && !(value isa Bool) || throw(ArgumentError(
                "expected_header '$key' must be an integer, got $(typeof(value))",
            ))
        else
            value isa Real && !(value isa Bool) && isfinite(value) ||
                throw(ArgumentError(
                    "expected_header '$key' must be a finite real, got $(repr(value))",
                ))
        end
        canonical[key] = value
    end
    return canonical
end

function _fibonacci_validate_mpo_sites(label::AbstractString, tensor::MPO,
                                       sites::AbstractVector{<:Index})
    length(tensor) == length(sites) || throw(ArgumentError(
        "$label has length $(length(tensor)), expected $(length(sites))",
    ))
    for (site_number, site) in pairs(sites)
        hasind(tensor[site_number], site) || throw(ArgumentError(
            "$label tensor $site_number does not carry saved site index $site",
        ))
        hasind(tensor[site_number], prime(site)) || throw(ArgumentError(
            "$label tensor $site_number does not carry the primed saved site index",
        ))
    end
    return nothing
end

function _fibonacci_validate_hamiltonian_for_save(H::TBHamiltonian)
    H.position_space isa FibonacciPositionSpace || throw(ArgumentError(
        "save_fibonacci_hamiltonian requires a TBHamiltonian with " *
        "FibonacciPositionSpace, got $(typeof(H.position_space))",
    ))
    H.L >= 2 || throw(ArgumentError("Fibonacci Hamiltonian L must be at least 2"))
    length(H.sites) == H.L || throw(ArgumentError(
        "Fibonacci Hamiltonian is not position-only: length(sites)=$(length(H.sites)), L=$(H.L)",
    ))
    expected_N = fibonacci_site_count(H.L)
    H.N == expected_N || throw(ArgumentError(
        "Fibonacci Hamiltonian has N=$(H.N), expected F_(L+2)=$expected_N for L=$(H.L)",
    ))
    isfinite(H.scale) && H.scale > 0 || throw(ArgumentError(
        "Fibonacci Hamiltonian scale must be finite and positive, got $(H.scale)",
    ))
    isfinite(H.center) || throw(ArgumentError(
        "Fibonacci Hamiltonian center must be finite, got $(H.center)",
    ))
    projector = H.position_space.projector
    _fibonacci_validate_mpo_sites("Hamiltonian MPO", H.mpo, H.sites)
    _fibonacci_validate_mpo_sites("physical projector", projector, H.sites)
    return projector
end

function _fibonacci_attr_value(attributes, key::AbstractString,
                               mismatches::Vector{String})
    if key in keys(attributes)
        return read(attributes[key])
    end
    push!(mismatches, "$key: missing")
    return nothing
end

function _fibonacci_metadata_matches(stored_value, stored_type,
                                     expected_value, expected_type)
    return stored_type == expected_type && isequal(stored_value, expected_value)
end

"""
    save_fibonacci_hamiltonian(path, H; metadata=Dict(), overwrite=false) -> path

Atomically save a projected Fibonacci `TBHamiltonian` to HDF5. The file stores
the Hamiltonian MPO, its exact site indices, the physical-subspace projector,
`L`, `N`, `scale`, `center`, and scalar model/build metadata.

Metadata keys may be strings or symbols. Metadata values retain an explicit
type tag and must be scalar HDF5-compatible values: fixed-width integers,
`Float32`/`Float64`, `ComplexF32`/`ComplexF64`, `Bool`, strings, or symbols.

Data is first written and header-validated in a sibling temporary file, then
moved into place. Existing files are protected unless `overwrite=true` is
passed explicitly.
"""
function save_fibonacci_hamiltonian(path::AbstractString, H::TBHamiltonian;
                                    metadata=Dict(), overwrite::Bool=false)
    isempty(path) && throw(ArgumentError("cache path cannot be empty"))
    canonical_metadata = _fibonacci_canonical_metadata(metadata)
    projector = _fibonacci_validate_hamiltonian_for_save(H)

    target = abspath(path)
    isdir(target) && throw(ArgumentError(
        "Fibonacci Hamiltonian cache path is a directory: $target",
    ))
    ispath(target) && !isfile(target) && throw(ArgumentError(
        "Fibonacci Hamiltonian cache path is not a regular file: $target",
    ))
    isfile(target) && !overwrite && throw(ArgumentError(
        "refusing to overwrite existing Fibonacci Hamiltonian cache: $target; " *
        "pass overwrite=true to replace it explicitly",
    ))
    directory = dirname(target)
    mkpath(directory)
    temporary = target * ".tmp.$(getpid()).$(time_ns())"

    try
        HDF5.h5open(temporary, "w") do file
            write(file, "mpo", H.mpo)
            write(file, "sites", H.sites)
            write(file, "physical_projector", projector)

            attributes = HDF5.attributes(file)
            attributes["format"] = _FIBONACCI_HAMILTONIAN_FORMAT
            attributes["format_version"] = _FIBONACCI_HAMILTONIAN_FORMAT_VERSION
            attributes["L"] = Int64(H.L)
            attributes["N"] = Int64(H.N)
            attributes["scale"] = Float64(H.scale)
            attributes["center"] = Float64(H.center)
            attributes["site_count"] = Int64(length(H.sites))
            attributes["mpo_length"] = Int64(length(H.mpo))
            attributes["projector_length"] = Int64(length(projector))
            attributes["metadata_count"] = Int64(length(canonical_metadata))

            metadata_group = HDF5.create_group(file, "model_metadata")
            type_group = HDF5.create_group(file, "model_metadata_types")
            metadata_attributes = HDF5.attributes(metadata_group)
            type_attributes = HDF5.attributes(type_group)
            for key in sort!(collect(keys(canonical_metadata)))
                value, type_tag = canonical_metadata[key]
                metadata_attributes[key] = value
                type_attributes[key] = type_tag
            end
        end

        expected = Dict{String,Any}(
            key => value for (key, (value, _)) in canonical_metadata
        )
        # Recreate Symbol values for the checker so type tags are also tested.
        for (key, (_, type_tag)) in canonical_metadata
            type_tag == "Symbol" && (expected[key] = Symbol(expected[key]))
        end
        ok, message = check_fibonacci_hamiltonian(temporary; expected)
        ok || error("staged Fibonacci Hamiltonian cache failed validation:\n$message")
        mv(temporary, target; force=overwrite)
    catch
        ispath(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return path
end

function _check_fibonacci_hamiltonian_file(file, canonical_expected,
                                           canonical_expected_header)
    mismatches = String[]
    for name in _FIBONACCI_HAMILTONIAN_DATASETS
        haskey(file, name) || push!(mismatches, "payload '$name': missing")
    end

    attributes = HDF5.attributes(file)
    stored_format = _fibonacci_attr_value(attributes, "format", mismatches)
    !isnothing(stored_format) && stored_format != _FIBONACCI_HAMILTONIAN_FORMAT &&
        push!(mismatches,
              "format: expected=$(_FIBONACCI_HAMILTONIAN_FORMAT), stored=$stored_format")
    stored_version = _fibonacci_attr_value(attributes, "format_version", mismatches)
    !isnothing(stored_version) &&
        stored_version != _FIBONACCI_HAMILTONIAN_FORMAT_VERSION &&
        push!(mismatches,
              "format_version: expected=$(_FIBONACCI_HAMILTONIAN_FORMAT_VERSION), " *
              "stored=$stored_version")

    L = _fibonacci_attr_value(attributes, "L", mismatches)
    N = _fibonacci_attr_value(attributes, "N", mismatches)
    scale = _fibonacci_attr_value(attributes, "scale", mismatches)
    center = _fibonacci_attr_value(attributes, "center", mismatches)
    site_count = _fibonacci_attr_value(attributes, "site_count", mismatches)
    mpo_length = _fibonacci_attr_value(attributes, "mpo_length", mismatches)
    projector_length = _fibonacci_attr_value(attributes, "projector_length", mismatches)
    metadata_count = _fibonacci_attr_value(attributes, "metadata_count", mismatches)

    stored_header = Dict{String,Any}(
        "L" => L, "N" => N, "scale" => scale, "center" => center,
    )
    for key in sort!(collect(keys(canonical_expected_header)))
        stored_value = stored_header[key]
        isnothing(stored_value) && continue
        expected_value = canonical_expected_header[key]
        stored_value == expected_value || push!(
            mismatches,
            "header '$key': expected=$(repr(expected_value)), stored=$(repr(stored_value))",
        )
    end

    if !isnothing(L)
        L isa Integer && 2 <= L <= 90 ||
            push!(mismatches,
                  "L: expected an integer in the supported range 2:90, stored=$L")
    end
    if L isa Integer && 2 <= L <= 90 && !isnothing(N)
        expected_N = try
            fibonacci_site_count(L)
        catch
            nothing
        end
        isnothing(expected_N) || N == expected_N || push!(
            mismatches, "N: expected=$expected_N for L=$L, stored=$N",
        )
    end
    !isnothing(scale) &&
        (!(scale isa Real) || !isfinite(scale) || scale <= 0) &&
        push!(mismatches, "scale: expected a finite positive real, stored=$scale")
    !isnothing(center) &&
        (!(center isa Real) || !isfinite(center)) &&
        push!(mismatches, "center: expected a finite real, stored=$center")
    if L isa Integer && 2 <= L <= 90
        !isnothing(site_count) && site_count != L &&
            push!(mismatches, "site_count: expected=$L, stored=$site_count")
        !isnothing(mpo_length) && mpo_length != L &&
            push!(mismatches, "mpo_length: expected=$L, stored=$mpo_length")
        !isnothing(projector_length) && projector_length != L &&
            push!(mismatches, "projector_length: expected=$L, stored=$projector_length")
    end

    if haskey(file, "model_metadata") && haskey(file, "model_metadata_types")
        metadata_attributes = HDF5.attributes(file["model_metadata"])
        type_attributes = HDF5.attributes(file["model_metadata_types"])
        metadata_keys = Set(String.(collect(keys(metadata_attributes))))
        type_keys = Set(String.(collect(keys(type_attributes))))
        for key in sort!(collect(setdiff(metadata_keys, type_keys)))
            push!(mismatches, "metadata '$key': missing type tag")
        end
        for key in sort!(collect(setdiff(type_keys, metadata_keys)))
            push!(mismatches, "metadata type '$key': value is missing")
        end
        !isnothing(metadata_count) && metadata_count != length(metadata_keys) &&
            push!(mismatches,
                  "metadata_count: expected=$(length(metadata_keys)), stored=$metadata_count")

        for key in sort!(collect(keys(canonical_expected)))
            expected_value, expected_type = canonical_expected[key]
            if !(key in metadata_keys)
                push!(mismatches, "metadata '$key': missing")
                continue
            end
            stored_value = read(metadata_attributes[key])
            stored_type = key in type_keys ? read(type_attributes[key]) : nothing
            _fibonacci_metadata_matches(stored_value, stored_type,
                                        expected_value, expected_type) || push!(
                mismatches,
                "metadata '$key': expected=$(repr(expected_value)) " *
                "[$expected_type], stored=$(repr(stored_value)) [$stored_type]",
            )
        end
    elseif !isempty(canonical_expected)
        for key in sort!(collect(keys(canonical_expected)))
            push!(mismatches, "metadata '$key': missing")
        end
    end
    return mismatches
end

"""
    check_fibonacci_hamiltonian(path; expected=Dict(), expected_header=(;))
        -> (ok, message)

Cheaply validate a Fibonacci Hamiltonian cache without deserializing its MPOs.
The check covers the format/version, required payload names, core scalar
invariants, consistency with `F_(L+2)`, metadata type tags, and every key/value
in `expected`. `expected` is a subset match, so files may contain additional
metadata. `expected_header` independently accepts a `NamedTuple` or dictionary
subset of the core fields `L`, `N`, `scale`, and `center`.

Returns `(true, "")` on success. On failure it returns `(false, message)`, where
`message` contains all detected header/metadata mismatches when possible.
"""
function check_fibonacci_hamiltonian(path::AbstractString;
                                     expected=Dict(), expected_header=(;))
    canonical_expected = try
        _fibonacci_canonical_metadata(expected)
    catch error
        return (false, "invalid expected metadata: $(sprint(showerror, error))")
    end
    canonical_expected_header = try
        _fibonacci_canonical_expected_header(expected_header)
    catch error
        return (false, "invalid expected header: $(sprint(showerror, error))")
    end
    isfile(path) || return (false, "file not found: $path")

    mismatches = try
        HDF5.h5open(path, "r") do file
            _check_fibonacci_hamiltonian_file(
                file, canonical_expected, canonical_expected_header,
            )
        end
    catch error
        return (false,
                "unreadable Fibonacci Hamiltonian HDF5 file ($path): " *
                sprint(showerror, error))
    end
    return (isempty(mismatches), join(mismatches, '\n'))
end

"""
    load_fibonacci_hamiltonian(path; expected=Dict(), expected_header=(;))
        -> TBHamiltonian

Load a cache written by [`save_fibonacci_hamiltonian`](@ref). Header and
expected-metadata validation runs before the tensor payload is read. The
returned Hamiltonian has empty lazy caches, chain geometry, and a restored
`FibonacciPositionSpace` containing the saved physical projector.
"""
function load_fibonacci_hamiltonian(path::AbstractString;
                                    expected=Dict(), expected_header=(;))
    canonical_expected = try
        _fibonacci_canonical_metadata(expected)
    catch error
        throw(ArgumentError(
            "invalid expected metadata: $(sprint(showerror, error))",
        ))
    end
    canonical_expected_header = try
        _fibonacci_canonical_expected_header(expected_header)
    catch error
        throw(ArgumentError(
            "invalid expected header: $(sprint(showerror, error))",
        ))
    end
    isfile(path) || throw(ArgumentError(
        "invalid or mismatched Fibonacci Hamiltonian cache: $path\nfile not found: $path",
    ))

    local mpo, raw_sites, projector, L, N, scale, center
    try
        HDF5.h5open(path, "r") do file
            mismatches = _check_fibonacci_hamiltonian_file(
                file, canonical_expected, canonical_expected_header,
            )
            isempty(mismatches) || throw(ArgumentError(
                "invalid or mismatched Fibonacci Hamiltonian cache: $path\n" *
                join(mismatches, '\n'),
            ))

            mpo = read(file, "mpo", MPO)
            raw_sites = read(file, "sites", ITensors.IndexSet)
            projector = read(file, "physical_projector", MPO)
            attributes = HDF5.attributes(file)
            L = Int(read(attributes["L"]))
            N = Int(read(attributes["N"]))
            scale = Float64(read(attributes["scale"]))
            center = Float64(read(attributes["center"]))
        end
    catch error
        error isa ArgumentError && rethrow()
        throw(ArgumentError(
            "invalid or unreadable Fibonacci Hamiltonian cache: $path\n" *
            sprint(showerror, error),
        ))
    end
    sites = collect(raw_sites)
    length(sites) == L || error(
        "loaded Fibonacci cache is inconsistent: length(sites)=$(length(sites)), L=$L",
    )
    length(mpo) == L || error(
        "loaded Fibonacci cache is inconsistent: length(mpo)=$(length(mpo)), L=$L",
    )
    length(projector) == L || error(
        "loaded Fibonacci cache is inconsistent: length(projector)=$(length(projector)), L=$L",
    )
    N == fibonacci_site_count(L) || error(
        "loaded Fibonacci cache is inconsistent: N=$N for L=$L",
    )
    _fibonacci_validate_mpo_sites("loaded Hamiltonian MPO", mpo, sites)
    _fibonacci_validate_mpo_sites("loaded physical projector", projector, sites)

    H = TBHamiltonian(L, N, sites, mpo, _chain_geometry(),
                      scale, center,
                      nothing, nothing, nothing, nothing, 0, nothing)
    H.position_space = FibonacciPositionSpace(projector)
    return H
end
