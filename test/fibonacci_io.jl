using Test
using LinearAlgebra
using HDF5
using ITensors
using ITensorMPS
using TensorBinding

@testset "Fibonacci Hamiltonian HDF5 cache" begin
    H = TensorBinding.fibonacci_hamiltonian(
        4; A=1.2, B=0.7, model=:hopping, t=0.9, onsite=0.2,
        boundary=:open, cutoff=1e-12, maxdim=80, padding=1.08,
    )
    metadata = Dict{String,Any}(
        "A" => 1.2,
        "B" => 0.7,
        "model" => :hopping,
        "t" => 0.9,
        "onsite" => 0.2,
        "boundary" => :open,
        "cutoff" => 1e-12,
        "maxdim" => Int64(80),
        "padding" => 1.08,
    )

    mktempdir() do directory
        path = joinpath(directory, "fibonacci_H.h5")
        returned = TensorBinding.save_fibonacci_hamiltonian(
            path, H; metadata,
        )
        @test returned == path
        @test isfile(path)
        @test isempty(filter(name -> occursin(".tmp.", name), readdir(directory)))

        ok, message = TensorBinding.check_fibonacci_hamiltonian(
            path; expected=metadata,
        )
        @test ok
        @test isempty(message)

        expected_header = (;
            L=H.L, N=H.N, scale=H.scale, center=H.center,
        )
        @test TensorBinding.check_fibonacci_hamiltonian(
            path; expected_header,
        ) == (true, "")

        # Symbol keys and NamedTuple metadata normalize to the same cache keys.
        expected_subset = (model=:hopping, boundary=:open, maxdim=Int64(80))
        @test TensorBinding.check_fibonacci_hamiltonian(
            path; expected=expected_subset,
        ) == (true, "")

        loaded = TensorBinding.load_fibonacci_hamiltonian(
            path; expected=metadata, expected_header,
        )
        @test loaded.L == H.L
        @test loaded.N == H.N
        @test loaded.scale == H.scale
        @test loaded.center == H.center
        @test loaded.position_space isa TensorBinding.FibonacciPositionSpace
        @test all(loaded.sites .== H.sites)
        @test norm(loaded.mpo - H.mpo) / norm(H.mpo) < 1e-13
        loaded_projector = TensorBinding.physical_projector(loaded)
        original_projector = TensorBinding.physical_projector(H)
        @test norm(loaded_projector - original_projector) /
              norm(original_projector) < 1e-13
        @test real(tr(loaded_projector)) ≈ loaded.N atol=1e-12

        wrong_header = Dict{String,Any}(
            "L" => H.L + 1,
            "N" => H.N + 1,
            "scale" => H.scale + 0.5,
            "center" => H.center + 0.25,
        )
        ok, message = TensorBinding.check_fibonacci_hamiltonian(
            path; expected_header=wrong_header,
        )
        @test !ok
        for field in keys(wrong_header)
            @test occursin("header '$field'", message)
        end
        @test_throws ArgumentError TensorBinding.load_fibonacci_hamiltonian(
            path; expected_header=(L=H.L + 1,),
        )

        ok, message = TensorBinding.check_fibonacci_hamiltonian(
            path; expected_header=(model=:hopping,),
        )
        @test !ok
        @test occursin("invalid expected header", message)

        bad_value = copy(metadata)
        bad_value["A"] = 9.0
        ok, message = TensorBinding.check_fibonacci_hamiltonian(
            path; expected=bad_value,
        )
        @test !ok
        @test occursin("metadata 'A'", message)
        @test occursin("expected=9.0", message)
        @test_throws ArgumentError TensorBinding.load_fibonacci_hamiltonian(
            path; expected=bad_value,
        )

        # Equal numeric values with different construction types are distinct.
        bad_type = copy(metadata)
        bad_type["maxdim"] = Int32(80)
        ok, message = TensorBinding.check_fibonacci_hamiltonian(
            path; expected=bad_type,
        )
        @test !ok
        @test occursin("Int32", message)
        @test occursin("Int64", message)

        @test_throws ArgumentError TensorBinding.save_fibonacci_hamiltonian(
            path, H; metadata,
        )
        @test TensorBinding.save_fibonacci_hamiltonian(
            path, H; metadata, overwrite=true,
        ) == path
        @test TensorBinding.check_fibonacci_hamiltonian(
            path; expected=metadata,
        ) == (true, "")

        missing = joinpath(directory, "missing.h5")
        ok, message = TensorBinding.check_fibonacci_hamiltonian(missing)
        @test !ok
        @test occursin("file not found", message)
        @test_throws ArgumentError TensorBinding.load_fibonacci_hamiltonian(missing)

        malformed = joinpath(directory, "malformed.h5")
        HDF5.h5open(malformed, "w") do file
            HDF5.attributes(file)["format_version"] = Int64(99)
        end
        ok, message = TensorBinding.check_fibonacci_hamiltonian(malformed)
        @test !ok
        @test occursin("format_version", message)
        @test occursin("payload 'mpo': missing", message)
    end

    binary_H = TensorBinding.get_Hamiltonian("chain_1d", 1.0; L=3)
    mktempdir() do directory
        @test_throws ArgumentError TensorBinding.save_fibonacci_hamiltonian(
            joinpath(directory, "binary.h5"), binary_H,
        )
        @test_throws ArgumentError TensorBinding.save_fibonacci_hamiltonian(
            joinpath(directory, "bad_metadata.h5"), H;
            metadata=Dict("callback" => identity),
        )
    end
end
