using Test
using LinearAlgebra
using TensorBinding

@testset "GPU MPS spatial LDOS interface" begin
    H = TensorBinding.get_Hamiltonian("chain_1d", 1.0; L=3, scale=2.5)
    energies = [-1.0, 0.0, 1.0]

    @testset "moment-column reconstruction" begin
        moments = [1.0 2.0; 0.5 -1.0; -0.25 0.75]
        weights = [1.0 2.0 3.0; 0.5 -1.0 4.0; 2.0 0.25 -2.0]
        denom = [2.0, 4.0, 0.0]
        valid = [true, true, false]
        reconstructed = TensorBinding._reconstruct_ldos_moment_columns(
            moments, weights, denom, valid,
        )
        expected = transpose(weights) * moments
        expected[1, :] ./= denom[1]
        expected[2, :] ./= denom[2]
        expected[3, :] .= 0.0
        @test reconstructed == expected
        @test size(reconstructed) == (size(weights, 2), size(moments, 2))
        @test_throws DimensionMismatch TensorBinding._reconstruct_ldos_moment_columns(
            moments[1:2, :], weights, denom, valid,
        )
        @test_throws DimensionMismatch TensorBinding._reconstruct_ldos_moment_columns(
            moments, weights, denom[1:2], valid,
        )
        @test_throws DimensionMismatch TensorBinding._reconstruct_ldos_moment_columns(
            moments, weights, denom, valid[1:2],
        )
    end

    # These checks deliberately run before CUDA discovery: unsupported requests
    # should fail at the public API boundary, not deep inside the GPU backend.
    @test_throws ArgumentError TensorBinding.get_ldos_spatial_mps_gpu(
        H, 1, energies; x_groups=[1],
    )
    @test_throws ArgumentError TensorBinding.get_ldos_spatial_mps_gpu(
        H, 4, energies; x_groups=[1], reduce=:block,
    )
    @test_throws ArgumentError TensorBinding.get_ldos_spatial_mps_gpu(
        H, 4, energies; x_groups=[1], grid=true,
    )
    @test_throws ArgumentError TensorBinding.get_ldos_spatial_mps_gpu(
        H, 4, energies; x_groups=[1], ordering=:conumber,
    )
    @test_throws ArgumentError TensorBinding.get_ldos_spatial_mps_gpu(
        H, 4, energies; x_groups=[1], spin_proj=true,
    )
    @test_throws ArgumentError TensorBinding.get_ldos_spatial_mps_gpu(
        H, 4, energies; x_groups=Vector{Vector{Int}}(),
    )
    @test_throws ArgumentError TensorBinding.get_ldos_spatial_mps_gpu(
        H, 4, energies; x_groups=[[0]],
    )
    Haux = TensorBinding.get_Hamiltonian("ssh_sublattice", (t=1.0, d=0.2); L=3)
    @test_throws ArgumentError TensorBinding.get_ldos_spatial_mps_gpu(
        Haux, 4, energies; x_groups=[1],
    )

    cuda_functional = false
    cuda_error = nothing
    if Base.find_package("CUDA") !== nothing
        try
            @eval using CUDA
            cuda_functional = CUDA.functional()
        catch err
            cuda_error = err
        end
    end

    if cuda_functional
        @testset "projected Fibonacci CPU/GPU agreement" begin
            Hf = TensorBinding.fibonacci_hamiltonian(
                4; A=1.0, B=2.0, model=:onsite, t=0.6,
                boundary=:open,
                cutoff=1e-12, maxdim=100,
            )
            @test Hf.center != 0.0  # exercises the projected shift H - center*P
            groups = [[1, 2], [4], [7, 8]]
            Ncheb = 8
            ω = collect(range(-0.1, 3.1; length=9))

            cpu = TensorBinding.get_ldos_spatial(
                Hf, Ncheb, ω;
                mode=:mps, x_groups=groups,
                maxdim=100, cutoff=1e-10,
            )
            gpu, moments, linkdims = TensorBinding.get_ldos_spatial_mps_gpu(
                Hf, Ncheb, ω;
                x_groups=groups,
                type=ComplexF32, maxdim=100, cutoff=1e-6,
                return_maxlinkdim=true,
                return_moments=true,
            )

            @test size(gpu) == (length(ω), length(groups))
            @test size(moments) == (Ncheb, length(groups))
            @test length(linkdims) == length(groups)
            @test all(>=(1), linkdims)
            @test gpu ≈ cpu rtol=5e-4 atol=5e-5

            dense = TensorBinding._dense_fibonacci_hamiltonian(
                4; A=1.0, B=2.0, model=:onsite, t=0.6, boundary=:open,
            )
            decomposition = eigen(Hermitian(dense))
            scaled_eigenvalues = clamp.(
                (decomposition.values .- Hf.center) ./ Hf.scale, -1.0, 1.0,
            )
            eigenangles = acos.(scaled_eigenvalues)
            dense_site_moments(site) = [
                sum(
                    abs2.(decomposition.vectors[site, :]) .*
                    cos.(n .* eigenangles)
                ) for n in 0:(Ncheb - 1)
            ]
            dense_group_moments = hcat([
                sum(
                    dense_site_moments(x) for x in group
                ) ./ length(group)
                for group in groups
            ]...)
            @test moments ≈ dense_group_moments rtol=5e-4 atol=5e-5

            ω_scaled = (ω .- Hf.center) ./ Hf.scale
            W, denom = TensorBinding._dos_weight_matrix(Ncheb, ω_scaled)
            reconstructed = TensorBinding._reconstruct_ldos_moment_columns(
                moments, W, denom, abs.(ω_scaled) .< 1.0,
            )
            @test reconstructed ≈ gpu rtol=5e-13 atol=5e-13
        end

        @testset "ordinary binary position space" begin
            groups = [[1], [3, 4]]
            cpu = TensorBinding.get_ldos_spatial(
                H, 6, energies;
                mode=:mps, x_groups=groups,
                maxdim=40, cutoff=1e-10,
            )
            gpu = TensorBinding.get_ldos_spatial_mps_gpu(
                H, 6, energies;
                x_groups=groups,
                type=ComplexF32, maxdim=40, cutoff=1e-6,
            )
            @test gpu ≈ cpu rtol=5e-4 atol=5e-5
        end
    else
        @info "Skipping CUDA-functional GPU MPS LDOS comparisons" exception=cuda_error
        @test true
    end
end
