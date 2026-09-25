using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test

# Regressions for the audited GPU bugs in src/gpu/GPU_tk.jl: real-typed one-hot
# extraction, `pointavg` on every trajectory sample, and the real-type guard in
# get_bands_gpu. They need a functional CUDA.jl and are skipped without one.
@testset "GPU bug fixes" begin
    cuda_functional = false
    if Base.find_package("CUDA") !== nothing
        try
            @eval using CUDA
            cuda_functional = CUDA.functional()
        catch
        end
    end

    if !cuda_functional
        @test_skip "CUDA.jl not functional"
    else
        H = TensorBinding.get_Hamiltonian("chain_1d", 1.0; L=3, scale=2.5)
        ωs = collect(range(-1.5, 1.5; length=5))

        @testset "real-typed get_ldos_spatial_gpu" begin
            ref = TensorBinding.get_ldos_spatial_gpu(H, 16, ωs; type=ComplexF64)
            @test all(isfinite, ref)
            for (T, tol) in ((Float64, 1e-10), (Float32, 1e-4))
                ldos = TensorBinding.get_ldos_spatial_gpu(H, 16, ωs; type=T)
                @test maximum(abs.(ldos .- ref)) < tol
            end
        end

        @testset "get_bands_gpu rejects real types" begin
            @test_throws ArgumentError TensorBinding.get_bands_gpu(H, 16, ωs; type=Float64, num_x=4)
        end

        @testset "pointavg applies to every sample" begin
            psi0 = TensorBinding.binary_to_MPS(2, H.L, H.sites)
            kw = (nsteps=4, dt=0.2, x_groups=[[1, 2], [3, 4], [5, 6, 7, 8]],
                  pointavg=:abs2, dtype=ComplexF64)
            re = TensorBinding.get_state_amplitude_trajectory_gpu(H, psi0; component=:real, kw...)
            im = TensorBinding.get_state_amplitude_trajectory_gpu(H, psi0; component=:imag, kw...)
            # |amplitude|^2 averages are real, so the component must not matter at any step.
            @test size(re.amplitude, 1) >= 3
            @test re.amplitude ≈ im.amplitude
            @test all(>=(0), re.amplitude)
        end
    end
end
