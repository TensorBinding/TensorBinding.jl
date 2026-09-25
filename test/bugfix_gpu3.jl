using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_bands, get_bands_gpu, add_spin!

# get_bands_gpu in grid mode (no kpath) must plan the same k-grid as the CPU
# get_bands on models with auxiliary indices. H.L already counts only position
# qubits, so subtracting the aux indices from it again shrank L_pos by one and
# changed the k-labels (and, in 2D, the number of columns). Needs CUDA.jl.
@testset "get_bands_gpu k-grid on aux models" begin
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
        ωs = collect(range(-2.5, 2.5; length=5))
        kw = (cutoff=1e-12, maxdim=200)

        # sublattice aux index (postpended); scale given, so no DMRG estimate
        Hsl = TensorBinding.get_Hamiltonian("honeycomb", 1.0; L=4, scale=3.2)
        # spin aux index (prepended); add_spin! resets the scale, so set it again
        Hsp = TensorBinding.get_Hamiltonian("square_2d", 1.0; L=4)
        add_spin!(Hsp)
        Hsp.scale = 4.4
        # 1D spinful chain: same column count either way, but the k-range halved
        Hch = TensorBinding.get_Hamiltonian("chain_1d", 1.0; L=3)
        add_spin!(Hch)
        Hch.scale = 2.5
        # control: no aux index, already consistent before the fix
        H0 = TensorBinding.get_Hamiltonian("square_2d", 1.0; L=4, scale=4.4)

        for H in (Hsl, Hsp, Hch, H0)
            cpu = get_bands(H, 16, ωs; kw..., num_x=4)
            gpu = get_bands_gpu(H, 16, ωs; kw..., num_x=4, type=ComplexF64)
            @test size(gpu) == size(cpu)
            @test size(gpu) == size(cpu) && isapprox(gpu, cpu; atol=1e-8)
        end
    end
end
