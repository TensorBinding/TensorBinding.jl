using Test
using LinearAlgebra
using TensorBinding
using ITensors
using ITensorMPS

function _projected_fibonacci_matrix(H)
    basis = [TensorBinding.physical_site_state(H, x) for x in 1:H.N]
    matrix = zeros(ComplexF64, H.N, H.N)
    for j in eachindex(basis)
        Hket = apply(H.mpo, basis[j]; cutoff=1e-13, maxdim=300)
        for i in eachindex(basis)
            matrix[i, j] = inner(basis[i], Hket)
        end
    end
    return matrix
end

function _dense_kpm_moments(decomposition, Ncheb, center, scale; site=nothing)
    scaled = clamp.((decomposition.values .- center) ./ scale, -1.0, 1.0)
    angles = acos.(scaled)
    weights = isnothing(site) ? ones(length(scaled)) :
        abs2.(decomposition.vectors[site, :])
    return [sum(weights .* cos.(n .* angles)) for n in 0:(Ncheb - 1)]
end

@testset "Fibonacci projected position space" begin
    H = TensorBinding.fibonacci_hamiltonian(4; A=1.0, B=2.0)
    H_dispatch = TensorBinding.get_Hamiltonian(
        "fibonacci", (A=1.0, B=2.0); L=4,
    )

    @test H.N == 8
    @test TensorBinding.ambient_dimension(H) == 16
    @test TensorBinding.ambient_dimension(H) isa BigInt
    @test H.position_space isa TensorBinding.FibonacciPositionSpace
    @test H_dispatch.N == H.N
    @test H.scale > 0
    @test H.center == 0

    P = TensorBinding.physical_projector(H)
    @test real(tr(P)) ≈ H.N atol=1e-12
    @test norm(apply(P, P; cutoff=1e-13) - P) / norm(P) < 1e-12

    @testset "TN and dense construction" begin
        for model in (:onsite, :hopping), boundary in (:open, :periodic)
            parameters = (; A=1.2, B=0.7, t=0.9, onsite=0.2,
                           model, boundary)
            Htn = TensorBinding.fibonacci_hamiltonian(4; parameters...)
            Hdense = TensorBinding._dense_fibonacci_hamiltonian(4; parameters...)
            matrix = _projected_fibonacci_matrix(Htn)
            @test maximum(abs.(matrix .- Hdense)) < 1e-11
            @test norm(matrix - matrix') < 1e-11

            expected_wrap = boundary === :open ? 0.0 :
                (model === :onsite ? parameters.t :
                 (iszero(TensorBinding.fibonacci_zeckendorf_digits(Htn.N - 1, 4)[end]) ?
                  parameters.A : parameters.B))
            @test matrix[end, 1] ≈ expected_wrap atol=1e-11
        end
    end

    @test_throws ArgumentError TensorBinding.fibonacci_hamiltonian(
        4; A=1 + 1im, B=2.0, model=:onsite,
    )
    Hcomplex = TensorBinding.fibonacci_hamiltonian(
        4; A=1 + 0.2im, B=2 - 0.1im, model=:hopping,
    )
    complex_matrix = _projected_fibonacci_matrix(Hcomplex)
    @test norm(complex_matrix - complex_matrix') < 1e-11
    @test_throws ArgumentError TensorBinding.get_Hamiltonian(
        "fibonacci", (A=1.0,); L=4,
    )

    @testset "KPM projector and spectra" begin
        Ncheb = 8
        Tn, _, _ = TensorBinding.KPM_Tn(
            H, Ncheb; maxdim=100, cutoff=1e-12,
        )
        @test real(tr(Tn[1])) ≈ H.N atol=1e-10
        @test norm(Tn[1] - P) < 1e-12

        dense = TensorBinding._dense_fibonacci_hamiltonian(
            4; A=1.0, B=2.0,
        )
        decomposition = eigen(Hermitian(dense))
        dense_moments = _dense_kpm_moments(
            decomposition, Ncheb, H.center, H.scale,
        )
        tn_moments = real.(tr.(Tn[1:Ncheb]))
        @test maximum(abs.(tn_moments .- dense_moments)) < 1e-8

        energies = collect(range(-3.5, 3.5; length=7))
        dos_tn = TensorBinding.get_dos_trace(
            H, Ncheb, energies; maxdim=100, cutoff=1e-12,
        )
        dos_dense = [
            TensorBinding.get_ldos_from_mun(
                dense_moments, Ncheb, (energy - H.center) / H.scale,
            ) for energy in energies
        ]
        @test maximum(abs.(dos_tn .- dos_dense)) < 1e-8

        ldos_tn = TensorBinding.get_ldos_spatial(
            H, Ncheb, energies;
            mode=:mps, ordering=:conumber,
            maxdim=100, cutoff=1e-12,
        )
        permutation = TensorBinding.site_permutation(H; ordering=:conumber)
        ldos_dense = zeros(length(energies), H.N)
        for site in 1:H.N
            moments = _dense_kpm_moments(
                decomposition, Ncheb, H.center, H.scale; site,
            )
            for (i, energy) in pairs(energies)
                ldos_dense[i, site] = TensorBinding.get_ldos_from_mun(
                    moments, Ncheb, (energy - H.center) / H.scale,
                )
            end
        end
        @test maximum(abs.(ldos_tn .- ldos_dense[:, permutation])) < 1e-8

        dos_stochastic = TensorBinding.get_dos_stochastic(
            H, 4, [0.0]; N_sample=100, seed=7,
            maxdim=60, cutoff=1e-12,
        )
        dos_exact = TensorBinding.get_dos_trace(
            H, 4, [0.0]; maxdim=60, cutoff=1e-12,
        )
        @test all(isfinite, dos_stochastic)
        @test abs(dos_stochastic[1] - dos_exact[1]) < 0.35 * max(abs(dos_exact[1]), 1.0)
    end

    @testset "Conumbering and guards" begin
        axis = TensorBinding.site_axis(H; ordering=:conumber)
        permutation = TensorBinding.site_permutation(H; ordering=:conumber)
        reflected = TensorBinding.site_permutation(
            H; ordering=:conumber, orientation=:reversed,
        )
        @test axis == collect(-4:3)
        @test sort(permutation) == collect(1:H.N)
        @test sort(reflected) == collect(1:H.N)
        @test permutation != reflected

        classes = TensorBinding.fibonacci_site_environment.(4, permutation)
        @test all(!=(:atomic), classes[1:3])
        @test classes[4:5] == fill(:atomic, 2)
        @test all(!=(:atomic), classes[6:8])
        for Ltest in 4:9
            Ntest = TensorBinding.fibonacci_site_count(Ltest)
            sites_by_c = sortperm([
                TensorBinding.fibonacci_conumber(
                    Ltest, site; centered=false,
                ) for site in 1:Ntest
            ])
            environments = TensorBinding.fibonacci_site_environment.(
                Ltest, sites_by_c,
            )
            molecular_count = Int(TensorBinding.fibonacci_number(Ltest))
            atomic_count = Int(TensorBinding.fibonacci_number(Ltest - 1))
            @test all(!=(:atomic), environments[1:molecular_count])
            @test environments[(molecular_count + 1):(molecular_count + atomic_count)] ==
                  fill(:atomic, atomic_count)
            @test all(!=(:atomic), environments[(molecular_count + atomic_count + 1):end])
            @test all(site -> TensorBinding.fibonacci_site_from_conumber(
                              Ltest,
                              TensorBinding.fibonacci_conumber(Ltest, site),
                          ) == site, 1:Ntest)
        end

        Llarge = 43
        large_partition = TensorBinding.fibonacci_rg_partition(Llarge)
        @test TensorBinding.fibonacci_site_count(Llarge) == 1_134_903_170
        @test large_partition.molecular_count == 433_494_437
        @test large_partition.atomic_count == 267_914_296

        # The largest Fibonacci register whose physical site count fits Int64
        # still needs overflow-safe modular shifts and a BigInt ambient size.
        Lmax = 90
        Nmax = TensorBinding.fibonacci_site_count(Lmax)
        site_at_high_raw_rank = TensorBinding.fibonacci_site_from_conumber(
            Lmax, Nmax - 1; alignment=:raw, centered=false,
        )
        atomic_conumber = TensorBinding.fibonacci_conumber(
            Lmax, site_at_high_raw_rank; alignment=:atomic, centered=false,
        )
        @test 0 <= atomic_conumber < Nmax
        @test TensorBinding.fibonacci_site_from_conumber(
            Lmax, atomic_conumber; alignment=:atomic, centered=false,
        ) == site_at_high_raw_rank

        Hwide = deepcopy(H)
        Hwide.L = Lmax
        @test TensorBinding.ambient_dimension(Hwide) == big(2)^Lmax

        deep = TensorBinding.fibonacci_rg_partition(Llarge; depth=13)
        @test deep.effective_L == 4
        @test deep.window_count == 8
        @test deep.molecular_count == 3
        @test deep.atomic_count == 2
        deep_sites = [TensorBinding.fibonacci_site_from_conumber(
                          Llarge, c; centered=false,
                      ) for c in deep.window_ranks]
        @test all(==(:atomic), TensorBinding.fibonacci_site_environment.(
                                      Llarge, deep_sites,
                                  ))
        @test all(>=(13), TensorBinding.fibonacci_atomic_depth.(
                               Llarge, deep_sites,
                           ))
        @test_throws ArgumentError TensorBinding.get_ldos_spatial(
            H, 4, [0.0]; ordering=:conumber, num_x=4,
        )
        @test_throws ArgumentError TensorBinding.add_onsite!(H, 0.1)
        @test_throws ArgumentError TensorBinding.get_bands(H, 4, 1, [0.0])
    end

    Hbinary = TensorBinding.get_Hamiltonian("chain_1d", 1.0; L=3)
    @test Hbinary.position_space isa TensorBinding.BinaryPositionSpace
    @test real(tr(TensorBinding.physical_projector(Hbinary))) ≈ Hbinary.N atol=1e-12
    @test TensorBinding.site_axis(Hbinary) == collect(0:7)
end
