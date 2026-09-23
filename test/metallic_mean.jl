using Test
using LinearAlgebra
using TensorBinding
using ITensors
using ITensorMPS

# Substitution fixed point A -> A^m B, B -> A: an oracle independent of the
# numeration system used by the package.
function _metallic_mean_word(m::Integer, n::Integer)
    word = "A"
    while length(word) < n
        word = join(c == 'A' ? "A"^m * "B" : "A" for c in word)
    end
    return word[1:n]
end

function _projected_mm_matrix(H)
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

function _mm_dense_kpm_moments(decomposition, Ncheb, center, scale; site=nothing)
    scaled = clamp.((decomposition.values .- center) ./ scale, -1.0, 1.0)
    angles = acos.(scaled)
    weights = isnothing(site) ? ones(length(scaled)) :
        abs2.(decomposition.vectors[site, :])
    return [sum(weights .* cos.(n .* angles)) for n in 0:(Ncheb - 1)]
end

@testset "Metallic-mean projected position space" begin
    @testset "numeration system and word" begin
        for m in 1:4, L in 2:5
            N = TensorBinding.metallic_mean_site_count(m, L)
            @test N == m * TensorBinding.metallic_mean_site_count(m, L - 1) +
                       TensorBinding.metallic_mean_site_count(m, L - 2)
            weights = [Int(TensorBinding.metallic_mean_number(m, L - p)) for p in 1:L]
            strings = [TensorBinding.metallic_mean_digits(m, n, L) for n in 0:(N - 1)]
            @test all(d -> all(0 .<= d .<= m), strings)
            @test all(d -> all(i -> !(d[i] == m && d[i + 1] != 0), 1:(L - 1)), strings)
            @test [sum(d .* weights) for d in strings] == collect(0:(N - 1))
            @test issorted(strings)
            brute = [digits_vec for s in 0:((m + 1)^L - 1)
                     for digits_vec in ([(s ÷ (m + 1)^(L - p)) % (m + 1) for p in 1:L],)
                     if all(i -> !(digits_vec[i] == m && digits_vec[i + 1] != 0), 1:(L - 1))]
            @test sort(brute) == strings

            letters = [d[end] == m ? 'B' : 'A' for d in strings]
            @test join(letters) == _metallic_mean_word(m, N)
            @test [TensorBinding.metallic_mean_bond_symbol(m, L, b) for b in 1:N] ==
                  [c == 'A' ? :A : :B for c in letters]
        end
        @test_throws ArgumentError TensorBinding.metallic_mean_digits(2, 17, 3)
        @test_throws ArgumentError TensorBinding.metallic_mean_number(0, 3)
        @test_throws BoundsError TensorBinding.metallic_mean_bond_symbol(2, 3, 18)
    end

    @testset "m = 1 reproduces the Fibonacci chain" begin
        for L in 2:6
            N = TensorBinding.fibonacci_site_count(L)
            @test TensorBinding.metallic_mean_site_count(1, L) == N
            @test all(n -> TensorBinding.metallic_mean_digits(1, n, L) ==
                           TensorBinding.fibonacci_zeckendorf_digits(n, L), 0:(N - 1))
        end
        params = (; A=1.2, B=0.7, t=0.9, onsite=0.2)
        for model in (:onsite, :hopping), boundary in (:open, :periodic)
            dense_mm = TensorBinding._dense_metallic_mean_hamiltonian(
                1, 4; params..., model, boundary,
            )
            dense_fib = TensorBinding._dense_fibonacci_hamiltonian(
                4; params..., model, boundary,
            )
            @test dense_mm == dense_fib
            Hmm = TensorBinding.metallic_mean_hamiltonian(1, 4; params..., model, boundary)
            @test Hmm.N == 8
            @test all(s -> dim(s) == 2, Hmm.sites)
            @test maximum(abs.(_projected_mm_matrix(Hmm) .- dense_fib)) < 1e-11
        end
    end

    @testset "TN construction matches the dense oracle" begin
        for m in (2, 3), model in (:onsite, :hopping), boundary in (:open, :periodic)
            parameters = (; A=1.2, B=0.7, t=0.9, onsite=0.2, model, boundary)
            Htn = TensorBinding.metallic_mean_hamiltonian(m, 3; parameters...)
            Hdense = TensorBinding._dense_metallic_mean_hamiltonian(m, 3; parameters...)
            @test Htn.N == TensorBinding.metallic_mean_site_count(m, 3)
            @test Htn.position_space isa TensorBinding.MetallicMeanPositionSpace
            @test Htn.position_space.m == m
            @test TensorBinding.ambient_dimension(Htn) == big(m + 1)^3
            @test TensorBinding.ambient_dimension(Htn) isa BigInt
            @test length(Htn.sites) == 3 && all(s -> dim(s) == m + 1, Htn.sites)
            @test Htn.scale > 0
            matrix = _projected_mm_matrix(Htn)
            @test maximum(abs.(matrix .- Hdense)) < 1e-11
            @test norm(matrix - matrix') < 1e-11
            expected_wrap = boundary === :open ? 0.0 :
                (model === :onsite ? parameters.t :
                 (TensorBinding.metallic_mean_digits(m, Htn.N - 1, 3)[end] == m ?
                  parameters.B : parameters.A))
            @test matrix[end, 1] ≈ expected_wrap atol=1e-11
        end

        Hcomplex = TensorBinding.metallic_mean_hamiltonian(
            2, 3; A=1 + 0.2im, B=2 - 0.1im, model=:hopping,
        )
        complex_matrix = _projected_mm_matrix(Hcomplex)
        @test norm(complex_matrix - complex_matrix') < 1e-11
        @test_throws ArgumentError TensorBinding.metallic_mean_hamiltonian(
            2, 3; A=1 + 1im, B=2.0, model=:onsite,
        )
        @test_throws ArgumentError TensorBinding.metallic_mean_hamiltonian(0, 3; A=1.0, B=2.0)
        @test_throws ArgumentError TensorBinding.metallic_mean_hamiltonian(2, 1; A=1.0, B=2.0)
        @test_throws ArgumentError TensorBinding.metallic_mean_hamiltonian(
            2, 3; A=1.0, B=2.0, boundary=:twisted,
        )
    end

    # Silver-mean hopping chain: 17 physical sites inside a 27-state register.
    H = TensorBinding.metallic_mean_hamiltonian(2, 3; A=1.0, B=2.0)
    P = TensorBinding.physical_projector(H)
    @test H.N == 17
    @test real(tr(P)) ≈ H.N atol=1e-12
    @test norm(apply(P, P; cutoff=1e-13) - P) / norm(P) < 1e-12
    @test H.center == 0
    @test occursin("TBHamiltonian", sprint(show, H))

    @testset "projector-aware KPM" begin
        Ncheb = 8
        Tn, _, _ = TensorBinding.KPM_Tn(H, Ncheb; maxdim=100, cutoff=1e-12)
        @test real(tr(Tn[1])) ≈ H.N atol=1e-10
        @test norm(Tn[1] - P) < 1e-12

        dense = TensorBinding._dense_metallic_mean_hamiltonian(2, 3; A=1.0, B=2.0)
        decomposition = eigen(Hermitian(dense))
        dense_moments = _mm_dense_kpm_moments(decomposition, Ncheb, H.center, H.scale)
        @test maximum(abs.(real.(tr.(Tn[1:Ncheb])) .- dense_moments)) < 1e-8

        energies = collect(range(-3.5, 3.5; length=7))
        dos_tn = TensorBinding.get_dos_trace(H, Ncheb, energies; maxdim=100, cutoff=1e-12)
        dos_dense = [
            TensorBinding.get_ldos_from_mun(
                dense_moments, Ncheb, (energy - H.center) / H.scale,
            ) for energy in energies
        ]
        @test maximum(abs.(dos_tn .- dos_dense)) < 1e-8

        ldos_dense = zeros(length(energies), H.N)
        for site in 1:H.N
            moments = _mm_dense_kpm_moments(decomposition, Ncheb, H.center, H.scale; site)
            for (i, energy) in pairs(energies)
                ldos_dense[i, site] = TensorBinding.get_ldos_from_mun(
                    moments, Ncheb, (energy - H.center) / H.scale,
                )
            end
        end
        for mode in (:mps, :mpo)
            ldos_tn = TensorBinding.get_ldos_spatial(
                H, Ncheb, energies; mode, maxdim=100, cutoff=1e-12,
            )
            @test size(ldos_tn) == (length(energies), H.N)
            @test maximum(abs.(ldos_tn .- ldos_dense)) < 1e-8
        end
        probe = 5
        ldos_online = TensorBinding.get_ldos_online(
            H, Ncheb, probe, energies; maxdim=100, cutoff=1e-12,
        )
        @test maximum(abs.(ldos_online .- ldos_dense[:, probe])) < 1e-8

        dos_stochastic = TensorBinding.get_dos_stochastic(
            H, 4, [0.0]; N_sample=60, seed=7, maxdim=60, cutoff=1e-12,
        )
        dos_exact = TensorBinding.get_dos_trace(H, 4, [0.0]; maxdim=60, cutoff=1e-12)
        @test all(isfinite, dos_stochastic)
        @test abs(dos_stochastic[1] - dos_exact[1]) < 0.35 * max(abs(dos_exact[1]), 1.0)
    end

    @testset "interface and guards" begin
        Hd = TensorBinding.get_Hamiltonian("metallic_mean", (A=1.0, B=2.0); L=3, m=2)
        @test Hd.N == H.N
        @test Hd.position_space isa TensorBinding.MetallicMeanPositionSpace
        @test_throws ArgumentError TensorBinding.get_Hamiltonian(
            "metallic_mean", (A=1.0, B=2.0); L=3,
        )
        @test_throws ArgumentError TensorBinding.get_Hamiltonian(
            "metallic_mean", (A=1.0,); L=3, m=2,
        )
        @test_throws ArgumentError TensorBinding.get_Hamiltonian(
            "metallic_mean", (A=1.0, B=2.0, foo=1); L=3, m=2,
        )

        @test TensorBinding.site_axis(H) == collect(0:(H.N - 1))
        @test TensorBinding.site_permutation(H) == collect(1:H.N)
        @test_throws ArgumentError TensorBinding.site_axis(H; ordering=:conumber)
        @test_throws ArgumentError TensorBinding.get_ldos_spatial(
            H, 4, [0.0]; ordering=:conumber,
        )
        @test_throws ArgumentError TensorBinding.add_onsite!(H, 0.1)
        @test_throws ArgumentError TensorBinding.add_hopping!(H, 0.1)
        @test_throws ArgumentError TensorBinding.get_bands(H, 4, 1, [0.0])
        @test_throws BoundsError TensorBinding.physical_site_state(H, H.N + 1)
        for x in 1:H.N
            psi = TensorBinding.physical_site_state(H, x)
            @test norm(psi) ≈ 1
            @test abs(inner(psi, apply(P, psi))) ≈ 1 atol=1e-12
        end
    end
end
