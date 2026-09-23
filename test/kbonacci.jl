using Test
using LinearAlgebra
using TensorBinding
using ITensors
using ITensorMPS

# Substitution fixed point a_i -> a_1 a_(i+1) (i < k), a_k -> a_1 on the letters
# A, B, C, …: an oracle independent of the numeration system used by the package.
function _kbonacci_word(k::Integer, n::Integer)
    letters = ['A' + i for i in 0:(k - 1)]
    word = "A"
    while length(word) < n
        word = join(c == letters[k] ? "A" :
                    "A" * letters[findfirst(==(c), letters) + 1] for c in word)
    end
    return word[1:n]
end

_kb_trailing_ones(d) = (r = 0; for x in reverse(d); x == 1 ? (r += 1) : break; end; r)

function _projected_kb_matrix(H)
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

function _kb_dense_kpm_moments(decomposition, Ncheb, center, scale; site=nothing)
    scaled = clamp.((decomposition.values .- center) ./ scale, -1.0, 1.0)
    angles = acos.(scaled)
    weights = isnothing(site) ? ones(length(scaled)) :
        abs2.(decomposition.vectors[site, :])
    return [sum(weights .* cos.(n .* angles)) for n in 0:(Ncheb - 1)]
end

@testset "k-bonacci projected position space" begin
    @testset "numeration system and word" begin
        @test [Int(TensorBinding.kbonacci_number(2, n)) for n in 0:8] == [1, 2, 3, 5, 8, 13, 21, 34, 55]
        @test [Int(TensorBinding.kbonacci_number(3, n)) for n in 0:8] == [1, 2, 4, 7, 13, 24, 44, 81, 149]
        @test [Int(TensorBinding.kbonacci_number(4, n)) for n in 0:8] == [1, 2, 4, 8, 15, 29, 56, 108, 208]
        @test TensorBinding.kbonacci_number(3, 60) isa BigInt
        for k in 2:5, L in 2:6
            N = TensorBinding.kbonacci_site_count(k, L)
            @test N == (L < k ? 2^L : sum(TensorBinding.kbonacci_site_count(k, L - i) for i in 1:k))
            weights = [Int(TensorBinding.kbonacci_number(k, L - p)) for p in 1:L]
            strings = [TensorBinding.kbonacci_digits(k, n, L) for n in 0:(N - 1)]
            @test all(d -> all(0 .<= d .<= 1), strings)
            @test all(d -> !occursin("1"^k, join(d)), strings)
            @test [sum(d .* weights) for d in strings] == collect(0:(N - 1))
            @test issorted(strings)
            brute = sort([[(s >> (L - p)) & 1 for p in 1:L] for s in 0:(2^L - 1)
                          if !occursin("1"^k, string(s; base=2, pad=L))])
            @test brute == strings
            @test strings[end] == [(p % k == 0) ? 0 : 1 for p in 1:L]

            letters = [TensorBinding.kbonacci_bond_symbol(k, L, b) for b in 1:N]
            @test join(string.(letters)) == _kbonacci_word(k, N)
            @test letters == [TensorBinding.kbonacci_letters(k)[_kb_trailing_ones(d) + 1] for d in strings]
        end
        @test TensorBinding.kbonacci_letters(3) == [:A, :B, :C]
        @test_throws ArgumentError TensorBinding.kbonacci_digits(3, 13, 4)
        @test_throws ArgumentError TensorBinding.kbonacci_number(1, 3)
        @test_throws ArgumentError TensorBinding.kbonacci_number(3, -1)
        @test_throws ArgumentError TensorBinding.kbonacci_letters(1)
        @test_throws BoundsError TensorBinding.kbonacci_bond_symbol(3, 4, 14)
    end

    @testset "k = 2 reproduces the Fibonacci chain" begin
        for L in 2:7
            N = TensorBinding.fibonacci_site_count(L)
            @test TensorBinding.kbonacci_site_count(2, L) == N
            @test all(n -> TensorBinding.kbonacci_digits(2, n, L) ==
                           TensorBinding.fibonacci_zeckendorf_digits(n, L), 0:(N - 1))
        end
        for model in (:onsite, :hopping), boundary in (:open, :periodic)
            dense_kb = TensorBinding._dense_kbonacci_hamiltonian(
                2, 4; values=(1.2, 0.7), t=0.9, onsite=0.2, model, boundary,
            )
            dense_fib = TensorBinding._dense_fibonacci_hamiltonian(
                4; A=1.2, B=0.7, t=0.9, onsite=0.2, model, boundary,
            )
            @test dense_kb == dense_fib
            Hkb = TensorBinding.kbonacci_hamiltonian(
                2, 4; values=(1.2, 0.7), t=0.9, onsite=0.2, model, boundary,
            )
            Hfib = TensorBinding.fibonacci_hamiltonian(
                4; A=1.2, B=0.7, t=0.9, onsite=0.2, model, boundary,
            )
            @test Hkb.N == 8
            @test all(s -> dim(s) == 2, Hkb.sites)
            @test Hkb.scale ≈ Hfib.scale && Hkb.center ≈ Hfib.center
            @test maximum(abs.(_projected_kb_matrix(Hkb) .- dense_fib)) < 1e-11
        end
    end

    @testset "TN construction matches the dense oracle" begin
        for k in (3, 4), model in (:onsite, :hopping), boundary in (:open, :periodic)
            values = (1.2, 0.7, 0.4, 1.5)[1:k]
            parameters = (; values, t=0.9, onsite=0.2, model, boundary)
            Htn = TensorBinding.kbonacci_hamiltonian(k, 4; parameters...)
            Hdense = TensorBinding._dense_kbonacci_hamiltonian(k, 4; parameters...)
            @test Htn.N == TensorBinding.kbonacci_site_count(k, 4)
            @test Htn.N == (k == 3 ? 13 : 15)
            @test Htn.position_space isa TensorBinding.KBonacciPositionSpace
            @test Htn.position_space.k == k
            @test TensorBinding.ambient_dimension(Htn) == big(2)^4
            @test TensorBinding.ambient_dimension(Htn) isa BigInt
            @test length(Htn.sites) == 4 && all(s -> dim(s) == 2, Htn.sites)
            @test Htn.scale > 0
            matrix = _projected_kb_matrix(Htn)
            @test maximum(abs.(matrix .- Hdense)) < 1e-11
            @test norm(matrix - matrix') < 1e-11
            last_letter = _kb_trailing_ones(TensorBinding.kbonacci_digits(k, Htn.N - 1, 4)) + 1
            expected_wrap = boundary === :open ? 0.0 :
                (model === :onsite ? parameters.t : values[last_letter])
            @test matrix[end, 1] ≈ expected_wrap atol=1e-11
        end

        # k > L: every binary string is admissible and the chain has 2^L sites.
        Hall = TensorBinding.kbonacci_hamiltonian(5, 3; values=(1, 2, 3, 4, 5))
        @test Hall.N == 8
        @test maximum(abs.(_projected_kb_matrix(Hall) .-
                           TensorBinding._dense_kbonacci_hamiltonian(5, 3; values=(1, 2, 3, 4, 5)))) < 1e-11

        Hcomplex = TensorBinding.kbonacci_hamiltonian(
            3, 4; values=(1 + 0.2im, 2 - 0.1im, 0.5im), model=:hopping,
        )
        complex_matrix = _projected_kb_matrix(Hcomplex)
        @test norm(complex_matrix - complex_matrix') < 1e-11
        @test_throws ArgumentError TensorBinding.kbonacci_hamiltonian(
            3, 4; values=(1 + 1im, 2.0, 3.0), model=:onsite,
        )
        @test_throws ArgumentError TensorBinding.kbonacci_hamiltonian(1, 4; values=(1.0,))
        @test_throws ArgumentError TensorBinding.kbonacci_hamiltonian(3, 1; values=(1.0, 2.0, 3.0))
        @test_throws ArgumentError TensorBinding.kbonacci_hamiltonian(3, 4; values=(1.0, 2.0))
        @test_throws ArgumentError TensorBinding.kbonacci_hamiltonian(
            3, 4; values=(1.0, 2.0, 3.0), boundary=:twisted,
        )
        @test_throws ArgumentError TensorBinding.kbonacci_hamiltonian(
            3, 4; values=(1.0, 2.0, 3.0), model=:mixed,
        )
    end

    # Tribonacci hopping chain with the paper's t_A/t_B = t_B/t_C = 0.8, t_C = 1:
    # 24 physical sites inside a 32-state register.
    H = TensorBinding.kbonacci_hamiltonian(3, 5; values=(0.64, 0.8, 1.0))
    P = TensorBinding.physical_projector(H)
    @test H.N == 24
    @test real(tr(P)) ≈ H.N atol=1e-12
    @test norm(apply(P, P; cutoff=1e-13) - P) / norm(P) < 1e-12
    @test H.center == 0
    @test occursin("TBHamiltonian", sprint(show, H))

    @testset "projector-aware KPM" begin
        Ncheb = 8
        Tn, _, _ = TensorBinding.KPM_Tn(H, Ncheb; maxdim=100, cutoff=1e-12)
        @test real(tr(Tn[1])) ≈ H.N atol=1e-10
        @test norm(Tn[1] - P) < 1e-12

        dense = TensorBinding._dense_kbonacci_hamiltonian(3, 5; values=(0.64, 0.8, 1.0))
        decomposition = eigen(Hermitian(dense))
        # Hopping model, zero onsite, even N: the spectrum is exactly chiral.
        @test maximum(abs.(decomposition.values .+ reverse(decomposition.values))) < 1e-12
        dense_moments = _kb_dense_kpm_moments(decomposition, Ncheb, H.center, H.scale)
        @test maximum(abs.(real.(tr.(Tn[1:Ncheb])) .- dense_moments)) < 1e-8

        energies = collect(range(-1.9, 1.9; length=7))
        dos_tn = TensorBinding.get_dos_trace(H, Ncheb, energies; maxdim=100, cutoff=1e-12)
        dos_dense = [
            TensorBinding.get_ldos_from_mun(
                dense_moments, Ncheb, (energy - H.center) / H.scale,
            ) for energy in energies
        ]
        @test maximum(abs.(dos_tn .- dos_dense)) < 1e-8

        ldos_dense = zeros(length(energies), H.N)
        for site in 1:H.N
            moments = _kb_dense_kpm_moments(decomposition, Ncheb, H.center, H.scale; site)
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
        probe = 7
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
        Hletters = TensorBinding.get_Hamiltonian("kbonacci", (A=0.64, B=0.8, C=1.0); L=5, k=3)
        Hvalues = TensorBinding.get_Hamiltonian("kbonacci", (values=(0.64, 0.8, 1.0),); L=5, k=3)
        @test Hletters.N == H.N && Hvalues.N == H.N
        @test Hletters.position_space isa TensorBinding.KBonacciPositionSpace
        @test Hletters.position_space.k == 3
        @test maximum(abs.(_projected_kb_matrix(Hletters) .- _projected_kb_matrix(Hvalues))) < 1e-11
        @test_throws ArgumentError TensorBinding.get_Hamiltonian(
            "kbonacci", (A=0.64, B=0.8, C=1.0); L=5,
        )
        @test_throws ArgumentError TensorBinding.get_Hamiltonian(
            "kbonacci", (A=0.64, B=0.8); L=5, k=3,
        )
        @test_throws ArgumentError TensorBinding.get_Hamiltonian(
            "kbonacci", (A=0.64, B=0.8, C=1.0, values=(1, 2, 3)); L=5, k=3,
        )
        @test_throws ArgumentError TensorBinding.get_Hamiltonian(
            "kbonacci", (A=0.64, B=0.8, C=1.0, foo=1); L=5, k=3,
        )
        @test_throws ArgumentError TensorBinding.get_Hamiltonian(
            "kbonacci", (A=0.64, B=0.8, C=1.0); L=5, k=3, ref_sites=siteinds("Qubit", 5),
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
