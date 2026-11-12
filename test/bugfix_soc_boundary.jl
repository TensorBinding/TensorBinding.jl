using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Logging, Test
using TensorBinding: get_Hamiltonian, add_soc!, get_matrix

# add_soc!(type=:rashba) always built the periodic chain term (the cyclic ±1 shifts of
# generate_kin_u/d), so on an open chain, the default of get_Hamiltonian("chain_1d"), it
# added a Rashba bond between the last and the first site that the kinetic term does not
# have. boundary=:open drops that bond; the default :periodic keeps the v0.2.0 term.
# The dense references are λ·kron(iσ_y, K_u − K_u†): the spin is prepended
# (H.sites = [spin, pos…]) and get_matrix is big-endian, so kron(spin, position).

@testset "add_soc! Rashba boundary" begin
    L, λ = 3, 0.1
    N    = 2^L
    iσy  = ComplexF64[0 1; -1 0]                        # the :iSy spin factor
    Ku_open = zeros(ComplexF64, N, N)
    for n in 1:N-1
        Ku_open[n + 1, n] = 1                           # |n+1⟩⟨n|
    end
    Ku_per = copy(Ku_open)
    Ku_per[1, N] = 1                                    # |1⟩⟨N|, the wrap bond
    rashba(K) = λ * kron(iσy, K - K')

    chain() = get_Hamiltonian("chain_1d", 1.0; L=L)    # open chain
    dense(H) = get_matrix(H.mpo, H.sites)
    Hc = chain()
    K0 = kron(Matrix{ComplexF64}(I, 2, 2), dense(Hc))  # kinetic term on [spin, pos…]

    Ho = chain(); add_soc!(Ho, λ; boundary=:open)
    Hp = chain(); add_soc!(Hp, λ; boundary=:periodic)
    Hd = chain(); add_soc!(Hd, λ)                       # default
    @test Ho.sites[1] == Ho.spin_s                      # the layout the references assume

    Mo, Mp, Md = dense(Ho), dense(Hp), dense(Hd)
    @test Mo ≈ K0 + rashba(Ku_open) atol=1e-10
    @test Mp ≈ K0 + rashba(Ku_per)  atol=1e-10
    @test Md == Mp                                      # the default is the v0.2.0 term
    @test Mo ≈ Mo' atol=1e-12                           # Hermitian
    # The two boundaries differ only on the wrap bond (positions N ↔ 1, every spin block);
    # elsewhere only by round-off, as the two MPO sums are truncated differently.
    wrap = falses(2N, 2N)
    for σ in 0:1, σ′ in 0:1
        wrap[σ * N + 1, σ′ * N + N] = true
        wrap[σ * N + N, σ′ * N + 1] = true
    end
    @test norm((Mp - Mo)[.!wrap]) < 1e-10
    @test norm((Mp - Mo)[wrap]) ≈ 2λ                    # λ on each of the 4 wrap entries

    # Aliases, and an unsupported boundary is an error for every type, before any change.
    Hobc = chain(); add_soc!(Hobc, λ; boundary=:obc)
    @test dense(Hobc) == Mo
    Hbad = chain()
    @test_throws ErrorException add_soc!(Hbad, λ; boundary=:twisted)
    @test_throws ErrorException add_soc!(Hbad, λ; type=:ising, boundary=:twisted)
    @test Hbad.spin_s === nothing

    # 2D: the term runs along the linear position index only, and says so.
    H2 = get_Hamiltonian("square_2d", 1.0; L=2)
    @test H2.Lx == 1
    @test_logs (:warn, r"2D Hamiltonian") add_soc!(H2, λ)
    @test_logs min_level=Logging.Warn add_soc!(chain(), λ; boundary=:open)
end
