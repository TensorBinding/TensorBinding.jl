using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: hopping2MPO, qtci_matrix_to_MPO, pairing2MPO, get_matrix,
                     get_Hamiltonian, build_tdvp_propagator_mpo

@testset "QTCI initial pivots are passed positionally" begin
    # hopping2MPO / qtci_matrix_to_MPO passed `initialpivots` as a keyword, which QuanticsTCI
    # forwards to TCI.optimize! -> MethodError whenever initial_positions was non-empty.
    L = 3; N = 2^L
    sites = siteinds("Qubit", L)
    f(i, j) = abs(i - j) == 1 ? -1.0 : (i == j ? 0.3 * i : 0.0)
    A  = [f(i, j) for i in 1:N, j in 1:N]
    M0 = get_matrix(hopping2MPO(f, N, sites), sites)
    @test M0 ≈ A atol=1e-8
    for piv in ([(i, i) for i in 1:N], [[i, i + 1] for i in 1:N-1])   # tuples and vectors
        @test get_matrix(hopping2MPO(f, N, sites; initial_positions=piv), sites) ≈ M0 atol=1e-8
        @test get_matrix(qtci_matrix_to_MPO(f, L, sites; initial_positions=piv), sites) ≈ M0 atol=1e-8
    end
    Mf = hopping2MPO(f, N, sites; unfoldingscheme=:fused, initial_positions=[(1, 2), (N, N)])
    @test get_matrix(Mf, sites) ≈ A atol=1e-8

    g(i, j) = abs(i - j) == 1 ? 0.4im * sign(j - i) : 0.0im
    P = pairing2MPO(g, N, sites; initial_positions=[(i, i + 1) for i in 1:N-1])
    @test get_matrix(P, sites) ≈ [g(i, j) for i in 1:N, j in 1:N] atol=1e-8

    # build_tdvp_propagator_mpo seeds the N diagonal pivots by default
    for Lp in (3, 4)
        H  = get_Hamiltonian("chain_1d", 1.0; L=Lp)
        U  = build_tdvp_propagator_mpo(H, 0.05)
        @test U isa MPO && length(U) == Lp
        U0 = build_tdvp_propagator_mpo(H, 0.05; use_diagonal_pivots=false)
        @test get_matrix(U, H.sites) ≈ get_matrix(U0, H.sites) atol=1e-3
    end
end
