using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, honeycomb_positions, haldane_hoppingf, get_matrix,
                     bilayer_hamiltonian, add_onsite!

@testset "Haldane preset is Hermitian" begin
    # get_Hamiltonian("haldane") used to throw (haldane_hoppingf was deleted); restored with
    # the (-1)^i sublattice sign it would be non-Hermitian on honeycomb_positions' layout.
    p = (t2=0.2, phi=0.7, M=0.3)
    for (L, Lx) in ((4, 2), (5, 3))
        rs = honeycomb_positions(L; Lx=Lx)
        H  = get_Hamiltonian("haldane", p; L=L, rs=rs)
        @test norm(H.mpo - swapprime(dag(H.mpo), 0, 1)) / norm(H.mpo) < 1e-10
        A = [haldane_hoppingf(rs[i, :], rs[j, :], i, j; p...) for i in 1:H.N, j in 1:H.N]
        @test get_matrix(H.mpo, H.sites) ≈ A atol=1e-8
        # Semenoff mass: opposite sign across every nearest-neighbour bond
        nn = [(i, j) for i in 1:H.N, j in 1:H.N if A[i, j] ≈ -1]
        @test !isempty(nn) && all(A[i, i] ≈ -A[j, j] for (i, j) in nn)
    end

    # Layered add_onsite! builds its per-layer term on a keyword copy of H
    Hb = bilayer_hamiltonian(:honeycomb, 1, 1; sublattice=true)
    M0 = get_matrix(Hb.mpo, Hb.sites)
    add_onsite!(Hb, 0.7; layer=1, sublat=1)
    dM = get_matrix(Hb.mpo, Hb.sites) - M0
    @test dM ≈ Diagonal(diag(dM)) atol=1e-10
    @test count(d -> isapprox(d, 0.7; atol=1e-8), diag(dM)) == 2^Hb.L
    @test count(d -> abs(d) < 1e-8, diag(dM)) == size(dM, 1) - 2^Hb.L
end
