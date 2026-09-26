using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, add_spin!, _project_spin_sector,
                     get_rpa_susceptibility, get_magnon_susceptibility

@testset "RPA Dyson drivers on spin-projected and sublattice sites" begin
    kw = (; P_method=:kpm, Ncheb=8, maxdim=20, η=0.1, rpa_nsweeps=2, rpa_maxdim=20)
    dense(χ) = array(prod(χ), siteinds(χ)...)

    # mode=:magnetic: Π and MPOV live on the 3 position qubits, not the 4 spinful
    # H.sites the Dyson solve used to be handed.
    Hs = get_Hamiltonian("chain_1d", 1.0; L=3, scale=2.2)
    add_spin!(Hs)
    V  = 0.5 * MPO(_project_spin_sector(Hs, 1).sites, "Id")
    χr = get_rpa_susceptibility(Hs, V, 0.3; mode=:magnetic, kw...)
    χm = get_magnon_susceptibility(Hs, V, 0.3; kw...)
    @test length(χr) == 2 * Hs.L
    @test all(isfinite, dense(χr)) && norm(χr) > 0
    # Same bubble get_bubble_mpo(H_↑, H_↓, ω) and the same Dyson solve.
    @test dense(χr) ≈ dense(χm) rtol=1e-6

    # Kagome: the dim-3 sublattice index is a site beyond H.L (the magnon driver used
    # 2·H.L sites) and not a qubit (both drivers used siteinds("Qubit", …)).
    Hk = get_Hamiltonian("kagome", 1.0; L=2, Lx=1, Ly=1)
    χc = get_rpa_susceptibility(Hk, 0.5 * MPO(Hk.sites, "Id"), 0.3; kw...)
    @test dim.(siteinds(χc)) == [2, 2, 2, 2, 3, 3]
    @test all(isfinite, dense(χc))
    add_spin!(Hk)
    Vk  = 0.5 * MPO(_project_spin_sector(Hk, 1).sites, "Id")
    χkm = get_magnon_susceptibility(Hk, Vk, 0.3; kw...)
    χkr = get_rpa_susceptibility(Hk, Vk, 0.3; mode=:magnetic, kw...)
    @test dim.(siteinds(χkm)) == dim.(siteinds(χkr)) == [2, 2, 2, 2, 3, 3]
    @test all(isfinite, dense(χkm)) && all(isfinite, dense(χkr))
end
