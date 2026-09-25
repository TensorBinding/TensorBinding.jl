using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, add_interaction!, get_scf, hermitize,
                     exciton_hamiltonian, bilayer_hamiltonian, add_hopping_2D!,
                     _bdg_from_pairing, _triplet_equalspin_bdg, _pairing_profile_mps

# These builders copied a Hamiltonian with the positional compatibility
# constructors, which dropped Lx, interaction_mpo, fock_mpo and position_space.
@testset "Lossless Hamiltonian copies in SCF, NH, exciton" begin
    H0 = get_Hamiltonian("square_2d", 1.0; L=4, scale=4.5)
    add_interaction!(H0, 1.0)
    @test H0.Lx == 2

    # Mean-field copy of H0 (scf_meanfield via get_scf) keeps Lx and the kernel
    res = get_scf(H0, :cdw; density_method=:mcweeny, scale=5.0, maxiters=1,
                  purif_maxiter=20, purif_tol=1e-4, tol=1e-3, maxdim=40,
                  verbose=false)
    @test res.ham.Lx == H0.Lx
    @test res.ham.interaction_mpo === H0.interaction_mpo

    # BdG builders: same position register, so Lx and the kernel carry over
    delta = _pairing_profile_mps(0.1, H0.L, H0.sites)
    for Hbdg in (_bdg_from_pairing(H0, delta),
                 _triplet_equalspin_bdg(H0, delta, delta))
        @test Hbdg.Lx == H0.Lx
        @test Hbdg.interaction_mpo isa MPO
        @test Hbdg.nambu_s !== nothing && Hbdg.spin_s !== nothing
    end

    # Hermitized NH block keeps Lx; the interaction stays on NH.parent
    NH = hermitize(H0)
    @test NH.hermitized.Lx == H0.Lx
    @test NH.hermitized.interaction_mpo === nothing

    # Exciton Hamiltonian keeps the per-sector Lx, not the one-body kernel
    H_c = get_Hamiltonian("square_2d", 1.0; L=4)
    H_v = get_Hamiltonian("square_2d", 1.0; L=4)
    add_interaction!(H_c, 1.0)
    Hx = exciton_hamiltonian(H_c, H_v, x -> 1.0)
    @test Hx.Lx == H_c.Lx
    @test Hx.interaction_mpo === nothing
    @test length(Hx.sites) == 2 * H_c.L

    # Layered add_hopping_2D! builds its per-layer term through a temporary copy
    Hb = bilayer_hamiltonian(:square, 2, 2)
    mpo_before = copy(Hb.mpo)
    add_hopping_2D!(Hb, 0.1; Lx=2, Ly=2, nn=2, lattice=:square)
    @test norm(Hb.mpo - mpo_before) > 0
end
