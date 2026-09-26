using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, add_spin!, add_interaction!, _project_spin_sector,
                     get_rpa_susceptibility_wynn, get_magnon_susceptibility_wynn,
                     get_bubble_mpo_haydock

@testset "RPA Wynn drivers, Haydock bubble and spin projection" begin
    # Two frequencies: the second one used to hit an undefined `nq` in both drivers.
    ωs = [0.3, 0.7]
    kw = (; P_method=:kpm, Ncheb=10, maxdim=20, η=0.1, K_max=2, maxdim_apply=20)

    # Charge channel: get_bubble_mpo(H, H, ω), so both registers start on H.sites.
    H = get_Hamiltonian("chain_1d", 1.0; L=3, scale=2.2)
    cp, cw = get_rpa_susceptibility_wynn(H, 0.5 * MPO(H.sites, "Id"), ωs; kw...)
    @test size(cp) == (3, 2, H.N)
    @test size(cw) == (1, 2, H.N)
    @test all(isfinite, cp) && !iszero(cp[:, 2, :])

    # Transverse spin channel, through both drivers.
    Hs = get_Hamiltonian("chain_1d", 1.0; L=3, scale=2.2)
    add_spin!(Hs)
    V  = 0.5 * MPO(_project_spin_sector(Hs, 1).sites, "Id")
    cm, _ = get_magnon_susceptibility_wynn(Hs, V, ωs; kw...)
    cr, _ = get_rpa_susceptibility_wynn(Hs, V, ωs; mode=:magnetic, kw...)
    @test size(cm) == size(cr) == (3, 2, Hs.N)
    @test all(isfinite, cm) && all(isfinite, cr)

    # Spin projection keeps the fields the positional constructor used to drop.
    H2 = get_Hamiltonian("square_2d", 1.0; L=4)
    add_spin!(H2)
    add_interaction!(H2, 2.0)
    Hup = _project_spin_sector(H2, 1)
    @test Hup.spin_s === nothing && length(Hup.sites) == H2.L
    @test Hup.Lx == H2.Lx
    @test Hup.interaction_mpo === H2.interaction_mpo
    @test Hup.position_space === H2.position_space

    # Haydock bubble: one MPO per frequency, on H1.sites.
    bubbles = get_bubble_mpo_haydock(H, H, ωs; N_steps=4, P_method=:kpm, Ncheb=10,
                                     maxdim=20, η=0.1)
    @test length(bubbles) == 2
    @test all(b -> all(n -> hasind(b[n], H.sites[n]), 1:H.L), bubbles)
end
