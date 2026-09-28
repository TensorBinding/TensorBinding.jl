using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, honeycomb_sublattice_hamiltonian, bilayer_hamiltonian,
                     add_spin!, add_zeeman!, add_soc!, add_superconductivity!, get_matrix

# add_zeeman!, add_soc! and add_superconductivity! built their terms as
# [spin, (nambu,) positions] on the side H.aux_side, so they failed on a Hamiltonian with
# a layer or sublattice index (the MPO sum threw on mismatched sites), and the pairing
# was lifted to the spin on the side of the Nambu index, not of the spin. The terms are
# now lifted to the site layout of H.sites (_lift_to_aux_sites). The dense checks use
# spectra, which do not depend on the order of the sites.

spectrum5(H) = sort(real(eigvals(Hermitian(Matrix(get_matrix(H.mpo, H.sites))))))

@testset "Zeeman and SOC terms on sublattice and layered Hamiltonians" begin
    h = 0.3
    for build in (() -> honeycomb_sublattice_hamiltonian(1, 1, 1.0),        # [pos…, sublat]
                  () -> bilayer_hamiltonian(:square, 1, 1; t_inter=0.4))    # [layer, pos…]
        H0 = build()
        E0 = spectrum5(H0)
        H  = build()
        add_zeeman!(H, h)                                  # spin on the side of H.aux_side
        @test length(H.sites) == length(H0.sites) + 1
        @test spectrum5(H) ≈ sort([E0 .+ h / 2; E0 .- h / 2]) atol=1e-10
        # Ising SOC λ S_z adds to the Zeeman field on the same spin site.
        add_soc!(H, 0.1; type=:ising)
        @test spectrum5(H) ≈ sort([E0 .+ (h + 0.1) / 2; E0 .- (h + 0.1) / 2]) atol=1e-10
    end
end

@testset "Pairing lifted to the spin on the spin's own side" begin
    Δ = 0.25
    spec(spin_pos, nambu_pos) = begin
        H = get_Hamiltonian("chain_1d", 1.0; L=2, scale=3.0)
        add_spin!(H; position=spin_pos)
        add_superconductivity!(H, Δ; position=nambu_pos)   # s-wave singlet
        spectrum5(H)
    end
    ref = spec(:pre, :pre)                                 # [nambu, spin, pos…]
    @test spec(:post, :post) ≈ ref atol=1e-10              # [pos…, spin, nambu]
    @test spec(:post, :pre) ≈ ref atol=1e-10               # [nambu, pos…, spin]: threw
    @test spec(:pre, :post) ≈ ref atol=1e-10               # [spin, pos…, nambu]: threw
    # s-wave BdG of a chain: ±sqrt(ε² + Δ²), each twice (spin).
    Hc = get_Hamiltonian("chain_1d", 1.0; L=2)
    ε  = eigvals(Hermitian(Matrix(get_matrix(Hc.mpo, Hc.sites))))
    E  = sqrt.(real(ε) .^ 2 .+ Δ^2)
    @test ref ≈ sort([E; E; -E; -E]) atol=1e-10
end

@testset "Twisted bilayer: complex t_inter is Hermitian" begin
    # V_lk was the transpose of V_kl (the adjoint only for a real t_inter), compressed as
    # Float64 (a complex t_inter threw an InexactError).
    H = TensorBinding.twisted_bilayer_hamiltonian(:square, 1, 1, 10.0;
                                                  t_inter=0.3 * cis(0.7), tol=1e-10)
    A = Matrix(get_matrix(H.mpo, H.sites))
    @test norm(A - A') < 1e-8 * norm(A)
    @test norm(imag(A)) > 1e-3                 # the phase of t_inter is there
end
