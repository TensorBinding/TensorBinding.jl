using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, add_spin!, add_zeeman!, add_soc!, add_superconductivity!,
                     bdg_hamiltonian, bdg_spin_hamiltonian, pairingNNN, shift_mpo, nambu_index,
                     spin_index, constant_mps

# The BdG hole block in the Nambu basis (c↑, c↓, c†↑, c†↓) of the package (singlet pairing
# iσ_y ⊗ Δ) is −H*. Up to v0.2.0 every builder used τz ⊗ H, whose hole block −H is wrong
# whenever H has imaginary entries (a Zeeman field along y: the BdG spectrum was off by
# 1.4e-2 for h = 0.3, Δ = 0.25 on an L = 2 spinful chain).
#
# Dense matrices: rows on the primed indices, the first MPO site the most significant, so
# a prepended Nambu site gives the blocks [particle-particle  pair; pair'  hole-hole].

const _TB = TensorBinding
_dense(M::MPO) = _TB._mpo_dense_matrix(M)
const _iσy = ComplexF64[0 1; -1 0]

# Hermitian, purely imaginary hopping: 0.3i (K_u − K_d), a Peierls-like phase term.
_imag_hop(sites) = 0.3im * +(shift_mpo(sites, 1; cyclic=false), -1 * shift_mpo(sites, -1; cyclic=false); cutoff=1e-14)

@testset "BdG hole block is −H*" begin
    Δ = 0.25

    @testset "add_superconductivity! with a Zeeman field along $dir" for dir in (:x, :y, :z)
        H = get_Hamiltonian("chain_1d", 1.0; L=2, scale=4.0)
        add_spin!(H)
        add_zeeman!(H, 0.3; direction=dir)
        H0 = _dense(H.mpo)                                 # normal state on [spin, pos…]
        add_superconductivity!(H, Δ)
        @test H.sites[1] == H.nambu_s
        D = Δ * kron(_iσy, Matrix{ComplexF64}(I, size(H0, 1) ÷ 2, size(H0, 1) ÷ 2))
        B = _dense(H.mpo)
        @test norm(B - [H0 D; D' -conj(H0)]) < 1e-10
        @test norm(B - B') < 1e-10
        if dir === :y   # the v0.2.0 model, τz ⊗ H, is visibly different
            @test norm(B - [H0 D; D' -H0]) > 0.1
        end
    end

    # A term added after the BdG doubling enters as it would before it.
    @testset "$what added before or after add_superconductivity!" for what in (:zeeman_y, :zeeman_z, :soc_custom_y)
        add_term!(H) = what === :zeeman_y ? add_zeeman!(H, 0.3; direction=:y) :
                       what === :zeeman_z ? add_zeeman!(H, 0.3; direction=:z) :
                                            add_soc!(H, 0.2; type=:custom, direction=:y)
        Hb = get_Hamiltonian("chain_1d", 1.0; L=2, scale=4.0); add_spin!(Hb)
        add_term!(Hb); add_superconductivity!(Hb, Δ)
        Ha = get_Hamiltonian("chain_1d", 1.0; L=2, scale=4.0); add_spin!(Ha)
        add_superconductivity!(Ha, Δ); add_term!(Ha)
        @test norm(_dense(Ha.mpo) - _dense(Hb.mpo)) < 1e-10
    end

    @testset "bdg_hamiltonian with a complex H_kin" begin
        Hc = get_Hamiltonian("chain_1d", 1.0; L=2)
        s  = Hc.sites
        Hk = +(Hc.mpo, _imag_hop(s); cutoff=1e-14)
        P  = pairingNNN(2, s, Δ * MPO(s, "Id"), 1)
        nb = nambu_index()
        K, Pd = _dense(Hk), _dense(P)
        @test norm(imag(K)) > 0.1
        @test norm(_dense(bdg_hamiltonian(Hk, P, nb)) - [K Pd; Pd' -conj(K)]) < 1e-10
    end

    @testset "bdg_spin_hamiltonian with complex H_kin" begin
        s  = siteinds("Qubit", 2)
        Hu = +(0.2 * MPO(s, "Id"), _imag_hop(s); cutoff=1e-14)
        Hd = +(-0.1 * MPO(s, "Id"), 2 * _imag_hop(s); cutoff=1e-14)
        P  = Δ * MPO(s, "Id")
        sp, nb = spin_index(), nambu_index()
        U, Dn, Pd = _dense(Hu), _dense(Hd), _dense(P)
        Pup, Pdn = ComplexF64[1 0; 0 0], ComplexF64[0 0; 0 1]
        Hn = kron(Pup, U) + kron(Pdn, Dn)
        Dp = kron(_iσy, Pd)
        @test norm(_dense(bdg_spin_hamiltonian(Hu, Hd, P, sp, nb)) - [Hn Dp; Dp' -conj(Hn)]) < 1e-10
    end

    @testset "equal-spin triplet BdG (SCF) with a complex H0" begin
        H0 = get_Hamiltonian("chain_1d", 1.0; L=2, scale=3.0)
        H0.mpo = +(H0.mpo, _imag_hop(H0.sites); cutoff=1e-14)
        du = constant_mps(H0.sites, 0.1)
        dd = constant_mps(H0.sites, 0.05)
        Hb = _TB._triplet_equalspin_bdg(H0, du, dd; distance=1, mu=0.1, scale=3.0)
        pos = Hb.sites[3:end]
        @test pos == H0.sites                              # [nambu, spin, pos…]
        K  = _dense(+(H0.mpo, -0.1 * MPO(pos, "Id"); cutoff=1e-14))
        Pu = _dense(_TB._pwave_bond_mpo(du, pos, 1))
        Pd = _dense(_TB._pwave_bond_mpo(dd, pos, 1))
        Pup, Pdn = ComplexF64[1 0; 0 0], ComplexF64[0 0; 0 1]
        Hn = kron(Pup, K) + kron(Pdn, K)
        Dp = kron(Pup, Pu) + kron(Pdn, Pd)
        @test norm(_dense(Hb.mpo) - [Hn Dp; Dp' -conj(Hn)]) < 1e-8
    end

    # Real Hamiltonians keep the one-term τz ⊗ H build, so their MPOs are unchanged
    # (exactly equal dense matrices; the two-term build would differ by rounding).
    @testset "real H: τz ⊗ H" begin
        Hc = get_Hamiltonian("chain_1d", 1.0; L=2)
        M  = Hc.mpo
        @test _TB._mpo_is_real(M)
        @test !_TB._mpo_is_real(_imag_hop(Hc.sites))
        nb = nambu_index()
        @test _dense(_TB._nambu_normal_block(M, nb, :pre; cutoff=1e-8)) ==
              _dense(_TB.prepend_nambu(M, nb, :tz))
        Mc = +(M, _imag_hop(Hc.sites); cutoff=1e-14)
        N  = _dense(_TB._nambu_normal_block(Mc, nb, :post; cutoff=1e-12))   # Nambu index last
        K  = _dense(Mc)
        @test norm(N - kron(K, ComplexF64[1 0; 0 0]) - kron(-conj(K), ComplexF64[0 0; 0 1])) < 1e-10
    end
end
