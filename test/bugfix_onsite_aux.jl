using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, add_onsite!, add_spin!, add_superconductivity!, add_zeeman!

# add_onsite! threw on a Hamiltonian with a spin or Nambu index (v0.2.0): the potential was
# built on the position (and sublattice) sites only. It now acts as the identity on the
# spin and as τ_z ⊗ V on the Nambu index, so adding it after add_spin! or
# add_superconductivity! gives what adding it before did.

const _TBo = TensorBinding
_dense(H) = _TBo._mpo_dense_matrix(H.mpo)

@testset "add_onsite! on spinful and BdG Hamiltonians" begin
    L = 3; N = 2^L
    f(n) = 0.1 * n - 0.3
    chain() = get_Hamiltonian("chain_1d", 1.0; L)

    # spinful, with the spin in front (the default) and behind
    for pos in (:pre, :post)
        Hb = chain(); add_onsite!(Hb, f); add_spin!(Hb; position=pos)
        Ha = chain(); add_spin!(Ha; position=pos); add_onsite!(Ha, f)
        @test _dense(Ha) ≈ _dense(Hb) atol=1e-10
    end
    Hs = chain(); add_spin!(Hs); add_onsite!(Hs, f)          # spin = most significant site
    @test _dense(Hs) ≈ kron(Matrix(1.0I, 2, 2), _dense(chain()) + Diagonal(f.(0:N-1))) atol=1e-10

    # spinful BdG with a Zeeman field along y (complex), the potential after the doubling
    function bdg(onsite_first)
        H = chain()
        onsite_first && add_onsite!(H, f)
        add_zeeman!(H, 0.3; direction=:y)
        add_superconductivity!(H, 0.25)
        onsite_first || add_onsite!(H, f)
        return H
    end
    @test _dense(bdg(false)) ≈ _dense(bdg(true)) atol=1e-10

    # spinless BdG (Kitaev chain)
    Hk1 = chain(); add_onsite!(Hk1, f); add_superconductivity!(Hk1, 0.2; type=:pwave)
    Hk2 = chain(); add_superconductivity!(Hk2, 0.2; type=:pwave); add_onsite!(Hk2, f)
    @test _dense(Hk2) ≈ _dense(Hk1) atol=1e-10
    # a complex on-site term is not Hermitian, and the BdG hole block needs a Hermitian H
    @test_throws ErrorException add_onsite!(Hk2, fill(0.1im, N))
    @test_throws ErrorException add_onsite!(Hk2, 0.1im)              # not an InexactError
    @test_throws ErrorException add_onsite!(Hk2, n -> 0.1im * (n + 1))

    # sublattice and spin: SSH, the potential on sublattice 1
    Hs1 = get_Hamiltonian("ssh_sublattice", (t=1.0, d=0.2); L=2)
    Hs2 = deepcopy(Hs1)
    add_onsite!(Hs1, n -> 0.1 * n; sublat=1); add_spin!(Hs1)
    add_spin!(Hs2); add_onsite!(Hs2, n -> 0.1 * n; sublat=1)
    @test _dense(Hs2) ≈ _dense(Hs1) atol=1e-10
end
