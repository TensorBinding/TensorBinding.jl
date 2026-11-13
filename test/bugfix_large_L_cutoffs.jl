using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Logging
using TensorBinding: get_Hamiltonian, add_onsite!, binary_to_MPS

# ITensors' cutoffs are relative to ‖H‖², which grows with the system: with the fixed 1e-8
# of v0.2.0 the 1D builders and the term adders dropped the bond across the middle of a
# chain (n = 2^(L-1) - 1 → 2^(L-1), whose carry flips every qubit) from L = 26 on, and
# get_Hamiltonian("custom") with nearest-neighbour hops threw (its self-check caught the
# missing bond). The default cutoff is now min(1e-8, 0.1/D), D the dimension of the site
# space; an explicit value is used as given and warns above 0.3/D.

const _TBc = TensorBinding
_el(H, a, b) = real(inner(binary_to_MPS(a, H.L, H.sites)', H.mpo, binary_to_MPS(b, H.L, H.sites)))
_middle(H) = _el(H, 2^(H.L - 1) - 1, 2^(H.L - 1))     # the bond across the middle
_wrap(H)   = _el(H, 0, 2^H.L - 1)                     # zero on an open chain

@testset "Size-aware truncation cutoff of the Hamiltonian builders" begin
    dc = _TBc._default_mpo_cutoff
    @test dc(2.0^23) == 1e-8 && dc(2.0^10) == 1e-8           # unchanged up to D = 2^23
    @test dc(2.0^24) ≈ 0.1 / 2.0^24
    @test _TBc._mpo_cutoff(nothing, 2.0^26, "f") ≈ 0.1 / 2.0^26
    @test (@test_logs min_level=Logging.Warn _TBc._mpo_cutoff(1e-8, 2.0^24, "f")) == 1e-8
    @test (@test_logs (:warn, r"single matrix element") _TBc._mpo_cutoff(1e-8, 2.0^26, "g")) == 1e-8

    L = 26
    H = @test_logs min_level=Logging.Warn get_Hamiltonian("chain_1d", 1.0; L)
    @test _middle(H) ≈ 1.0 atol=1e-8
    @test abs(_wrap(H)) < 1e-8
    Hold = @test_logs (:warn, r"single matrix element") get_Hamiltonian("chain_1d", 1.0; L, tol=1e-8)
    # what v0.2.0 built: at L = 26, 1e-8 removes one of the two (degenerate) directions of
    # the bond, which one being up to the SVD's rounding, so check both
    m = 2^(L - 1)
    @test max(abs(_el(Hold, m - 1, m) - 1), abs(_el(Hold, m, m - 1) - 1)) > 0.5

    add_onsite!(H, n -> 0.1 * cos(0.3n))                        # with 1e-8 it cut the chain again
    @test _middle(H) ≈ 1.0 atol=1e-8

    # the QTCI presets (hopping2MPO and their own sums) and "custom", which threw
    for (geom, p, kw) in (("uniform", 1.0, (;)),
                          ("aah", (V=0.5, phi=0.0, t=1.0), (;)),
                          ("custom", (i, j) -> abs(i - j) == 1 ? 1.0 : 0.0, (; scale=2.5)))
        @test abs(_middle(get_Hamiltonian(geom, p; L, kw...))) ≈ 1.0 atol=1e-6
    end

    # the non-Hermitian adders cut the chain again too
    Hn = get_Hamiltonian("chain_1d", 1.0; L)
    TensorBinding.add_nh_onsite!(Hn, 0.1im)
    @test _el(Hn, m - 1, m) ≈ 1.0 atol=1e-8
    @test _el(Hn, m, m - 1) ≈ 1.0 atol=1e-8

    # one explicit risky value, one warning (add_zeeman! used to warn under add_spin! too)
    Hz = get_Hamiltonian("chain_1d", 1.0; L)
    @test_logs (:warn, r"^add_zeeman!") TensorBinding.add_zeeman!(Hz, 0.1; tol=1e-8)

    # 2D: a long carry repeats along a whole row or column, so the cutoff scales with the
    # longer side and an explicit 1e-8 neither warns nor changes at L = 26
    @test_logs min_level=Logging.Warn get_Hamiltonian("square_2d", 1.0; L, Lx=13, tol=1e-8)
    @test TensorBinding._cutoff_dim(get_Hamiltonian("square_2d", 1.0; L=8, Lx=3)) == 2.0^5

    # kinetic_1d_nn's own sum: 1e-12 cut one direction of the bond at L = 39, both at 40.
    # At L = 40 (‖H‖ ≈ 1.5e6) the kept bond carries rounding noise of ~5e-9 here and
    # 1.7e-8 on Linux with Julia 1.10, so these checks use 1e-6; a cut bond is off by ≥ 0.5.
    H40 = get_Hamiltonian("chain_1d", 1.0; L=40)
    m40 = 2^39
    @test _el(H40, m40 - 1, m40) ≈ 1.0 atol=1e-6
    @test _el(H40, m40, m40 - 1) ≈ 1.0 atol=1e-6

    # a weak T-junction, a single-site term, is no longer truncated away (t_j = 0.02 at L = 16)
    Ht = TensorBinding.tjunction_hamiltonian(16, 1.0, 0.02)
    st(n, b) = MPS(Ht.sites, [[((n >> (16 - k)) & 1) + 1 for k in 1:16]; b])
    @test real(inner(st(0, 1)', Ht.mpo, st(0, 2))) ≈ 0.02 atol=1e-10
    # ... and without a junction the cap does not apply (a zero cap kept every noise vector)
    @test maxlinkdim(TensorBinding.tjunction_hamiltonian(10, 1.0, 0.0).mpo) <= 4

    # the hermitized block grows with |z|: its default scales with ‖H‖²/‖zI − H‖², so the
    # middle bond survives at z = 4 (0.92/0.08 at z = 3 and 1e-7 at z = 4 without it)
    Hc = get_Hamiltonian("chain_1d", 1.0; L)
    Hh = TensorBinding.hermitized_hamiltonian(Hc; z = 4.0)
    sb(n, b) = MPS(Hh.sites, [[((n >> (L - k)) & 1) + 1 for k in 1:L]; b])
    @test real(inner(sb(m - 1, 1)', Hh.mpo, sb(m, 2))) ≈ -1.0 atol=1e-8
    @test real(inner(sb(m, 1)', Hh.mpo, sb(m - 1, 2))) ≈ -1.0 atol=1e-8

    # pairingNNN summed at a fixed 1e-12, like kinetic_1d_nn
    s40 = H40.sites
    P40 = TensorBinding.pairingNNN(40, s40, 0.2 * MPO(s40, "Id"), 1)
    @test _el(TensorBinding.TBHamiltonian(H40; mpo = P40), m40 - 1, m40) ≈ 0.2 atol=1e-6
end
