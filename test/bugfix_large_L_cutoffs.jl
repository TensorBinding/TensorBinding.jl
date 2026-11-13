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
    @test abs(_middle(Hold) - 1.0) > 0.5                      # what v0.2.0 built

    add_onsite!(H, n -> 0.1 * cos(0.3n))                        # with 1e-8 it cut the chain again
    @test _middle(H) ≈ 1.0 atol=1e-8

    # the QTCI presets (hopping2MPO and their own sums) and "custom", which threw
    for (geom, p, kw) in (("uniform", 1.0, (;)),
                          ("aah", (V=0.5, phi=0.0, t=1.0), (;)),
                          ("custom", (i, j) -> abs(i - j) == 1 ? 1.0 : 0.0, (; scale=2.5)))
        @test abs(_middle(get_Hamiltonian(geom, p; L, kw...))) ≈ 1.0 atol=1e-6
    end
end
