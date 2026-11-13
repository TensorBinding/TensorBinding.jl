using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Logging, Random
using TensorBinding: get_Hamiltonian, add_onsite!, build_K, get_matrix

# build_K's cutoff_K was a fixed 1e-8, relative to ‖K‖², so at small η the truncation
# dropped the η² I term of K = (H − ω)² + η² (η = 1e-2 on this chain: ‖ΔK‖ = 9.7e-5 ≈ η²),
# and in large systems the terms that live on a few sites (the end bond of an open chain
# from L = 13). The default is now min(1e-8, 1e-4 η⁴, 0.1/D²): 1e-8 as before for η ≥ 0.1
# (and η = 0) up to L = 11, smaller below; an explicit cutoff_K ≥ η⁴ or > 0.3/D² is used
# as given and warns.
@testset "build_K: cutoff_K default scales with η and system size" begin
    H = get_Hamiltonian("chain_1d", 1.0; L=5)
    Random.seed!(11)
    add_onsite!(H, n -> 0.3 * cos(2π * 0.618 * n))
    Hd = get_matrix(H.mpo, H.sites)
    ω  = 0.05
    Kex(η) = (Hd - ω * I)^2 + η^2 * I
    Kd(η; kw...) = get_matrix(build_K(H.mpo, H.sites, ω, η; kw...), H.sites)

    η = 1e-2
    @test opnorm(Kd(η) - Kex(η)) < 1e-3 * η^2
    Kold = @test_logs (:warn, r"cutoff_K = 1.0e-8 is not below η⁴") Kd(η; cutoff_K=1e-8)
    @test opnorm(Kold - Kex(η)) > 0.5 * η^2          # the v0.2.0 default lost most of η²

    # η ≥ 0.1 and η = 0: the default is 1e-8, as before, bit for bit and without warning.
    for η in (0.1, 0.3, 0.0)
        @test (@test_logs min_level=Logging.Warn Kd(η)) == Kd(η; cutoff_K=1e-8)
    end
    dc = TensorBinding._default_cutoff_K
    @test dc(0.05, 2.0^5) ≈ 1e-4 * 0.05^4
    @test dc(0.1, 2.0^11) == 1e-8 == dc(0.0, 2.0^11)          # unchanged up to L = 11
    @test dc(0.1, 2.0^12) ≈ 0.1 / 4.0^12
    @test dc(0.0, 2.0^16) ≈ 0.1 / 4.0^16
    @test dc(0.01, 2.0^16) ≈ 1e-12                             # η bound below the size bound

    # Large systems: a cutoff above about 0.5·4^(-L) drops the end bond of an open chain
    # (<0|K|0> = 2.01 instead of 1.01 at L = 16 with 1e-8, the v0.2.0 default). The default
    # (here 0.1·4^(-16)) keeps it without a warning; an explicit 1e-8 warns and loses it.
    L16 = 16
    H16 = get_Hamiltonian("chain_1d", 1.0; L=L16)
    ψ0  = TensorBinding.binary_to_MPS(0, L16, H16.sites)
    K00(K) = real(inner(ψ0', K, ψ0))
    K16 = @test_logs min_level=Logging.Warn build_K(H16.mpo, H16.sites, ω, 0.1)
    @test abs(K00(K16) - (ω^2 + 1 + 0.1^2)) < 1e-8
    Kold = @test_logs (:warn, r"live on a few sites") build_K(H16.mpo, H16.sites, ω, 0.1;
                                                             cutoff_K = 1e-8)
    @test abs(K00(Kold) - (ω^2 + 1 + 0.1^2)) > 0.9             # the end bond is lost

    # ‖K‖ grows with ω while the end bond does not: the default scales with
    # r = min(1, ‖H‖²/‖H − ω‖²) (r ≈ 1/3 here). At ω = 2 a plain 0.1/D² lost the end bond
    # from L = 12 (<0|K|0> off by 1).
    @test dc(0.1, 2.0^14, 1/3) ≈ 0.1 / 9 / 4.0^14
    H14 = get_Hamiltonian("chain_1d", 1.0; L=14)
    ψ14 = TensorBinding.binary_to_MPS(0, 14, H14.sites)
    K14 = build_K(H14.mpo, H14.sites, 2.0, 0.1)
    @test abs(real(inner(ψ14', K14, ψ14)) - (1 + 2.0^2 + 0.1^2)) < 1e-6
end
