using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Logging, Random
using TensorBinding: get_Hamiltonian, add_onsite!, build_K, get_matrix

# build_K's cutoff_K was a fixed 1e-8, relative to ‖K‖², so at small η the truncation
# dropped the η² I term of K = (H − ω)² + η² (η = 1e-2 on this chain: ‖ΔK‖ = 9.7e-5 ≈ η²).
# The default is now min(1e-8, 1e-4 η⁴): 1e-8 as before for η ≥ 0.1 (and η = 0), smaller
# below; an explicit cutoff_K ≥ η⁴ is used as given and warns.
@testset "build_K: cutoff_K default scales with η" begin
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
    @test TensorBinding._default_cutoff_K(0.05) ≈ 1e-4 * 0.05^4
end
