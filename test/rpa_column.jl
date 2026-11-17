using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Logging, Random, Test
using TensorBinding: get_Hamiltonian, get_bubble_mpo, get_rpa_susceptibility,
                     get_rpa_susceptibility_column, rpa_wynn_from_bubbles, custom_mpo,
                     _rpa_column_solve, _basis_state_mps

# One column of the RPA susceptibility, (I − Π₀V)x = Π₀|j⟩ on the L sites of Π₀, and the
# warning of the Wynn series near ρ(VΠ₀) = 1.
@testset "RPA: one column of χ; Wynn series near the instability" begin
    dmat(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
    dvec(ψ, s) = vec(Array(prod(ψ), s...))
    function mpo_of(A, s)
        d = dim.(s)
        return MPO(ITensor(reshape(A, d..., d...), prime.(s)..., s...), s; cutoff=1e-15)
    end

    # Exact Lindhard bubble of a half-filled 32-site chain, V = U·1, U = α U_c.
    H  = get_Hamiltonian("chain_1d", 1.0; L=5)
    s  = H.sites
    E, ψ = eigen(Hermitian(dmat(H.mpo, s)))
    D  = length(E)
    f  = Float64.(E .< 0)
    W  = reshape([ψ[i, a] * conj(ψ[i, b]) for i in 1:D, a in 1:D, b in 1:D], D, D * D)
    bubble(ω, η) = W * Diagonal(vec([(f[b] - f[a]) / (ω + im * η - E[b] + E[a])
                                     for a in 1:D, b in 1:D])) * W'
    μ  = eigvals(bubble(0.0, 0.03))
    Uc = 1 / real(μ[argmax(abs.(real.(μ)))])
    Πd = bubble(0.0, 0.03)
    Π  = mpo_of(Πd, s)
    j  = 17
    bj = _basis_state_mps(j - 1, s)
    for α in (0.3, 0.999)                        # condition number 1.4 and 10³
        V  = α * Uc * MPO(s, "Id")
        χx = (I - α * Uc * Πd) \ Πd
        col = χx * dvec(bj, s)
        r = @test_logs _rpa_column_solve(Π, V, s, bj; nsweeps=40, maxdim=400, cutoff=1e-12,
                                         residual_warn=1e-2, return_info=true)
        @test norm(dvec(r.chi, s) - col) / norm(col) < 1e-5
        @test r.residual < 1e-3
    end

    # Π₀|j⟩ = 0: x = 0 with residual 0, and no warning (it was 0.89, "not converged")
    r0 = @test_logs _rpa_column_solve(0.0 * Π, 0.3Uc * MPO(s, "Id"), s, bj; nsweeps=40,
                                      maxdim=400, cutoff=1e-12, residual_warn=1e-2,
                                      return_info=true)
    @test r0.residual == 0 && norm(r0.chi) == 0

    # The Wynn series warns where it converges slowly (‖Tₙ‖/‖Tₙ₋₁‖ ≈ ρ(VΠ₀) > 0.9).
    @test_logs rpa_wynn_from_bubbles([Π], 0.3Uc * MPO(s, "Id"); K_max=6)
    @test_logs (:warn, r"Wynn series") match_mode=:any rpa_wynn_from_bubbles(
        [Π], 0.999Uc * MPO(s, "Id"); K_max=6)

    # Through the bubble: column j of get_rpa_susceptibility's χ, on a 4-site chain with
    # the exact occupied-state projector cached (so the bubble itself is exact).
    Hc = get_Hamiltonian("chain_1d", 1.0; L=2, scale=2.2)
    F  = eigen(Hermitian(dmat(Hc.mpo, Hc.sites)))
    Hc._density_cache = mpo_of(F.vectors * Diagonal(F.values .< Hc.center) * F.vectors',
                               Hc.sites)
    sc = Hc.sites
    kw = (; GF_method=:krylov, η=0.2, krylov_nsweeps=10, krylov_maxdim=64,
            krylov_cutoff=1e-12, cutoff=1e-12, maxdim=64)
    V  = +(0.5 * MPO(sc, "Id"), 0.2 * MPO(sc, ["Z", "Id"]); cutoff=1e-14)
    χ  = dmat(custom_mpo(get_rpa_susceptibility(Hc, V, 0.3; kw..., rpa_nsweeps=6,
                                                rpa_maxdim=64, rpa_cutoff=1e-12), sc), sc)
    for jj in (1, 3)
        x = get_rpa_susceptibility_column(Hc, V, 0.3, jj; kw...)
        @test dvec(x, sc) ≈ χ * dvec(_basis_state_mps(jj - 1, sc), sc) rtol=1e-8
    end
    @test_throws BoundsError get_rpa_susceptibility_column(Hc, V, 0.3, 5; kw...)
    @test_throws ErrorException get_rpa_susceptibility_column(Hc, V, 0.3, 1; mode=:spin)
end
