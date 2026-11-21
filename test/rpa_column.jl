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
        @test norm(dvec(r.chi, s) - col) / norm(col) < 1e-9
        @test r.residual < 1e-3
    end

    # Π₀|j⟩ = 0: x = 0 with residual 0, and no warning (it was 0.2–0.9, "not converged"),
    # for Π₀ = 0 and for a Π₀ that annihilates |j⟩, where Π₀|j⟩ comes out at ~1e-15
    ej = dvec(bj, s)
    for Πz in (0.0 * Π, mpo_of(Πd * (I - ej * ej'), s))
        r0 = @test_logs _rpa_column_solve(Πz, 0.3Uc * MPO(s, "Id"), s, bj; nsweeps=40,
                                          maxdim=400, cutoff=1e-12, residual_warn=1e-2,
                                          return_info=true)
        @test r0.residual == 0 && norm(r0.chi) == 0
    end

    # The operator Π₀V is built at min(cutoff, 1e-14): at the solve cutoff 1e-12 it was
    # off by √cutoff, which limited the column to 4e-7 here (128 sites, U = 0.999 U_c;
    # the 32-site case above does not see it).
    let H7 = get_Hamiltonian("chain_1d", 1.0; L=7)
        s7 = H7.sites
        E7, ψ7 = eigen(Hermitian(ITensors.@disable_warn_order dmat(H7.mpo, s7)))
        D7 = length(E7)
        f7 = Float64.(E7 .< 0)
        W7 = reshape([ψ7[i, a] * conj(ψ7[i, b]) for i in 1:D7, a in 1:D7, b in 1:D7], D7, D7^2)
        Π7d = W7 * Diagonal(vec([(f7[b] - f7[a]) / (0.03im - E7[b] + E7[a])
                                 for a in 1:D7, b in 1:D7])) * W7'
        μ7  = eigvals(Π7d)
        U7  = 0.999 / real(μ7[argmax(abs.(real.(μ7)))])
        b7  = _basis_state_mps(16, s7)
        col = (I - U7 * Π7d) \ (Π7d * (ITensors.@disable_warn_order dvec(b7, s7)))
        r7  = _rpa_column_solve((ITensors.@disable_warn_order mpo_of(Π7d, s7)),
                                U7 * MPO(s7, "Id"), s7, b7; nsweeps=40, maxdim=400,
                                cutoff=1e-12, residual_warn=1e-2, return_info=true)
        @test norm((ITensors.@disable_warn_order dvec(r7.chi, s7)) - col) / norm(col) < 1e-7
    end

    # The Wynn series warns where it converges slowly: (‖T_K‖/‖T_{K−2}‖)^½ estimates
    # ρ(VΠ₀) from below, 1–6 % low at K = 6 and 25–40 % low at K = 2, with a threshold for
    # each. K_max < 2 has no ratio to take, and no check.
    @test_logs rpa_wynn_from_bubbles([Π], 0.3Uc * MPO(s, "Id"); K_max=6)
    @test_logs (:warn, r"Wynn series") match_mode=:any rpa_wynn_from_bubbles(
        [Π], 0.999Uc * MPO(s, "Id"); K_max=6)
    for K in (2, 3, 4, 5)
        @test_logs rpa_wynn_from_bubbles([Π], 0.3Uc * MPO(s, "Id"); K_max=K)
        @test_logs (:warn, r"Wynn series") match_mode=:any rpa_wynn_from_bubbles(
            [Π], 0.999Uc * MPO(s, "Id"); K_max=K)
    end
    @test_logs rpa_wynn_from_bubbles([Π], 0.999Uc * MPO(s, "Id"); K_max=1)

    # Through the bubble: column j of get_rpa_susceptibility's χ, on a 4-site chain with
    # the exact occupied-state projector cached (so the bubble itself is exact).
    Hc = get_Hamiltonian("chain_1d", 1.0; L=2, scale=2.2)
    F  = eigen(Hermitian(dmat(Hc.mpo, Hc.sites)))
    Hc._density_cache = mpo_of(F.vectors * Diagonal(F.values .< Hc.center) * F.vectors',
                               Hc.sites)
    sc = Hc.sites
    kw = (; GF_method=:krylov, η=0.2, krylov_nsweeps=10, krylov_maxdim=64,
            krylov_cutoff=1e-12, cutoff=1e-12, maxdim=64)
    # A non-symmetric V, so that χ ≠ χᵀ and a column is told from a row (with a diagonal V
    # and the symmetric Π₀ of a real chain, χ is symmetric and a transposed solve passes).
    V  = mpo_of(0.5I + 0.2 * [0 1 0 0; 0 0 0 0; 0.3 0 0 1; 0 0 0 0], sc)
    χ  = dmat(custom_mpo(get_rpa_susceptibility(Hc, V, 0.3; kw..., rpa_nsweeps=6,
                                                rpa_maxdim=64, rpa_cutoff=1e-12), sc), sc)
    @test norm(χ - transpose(χ)) / norm(χ) > 0.01
    for jj in (1, 3)
        ej = dvec(_basis_state_mps(jj - 1, sc), sc)
        x  = get_rpa_susceptibility_column(Hc, V, 0.3, jj; kw...)
        @test dvec(x, sc) ≈ χ * ej rtol=1e-8
        @test !isapprox(dvec(x, sc), transpose(χ) * ej; rtol=1e-3)
    end
    # `j` as an MPS, return_info, and the ASCII names of the broadening and Fermi energy
    b3 = _basis_state_mps(2, sc)
    r3 = get_rpa_susceptibility_column(Hc, V, 0.3, b3; kw..., return_info=true)
    @test dvec(r3.chi, sc) ≈ χ * dvec(b3, sc) rtol=1e-8
    @test r3.residual < 1e-3 && r3.residual >= r3.residual_floor
    kwa = Base.structdiff(kw, (; η=0))
    xa  = get_rpa_susceptibility_column(Hc, V, 0.3, 3; kwa..., eta=0.2, fermi=0.0)
    @test dvec(xa, sc) ≈ χ * dvec(b3, sc) rtol=1e-8
    @test_throws ArgumentError get_rpa_susceptibility_column(Hc, V, 0.3, 3; kw..., eta=0.2)
    @test_throws ArgumentError get_rpa_susceptibility_column(Hc, V, 0.3, 3; kw..., fermi=0.0,
                                                             ϵF=0.0)
    @test_throws BoundsError get_rpa_susceptibility_column(Hc, V, 0.3, 5; kw...)
    @test_throws BoundsError get_rpa_susceptibility_column(Hc, V, 0.3, 0; kw...)
    @test_throws ArgumentError get_rpa_susceptibility_column(Hc, V, 0.3, true; kw...)
    @test_throws ErrorException get_rpa_susceptibility_column(Hc, V, 0.3, 1; mode=:spin)
    # a spinless H in the magnetic channel is refused as such, whatever j
    @test_throws ErrorException get_rpa_susceptibility_column(Hc, V, 0.3, 99; mode=:magnetic)

    # The Krylov bubble takes the start and the residual threshold of get_green_krylov:
    # the start of v0.2.1 is reachable, and the check can be silenced.
    kwk = (; GF_method=:krylov, η=0.2, krylov_nsweeps=2, krylov_maxdim=4, cutoff=1e-12,
             maxdim=64)
    @test_logs (:warn, r"get_green_krylov: G is not converged") match_mode=:any get_bubble_mpo(
        Hc, Hc, 0.3; kwk...)
    Π1 = @test_logs get_bubble_mpo(Hc, Hc, 0.3; kwk..., krylov_residual_warn=Inf)
    Π2 = @test_logs get_bubble_mpo(Hc, Hc, 0.3; kwk..., krylov_residual_warn=Inf,
                                   krylov_start=:identity)
    @test Π1 isa MPO && Π2 isa MPO
    @test dmat(Π1, sc) != dmat(Π2, sc)
    # krylov_cutoff = nothing is the size-aware default of get_green_krylov, 1e-8 at this
    # size (the doubled register has 2^4 states)
    Π3 = get_bubble_mpo(Hc, Hc, 0.3; kwk..., krylov_residual_warn=Inf, krylov_cutoff=nothing)
    @test dmat(Π3, sc) == dmat(Π1, sc)
end
