using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Logging, Random, Test
using TensorBinding: get_Hamiltonian, add_onsite!, get_green_krylov

# get_green_krylov started from |I⟩⟩ with a 2-vector global Krylov expansion (up to v0.2.1),
# and the two-site sweeps keep the rank their start spans. On a short-ranged G (ω outside
# the band) of a chain with an impurity, the default settings returned the G without the
# impurity's correction, 9 % off, with nothing to show it. The default start is now
# |I⟩⟩ + 0.1·(random MPS, χ 8) with an 8-vector expansion, and the residual of the solve is
# checked: when ‖G(z − H) − 1‖_F exceeds `residual_warn` (0.1) a warning says that G is not
# converged (bugfix_green_krylov_size.jl has the large-system side of the check).
@testset "get_green_krylov: random start and residual check" begin
    dmat(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
    relerr(G, Gx, s) = norm(dmat(G, s) - Gx) / norm(Gx)

    H = get_Hamiltonian("chain_1d", 1.0; L=6)                 # 64 sites, band [−2, 2]
    v = zeros(64); v[64 ÷ 3 + 1] = 2.0
    add_onsite!(H, v; tol=1e-14)
    s  = H.sites
    Hd = dmat(H.mpo, s)
    ω = 2.3
    for η in (0.1, 0.01)
        Gx = inv((ω + im * η) * I - Hd)
        # the start of v0.2.1: wrong, and the residual check now says so
        old = @test_logs (:warn, r"not converged") match_mode=:any get_green_krylov(
                  H, ω; η, start=:identity, return_info=true)
        @test relerr(old.G, Gx, s) > 1e-2
        @test old.residual > 1e-2
        @test old.residual_abs ≈ 8 * old.residual      # ‖|I⟩⟩‖ = √64
        # the new default start, with GMRES and with MINRES
        for solver in (:gmres, :minres)
            new = get_green_krylov(H, ω; η, solver, return_info=true)
            @test relerr(new.G, Gx, s) < 1e-5
            @test new.residual < 1e-5
        end
    end

    # The random start does not depend on the global RNG; `rng` changes it (another start,
    # so another G within the accuracy of the solve).
    Random.seed!(1); G1 = get_green_krylov(H, 0.5; η=0.1)
    Random.seed!(2); G2 = get_green_krylov(H, 0.5; η=0.1)
    @test dmat(G1, s) ≈ dmat(G2, s) rtol=1e-12
    G3 = get_green_krylov(H, 0.5; η=0.1, rng=Random.Xoshiro(7))
    @test dmat(G3, s) != dmat(G1, s)
    @test dmat(G3, s) ≈ inv((0.5 + 0.1im) * I - Hd) rtol=1e-6

    # residual_warn = Inf: no check, no warning, the plain MPO
    G = @test_logs get_green_krylov(H, ω; η=0.1, start=:identity, residual_warn=Inf)
    @test G isa MPO
    @test_throws ArgumentError get_green_krylov(H, ω; start=:zero)
end
