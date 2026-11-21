using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Random, Test
using TensorBinding: get_Hamiltonian, get_green_krylov, _minres_shifted, _hermiticity_defect

# get_green_krylov(solver=:minres): MINRES as the local solver of the linsolve sweeps,
# for Hermitian H. The default (:gmres) is unchanged and pinned by golden_dynamics.jl.
@testset "get_green_krylov: MINRES local solver" begin
    # dense matrix of an MPO; site 1 fastest, rows = primed indices
    dmat(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))

    # The shifted-Hermitian MINRES on dense vectors, against the direct solve.
    rng = MersenneTwister(5)
    n = 60
    A = randn(rng, ComplexF64, n, n); A = (A + A') / 2
    b = randn(rng, ComplexF64, n)
    x0 = zeros(ComplexF64, n)
    for (a0, a1) in ((0.3 + 0.05im, -1.0), (0.3 + 1e-3im, -1.0), (1.5, 2.0 + 0.5im))
        x, info = _minres_shifted(v -> A * v, b, x0, a0, a1; tol=1e-11, maxiter=5000)
        @test info.converged
        @test x ≈ (a0 * I + a1 * A) \ b rtol=1e-8
    end
    # from a nonzero start, and the budget: maxiter products after the residual of x0
    xs = randn(rng, ComplexF64, n)
    x, info = _minres_shifted(v -> A * v, b, xs, 0.2 + 0.1im, -1.0; tol=1e-11, maxiter=5000)
    @test x ≈ ((0.2 + 0.1im) * I - A) \ b rtol=1e-8
    x, info = _minres_shifted(v -> A * v, b, x0, 0.2 + 0.1im, -1.0; maxiter=3)
    @test !info.converged && info.numops == 4

    # The Green's function of a chain, real and complex Hermitian, against the inverse.
    z  = 0.4 + 0.3im
    H  = get_Hamiltonian("chain_1d", 1.0; L=4)
    s  = H.sites
    Hc = +(H.mpo, 0.3 * MPO(ComplexF64, s, ["Id", "Y", "Id", "Id"]); cutoff=1e-14)
    @test _hermiticity_defect(H.mpo) < 1e-7
    @test _hermiticity_defect(Hc) < 1e-7
    Gm = get_green_krylov(H, 0.4; η=0.3, nsweeps=6, solver=:minres)
    @test dmat(Gm, s) ≈ inv(z * I - dmat(H.mpo, s)) rtol=1e-8
    Gc = get_green_krylov(Hc, s, 0.4; η=0.3, nsweeps=6, solver=:minres)
    @test dmat(Gc, s) ≈ inv(z * I - dmat(Hc, s)) rtol=1e-8
    # small η, near an eigenvalue of the L = 5 chain (2cos(πk/33))
    H5 = get_Hamiltonian("chain_1d", 1.0; L=5)
    ω  = 2cos(π * 7 / 33) + 0.02
    G5 = get_green_krylov(H5, ω; η=1e-3, nsweeps=8, maxdim=64, solver=:minres)
    @test dmat(G5, H5.sites) ≈ inv((ω + 1e-3im) * I - dmat(H5.mpo, H5.sites)) rtol=1e-6

    # A non-Hermitian H is refused (GMRES still takes it), and so is an unknown solver.
    Hn = +(H.mpo, 0.3im * MPO(s, ["Z", "Id", "Id", "Id"]); cutoff=1e-14)
    @test _hermiticity_defect(Hn) > 0.1
    @test_throws ArgumentError get_green_krylov(Hn, s, 0.4; η=0.3, solver=:minres)
    @test dmat(get_green_krylov(Hn, s, 0.4; η=0.3, nsweeps=6), s) ≈
          inv(z * I - dmat(Hn, s)) rtol=1e-8
    @test_throws ArgumentError get_green_krylov(H, 0.4; solver=:cg)
end
