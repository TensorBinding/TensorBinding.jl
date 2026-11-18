using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Logging, Random, Test
using TensorBinding: get_Hamiltonian, add_onsite!, probe_state, get_correction_vector,
                     get_green_krylov, interleave_mpo, _vec_mps_from_mpo, _linsolve_residual,
                     _norm2_Ax, _opnorm_estimate, _RESIDUAL_MAX_ELEMENTS

# The residual ‖(a0 + a1·A)x − b‖/‖b‖ behind the checks of get_green_krylov, the
# correction vectors and the RPA Dyson solves. From inner products its terms cancel to
# ~√eps of the largest, a floor that grows with ‖x‖ (as 1/η for a resolvent): a wrong x
# could read as residual 0, and the exact x as 0.04. The value is now an upper bound that
# never goes below that floor, and where the floor leaves the comparison with the
# threshold open the residual is evaluated exactly, from a direct sum.
@testset "linsolve residual: rounding floor, exact evaluation, size guard" begin
    dmat(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
    dvec(ψ, s) = vec(Array(prod(ψ), s...))
    tomps(v, s) = MPS(reshape(v, dim.(s)...), s; cutoff=1e-30)

    # A 64-site chain with a bound level below the band; ω on the level.
    H = get_Hamiltonian("chain_1d", 1.0; L=6)
    v = zeros(64); v[20] = -2.0
    add_onsite!(H, v; tol=1e-14)
    s, Hd = H.sites, dmat(H.mpo, H.sites)
    b  = probe_state(H, 20); bv = dvec(b, s)
    E0 = eigvals(Hermitian(Hd))[1]
    rng = Xoshiro(3)
    truth(x, z) = norm((z * I - Hd) * dvec(x, s) - bv) / norm(bv)
    function perturbed(z, δ)                    # the exact solution plus δ‖x‖·(unit noise)
        xv = (z * I - Hd) \ bv
        p  = randn(rng, ComplexF64, 64)
        return tomps(xv + δ * norm(xv) * p / norm(p), s)
    end

    # ⟨Ax|Ax⟩ one tensor at a time against ITensorMPS's contraction
    y = perturbed(complex(0.3, 0.05), 0.1)
    @test _norm2_Ax(H.mpo, y) ≈ real(inner(H.mpo, y, H.mpo, y)) rtol=1e-12

    # Well above the floor (η = 0.05) the inner products give the dense residual.
    z = complex(E0, 0.05)
    x = perturbed(z, 1e-3)
    r = _linsolve_residual(H.mpo, b, x, z, -1.0; threshold=0.02)
    @test r.method === :inner && r.floor < 1e-5
    @test r.residual ≈ truth(x, z) rtol=1e-6

    # η = 1e-6: the floor (≈ 0.1) is above rtol = 0.02.
    z = complex(E0, 1e-6)
    x = perturbed(z, 0.0)                       # the exact x: true residual ~1e-9
    r = _linsolve_residual(H.mpo, b, x, z, -1.0)
    @test r.method === :inner && r.floor > 0.02
    @test r.residual >= r.floor                 # an upper bound, not noise below the floor
    r = _linsolve_residual(H.mpo, b, x, z, -1.0; threshold=0.02)
    @test r.method === :exact && r.residual < 1e-6
    for _ in 1:5                                # wrong vectors, true residual ≈ 0.05–0.3
        x = perturbed(z, 5e-8)
        r = _linsolve_residual(H.mpo, b, x, z, -1.0; threshold=0.02)
        @test r.method === :exact
        @test r.residual ≈ truth(x, z) rtol=1e-4
        @test r.residual > 0.02
    end

    # The vectorized system of get_green_krylov (a0 = 0), whose operator has a bond
    # without a link index: the exact path runs on it (the exact solution, whose
    # residual the inner products cannot tell from their floor).
    Hg = get_Hamiltonian("chain_1d", 1.0; L=4)
    sg = Hg.sites
    s2 = siteinds("Qubit", 8)
    Ag = interleave_mpo(swapprime(complex(2.3, 0.1) * MPO(sg, "Id") - Hg.mpo, 0 => 1), s2, 0)
    bg = _vec_mps_from_mpo(MPO(sg, "Id"), s2)
    @test any(j -> linkind(Ag, j) === nothing, 1:7)
    Agd = ITensors.@disable_warn_order dmat(Ag, s2)
    xg  = tomps(Agd \ (ITensors.@disable_warn_order dvec(bg, s2)), s2)
    rg  = _linsolve_residual(Ag, bg, xg, 0.0, 1.0; threshold=1e-9)
    @test rg.method === :exact && rg.residual < 1e-10
    res = get_green_krylov(Hg, 2.3; η=0.1, return_info=true)
    @test res.residual >= res.residual_floor > 0

    # The scale of the floor is the norm of A, not its rms singular value: one strong
    # bond on a long chain is 500 times the rms value, and the rounding of ⟨Ax|Ax⟩ on a
    # state that sits on it follows the bond. The power-iteration estimate finds it.
    L20 = 20
    H20 = get_Hamiltonian("chain_1d", 1.0; L=L20)
    P20 = MPO(H20.sites, [d == 0 ? "Proj0" : "Proj1"
                          for d in reverse(digits(2^(L20 - 1) + 5; base=2, pad=L20))])
    A20 = +(H20.mpo, 1e4 * P20; alg="directsum")
    @test sqrt(real(inner(A20, A20)) / 2.0^L20) < 20
    @test _opnorm_estimate(A20, H20.sites) ≈ 1e4 rtol=1e-3
    @test 1.9 < _opnorm_estimate(H20.mpo, H20.sites) <= 2.0 + 1e-8

    # Above the size bound the check is skipped with one message, never a warning: the
    # correction vector reports residual NaN and converged = false.
    old = _RESIDUAL_MAX_ELEMENTS[]
    try
        _RESIDUAL_MAX_ELEMENTS[] = 10.0
        r = _linsolve_residual(H.mpo, b, x, z, -1.0; threshold=0.02)
        @test r.method === :skipped && isnan(r.residual)
        cv = @test_logs (:info, r"residual check was skipped") match_mode=:any get_correction_vector(
                 H, b, 0.3; eta=0.05)
        @test isnan(cv.residual) && !cv.converged
        G = @test_logs (:info, r"residual check was skipped") match_mode=:any get_green_krylov(
                Hg, 2.3; η=0.1)
        @test G isa MPO
    finally
        _RESIDUAL_MAX_ELEMENTS[] = old
    end
end
