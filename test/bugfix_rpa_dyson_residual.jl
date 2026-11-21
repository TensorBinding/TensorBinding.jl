using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Logging, Test
using TensorBinding: get_Hamiltonian, rpa_from_bubble_diag, custom_mpo, _rpa_pair_sites

# The Dyson solve (I − Π₀V)χ = Π₀ near the RPA instability: at the default cutoff 1e-8 the
# truncation limits χ to about κ·1e-5 (κ the condition number of I − Π₀V), with nothing to
# show it. The solve now checks its relative residual and warns above `residual_warn`.
# Π₀ is the exact Lindhard bubble of a half-filled 32-site chain, V = U·1 with U = α U_c.
@testset "rpa_from_bubble_diag: residual check near the instability" begin
    dmat(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
    function mpo_of(A, s)                    # exact MPO of the dense matrix A on `s`
        d = dim.(s)
        return MPO(ITensor(reshape(A, d..., d...), prime.(s)..., s...), s; cutoff=1e-15)
    end
    H  = get_Hamiltonian("chain_1d", 1.0; L=5)
    s  = H.sites
    E, ψ = eigen(Hermitian(dmat(H.mpo, s)))
    D  = length(E)
    f  = Float64.(E .< 0)
    W  = reshape([ψ[i, a] * conj(ψ[i, b]) for i in 1:D, a in 1:D, b in 1:D], D, D * D)
    bubble(ω, η) = W * Diagonal(vec([(f[b] - f[a]) / (ω + im * η - E[b] + E[a])
                                     for a in 1:D, b in 1:D])) * W'
    η  = 0.03
    μ  = eigvals(bubble(0.0, η))
    Uc = 1 / real(μ[argmax(abs.(real.(μ)))])
    Πd = bubble(0.0, η)
    Π  = mpo_of(Πd, s)
    fs = _rpa_pair_sites(s)
    relerr(r, χx) = norm(dmat(custom_mpo(r.chi, s), s) - χx) / norm(χx)

    # far from the instability: converged, no warning
    V  = 0.3Uc * MPO(s, "Id")
    χx = (I - 0.3Uc * Πd) \ Πd
    r  = @test_logs rpa_from_bubble_diag(Π, V, fs, s; return_info=true)
    @test r.residual < 1e-3 && relerr(r, χx) < 1e-3

    # κ = 10³: the default cutoff leaves 5e-3, and the residual check warns
    V  = 0.999Uc * MPO(s, "Id")
    χx = (I - 0.999Uc * Πd) \ Πd
    r  = @test_logs (:warn, r"not converged") match_mode=:any rpa_from_bubble_diag(
             Π, V, fs, s; return_info=true)
    @test r.residual > 1e-2 && relerr(r, χx) > 1e-3
    # a tight cutoff converges, without a warning
    r  = @test_logs rpa_from_bubble_diag(Π, V, fs, s; cutoff=1e-12, nsweeps=40,
                                         return_info=true)
    @test r.residual < 1e-3 && relerr(r, χx) < 1e-5
    # the default return is vec(χ) as before; residual_warn = Inf skips the check
    @test (@test_logs rpa_from_bubble_diag(Π, V, fs, s; residual_warn=Inf)) isa MPS
    # Π₀ = 0: χ = 0 with residual 0 (it was NaN), and no warning
    r0 = @test_logs rpa_from_bubble_diag(0.0 * Π, V, fs, s; return_info=true)
    @test r0.residual == 0 && norm(r0.chi) == 0
end
