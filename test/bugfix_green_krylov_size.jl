using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Logging, Test
using TensorBinding: get_Hamiltonian, get_green_krylov, _green_krylov_cutoff

# get_green_krylov on a large system. The cutoff of the sweeps is relative to ‖|G⟩⟩‖² ∝ D
# (D sites), so a fixed 1e-8 truncated away what sets one row of G apart from the others:
# from 2^20 sites the end rows of a clean open chain came back as bulk rows, G_11 50 % off.
# The residual check could not see it either: relative to ‖|I⟩⟩‖ = √D, one wrong row costs
# 1/√D (8e-4 here, under the threshold 1e-2). The default cutoff now scales as 1/D beyond
# 2^10 sites, and the warning is on the absolute residual ‖G(z − H) − 1‖_F, which bounds
# every row's.
@testset "get_green_krylov: size-aware cutoff, absolute residual" begin
    bstate(s, p) = MPS(s, [string(b) for b in reverse(digits(p - 1; base=2, pad=length(s)))])
    el(G, s, i, j) = inner(bstate(s, i)', G, bstate(s, j))

    @test _green_krylov_cutoff(2.0^6)  == 1e-8          # unchanged up to 2^10 sites
    @test _green_krylov_cutoff(2.0^10) == 1e-8
    @test _green_krylov_cutoff(2.0^20) == 1e-8 / 2^10

    # A clean open chain of 2^20 sites, ω outside the band: analytic elements.
    L = 20; N = 2^L
    H = get_Hamiltonian("chain_1d", 1.0; L)
    s = H.sites
    m = N ÷ 2                                      # the bond m ↔ m+1 flips every bit
    h = real(el(H.mpo, s, m, m + 1))
    ω, η = 2.3, 0.1
    z  = ω + im * η
    sq = sqrt(z - 2) * sqrt(z + 2)
    gs = (z - sq) / 2                              # end site of a semi-infinite chain
    g0 = 1 / sq                                    # bulk diagonal
    g1 = g0 * (z - sq) / 2h                        # bulk nearest neighbour

    # the default call: chain ends and middle bond right, and no warning
    r = @test_logs get_green_krylov(H, ω; η, return_info=true)
    @test el(r.G, s, 1, 1) ≈ gs rtol=1e-6
    @test el(r.G, s, N, N) ≈ gs rtol=1e-6
    @test el(r.G, s, m, m) ≈ g0 rtol=1e-6
    @test el(r.G, s, m, m + 1) ≈ g1 rtol=1e-6
    @test el(r.G, s, m + 1, m) ≈ g1 rtol=1e-6
    @test r.residual_abs ≈ r.residual * sqrt(N)
    @test r.residual_abs < 0.01

    # the fixed cutoff of v0.2.1: bulk rows at the chain ends. The relative residual is
    # under the old threshold, the absolute one is not, and the call warns.
    old = @test_logs (:warn, r"not converged") match_mode=:any get_green_krylov(
              H, ω; η, cutoff=1e-8, return_info=true)
    @test abs(el(old.G, s, 1, 1) - gs) / abs(gs) > 0.1
    @test el(old.G, s, 1, 1) ≈ g0 rtol=1e-3
    @test old.residual < 1e-2
    @test old.residual_abs > 0.1

    # 2^36 sites. The start: the truncated sum |I⟩⟩ + 0.1·r drops the random part there,
    # so it is a direct sum beyond 2^30 sites. The check: the rounding floor of the inner
    # products (0.1 in ‖G(z − H) − 1‖_F) is at the threshold, so the residual is
    # evaluated exactly, from a direct sum on an operator with a bond without link index.
    # (KrylovKit may warn about a local solve that stops at 1.1e-10: only the warning of
    # the check is looked for.)
    L = 36; N = 2^L
    H = get_Hamiltonian("chain_1d", 1.0; L)
    s = H.sites
    io = IOBuffer()
    r = with_logger(() -> get_green_krylov(H, ω; η, return_info=true), SimpleLogger(io))
    @test !occursin("G is not converged", String(take!(io)))
    @test el(r.G, s, 1, 1) ≈ gs rtol=1e-6
    @test el(r.G, s, N, N) ≈ gs rtol=1e-6
    @test r.residual_abs < 0.01                    # the floor of the inner products: 0.1
    @test r.residual_floor * sqrt(N) < 1e-4        # the exact evaluation's
end
