using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, chebyshev2d_gf_coeffs, get_bubble_mpo, get_bubble_mpo_cheb2d,
                     get_bubble_mpo_cheb2d_tucker, get_bubble_mpo_haydock, _get_density_matrix,
                     mcweeny_purify, haydock_cf, eval_haydock_cf, haydock_resolve_mpo,
                     wynn_epsilon, _kpm_kernel

# Dense ground truth on 2-qubit (4-site) chains; site 1 fastest, rows = primed indices.
densemat(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
chain2() = get_Hamiltonian("chain_1d", 1.0; L=2, scale=2.2)      # E = ±0.618, ±1.618
# The same chain with a complex Hermitian term 0.3·Y on qubit 2 (and 0.2·Z on qubit 1).
function chain2_cplx()
    H = chain2()
    s = H.sites
    H.mpo = +(H.mpo, 0.3 * MPO(ComplexF64, s, ["Id", "Y"]), 0.2 * MPO(s, ["Z", "Id"]);
              cutoff=1e-14)
    H.scale, H.center = 2.8, 0.1
    return H
end
eigprojs(Hd) = (F = eigen(Hermitian(Hd)); (F.values, [F.vectors[:, a] * F.vectors[:, a]' for a in eachindex(F.values)]))

@testset "cheb2d: correctly normalised 2D Chebyshev coefficients" begin
    # Σ c_mn T_m(x) T_n(y) must reproduce f(x, y) = 1/(ω + iη − (s₂y + c₂ − s₁x − c₁)),
    # not f/4 (the coefficients were divided by (2N)² instead of N²).
    ω, η, N = 0.3, 1.0, 40
    C = chebyshev2d_gf_coeffs(ω, 1.1, 0.1, 0.9, -0.2, η, N)
    T(n, t) = cos(n * acos(t))
    for (x, y) in ((0.31, -0.47), (-0.8, 0.05), (0.99, 0.99))
        f = 1 / (ω + im * η - (0.9y - 0.2 - 1.1x - 0.1))
        @test sum(C[m+1, n+1] * T(m, x) * T(n, y) for m in 0:N-1, n in 0:N-1) ≈ f rtol=1e-8
    end

    # The bubble itself against its documented formula with the exact f:
    # Π = Σ_ab f(ε_a, ε_b) [(P_a P) ⊙ P_b − P_a ⊙ (P_b P)] with P the code's density matrix
    # (the transposes of the documented formula drop out for this real H).
    # (For a real H this is −1 × get_bubble_mpo, a sign difference kept by decision
    # (2026-10-04); see the note in test/data/generate_rpa_golden.jl.)
    ω, η, Nc = 0.4, 1.0, 20       # interpolation error ~1e-4 here; the old bubble was f/4
    H  = chain2()
    kw = (; P_method=:kpm, Ncheb=Nc, maxdim=100, cutoff=1e-12, η=η)
    P  = densemat(_get_density_matrix(H, 0.0, :kpm, Nc, 100, 1e-12, :mcweeny, 40, 30, 1e-5, false), H.sites)
    E, Pr = eigprojs(densemat(H.mpo, H.sites))
    ref = sum(((Pr[a] * P) .* Pr[b] .- Pr[a] .* (Pr[b] * P)) ./ (ω + im * η - (E[b] - E[a]))
              for a in 1:4, b in 1:4)
    Π = densemat(get_bubble_mpo_cheb2d(H, H, [ω]; kw...)[1], H.sites)
    @test Π ≈ ref rtol=1e-3
end

@testset "RPA density matrices take the Fermi level" begin
    H  = chain2()
    Hd = densemat(H.mpo, H.sites)
    F  = eigen(Hermitian(Hd))
    Pocc(ϵ) = F.vectors * Diagonal(F.values .< ϵ) * F.vectors'
    dm(ϵF, P_method; purify_method=:mcweeny) =
        _get_density_matrix(H, ϵF, P_method, 60, 100, 1e-12, purify_method, 100, 40, 1e-10, false)

    # ϵF = 0.8 lies between 0.618 and 1.618: three occupied states, where the purification
    # used to return the half-filled projector for every ϵF.
    ρ = densemat(dm(0.8, :purification), H.sites)
    @test norm(ρ - Pocc(0.8)) < 1e-6
    @test real(tr(ρ)) ≈ 3 atol=1e-6
    H._density_cache = nothing
    @test norm(densemat(dm(0.8, :kpm), H.sites) - Pocc(0.8)) < 0.1      # KPM: Jackson-smeared step

    # ϵF = 0 is the old half-filled purification, bit for bit.
    ρ0 = dm(0.0, :purification)
    H._density_cache = nothing
    ρm = mcweeny_purify(H; maxiters=40, maxdim=100, cutoff=1e-12, tol=1e-10)
    H._density_cache = nothing
    @test densemat(ρ0, H.sites) == densemat(ρm, H.sites)

    # SP2 fixes the filling at half the states (_half_filling; = H.N ÷ 2 on this chain) and
    # cannot honour a Fermi level.
    @test_throws ArgumentError dm(0.8, :purification; purify_method=:sp2)
    @test _get_density_matrix(H, 0.0, :purification, 60, 100, 1e-12, :sp2, 40, 30, 1e-5,
                              false) isa MPO
    H._density_cache = nothing

    # The bubble drivers hand ϵF on: at ϵF = 0.8 the purified bubble is the one of Pocc(0.8).
    ω, η = 0.3, 0.5
    Π08 = densemat(get_bubble_mpo(H, H, ω; ϵF=0.8, Ncheb=60, maxdim=100, cutoff=1e-12, η=η,
                                  purify_maxdim=100, purify_tol=1e-10, purify_maxiters=40), H.sites)
    H._density_cache = nothing
    Π00 = densemat(get_bubble_mpo(H, H, ω; ϵF=0.0, Ncheb=60, maxdim=100, cutoff=1e-12, η=η,
                                  purify_maxdim=100, purify_tol=1e-10, purify_maxiters=40), H.sites)
    H._density_cache = nothing
    Πk  = densemat(get_bubble_mpo(H, H, ω; ϵF=0.8, P_method=:kpm, Ncheb=60, maxdim=100,
                                  cutoff=1e-12, η=η), H.sites)
    @test norm(Π08 - Π00) > 0.1 * norm(Π00)
    @test Π08 ≈ Πk rtol=0.1
end

@testset "haydock_cf: Frobenius inner product, complex Hermitian seeds" begin
    z = 0.3 + 0.2im
    for (H, seed) in ((chain2_cplx(), s -> +(MPO(ComplexF64, s, ["Z", "Id"]),
                                             0.5 * MPO(ComplexF64, s, ["X", "Y"]); cutoff=1e-14)),
                      # real H and seed, but H·seed is not symmetric: Tr[conj(A)·B] ≠ Tr[A†B]
                      (chain2(), s -> +(MPO(s, ["Z", "Id"]), 0.5 * MPO(s, ["X", "X"]); cutoff=1e-14)))
        S  = seed(H.sites)
        Hd = densemat(H.mpo, H.sites); Sd = densemat(S, H.sites)
        a, b, basis, norm0 = haydock_cf(H.mpo, S, 16; maxdim=64, cutoff=1e-14)
        @test norm0 ≈ norm(Sd)
        R = (z * I - Hd) \ Sd
        # 16 steps span the whole Krylov space of a 4×4 problem: the resolvent is exact.
        @test densemat(haydock_resolve_mpo(a, b, basis, z; maxdim=64, cutoff=1e-14), H.sites) ≈ R rtol=1e-8
        @test eval_haydock_cf(a, b, z) ≈ tr(Sd' * R) rtol=1e-8
    end

    # The Hermitian, purely imaginary seed Y ⊗ I threw a DomainError (sqrt(-4)).
    H = chain2()
    a, b, basis, norm0 = haydock_cf(H.mpo, MPO(ComplexF64, H.sites, ["Y", "Id"]), 3;
                                    maxdim=20, cutoff=1e-12)
    @test norm0 ≈ 2.0
    @test all(isfinite, a) && all(isfinite, b)

    # The Haydock bubble on a complex H equals the dense Lindhard bubble of get_bubble_mpo:
    # Σ_ab (f_b − f_a)/(z − (ε_b − ε_a)) (P_a)ᵀ ⊙ P_b.
    H = chain2_cplx()
    ω, η = 0.4, 0.3
    E, Pr = eigprojs(densemat(H.mpo, H.sites))
    f = E .< 0.1        # McWeeny's level is H.center + ϵF = 0.1 (no eigenvalue near it)
    ref = sum((f[b] - f[a]) / (ω + im * η - (E[b] - E[a])) .* transpose(Pr[a]) .* Pr[b]
              for a in 1:4, b in 1:4)
    Πh = get_bubble_mpo_haydock(H, H, [ω]; N_steps=16, η=η, maxdim=100, cutoff=1e-14,
                                purify_maxdim=100, purify_tol=1e-12, purify_maxiters=60)[1]
    @test densemat(Πh, H.sites) ≈ ref rtol=1e-6
end

@testset "wynn_epsilon: converged sequences return their limit" begin
    # Exactly converged: the estimates used to be the 1e30 sentinel.
    @test wynn_epsilon(fill(1.0, 5)) == ComplexF64[1.0, 1.0]
    @test wynn_epsilon(cumsum([0.5^k for k in 0:6])) ≈ fill(2.0, 3) rtol=1e-14
    # A genuinely singular table (arithmetic sequence: Δ² = 0) keeps the sentinel.
    @test wynn_epsilon([1, 2, 3, 4, 5]) == ComplexF64[1e30, 1e30]
    # A sequence without equal entries is untouched: ε₂ is Aitken's Δ².
    s = cumsum([(-1.0)^k / (k + 1) for k in 0:2])
    @test wynn_epsilon(s)[1] ≈ s[2] + 1 / (1 / (s[3] - s[2]) - 1 / (s[2] - s[1]))
end

@testset "cheb2d: textbook Jackson kernel (shared _kpm_kernel)" begin
    @test !isdefined(TensorBinding, :_jackson_kernel)
    for N in (1, 5, 9, 51)
        g = _kpm_kernel(N + 1, :jackson)[1:N] ./ (N + 1)
        textbook = [((N - n + 1) * cos(π * n / (N + 1)) +
                     sin(π * n / (N + 1)) * cot(π / (N + 1))) / (N + 1) for n in 0:N-1]
        @test g ≈ textbook rtol=1e-12
        @test g[1] ≈ 1
    end

    # The Tucker bubble at full rank with kernel=:jackson is Σ g_m g_n c_mn D_mn.
    ω, η, Nc = 0.4, 0.3, 8
    H  = chain2()
    kw = (; P_method=:kpm, Ncheb=Nc, maxdim=100, cutoff=1e-12, η=η)
    Ht = (densemat(H.mpo, H.sites) - H.center * I) / H.scale
    T  = [Matrix{ComplexF64}(I, 4, 4), Ht]
    for _ in 3:Nc+1
        push!(T, 2Ht * T[end] - T[end-1])
    end
    P = densemat(_get_density_matrix(H, 0.0, :kpm, Nc, 100, 1e-12, :mcweeny, 40, 30, 1e-5, false), H.sites)
    C = chebyshev2d_gf_coeffs(ω, H.scale, H.center, H.scale, H.center, η, Nc + 1)
    g = _kpm_kernel(Nc + 2, :jackson)[1:Nc+1] ./ (Nc + 2)
    ref = sum(g[m] * g[n] * C[m, n] .* ((T[m] * P) .* T[n] .- T[m] .* (T[n] * P))
              for m in 1:Nc+1, n in 1:Nc+1)
    Πt = get_bubble_mpo_cheb2d_tucker(H, H, [ω]; kernel=:jackson, tucker_tol=1e-14,
                                      tucker_maxrank=Nc + 1, hooi_iters=0, coeff_tol=0.0, kw...)[1]
    @test densemat(Πt, H.sites) ≈ ref rtol=1e-6
end

# Everything `f()` writes to standard output.
function printed(f)
    path, io = mktemp()
    try
        redirect_stdout(f, io)
        close(io)
        return read(path, String)
    finally
        isopen(io) && close(io)
        rm(path; force=true)
    end
end

@testset "RPA P_method=:kpm honours verbose" begin
    H = chain2()
    quiet = printed(() -> get_bubble_mpo(H, H, 0.3; P_method=:kpm, Ncheb=10, maxdim=20, η=0.1))
    @test !occursin("Computed T_", quiet)
    @test isempty(printed(() -> _get_density_matrix(H, 0.0, :kpm, 10, 20, 1e-8, :mcweeny,
                                                    40, 30, 1e-5, false)))
    loud = printed(() -> get_bubble_mpo(H, H, 0.3; P_method=:kpm, Ncheb=10, maxdim=20, η=0.1,
                                        verbose=true))
    @test occursin("Computed T_", loud) && occursin("Polarization bubble:", loud)
end
