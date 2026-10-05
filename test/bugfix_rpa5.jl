using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, get_bubble_mpo_cheb2d, get_bubble_mpo_cheb2d_tucker,
                     get_bubble_diag_cheb2d, get_bubble_diag_cheb2d_svd,
                     get_bubble_diag_cheb2d_tucker, get_bubble_mpo_haydock, get_spect_k,
                     _get_density_matrix, _cheb2d_setup, _hadamard_difference, _transpose_mpo

# Dense ground truth on 2-qubit (4-site) chains; site 1 fastest, rows = primed indices.
densemat(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
chain2() = get_Hamiltonian("chain_1d", 1.0; L=2, scale=2.2)
# The chain with a complex Hermitian term 0.3·Y on qubit 2 (and 0.2·Z on qubit 1).
function chain2_cplx()
    H = chain2()
    s = H.sites
    H.mpo = +(H.mpo, 0.3 * MPO(ComplexF64, s, ["Id", "Y"]), 0.2 * MPO(s, ["Z", "Id"]);
              cutoff=1e-14)
    H.scale, H.center = 2.8, 0.1
    return H
end
eigprojs(Hd) = (F = eigen(Hermitian(Hd)); (F.values, [F.vectors[:, a] * F.vectors[:, a]' for a in eachindex(F.values)]))

@testset "cheb2d on a complex H: Lindhard structure (P_a)ᵀ ⊙ P_b" begin
    s = siteinds("Qubit", 2)
    M = random_mpo(s) + 0.4im * MPO(ComplexF64, s, ["Y", "X"])
    @test densemat(_transpose_mpo(M), s) == transpose(densemat(M, s))

    # The documented formula with the exact f, for the code's density matrix P:
    # Π = Σ_ab f(ε_a, ε_b) [(P_a P)ᵀ ⊙ P_b − P_aᵀ ⊙ (P_b P)], f = 1/(ω + iη − (ε_b − ε_a)).
    # The Hadamard product of the untransposed factors (the old bubble) is 0.4 away.
    ω, η, Nc = 0.4, 1.0, 20
    H  = chain2_cplx()
    kw = (; P_method=:kpm, Ncheb=Nc, maxdim=100, cutoff=1e-12, η=η)
    P  = densemat(_get_density_matrix(H, 0.0, :kpm, Nc, 100, 1e-12, :mcweeny, 40, 30, 1e-5, false), H.sites)
    E, Pr = eigprojs(densemat(H.mpo, H.sites))
    ref = sum((transpose(Pr[a] * P) .* Pr[b] .- transpose(Pr[a]) .* (Pr[b] * P)) ./
              (ω + im * η - (E[b] - E[a])) for a in 1:4, b in 1:4)
    Π = densemat(get_bubble_mpo_cheb2d(H, H, [ω]; kw...)[1], H.sites)
    @test Π ≈ ref rtol=1e-3

    # Particle conservation: every row and column of Π sums to zero (exactly per (m, n)
    # term, since T_n(H̃), P commute); the old bubble's row sums were half its norm.
    @test norm(sum(Π; dims=2)) < 1e-8 * norm(Π)
    @test norm(sum(Π; dims=1)) < 1e-8 * norm(Π)

    # Tucker at full rank without a kernel is the plain sum; the k-space diagonals are the
    # QFT diagonal of the MPO bubble (all five bubbles share the Hadamard step).
    Πt = get_bubble_mpo_cheb2d_tucker(H, H, [ω]; kernel=:none, tucker_tol=1e-14,
                                      tucker_maxrank=Nc + 1, hooi_iters=0, coeff_tol=0.0, kw...)[1]
    @test densemat(Πt, H.sites) ≈ Π rtol=1e-6
    dk = get_spect_k(get_bubble_mpo_cheb2d(H, H, [ω]; kw...)[1])
    for d in (get_bubble_diag_cheb2d(H, H, [ω]; kw...)[1],
              get_bubble_diag_cheb2d_svd(H, H, [ω]; kernel=:none, svd_tol=1e-14,
                                         svd_maxrank=Nc + 1, kw...)[1],
              get_bubble_diag_cheb2d_tucker(H, H, [ω]; kernel=:none, tucker_tol=1e-14,
                                            tucker_maxrank=Nc + 1, hooi_iters=0,
                                            coeff_tol=0.0, kw...)[1])
        @test reshape(Array(prod(d), siteinds(d)...), :) ≈ dk rtol=1e-6
    end
end

@testset "cheb2d on a complex H: −1 × the Haydock (Lindhard) bubble" begin
    # Same purified density for both (the second call reuses H._density_cache); 16
    # Haydock steps are exact on this 16-dimensional problem, cheb2d is ~1e-4 off.
    ω, η = 0.4, 1.0
    H  = chain2_cplx()
    kw = (; η=η, maxdim=100, cutoff=1e-12, purify_maxdim=100, purify_tol=1e-12,
            purify_maxiters=60)
    Πc = densemat(get_bubble_mpo_cheb2d(H, H, [ω]; Ncheb=20, kw...)[1], H.sites)
    Πh = densemat(get_bubble_mpo_haydock(H, H, [ω]; N_steps=16, kw...)[1], H.sites)
    @test Πc ≈ -Πh rtol=1e-3
end

@testset "cheb2d on a real H: the transpose is skipped" begin
    ω = 0.4
    kw = (; Ncheb=6, maxdim=40, cutoff=1e-12, ϵF=0.0, P_method=:kpm, purify_method=:mcweeny,
            purify_maxdim=40, purify_maxiters=30, purify_tol=1e-5, η=0.5, verbose=false)
    Hr = chain2()
    Sr = _cheb2d_setup(Hr, Hr, [ω], "f", "t"; diagonal=false, lowrank=false, kw...)
    Sc = _cheb2d_setup(chain2_cplx(), chain2_cplx(), [ω], "f", "t"; diagonal=false,
                       lowrank=false, kw...)
    @test Sr.transpose1 == false
    @test Sc.transpose1 == true
    # For a real H the H₁-side factors are symmetric: transposing them changes only
    # rounding, so skipping the transpose keeps the old results bit for bit.
    D(t) = densemat(_hadamard_difference(Sr.TP1[3], Sr.Tn2[2], Sr.Tn1[3], Sr.TP2[2],
                                         Sr.out_sites; transpose1=t, maxdim=40,
                                         cutoff=1e-12), Sr.out_sites)
    @test D(true) ≈ D(false) rtol=1e-10
end
