using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, add_spin!, add_zeeman!, _project_spin_sector,
                     get_bubble_mpo_cheb2d, get_bubble_mpo_cheb2d_tucker,
                     chebyshev2d_gf_coeffs, _get_density_matrix

@testset "cheb2d bubbles on spinful and sublattice Hamiltonians" begin
    ω, η, Nc = 0.4, 0.3, 8
    kw = (; P_method=:kpm, Ncheb=Nc, maxdim=100, cutoff=1e-12, η=η)
    densemat(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
    # The cheb2d formula Σ c_mn [(T_m P) ∘ T_n − T_m ∘ (T_n P)] on dense matrices,
    # with the same coefficients and density matrix as the MPO code.
    function cheb2d_dense(H)
        Ht = (densemat(H.mpo, H.sites) - H.center * I) / H.scale
        T  = [Matrix{ComplexF64}(I, size(Ht)), Ht]
        for _ in 3:Nc+1
            push!(T, 2Ht * T[end] - T[end-1])
        end
        P = densemat(_get_density_matrix(H, 0.0, :kpm, Nc, 100, 1e-12,
                                         :mcweeny, 40, 30, 1e-5, false), H.sites)
        C = chebyshev2d_gf_coeffs(ω, H.scale, H.center, H.scale, H.center, η, Nc + 1)
        return sum(C[m, n] .* ((T[m] * P) .* T[n] .- T[m] .* (T[n] * P))
                   for m in 1:Nc+1, n in 1:Nc+1)
    end

    # A spinful chain with a spin-mixing x field, and kagome with its dim-3 sublattice
    # index: H.sites has one index beyond the H.L position qubits, which the cheb2d
    # bubbles' siteinds("Qubit", H.L) output indices could not hold.
    Hs = get_Hamiltonian("chain_1d", 1.0; L=2, scale=2.2)
    add_zeeman!(Hs, 0.4; direction=:x)
    Hs.scale = 2.5
    Hk = get_Hamiltonian("kagome", 1.0; L=2, Lx=1, Ly=1)
    Hk.scale = 4.5
    for H in (Hs, Hk)
        ref = cheb2d_dense(H)
        Π   = get_bubble_mpo_cheb2d(H, H, [ω]; kw...)[1]
        Πt  = get_bubble_mpo_cheb2d_tucker(H, H, [ω]; kernel=:none, tucker_tol=1e-14,
                                           tucker_maxrank=Nc + 1, hooi_iters=0, kw...)[1]
        @test all(n -> hasinds(Π[n], H.sites[n], H.sites[n]'), eachindex(H.sites))
        @test densemat(Π, H.sites) ≈ ref rtol=1e-6
        @test densemat(Πt, H.sites) ≈ ref rtol=1e-5
    end

    # Spin-degenerate chain: both spin blocks are the spin-sector bubble (the path that
    # already worked) and the spin-flip blocks vanish.
    H2 = get_Hamiltonian("chain_1d", 1.0; L=2, scale=2.2)
    add_spin!(H2)
    H2.scale = 2.2
    Hu = _project_spin_sector(H2, 1)
    Hu.scale = 2.2
    A  = Array(prod(get_bubble_mpo_cheb2d(H2, H2, [ω]; kw...)[1]), prime.(H2.sites)..., H2.sites...)
    Au = Array(prod(get_bubble_mpo_cheb2d(Hu, Hu, [ω]; kw...)[1]), prime.(Hu.sites)..., Hu.sites...)
    @test A[1, :, :, 1, :, :] ≈ Au rtol=1e-6
    @test A[2, :, :, 2, :, :] ≈ Au rtol=1e-6
    @test norm(A[1, :, :, 2, :, :]) < 1e-10 * norm(A)

    # H1 and H2 on different site structures: an ArgumentError, not a failed assertion.
    @test_throws ArgumentError get_bubble_mpo_cheb2d(H2, Hu, [ω]; kw...)
    @test_throws ArgumentError get_bubble_mpo_cheb2d_tucker(H2, Hu, [ω]; kw...)
end
