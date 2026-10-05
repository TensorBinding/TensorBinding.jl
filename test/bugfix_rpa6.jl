using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Random, Test
using TensorBinding: get_Hamiltonian, add_zeeman!, get_bubble_mpo, get_magnon_bubble,
                     get_rpa_susceptibility, get_magnon_susceptibility, rpa_from_bubble_diag,
                     get_green_krylov, custom_mpo, replace_sites, _rpa_pair_sites,
                     _build_heff, _project_spin_sector

# Dense ground truth on 2-qubit registers; site 1 fastest, rows = primed indices.
densemat(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
function mpo_of(A, s)                        # exact MPO of the dense matrix A on `s`
    d = dim.(s)
    return MPO(ITensor(reshape(A, d..., d...), prime.(s)..., s...), s; cutoff=1e-15)
end
chain2() = get_Hamiltonian("chain_1d", 1.0; L=2, scale=2.2)      # E = ±0.618, ±1.618
function chain2_cplx()                       # + 0.3·Y on qubit 2, 0.2·Z on qubit 1
    H = chain2()
    s = H.sites
    H.mpo = +(H.mpo, 0.3 * MPO(ComplexF64, s, ["Id", "Y"]), 0.2 * MPO(s, ["Z", "Id"]);
              cutoff=1e-14)
    H.scale, H.center = 2.8, 0.1
    return H
end
# The exact occupied-state projector (level μ), cached so that every bubble uses it.
function cache_projector!(H, μ)
    F = eigen(Hermitian(densemat(H.mpo, H.sites)))
    H._density_cache = mpo_of(F.vectors * Diagonal(F.values .< μ) * F.vectors', H.sites)
    return F
end
# get_bubble_mpo's Lindhard bubble Σ_ab (f_b − f_a)/(z − (ε_b − ε_a)) (P_a)ᵀ ⊙ P_b
function lindhard(F, μ, z)
    E, U = F.values, F.vectors
    P = [U[:, a] * U[:, a]' for a in eachindex(E)]
    return sum(((E[b] < μ) - (E[a] < μ)) / (z - (E[b] - E[a])) .* transpose(P[a]) .* P[b]
               for a in eachindex(E), b in eachindex(E))
end

@testset "rpa_from_bubble_diag returns vec(χ), χ = (I − Π₀V)⁻¹Π₀" begin
    Random.seed!(11)
    kw = (; nsweeps=6, maxdim=64, cutoff=1e-14)
    for out in (siteinds("Qubit", 2), [siteind("Qubit", 1), Index(3, "Site,Sub,n=2")])
        D  = prod(dim.(out))
        Dπ = 0.3 * randn(ComplexF64, D, D)
        Dv = 0.5I + 0.2 * randn(ComplexF64, D, D)           # does not commute with Π₀
        fs = _rpa_pair_sites(out)
        x  = rpa_from_bubble_diag(mpo_of(Dπ, out), mpo_of(Dv, out), fs, out; kw...)
        χ  = (I - Dπ * Dv) \ Dπ
        @test siteinds(x) == fs
        @test densemat(custom_mpo(x, out), out) ≈ χ rtol=1e-10
        # The layout: χ[i, j] with i on the odd and j on the even sites.
        X = permutedims(Array(prod(x), fs...), [1:2:length(fs); 2:2:length(fs)])
        @test reshape(X, D, D) ≈ χ rtol=1e-10
    end
    # Π₀ and V proportional to the identity: χ = 0.3/(1 − 0.15)·I (the old output was
    # 0.3/0.85 in every entry, a rank-1 array).
    out = siteinds("Qubit", 2)
    x = rpa_from_bubble_diag(0.3 * MPO(ComplexF64, out, "Id"), 0.5 * MPO(out, "Id"),
                             _rpa_pair_sites(out), out; kw...)
    @test densemat(custom_mpo(x, out), out) ≈ (0.3 / 0.85) * I(4) rtol=1e-12
end

@testset "RPA drivers return the dense Dyson solution" begin
    ω, η = 0.3, 0.2
    kw  = (; GF_method=:krylov, η=η, krylov_nsweeps=10, krylov_maxdim=64,
             krylov_cutoff=1e-12, cutoff=1e-12, maxdim=64)
    dkw = (; rpa_nsweeps=6, rpa_maxdim=64, rpa_cutoff=1e-12)
    for H in (chain2(), chain2_cplx())
        cache_projector!(H, H.center)
        V  = +(0.5 * MPO(H.sites, "Id"), 0.2 * MPO(H.sites, ["Z", "Id"]); cutoff=1e-14)
        Πd = densemat(get_bubble_mpo(H, H, ω; kw...), H.sites)
        χ  = get_rpa_susceptibility(H, V, ω; kw..., dkw...)
        @test densemat(custom_mpo(χ, H.sites), H.sites) ≈
              (I - Πd * densemat(V, H.sites)) \ Πd rtol=1e-8
    end
    # Transverse spin channel: χ on the spin-projected sites.
    Hs = get_Hamiltonian("chain_1d", 1.0; L=2, scale=2.8)
    add_zeeman!(Hs, 0.4; direction=:z)
    Hs.scale = 2.8
    Hu = _project_spin_sector(Hs, 1)
    V  = 0.5 * MPO(Hu.sites, "Id")
    # The same bubble in all three calls: the RNG is reset before each (a DMRG estimate
    # of the sector scales draws from it).
    pk = (; kw..., P_method=:kpm, Ncheb=30)
    Random.seed!(7); Πd = densemat(get_magnon_bubble(Hs, ω; pk...), Hu.sites)
    Random.seed!(7); χm = get_magnon_susceptibility(Hs, V, ω; pk..., dkw...)
    Random.seed!(7); χr = get_rpa_susceptibility(Hs, V, ω; mode=:magnetic, pk..., dkw...)
    for χ in (χm, χr)
        @test densemat(custom_mpo(χ, Hu.sites), Hu.sites) ≈
              (I - Πd * densemat(V, Hu.sites)) \ Πd rtol=1e-8
    end
end

@testset "get_green_krylov: real two-register H_eff; Krylov bubble on real/complex H" begin
    # The bubble's H_eff = I⊗H − H⊗I of a real chain: from the vectorized identity the
    # sweeps stalled at a 48 % error, with the default settings as with more sweeps.
    z  = 0.4 + 0.3im
    H  = chain2()
    s1 = H.sites; s2 = sim.(s1)
    sc = reduce(vcat, [[a, b] for (a, b) in zip(s1, s2)])
    He = _build_heff(H.mpo, replace_sites(H.mpo, s2), s1, s2)
    Gex = inv(z * I - densemat(He, sc))
    G  = densemat(get_green_krylov(He, sc, 0.4; η=0.3), sc)
    @test G ≈ Gex rtol=1e-8
    # An L-site H is unchanged (it converged before too).
    H4 = get_Hamiltonian("chain_1d", 1.0; L=4)
    @test densemat(get_green_krylov(H4.mpo, H4.sites, 0.4; η=0.3, nsweeps=6), H4.sites) ≈
          inv(z * I - densemat(H4.mpo, H4.sites)) rtol=1e-8

    # get_bubble_mpo(GF_method=:krylov) against the dense Lindhard bubble; the real chain
    # used to be 59 % off, the complex one was right.
    ω, η = 0.4, 0.3
    for H in (chain2(), chain2_cplx())
        F  = cache_projector!(H, H.center)
        Π  = get_bubble_mpo(H, H, ω; GF_method=:krylov, η=η, maxdim=64, cutoff=1e-12)
        @test densemat(Π, H.sites) ≈ lindhard(F, H.center, ω + im * η) rtol=1e-8
    end
end
