using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, get_density, KPM_Tn, get_ldos, get_ldos_spectrum,
                     eval_mps, binary_to_MPS

# The caches stay with the operator and the window they were computed for (docs/dev/
# REORGANISATION_TODO.md, "Found by the v0.2.0 scope survey"): assigning `mpo`, `sites`,
# `position_space`, or a different `scale` or `center`, empties them; `deepcopy` keeps
# the record of what a cached density was computed for; `get_ldos(mode=:mps)` refuses a
# probe other than the cached one. Up to v0.1.1 each of these gave a stale result
# without an error. Dense ground truth on an L = 3 chain (E = 2cos(kπ/9), k = 1…8).

densemat_cs(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
diag_cs(ψ::MPS, N) = [real(eval_mps(ψ, x)) for x in 0:(N - 1)]
chain_cs(; scale = 2.5) = get_Hamiltonian("chain_1d", 1.0; L = 3, scale = scale)
staggered_cs(H) = +(H.mpo, 0.8 * MPO(H.sites, ["Z", "Id", "Id"]); cutoff = 1e-12)
levels_cs(H) = eigvals(Hermitian(densemat_cs(H.mpo, H.sites)))
function projector_cs(H, μ)
    F = eigen(Hermitian(densemat_cs(H.mpo, H.sites)))
    return F.vectors * Diagonal(F.values .< μ) * F.vectors'
end

@testset "Assigning the operator or the window empties the caches" begin
    kw  = (; maxdim = 100, cutoff = 1e-12)
    pkw = (; maxiters = 40, tol = 1e-10, kw...)

    # A new operator by assignment: get_density recomputes for it (it returned the
    # projector of the old operator, ‖ρ − P‖ = 1.07); the assigned window is kept.
    H = chain_cs()
    ρold = get_density(H; method = :mcweeny, ϵF = 0.0, pkw...)
    H.mpo = staggered_cs(H)
    @test H._density_cache === nothing
    @test H.scale == 2.5
    ρnew = get_density(H; method = :mcweeny, ϵF = 0.0, pkw...)
    @test ρnew !== ρold
    @test norm(densemat_cs(ρnew, H.sites) - projector_cs(H, 0.0)) < 1e-4   # stale: 1.07

    # The same after KPM_Tn: the list goes, so get_ldos_spectrum asks for a new one
    # (it returned the old operator's LDOS, 60 % off).
    H = chain_cs()
    KPM_Tn(H, 60; kw...)
    H.mpo = staggered_cs(H)
    @test H._tn_cache === nothing
    @test_throws ErrorException get_ldos_spectrum(H, [0.3])
    KPM_Tn(H, 60; kw...)
    Hf = chain_cs()
    Hf.mpo = staggered_cs(Hf)
    KPM_Tn(Hf, 60; kw...)
    @test diag_cs(get_ldos_spectrum(H, [0.3])[1], 8) ≈ diag_cs(get_ldos_spectrum(Hf, [0.3])[1], 8) atol = 1e-10

    # A new scale empties the list built in the old window (it was read in the new one,
    # 69 % off); the KPM density is then expanded at the new scale, as a fresh one.
    H = chain_cs()
    KPM_Tn(H, 80; kw...)
    H.scale = 5.0
    @test H._tn_cache === nothing
    @test H.scale == 5.0
    ρ  = get_density(H; method = :kpm, ϵF = 0.5, Ncheb = 80, kw...)
    ρf = get_density(chain_cs(; scale = 5.0); method = :kpm, ϵF = 0.5, Ncheb = 80, kw...)
    @test real(tr(ρ)) ≈ real(tr(ρf)) atol = 1e-10
    # assigning the stored value keeps the caches
    KPM_Tn(H, 20; kw...)
    H.scale = 5.0
    @test H._tn_cache !== nothing

    # A new center empties a cached McWeeny density, whose level is center + ϵF (the
    # cached one answered with 4 states where 5 lie below 0.5).
    H = chain_cs()
    get_density(H; method = :mcweeny, ϵF = 0.0, pkw...)
    H.center = 0.5
    @test H._density_cache === nothing
    ρc = get_density(H; method = :mcweeny, ϵF = 0.0, pkw...)
    @test real(tr(ρc)) ≈ count(<(0.5), levels_cs(H)) atol = 1e-6

    # Filling an undetermined window (_ensure_scale!) empties nothing: a density set by
    # hand stays.
    H = chain_cs()
    H.scale = 0.0
    ρhand = 0.5 * MPO(H.sites, "Id")
    H._density_cache = ρhand
    KPM_Tn(H, 10; kw...)
    @test H.scale > 0
    @test H._density_cache === ρhand

    # The copy constructor and the mutators behave as before.
    H = chain_cs()
    KPM_Tn(H, 20; kw...)
    @test TensorBinding.TBHamiltonian(H; mpo = staggered_cs(H))._tn_cache === nothing
    @test H._tn_cache !== nothing
    TensorBinding.add_onsite!(H, n -> 0.1 * n)
    @test H._tn_cache === nothing
    @test H.scale == 0.0
end

@testset "deepcopy keeps what a cached density was computed for" begin
    pkw = (; maxiters = 40, tol = 1e-10, maxdim = 100, cutoff = 1e-12)
    H = chain_cs()
    get_density(H; method = :mcweeny, ϵF = 0.0, pkw...)
    Hc = deepcopy(H)
    @test Hc._density_cache !== nothing
    # another level on the copy is recomputed (the copy answered ϵF = 0.8 with the
    # ϵF = 0 density, 4 states instead of 5)
    ρ8 = get_density(Hc; method = :mcweeny, ϵF = 0.8, pkw...)
    @test real(tr(ρ8)) ≈ count(<(0.8), levels_cs(Hc)) atol = 1e-6
    # the same level on the copy is reused
    Hc2 = deepcopy(H)
    @test get_density(Hc2; method = :mcweeny, ϵF = 0.0, pkw...) === Hc2._density_cache
    # a density set by hand still answers everything on the copy
    H = chain_cs()
    H._density_cache = 0.5 * MPO(H.sites, "Id")
    Hc = deepcopy(H)
    @test get_density(Hc; method = :kpm, Ncheb = 10) === Hc._density_cache
end

@testset "get_ldos(mode=:mps) refuses a probe other than the cached one" begin
    H  = chain_cs()
    ψ1 = binary_to_MPS(2, H.L, H.sites)
    ψ2 = binary_to_MPS(5, H.L, H.sites)
    KPM_Tn(H, 40; mode = :mps, psi0 = ψ1, maxdim = 100, cutoff = 1e-12)
    @test_throws ArgumentError get_ldos(H, 0.3; mode = :mps, psi0 = ψ2)
    v = get_ldos(H, 0.3; mode = :mps, psi0 = ψ1)
    @test v ≈ get_ldos(H, 0.3; mode = :mps, psi0 = 2.0 * ψ1)    # the norm does not matter
    Hf = chain_cs()
    KPM_Tn(Hf, 40; mode = :mps, psi0 = binary_to_MPS(5, Hf.L, Hf.sites), maxdim = 100,
           cutoff = 1e-12)
    @test get_ldos(Hf, 0.3; mode = :mps, psi0 = binary_to_MPS(5, Hf.L, Hf.sites)) isa Real
end
