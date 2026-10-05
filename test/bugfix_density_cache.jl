using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, get_density, _get_density_matrix, _get_projector,
                     KPM_Tn, get_ldos, get_ldos_spectrum

# The density cache answers only the projector it holds (docs/dev/REORGANISATION_TODO.md,
# "Found by the bug pass"): a density matrix cached on H is reused for the same method
# and the same Fermi level (McWeeny, KPM), filling (SP2), Chebyshev order and kernel
# (KPM), whatever the truncation keywords. Until the fix get_density, the RPA
# purification and _get_projector returned it for any ϵF, Nel or Ncheb. Dense ground
# truth on a 4-state chain, E = ±0.618, ±1.618. Also the Chebyshev lists, which share
# one order field between the MPO and MPS lists.

"Dense matrix of `M` over `sites` (site 1 fastest, rows = primed indices)."
densemat_dc(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
chain_dc() = get_Hamiltonian("chain_1d", 1.0; L=2, scale=2.2)

@testset "Density cache: reused only for the same projector" begin
    H = chain_dc()
    F = eigen(Hermitian(densemat_dc(H.mpo, H.sites)))
    Pocc(ϵ) = F.vectors * Diagonal(F.values .< ϵ) * F.vectors'
    pkw = (; maxiters=40, maxdim=100, cutoff=1e-12, tol=1e-10)

    # McWeeny: a second Fermi level is computed, not answered by the half-filled cache.
    H  = chain_dc()
    ρ0 = get_density(H; method=:mcweeny, ϵF=0.0, pkw...)
    ρ8 = get_density(H; method=:mcweeny, ϵF=0.8, pkw...)
    @test ρ8 !== ρ0
    @test norm(densemat_dc(ρ0, H.sites) - Pocc(0.0)) < 1e-6
    @test norm(densemat_dc(ρ8, H.sites) - Pocc(0.8)) < 1e-6
    @test get_density(H; method=:mcweeny, ϵF=0.8, pkw...) === ρ8
    @test get_density(H; method=:mcweeny, ϵF=0.8, maxdim=30) === ρ8   # truncation: no
    @test _get_projector(H; method=:mcweeny, fermi=0.8) === ρ8
    P0 = _get_projector(H; method=:mcweeny, fermi=0.0, maxdim=100)
    @test P0 !== ρ8
    @test norm(densemat_dc(P0, H.sites) - Pocc(0.0)) < 1e-4

    # SP2: another electron count is another projector.
    H  = chain_dc()
    ρ1 = get_density(H; method=:sp2, Nel=1, pkw...)
    ρ3 = get_density(H; method=:sp2, Nel=3, pkw...)
    @test ρ3 !== ρ1
    @test real(tr(ρ1)) ≈ 1 atol=1e-6
    @test norm(densemat_dc(ρ3, H.sites) - Pocc(0.8)) < 1e-6
    @test _get_projector(H; method=:sp2, Nel=3) === ρ3
    @test _get_projector(H; method=:sp2, Nel=1) !== ρ3

    # KPM: the key holds the order actually expanded and the Fermi level.
    kkw = (; maxdim=100, cutoff=1e-12)
    H   = chain_dc()
    k40 = get_density(H; method=:kpm, Ncheb=40, kkw...)
    k80 = get_density(H; method=:kpm, Ncheb=80, kkw...)
    @test k80 !== k40
    Hf  = chain_dc()
    @test densemat_dc(k80, H.sites) ≈
          densemat_dc(get_density(Hf; method=:kpm, Ncheb=80, kkw...), Hf.sites) atol=1e-12
    # a lower order is expanded on the cached 80-moment list at order 80: the same matrix
    @test get_density(H; method=:kpm, Ncheb=60, kkw...) === k80
    k8 = get_density(H; method=:kpm, ϵF=0.8, Ncheb=80, kkw...)
    @test k8 !== k80
    @test norm(densemat_dc(k8, H.sites) - Pocc(0.8)) < 0.1             # Jackson-smeared step

    # RPA purification: the bubble's density follows ϵF without clearing the cache.
    H = chain_dc()
    dm(ϵF; pm=:mcweeny) =
        _get_density_matrix(H, ϵF, :purification, 60, 100, 1e-12, pm, 100, 40, 1e-10, false)
    r0 = dm(0.0)
    r8 = dm(0.8)
    @test r8 !== r0
    @test norm(densemat_dc(r8, H.sites) - Pocc(0.8)) < 1e-6
    @test dm(0.8) === r8
    # SP2 refuses a Fermi level also when an SP2 density is cached
    s0 = dm(0.0; pm=:sp2)
    @test_throws ArgumentError dm(0.5; pm=:sp2)
    @test dm(0.0; pm=:sp2) === s0

    # A density matrix set by hand still answers every method and level.
    H = chain_dc(); sentinel = 0.5 * MPO(H.sites, "Id"); H._density_cache = sentinel
    @test get_density(H; method=:mcweeny, ϵF=0.8) === sentinel
    @test get_density(H; method=:kpm, Ncheb=10) === sentinel
    @test _get_projector(H; method=:sp2, Nel=1) === sentinel
end

@testset "Cached Chebyshev lists are read at their own order" begin
    # `_tn_Ncheb` is the order of the list built last, MPO or MPS; the readers of the
    # other list used it: silently fewer moments, or a BoundsError past its end.
    chain3() = get_Hamiltonian("chain_1d", 1.0; L=3, scale=2.5)
    probe(H) = (TensorBinding.binary_to_MPS(3, H.L, H.sites) +
                TensorBinding.binary_to_MPS(4, H.L, H.sites)) / sqrt(2)
    kw = (; maxdim=100, cutoff=1e-12)

    # MPS list at order 30, then an MPO list at 20: get_ldos(:mps) still reads 30 moments.
    H = chain3(); ψ = probe(H)
    KPM_Tn(H, 30; mode=:mps, psi0=ψ, kw...)
    ref = get_ldos(H, 0.3; mode=:mps, psi0=ψ)
    KPM_Tn(H, 20; kw...)
    @test get_ldos(H, 0.3; mode=:mps, psi0=ψ) == ref

    # MPO list at order 80, then an MPS list at 20: the MPO readers still expand 80.
    H = chain3()
    KPM_Tn(H, 80; kw...)
    s80 = get_ldos_spectrum(H, [0.3])[1]
    ρ80 = get_density(H; method=:kpm, Ncheb=80, kw...)
    H._density_cache = nothing
    KPM_Tn(H, 20; mode=:mps, psi0=probe(H), kw...)
    @test norm(get_ldos_spectrum(H, [0.3])[1] - s80) < 1e-12 * norm(s80)
    @test densemat_dc(get_density(H; method=:kpm, Ncheb=40, kw...), H.sites) ≈
          densemat_dc(ρ80, H.sites) atol=1e-12

    # MPO list at order 20, then an MPS list at 60: get_density(:kpm, Ncheb=40) rebuilds
    # the MPO list at 40 (it read 61 moments from the 21-entry list).
    H = chain3()
    KPM_Tn(H, 20; kw...)
    KPM_Tn(H, 60; mode=:mps, psi0=probe(H), kw...)
    Hf = chain3()
    @test densemat_dc(get_density(H; method=:kpm, Ncheb=40, kw...), H.sites) ≈
          densemat_dc(get_density(Hf; method=:kpm, Ncheb=40, kw...), Hf.sites) atol=1e-12
end
