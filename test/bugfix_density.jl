using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, KPM_Tn, get_density_from_Tn, get_density, get_matrix

# get_density_from_Tn must return the occupied-state projector θ(ϵF − H). Until v0.1.1 its
# Chebyshev coefficients had the wrong signs and it returned θ(H − ϵF), the empty states.
# A gapped SSH chain in its trivial phase (no zero-energy edge modes, gap |E| > 1) keeps the
# projector sharp, so the two are far apart.
@testset "KPM density matrix is the occupied-state projector" begin
    H  = get_Hamiltonian("ssh", (t=1.0, d=-0.5); L=4)           # 16 sites, trivial gap at E = 0
    Hd = Matrix(get_matrix(H.mpo, H.sites))
    F  = eigen(Hermitian(Hd))
    Pocc(ϵ) = F.vectors * Diagonal(F.values .< ϵ) * F.vectors'
    Ncheb = 120
    Tn, _, _ = KPM_Tn(H, Ncheb)

    ρ = Matrix(get_matrix(get_density_from_Tn(Tn, Ncheb; fermi=0.0, maxdim=64), H.sites))
    @test norm(ρ - Pocc(0.0)) < 0.05
    @test real(tr(ρ * Hd)) ≈ sum(F.values[F.values .< 0]) rtol=1e-2
    @test real(tr(ρ)) ≈ count(<(0), F.values) atol=0.05

    # Fermi level below / above the whole (rescaled) spectrum: empty / full.
    ρlo = Matrix(get_matrix(get_density_from_Tn(Tn, Ncheb; fermi=-0.99, maxdim=64), H.sites))
    ρhi = Matrix(get_matrix(get_density_from_Tn(Tn, Ncheb; fermi=0.99, maxdim=64), H.sites))
    @test real(tr(ρlo)) < 0.05
    @test real(tr(ρhi)) ≈ H.N atol=0.05

    # get_density(:kpm) agrees with McWeeny purification at the same Fermi level.
    ρk = Matrix(get_matrix(get_density(H; method=:kpm, ϵF=0.0, Ncheb=Ncheb, maxdim=64), H.sites))
    ρm = Matrix(get_matrix(get_density(H; method=:mcweeny, ϵF=0.0, maxdim=64), H.sites))
    @test norm(ρk - ρm) < 0.05
end
