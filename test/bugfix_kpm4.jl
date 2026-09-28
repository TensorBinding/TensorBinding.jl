using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, KPM_Tn, KPM_Tn_mps, get_ldos, get_ldos_spectrum,
                     get_matrix, fix_sites, extract_diagonal_to_mps, mps_to_diagonal_mpo,
                     spatial_sampling_plan, kspace_sampling_plan, ilinspace, add_spin!,
                     exciton_hamiltonian, get_exciton_ldos_spatial, get_ldos_spatial,
                     tjunction_hamiltonian, probe_state, eval_mps
const TB4 = TensorBinding

# Regression tests for the KPM, sampling and site-index bugs of the 2026-09 audit
# (docs/dev/REORGANISATION_TODO.md, "Found while fixing" and the characterization sweep).

# Dense vector of an MPS on `sites` (site 1 = most significant bit).
dense_mps4(ψ, sites) = [eval_mps(ψ, n) for n in 0:(prod(dim.(sites)) - 1)]

@testset "get_ldos(mode=:mps) does not scale with the probe norm" begin
    # KPM_Tn_mps caches T_n(H̃)|ψ₀⟩ for the normalised probe; the moments used the raw one,
    # so the LDOS grew linearly with ‖ψ₀‖.
    H = get_Hamiltonian("chain_1d", 1.0; L=3, scale=3.0)
    ψ = probe_state(H, 3)
    KPM_Tn_mps(H, 20, ψ)
    a1 = get_ldos(H, 0.3; mode=:mps, psi0=ψ)
    KPM_Tn_mps(H, 20, 2ψ)
    a2 = get_ldos(H, 0.3; mode=:mps, psi0=2ψ)
    @test a2 ≈ a1 rtol=1e-12
end

@testset "MPO site legs are read by prime level, not by storage order" begin
    # A non-symmetric operator, stored once as (s', s) and once as (s, s').
    s = siteinds("Qubit", 3)
    M = MPO(ITensor(Float64.(1:64), prime.(s)..., s...), s)
    M = MPO([permute(M[i], sort(collect(inds(M[i])); by=plev, rev=true)...)
             for i in eachindex(M)])                  # (s', s, links) storage
    Mk = MPO([permute(M[i], sort(collect(inds(M[i])); by=plev)...)
              for i in eachindex(M)])                 # (s, links, s') storage: ket first
    A = Matrix(get_matrix(M, s))
    @test A != transpose(A)
    @test Matrix(get_matrix(Mk, s)) ≈ A
    t = siteinds("Qubit", 3)
    # fix_sites used to take the second leg as the ket and transposed a ket-first MPO.
    @test Matrix(get_matrix(fix_sites(copy(M), t), t)) ≈ A
    @test Matrix(get_matrix(fix_sites(copy(Mk), t), t)) ≈ A
    # extract_diagonal_to_mps used to put a ket-first diagonal on the primed indices.
    d = extract_diagonal_to_mps(Mk)
    @test all(plev(only(siteinds(d, i))) == 0 for i in 1:3)
    @test dense_mps4(d, s) ≈ diag(A)

    # The Fibonacci projector T_0 is stored ket-first: get_ldos(:diag) and
    # get_ldos_spectrum threw on the first sum of diagonals.
    Hf = get_Hamiltonian("fibonacci", (A=1.0, B=2.0); L=4)
    KPM_Tn(Hf, 10)
    ω  = [-1.2, 0.4]
    Ad = [get_ldos(Hf, w) for w in ω]
    As = get_ldos_spectrum(Hf, ω)
    for (i, w) in enumerate(ω)
        Aw = get_ldos(Hf, w; mode=:mpo)                # the full spectral-weight MPO
        want = diag(Matrix(get_matrix(Aw, Hf.sites)))
        @test dense_mps4(Ad[i], Hf.sites) ≈ want atol=1e-8
        @test dense_mps4(As[i], Hf.sites) ≈ want atol=1e-8
    end
end

@testset "Empty sampling groups and one-site MPS" begin
    # spatial_sampling_plan took first() of every group, so an empty group was a
    # BoundsError before the callers' own checks.
    err = try spatial_sampling_plan(3; x_groups=[[1], Int[]]); nothing catch e; e end
    @test err isa ErrorException && occursin("at least one position", err.msg)
    Hx = exciton_hamiltonian("chain_1d", 1.0, x -> -1.0; L=2, scale=4.5)
    @test_throws ErrorException get_exciton_ldos_spatial(Hx, 6, [0.0]; X_groups=[[1], Int[]])

    # mps_to_diagonal_mpo threw a BoundsError on a one-site MPS (its GPU twin did not).
    s1 = siteinds("Qubit", 1)
    D  = mps_to_diagonal_mpo(MPS([ITensor([1.0, 2.0], s1[1])]), s1)
    @test Matrix(get_matrix(D, s1)) ≈ Diagonal([1.0, 2.0])
end

@testset "Exciton detection needs the bare 2L-site register" begin
    # A spinful L = 1 chain has 2 = 2L sites and was taken for an exciton register.
    Hs = get_Hamiltonian("chain_1d", 1.0; L=1, scale=3.0)
    add_spin!(Hs)
    @test length(Hs.sites) == 2 * Hs.L
    @test !TB4._is_exciton_register(Hs)
    @test !occursin("exciton", sprint(show, Hs))
    @test TB4._is_exciton_register(exciton_hamiltonian("chain_1d", 1.0, x -> -1.0; L=1,
                                                       scale=4.5))
end

@testset "k-space planners: ilinspace and the 2D diagonal cut" begin
    @test ilinspace(3, 10, 1) == [3]                  # was [0], outside the window
    @test ilinspace(0, 15, 1) == [0]
    # 2D: num_x points spread along the diagonal cut, not the first num_x of the grid.
    @test kspace_sampling_plan(4, 2; num_x=2).k_groups == [[0], [15]]
    @test kspace_sampling_plan(4, 2; num_x=4).k_groups == [[0], [5], [10], [15]]
    # A window narrower than the register used to trip ilinspace's assertion.
    p = kspace_sampling_plan(4, 2; num_x=8, xmin=1, ymin=1)
    @test p.num_x == 3 && p.k_groups == [[5], [10], [15]]
    p = kspace_sampling_plan(4, 2; num_x=3, xmax=2, ymax=2)
    @test p.k_groups == [[0], [5], [10]]
end

@testset "Grid LDOS maps reject T-junctions; :hodc names where it applies" begin
    Ht = tjunction_hamiltonian(2, 1.0, 1.0)
    err = try get_ldos_spatial(Ht, 4, [0.0]; grid=true, num_x=2); nothing catch e; e end
    @test err isa ErrorException && occursin("T-junction", err.msg)
    H = get_Hamiltonian("chain_1d", 1.0; L=3, scale=3.0)
    err = try get_ldos_spatial(H, 4, [0.0]; num_x=2, kernel=:hodc); nothing catch e; e end
    @test err isa ErrorException && occursin("eta and m_order", err.msg)
end
