using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Logging
using TensorBinding: get_Hamiltonian, add_onsite!, add_spin!, add_superconductivity!,
                     chern_marker, winding_marker, get_density, get_scf, KPM_Tn, get_density_from_Tn

# Regressions for the topology / purification / SCF items of the Tier 1 characterization
# sweep (docs/dev/REORGANISATION_TODO.md): the imaginary Chern markers, the SP2
# divergence and its default filling, the dead add_superconductivity! scale and the
# get_scf scale override, the deprecated index matching of rms_error, the density
# helpers' Chebyshev order, cutoff and cache rules, and (with CUDA.jl) the real element
# type of scf_magnetic_hubbard_gpu. Every system is tiny and compared with dense
# linear algebra.

const TB = TensorBinding

"Unprimed site index of every tensor of `mpo`, in chain order."
mpo_sites(mpo::MPO) = [only(filter(i -> plev(i) == 0, siteinds(mpo, k))) for k in eachindex(mpo)]
"Dense matrix of `mpo` over `sites` (site 1 = most significant digit)."
function dense_st4(mpo::MPO, sites = mpo_sites(mpo))
    T = ITensor(1.0)
    for k in eachindex(mpo)
        T *= mpo[k]
    end
    D = prod(dim, sites)
    return reshape(Array(T, prime.(reverse(sites))..., reverse(sites)...), D, D)
end
"Dense vector of `psi` (site 1 = most significant digit)."
function densev(psi::MPS)
    s = siteinds(psi)
    T = ITensor(1.0)
    for k in eachindex(psi)
        T *= psi[k]
    end
    return vec(Array(T, reverse(s)...))
end
"MPO of the dense matrix `M` over `sites` (the inverse of `dense_st4`)."
function mpo_from_dense(M::AbstractMatrix, sites)
    d = [dim(s) for s in reverse(sites)]
    T = ITensor(reshape(M, d..., d...), prime.(reverse(sites))..., reverse(sites)...)
    return MPO(T, sites; cutoff = 1e-15)
end
"Projector onto the `n` lowest eigenstates of the Hermitian part of `H`."
function lowest_projector(H::AbstractMatrix, n)
    F = eigen(Hermitian((H + H') / 2))
    V = F.vectors[:, 1:n]
    return V * V'
end

chain(L = 3; scale = 2.5) = get_Hamiltonian("chain_1d", 1.0; L = L, scale = scale)
ssh() = get_Hamiltonian("ssh_sublattice", (t = 1.0, d = -0.3); L = 3)   # 8 cells, 16 states

@testset "Chern marker: the Hermitian part of the marker operator (real values)" begin
    # A random complex Hermitian model on 2x2 unit cells with 2 atoms each (sites: two
    # position qubits and a dim-2 sublattice index), its exact half-filling projector as
    # an MPO, and the unit-cell coordinates the auto-derived geometry_uc functions give.
    pos = siteinds("Qubit", 2)
    sub = Index(2, "Sublattice")
    sites = [pos; sub]
    A = [cos(1.3i + 0.7j) + im * sin(0.9i - 0.4j) for i in 1:8, j in 1:8]
    Hd = (A + A') / 2
    Pd = lowest_projector(Hd, 4)
    P = mpo_from_dense(Pd, sites)
    @test dense_st4(P, sites) ≈ Pd atol = 1e-12
    xf = (i, _) -> Float64((i ÷ 2) % 2)
    yf = (i, _) -> Float64((i ÷ 2) ÷ 2)

    Qd = I - Pd
    xs = [xf(a - 1, 2) for a in 1:8]
    ys = [yf(a - 1, 2) for a in 1:8]
    for quenched in (true, false), sequential in (false, true)
        (sequential && !quenched) && continue
        Λ = 10.0
        C = TB.get_C_op_MPO_from_P(P, 2, sites, xf, yf; Lambda = Λ, maxdim = 64, cutoff = 1e-14,
                                   quenched = quenched, sequential = sequential)
        for uc in 1:4
            got = C(uc)
            code = 0.0im       # the operator assembled, 2πi (Q X P Y Q − P X Q Y P)
            mean_PQ = 0.0      # the mean of the Bianco–Resta P- and Q-form markers
            for s in 1:2
                α = 2(uc - 1) + s
                X = quenched ? Diagonal(Λ .* sin.((xs .- xs[α]) ./ Λ)) : Diagonal(xs)
                Y = quenched ? Diagonal(Λ .* sin.((ys .- ys[α]) ./ Λ)) : Diagonal(ys)
                code += 2im * π * (Qd * X * Pd * Y * Qd - Pd * X * Qd * Y * Pd)[α, α]
                mean_PQ += 2π * imag((Pd * X * Qd * Y * Pd)[α, α]) -
                           2π * imag((Qd * X * Pd * Y * Qd)[α, α])
            end
            @test got isa ComplexF64
            @test imag(got) == 0
            @test real(got) ≈ real(code) atol = 1e-10
            @test real(got) ≈ mean_PQ atol = 1e-10
        end
    end
    # The operator itself is not Hermitian here: its diagonal has an O(0.1) imaginary
    # part, which get_C returned until the fix.
    X = Diagonal(xs); Y = Diagonal(ys)
    @test maximum(abs.(imag.(diag(2im * π * (Qd * X * Pd * Y * Qd - Pd * X * Qd * Y * Pd))))) > 0.05

    # get_C on a trivial Semenoff honeycomb (real P): every local marker is 0, the
    # imaginary parts were ±0.13–0.16 per unit cell.
    H = get_Hamiltonian("honeycomb", 1.0; L = 2, Lx = 1, Ly = 1)
    add_onsite!(H, 0.4; sublat = 1)
    add_onsite!(H, -0.4; sublat = 2)
    H.scale = 3.5
    for quenched in (true, false)
        m = [chern_marker(H; method = :mcweeny, maxdim = 40, quenched = quenched)(uc) for uc in 1:H.N]
        @test all(iszero ∘ imag, m)
        @test maximum(abs, m) < 1e-10
    end
end

@testset "SP2: no divergence near convergence; default filling counts states" begin
    # 8-site chain, Nel = 3 (gapped: E₄ − E₃ = 0.65): the default tol 1e-5 lies below
    # the truncation floor, and the loop used to run away to NaN.
    H = chain()
    ρ0 = TB.purification_initial_guess(H.mpo, 2.5, H.sites)
    ρ = TB.sp2_purify(ρ0, 3)
    M = dense_st4(ρ, H.sites)
    Pex = lowest_projector(dense_st4(H.mpo, H.sites), 3)
    @test all(isfinite, M)
    @test norm(M - Pex) < 1e-3
    @test real(tr(M)) ≈ 3 atol = 1e-3

    # An invalid guess (scale 0.6 below the spectral radius 1.88: spectrum of ρ0 in
    # [−1.1, 2.1]) overflowed to NaN; it is now a clear error.
    @test_throws ErrorException TB.sp2_purify(TB.purification_initial_guess(H.mpo, 0.6, H.sites), 4)

    # A tol above the floor still ends the run first.
    ρa = TB.sp2_purify(ρ0, 4; tol = 1e-3)
    @test norm(dense_st4(ρa, H.sites) - lowest_projector(dense_st4(H.mpo, H.sites), 4)) < 1e-2

    # Default Nel: half the states (8 of the 16 on the 8-cell SSH chain), not half the
    # cells (4, quarter filling).
    Hs = ssh()
    @test TB._half_filling(Hs) == 8
    Hd = dense_st4(Hs.mpo, Hs.sites)
    for f in (H -> get_density(H; method = :sp2, maxdim = 30),
              H -> TB.sp2_purify(H; maxdim = 30),
              H -> TB._get_projector(H; method = :sp2, maxdim = 30))
        Hn = ssh()                      # fresh indices, same model and site order as Hs
        Md = dense_st4(f(Hn), Hn.sites)
        @test real(tr(Md)) ≈ 8 atol = 1e-2
        @test norm(Md - lowest_projector(Hd, 8)) < 1e-2
    end
    # winding_marker with the default SP2 filling agrees with McWeeny at half filling
    Wsp2 = winding_marker(ssh(); method = :sp2, maxdim = 30)
    Wmcw = winding_marker(ssh(); method = :mcweeny, maxdim = 30)
    @test [real(Wsp2(uc)) for uc in 1:8] ≈ [real(Wmcw(uc)) for uc in 1:8] atol = 1e-2
end

@testset "add_superconductivity! scale; get_scf leaves the scale to the driver" begin
    bdg_radius(H) = maximum(abs, eigvals(Hermitian(dense_st4(H.mpo, H.sites))))

    H = chain()                                  # spinless: s-wave redirects to p-wave
    add_superconductivity!(H, 0.2)
    @test H.scale ≈ 2.5 + 1.1 * 2 * 0.2
    @test H.center == 0
    @test bdg_radius(H) <= H.scale

    H = chain(2); add_spin!(H); H.scale = 2.5    # spinful s-wave
    add_superconductivity!(H, 0.3)
    @test H.scale ≈ 2.5 + 1.1 * 0.3
    @test bdg_radius(H) <= H.scale

    H = chain(2); add_onsite!(H, 0.4); H.scale = 2.0; H.center = 0.4
    add_superconductivity!(H, 0.2; type = :pwave)
    @test H.scale ≈ 0.4 + 2.0 + 1.1 * 2 * 0.2
    @test bdg_radius(H) <= H.scale

    # No bound known: a function Δ, or no scale on H (add_spin! resets it)
    H = chain(2); add_spin!(H); H.scale = 2.5
    add_superconductivity!(H, i -> 0.1 * i)
    @test H.scale == 0
    H = chain(2); add_spin!(H)
    add_superconductivity!(H, 0.3)
    @test H.scale == 0

    # get_scf(:magnetic) without `scale` runs scf_magnetic_hubbard with its default
    # (H0.scale), not a DMRG estimate per iteration.
    kw = (maxdim = 30, cutoff = 1e-10, purif_maxiter = 10, purif_tol = 1e-6, verbose = false)
    H0 = chain(); H0.scale = 4.0
    a = get_scf(H0, 2.0, :magnetic; method = :mcweeny, maxiters = 1, kw...)
    b = TB.scf_magnetic_hubbard(H0, 2.0; density_method = :mcweeny, max_scf_iter = 1, kw...)
    @test a.H_up.scale == b.H_up.scale == 4.0 * 1.05
    @test a.rms_error == b.rms_error
end

@testset "rms_error: explicit index matching, same value" begin
    s = siteinds("Qubit", 3)
    A = TB.binary_to_MPS(5, 3, s)
    B = TB.constant_mps(s, 0.35)
    r = @test_logs min_level = Logging.Warn TB.rms_error(A, B)
    @test r ≈ sqrt(sum(abs2, densev(A) - densev(B)) / 8) rtol = 1e-12
end

@testset "Density helpers: Chebyshev order, cutoff and cache method" begin
    gapped() = (H = chain(); add_onsite!(H, n -> 0.4 * (-1)^n); H.scale = 2.8; H)

    # A cached Chebyshev list shorter than Ncheb is rebuilt at Ncheb.
    H = gapped(); KPM_Tn(H, 10; maxdim = 30)
    P = TB._get_projector(H; method = :kpm, Ncheb = 60, maxdim = 30)
    @test H._tn_Ncheb == 60
    Pfresh = TB._get_projector(gapped(); method = :kpm, Ncheb = 60, maxdim = 30)
    @test dense_st4(P) ≈ dense_st4(Pfresh) atol = 1e-12
    # a longer one is used at its own order, as get_density does
    H = gapped(); KPM_Tn(H, 80; maxdim = 30)
    TB._get_projector(H; method = :kpm, Ncheb = 60, maxdim = 30)
    @test H._tn_Ncheb == 80

    # The expansion uses `cutoff`, like the Chebyshev list.
    H = gapped()
    Pc = TB._get_projector(H; method = :kpm, Ncheb = 30, maxdim = 30, cutoff = 1e-3)
    fermi = (0.0 - H.center) / H.scale
    ref = get_density_from_Tn(H._tn_cache, 30; fermi = fermi, maxdim = 30, cutoff = 1e-3)
    @test dense_st4(Pc, H.sites) == dense_st4(ref, H.sites)

    # get_density returns a cached density matrix only for the method that computed it.
    H = chain()
    ρm = get_density(H; method = :mcweeny, maxdim = 30)
    ρk = get_density(H; method = :kpm, Ncheb = 40, maxdim = 30)
    @test ρk !== ρm
    @test dense_st4(ρk) ≈ dense_st4(get_density(chain(); method = :kpm, Ncheb = 40, maxdim = 30)) atol = 1e-12
    @test get_density(H; method = :kpm, Ncheb = 40, maxdim = 30) === ρk
    @test TB._get_projector(H; method = :mcweeny, maxdim = 30) !== ρk
    @test get_density(H; method = :mcweeny) === H._density_cache
    # a density matrix cached by hand answers every method, as before
    H = chain(); sentinel = 0.5 * MPO(H.sites, "Id"); H._density_cache = sentinel
    @test get_density(H; method = :kpm) === sentinel
    @test TB._get_projector(H; method = :sp2) === sentinel
    # an unknown method is an error even with a cached matrix
    @test_throws ErrorException get_density(H; method = :nonsense)
end

@testset "GPU: real scf_magnetic_hubbard_gpu, real chern_marker_gpu, rms without warning" begin
    cuda_functional = false
    if Base.find_package("CUDA") !== nothing
        try
            @eval using CUDA
            cuda_functional = CUDA.functional()
        catch
        end
    end

    if !cuda_functional
        @test_skip "CUDA.jl not functional"
    else
        eltypes(x) = unique(eltype(TB.NDTensors.data(TB.NDTensors.storage(ITensors.tensor(x[i]))))
                            for i in 1:length(x))
        scf_chain() = (H = chain(); add_spin!(H); H.scale = 3.5; H)
        kw = (max_scf_iter = 2, purif_maxiter = 20, purif_tol = 1e-4, mix = 0.25, maxdim = 40,
              cutoff = 1e-10, verbose = false)
        ref = TB.scf_magnetic_hubbard_gpu(scf_chain(), 2.0; kw..., type = ComplexF64)
        res = TB.scf_magnetic_hubbard_gpu(scf_chain(), 2.0; kw..., type = Float64)
        for M in (res.H_up_mpo_gpu, res.density_up_mpo_gpu, res.rho_up_gpu, res.rho_dn_gpu)
            @test eltypes(M) == [Float64]
        end
        @test [h.rms_error for h in res.history] ≈ [h.rms_error for h in ref.history] rtol = 1e-8
        @test densev(res.rho_up) ≈ densev(ref.rho_up) atol = 1e-9

        s = siteinds("Qubit", 3)
        a = TB._to_gpu(TB.binary_to_MPS(5, 3, s), ComplexF64)
        b = TB._to_gpu(TB.constant_mps(s, 0.35), ComplexF64)
        r = @test_logs min_level = Logging.Warn TB._rms_error_gpu(a, b)
        @test r ≈ TB.rms_error(TB.binary_to_MPS(5, 3, s), TB.constant_mps(s, 0.35)) rtol = 1e-10

        H = get_Hamiltonian("honeycomb", 1.0; L = 2, Lx = 1, Ly = 1)
        add_onsite!(H, 0.4; sublat = 1); add_onsite!(H, -0.4; sublat = 2); H.scale = 3.5
        C = TB.chern_marker_gpu(H; maxdim = 40, cutoff = 1e-10, dtype = ComplexF64)
        @test all(uc -> imag(C(uc)) == 0 && abs(C(uc)) < 1e-8, 1:H.N)
    end
end

@testset "scf_meanfield refuses max_scf_iter < 1" begin
    # max_scf_iter = 0 ended in a MethodError (no best state to return) up to v0.1.1
    H = get_Hamiltonian("chain_1d", 1.0; L = 3, scale = 2.5)
    @test_throws ArgumentError get_scf(H, 1.0, :cdw; maxiters = 0, verbose = false)
end
