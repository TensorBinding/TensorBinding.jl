using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Random, Test
using TensorBinding: get_Hamiltonian, TBHamiltonian, exciton_hamiltonian, mpsexciton,
                     interleave_mpo, swap_every_other_legs, collapse_mpo_pairs,
                     conjugate_by_qft, conjugate_by_qft_exciton, get_green_krylov,
                     get_bubble_mpo, rpa_from_bubble_diag, hermitize,
                     hermitized_hamiltonian, nh_spectrum_grid, binary_to_MPS, eval_mps,
                     add_loss!
const TBX = TensorBinding

# Regression tests for the exciton, interleaving, non-Hermitian and GPU NH time-step
# bugs of the 2026-09 characterization sweep (docs/dev/REORGANISATION_TODO.md): the
# exciton kinetic terms on the wrong registers, interleave_mpo embedding transpose(op),
# the NH bookkeeping (Z_spatial, hermitize(NH), aux_side, scale=0.0) and the Float32
# step of rk4_step_dm_nh_gpu. Dense ground truth on 2–3 qubit registers throughout.

# Dense matrix of an MPO in the big-endian basis `basis` (basis[1] most significant):
# rows on the primed indices, columns on the unprimed ones.
function dmat_x(W::MPO, basis)
    T = ITensor(1.0)
    for i in eachindex(W); T *= W[i]; end
    D = prod(dim, basis)
    return reshape(Array(T, reverse(prime.(basis))..., reverse(basis)...), D, D)
end
dvec_x(ψ::MPS, basis) = (T = ITensor(1.0); for i in eachindex(ψ); T *= ψ[i]; end;
                         vec(Array(T, reverse(basis)...)))

# Exact MPO of the dense matrix A (in the basis `sites`), its tensors in the chain
# order `order`, each stored (s', s, links) or, with ketfirst, (s, s', links).
function mpo_x(A, sites; ketfirst=false, order=sites)
    T = ITensor(reshape(A, reverse(vcat(dim.(sites), dim.(sites)))...),
                reverse(prime.(sites))..., reverse(sites)...)
    M = MPO(T, order; cutoff=1e-15)
    key = ketfirst ? plev : (i -> -plev(i))
    return MPO([permute(M[i], sort(collect(inds(M[i])); by=key)...) for i in eachindex(M)])
end

# A GPU (or CPU) MPO copied to dense CPU tensors.
host_x(W::MPO) = MPO([(t = ITensors.dense(W[i]);
                       itensor(Array(ITensors.NDTensors.data(ITensors.NDTensors.storage(
                                   ITensors.tensor(t)))), inds(t)...)) for i in eachindex(W)])

# Hermitian, complex, not symmetric (no gauge makes it real).
randherm(n; seed) = (Random.seed!(seed); A = randn(ComplexF64, n, n); (A + A') / 2)

@testset "Exciton Hamiltonian: H_c on the electron sites, H_v on the hole sites" begin
    # H_c ≠ H_v, one of them complex: H_c ⊗ I − I ⊗ H_v used to come out as
    # I ⊗ H_cᵀ − H_vᵀ ⊗ I (electron register first).
    L, N = 2, 4
    Hc = get_Hamiltonian("chain_1d", 1.0; L=L)
    Hv = get_Hamiltonian("chain_1d", 0.5; L=L)
    Hv = TBHamiltonian(Hv; mpo=+(Hv.mpo, 0.25 * MPO(ComplexF64, Hv.sites, ["Y", "Id"]);
                                 cutoff=1e-14))
    Dc, Dv = dmat_x(Hc.mpo, Hc.sites), dmat_x(Hv.mpo, Hv.sites)
    @test Dv != transpose(Dv)
    Uf(x) = 0.7 + 0.1x
    Vf(x) = 0.2x^2 - 0.1
    Hx = exciton_hamiltonian(Hc, Hv, Uf; on_site=Vf)
    @test Hx.sites == [Hc.sites[1], Hv.sites[1], Hc.sites[2], Hv.sites[2]]
    Id, V = Matrix(I, N, N), Diagonal(Vf.(1:N))
    U = Diagonal([xe == xh ? -Uf(xe) : 0.0 for xe in 1:N for xh in 1:N])   # contact −Ufunc
    want = kron(Dc + V, Id) - kron(Id, Dv - V) + U
    @test dmat_x(Hx.mpo, [Hc.sites; Hv.sites]) ≈ want atol=1e-10
    # The contact term: ⟨x,x|H|x,x⟩ = ⟨x|H_c|x⟩ − ⟨x|H_v|x⟩ + 2V(x) − Ufunc(x).
    for x in 1:N
        ψ = mpsexciton(x, x, Hx.sites)
        @test real(inner(ψ', Hx.mpo, ψ)) ≈ real(Dc[x, x] - Dv[x, x]) + 2Vf(x) - Uf(x) atol=1e-10
    end
end

@testset "interleave_mpo embeds the operator, not its transpose" begin
    s, p = siteinds("Qubit", 2), siteinds("Qubit", 4)
    Random.seed!(11)
    A = randn(ComplexF64, 4, 4)
    for ketfirst in (false, true), n in (0, 1)
        E   = interleave_mpo(mpo_x(A, s; ketfirst), p, n)
        reg = n == 1 ? p[1:2:end] : p[2:2:end]
        oth = n == 1 ? p[2:2:end] : p[1:2:end]
        @test dmat_x(E, [reg; oth]) ≈ kron(A, Matrix(I, 4, 4)) atol=1e-12
    end

    # swap_every_other_legs reads the legs by prime level: same result for either storage.
    q  = siteinds("Qubit", 4)
    X  = randn(ComplexF64, 16, 16)
    ns = [Index(2, "New,n=$i") for i in 1:4]
    Y1 = dmat_x(swap_every_other_legs(mpo_x(X, q), ns), ns)
    Y2 = dmat_x(swap_every_other_legs(mpo_x(X, q; ketfirst=true), ns), ns)
    @test Y1 ≈ Y2 atol=1e-12
    @test Y1 != dmat_x(mpo_x(X, ns), ns)

    # Two-particle QFT = single-particle QFT on each register (A ⊗ B maps to
    # conjugate_by_qft(A) ⊗ conjugate_by_qft(B)); the swapprime that compensated the
    # transpose is gone.
    Hx = exciton_hamiltonian("chain_1d", 1.0, x -> 1.0; L=2, scale=5.0)
    se, sh = Hx.sites[1:2:end], Hx.sites[2:2:end]
    Ae, Bh = randn(ComplexF64, 4, 4), randn(ComplexF64, 4, 4)
    W  = mpo_x(kron(Ae, Bh), [se; sh]; order=Hx.sites)
    Wk = conjugate_by_qft_exciton(Hx, W; tol=1e-14, maxdim=200)
    ce = dmat_x(conjugate_by_qft(mpo_x(Ae, se); tol=1e-14), se)
    ch = dmat_x(conjugate_by_qft(mpo_x(Bh, sh); tol=1e-14), sh)
    @test dmat_x(Wk, [se; sh]) ≈ kron(ce, ch) atol=1e-10

    # Krylov Green's function of a complex, non-symmetric Hermitian H: (z − H)⁻¹, not
    # its transpose, whichever order the H tensors store their legs in.
    H0 = get_Hamiltonian("chain_1d", 1.0; L=2)
    Dh = randherm(4; seed=7)
    z  = 0.3 + 0.2im
    for ketfirst in (false, true)
        G = get_green_krylov(mpo_x(Dh, H0.sites; ketfirst), H0.sites, 0.3; η=0.2,
                             nsweeps=6, maxdim=32)
        @test dmat_x(G, H0.sites) ≈ inv(z * I - Dh) atol=1e-10
    end

    # Polarization bubble on the same H with the exact projector: the Lindhard
    # Π_ij = Σ_ab (f_b − f_a)/(z + E_a − E_b) ⟨a|n_i|b⟩⟨b|n_j|a⟩ (it used to be Πᵀ).
    E, ψ = eigen(Hermitian(Dh))
    Dp = ψ[:, E .< 0] * ψ[:, E .< 0]'
    H  = TBHamiltonian(H0; mpo=mpo_x(Dh, H0.sites), scale=1.1 * opnorm(Dh), center=0.0)
    H._density_cache = mpo_x(Dp, H.sites)
    f  = Float64.(E .< 0)
    ΠL = [sum((f[b] - f[a]) / (z + E[a] - E[b]) *
              conj(ψ[i, a]) * ψ[i, b] * conj(ψ[j, b]) * ψ[j, a] for a in 1:4, b in 1:4)
          for i in 1:4, j in 1:4]
    @test norm(ΠL - transpose(ΠL)) > 1e-2
    Π = get_bubble_mpo(H, H, 0.3; GF_method=:krylov, η=0.2, krylov_nsweeps=10,
                       krylov_maxdim=64, krylov_cutoff=1e-12, cutoff=1e-12, maxdim=64)
    @test dmat_x(Π, H.sites) ≈ ΠL atol=1e-8

    # Dyson solve: (I − Π₀V) χ = Π₀, not (I − Π₀V)ᵀ (they differ for complex Π₀, and
    # for a real V that does not commute with Π₀). The result is vec(χ), rows on the odd
    # and columns on the even sites (test/bugfix_rpa6.jl has the full check).
    Dπ = 0.3 * randn(ComplexF64, 4, 4)
    Dv = Diagonal([0.5, 0.8, 0.2, 0.6])
    out = siteinds("Qubit", 2)
    fs  = TBX._rpa_pair_sites(out)
    x   = rpa_from_bubble_diag(mpo_x(Dπ, out), mpo_x(Matrix{ComplexF64}(Dv), out), fs, out;
                               nsweeps=6, maxdim=32, cutoff=1e-14)
    χv  = reshape(dvec_x(x, [fs[2:2:end]; fs[1:2:end]]), 4, 4)   # [column reg, row reg]
    @test χv ≈ (I - Dπ * Dv) \ Dπ atol=1e-10
end

@testset "NH: Z_spatial keeps its imaginary part; hermitize(NH) and aux_side" begin
    H = get_Hamiltonian("chain_1d", 1.0; L=3)
    add_loss!(H, n -> 0.1 + 0.05n)

    # nh_spectrum_grid(:diag): Z_spatial was real(A(r, z)) while Z kept the complex sum.
    out = nh_spectrum_grid(H, (-1.0, 1.0), 2, (-0.5, -0.5), 1, 3; scale=4.0, maxdim=20,
                           mode=:diag)
    Z, Zs = out[3], out[4]
    @test eltype(Zs) == ComplexF64 && size(Zs) == (H.N, 1, 2)
    @test dropdims(sum(Zs; dims=1); dims=1) ≈ Z rtol=1e-10
    @test maximum(abs, imag(Zs)) > 1e-8
    A, _ = TBX._nh_diag_online(hermitize(H; z=1.0 - 0.5im, scale=4.0, maxdim=20), 3;
                               scale=4.0, maxdim=20)
    @test real.(Zs[:, 1, 2]) ≈ [eval_mps(A, i) for i in 0:H.N-1] rtol=1e-12   # real part as before

    # hermitize(NH) used to rebuild with the default convention and scale = 0.0.
    NH1 = hermitize(H; z=0.2 + 0.1im, convention=:H_minus_z, scale=2.0,
                    block_placement=:pre)
    @test NH1.convention === :H_minus_z && NH1.scale == 2.0
    NH2 = hermitize(NH1; z=-0.4im)
    ref = hermitized_hamiltonian(H; z=-0.4im, convention=:H_minus_z, block_placement=:pre)
    @test NH2.convention === :H_minus_z && NH2.scale == 2.0 && NH2.hermitized.scale == 2.0
    @test NH2.block_placement === :pre && NH2.z == -0.4im
    @test dmat_x(NH2.hermitized.mpo, NH2.hermitized.sites) ≈ dmat_x(ref.mpo, ref.sites)
    NH3 = hermitize(NH1; convention=:z_minus_H, scale=0.0)       # explicit keywords win
    @test NH3.convention === :z_minus_H && NH3.hermitized.scale == 0.0
    # The five-argument constructor records the hermitize defaults.
    NH5 = TBX.NonHermitianHamiltonian(H, 0.1 + 0.0im, NH1.block_s, NH1.hermitized, :pre)
    @test NH5.convention === :z_minus_H && NH5.scale == 0.0

    # aux_side is the side of the block index; L and N count the position register.
    for bp in (:post, :pre)
        Hh = hermitized_hamiltonian(H; z=0.3, block_placement=bp)
        @test Hh.aux_side === bp
        @test (Hh.L, Hh.N) == (H.L, H.N)
    end
end

@testset "_nh_resolve_scale: scale = 0.0 means not given, like scale = nothing" begin
    H = get_Hamiltonian("chain_1d", 1.0; L=3)
    add_loss!(H, n -> 0.1 + 0.05n)
    NHs = hermitize(H; z=0.3 - 0.2im, scale=3.0)
    @test TBX._nh_resolve_scale(NHs; scale=0.0) == TBX._nh_resolve_scale(NHs) == 3.0
    @test TBX._nh_resolve_scale(NHs; scale=1.5) == 1.5
    NH0 = hermitize(H; z=0.3 - 0.2im)                           # hermitized.scale == 0.0
    Random.seed!(5); a = TBX._nh_resolve_scale(NH0; maxdim=20)
    Random.seed!(5); b = TBX._nh_resolve_scale(NH0; scale=0.0, maxdim=20)
    @test a == b && a > 0
    @test_throws ErrorException TBX._nh_resolve_scale(NHs; scale=-1.0)
end

@testset "rk4_step_dm_nh_gpu: step coefficients in the precision of ρ" begin
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
        H = get_Hamiltonian("chain_1d", 1.0; L=3, scale=3.0)
        add_loss!(H, n -> 0.1 + 0.05n)
        ψ = binary_to_MPS(2, length(H.sites), H.sites)
        ρ = outer(ψ', ψ)
        dt, kw = 0.1, (maxdim=64, cutoff=1e-12)
        ref = dmat_x(TBX.rk4_step_dm_nh(_ -> H.mpo, ρ, 0.0, dt; kw...), H.sites)
        for (T, tol) in ((ComplexF64, 1e-12), (ComplexF32, 1e-5))
            Hg = TBX._to_gpu(H.mpo, T)
            r  = TBX.rk4_step_dm_nh_gpu(Hg, conj(swapprime(Hg, 0, 1)), TBX._to_gpu(ρ, T), dt;
                                        kw...)
            @test all(t -> eltype(t) == T, r)
            # ComplexF64 steps used Float32(dt / 2) etc.: ~1e-9 off the CPU step.
            @test dmat_x(host_x(r), H.sites) ≈ ref atol=tol
        end
    end
end
