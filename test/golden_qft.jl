using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: get_Hamiltonian, get_bands, conjugate_by_qft, conjugate_by_qft_exciton,
                     get_exciton_bands, get_exciton_continuum, project_aux, aux_site,
                     kpath_setup, kpath_2d, hsk_honeycomb, hsk_square, hsk_triangular,
                     add_spin!, add_superconductivity!, fibonacci_hamiltonian, TBHamiltonian

# Characterization ("golden") tests for src/physics/QFT_tk.jl.
#
# These tests pin what the QFT/band-structure code computes *today*, bugs
# included, so that the Tier 1 reorganisation (docs/dev/REORGANISATION_TODO.md:
# splitting QFT_tk.jl into Conjugation/Bands/KPath, moving the exciton spectra
# and the aux projection helpers) cannot silently change an output. The expected
# values live in test/data/qft_golden.jl, written by
# test/data/generate_qft_golden.jl from the cases defined below.
#
# Covered: conjugate_by_qft (MPO and TBHamiltonian methods), _embed_in_full_sites,
# _embed_displacement_in_full_sites, conjugate_by_qft_exciton, _eval_diag_mps,
# _kpm_weight_matrix (every kernel), kpath_2d, hsk_honeycomb/square/triangular,
# _hs_label, _hsk, kpath_setup, the low-level get_bands (1D/2D grids, num_avg,
# windows, k_groups_override, legacy sublattice masks, spin/Nambu/layer/sublattice
# projections, printinfo), get_bands(H, Ncheb, D, ω) on every aux model (spin,
# BdG, BdG+spin, layer, layer+sublattice, honeycomb, kagome, SSH, plain chain and
# square lattice; kpath honeycomb/square/triangular, kpath_Lx), get_bands(H,
# Ncheb, ω), get_exciton_bands (every momentum-selection alias, kernels,
# errors), get_exciton_continuum (random-phase and k_list probes, seeds,
# normalize, errors), project_aux, aux_site, _autoenable_proj, and the error
# branches of all of these. Functions that Tier 1 plans to delete (projop_1DSL,
# projop_2DSL, sample_diag, project_spin) are pinned in cases marked `requires`;
# those cases are skipped once the function is gone.
#
# Not pinned, because the process dies with a segfault instead of raising:
# get_bands on a spin index added with add_spin!(H; position=:post), and the
# low-level get_bands with sublat_side=:pre on a postpended sublattice index
# (both make project_aux(side=:pre) project a site that is not at position 1).
#
# Runtime: ~2 min from a cold start, almost all of it first-call compilation of
# the code under test (ITensorMPS arithmetic, apply, the QFT MPO, the MPS-KPM).
#
# A failure means an output changed. If that is a regression, fix the code. If
# it is intentional, regenerate the data file in the same commit as the change:
#
#     JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. test/data/generate_qft_golden.jl
#
# Comparison rules (see `mismatch`):
#   * every case records the fields it returned plus everything it printed to
#     stdout (`stdout`); the field names must be the same;
#   * floating-point scalars and arrays: same type/element type and shape, and
#     every entry within ATOL + RTOL * (largest |entry| of the recorded array);
#     NaN matches NaN;
#   * everything else (integers, strings, symbols, tuples of integers, nested
#     vectors of integers, `nothing`) must be `isequal` with the same type;
#   * a case recorded as throwing must still throw the same exception type with
#     the same first MESSAGE_PREFIX_CHARS characters of its message;
#   * the case names and their order must match the data file exactly.
#
# Every case runs after `Random.seed!(case.seed)`; every fixture is built on
# first use after `Random.seed!(its own seed)`, with the global RNG saved and
# restored around it (see `fx`), so the RNG stream a case sees does not depend on
# which fixtures earlier cases already built.

module QFTGoldenRunner

using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: get_Hamiltonian, get_bands, conjugate_by_qft, conjugate_by_qft_exciton,
                     get_exciton_bands, get_exciton_continuum, project_aux, aux_site,
                     kpath_setup, kpath_2d, hsk_honeycomb, hsk_square, hsk_triangular,
                     add_spin!, add_superconductivity!, fibonacci_hamiltonian, TBHamiltonian
const TB = TensorBinding

const RTOL = 1e-10
const ATOL = 1e-12
const MESSAGE_PREFIX_CHARS = 60
const DEFAULT_SEED = 1234

# ── Dense views of tensor-network outputs ──────────────────────────────────────

# Contract `T` with onehot(i => 1) for every dim-1 index outside `keep`; any
# other extra index is an error (the output no longer lives on `keep`).
function _drop_trivial(T::ITensor, keep)
    for i in inds(T)
        i in keep && continue
        dim(i) == 1 || error("dense view: unexpected index $i (dim $(dim(i)))")
        T *= onehot(i => 1)
    end
    return T
end

# The dense views below work core by core on plain arrays of rank <= 5, never on
# a 2L-index ITensor: NDTensors compiles its permutation/contraction kernels per
# tensor rank, and a 2L-rank tensor per system size would add ~5 s each.

_link(W, j) = (1 <= j < length(W)) ? commonind(W[j], W[j+1]) : nothing

"""
    densemat(W::MPO, sites) -> Matrix

Dense matrix of `W` on `sites`: rows are the primed indices, columns the
unprimed ones, site 1 varying fastest (column-major convention).
"""
function densemat(W::MPO, sites)
    length(W) == length(sites) || error("dense view: $(length(W)) tensors for $(length(sites)) sites")
    T = promote_type((eltype(W[j]) for j in 1:length(W))...)
    M = ones(T, 1, 1, 1)                        # (rows so far, cols so far, right link)
    for j in eachindex(sites)
        s, l, r = sites[j], _link(W, j - 1), _link(W, j)
        keep = filter(!isnothing, [l, r, s', s])
        C = Array(_drop_trivial(W[j], keep), keep...)
        C = reshape(C, isnothing(l) ? 1 : dim(l), isnothing(r) ? 1 : dim(r), dim(s), dim(s))
        R, K, dl = size(M)
        dr, d = size(C, 2), size(C, 3)
        N = reshape(reshape(M, R * K, dl) * reshape(C, dl, dr * d * d), R, K, dr, d, d)
        M = reshape(permutedims(N, (1, 4, 2, 5, 3)), R * d, K * d, dr)
    end
    size(M, 3) == 1 || error("dense view: open right link of dim $(size(M, 3))")
    return M[:, :, 1]
end

"""
    densevec(psi::MPS, sites) -> Vector

Dense vector of `psi` on `sites`, site 1 varying fastest.
"""
function densevec(psi::MPS, sites)
    T = promote_type((eltype(psi[j]) for j in 1:length(psi))...)
    v = ones(T, 1, 1)                           # (entries so far, right link)
    for j in eachindex(sites)
        s, l, r = sites[j], _link(psi, j - 1), _link(psi, j)
        keep = filter(!isnothing, [l, r, s])
        C = reshape(Array(_drop_trivial(psi[j], keep), keep...),
                    isnothing(l) ? 1 : dim(l), isnothing(r) ? 1 : dim(r), dim(s))
        R, dl = size(v)
        N = reshape(v * reshape(C, dl, :), R, size(C, 2), dim(s))
        v = reshape(permutedims(N, (1, 3, 2)), R * dim(s), size(C, 2))
    end
    return v[:, 1]
end

"""
    matsummary(A) -> NamedTuple

Scalar functionals and a few slices of a dense matrix too large to store whole.
"""
matsummary(A::AbstractMatrix) =
    (; size = size(A), frob = norm(A), trace = tr(A), total = sum(A),
       diag = diag(A), row1 = A[1, :], col1 = A[:, 1], row_last = A[end, :])

# Exact tensor train of `x` (entries indexed by per-site digits p_j in 0:q_j-1,
# site 1 fastest) by sequential SVDs of 2D matrices; singular values below
# 1e-13 of the largest are dropped. Returns cores C_j[left, p_j, right].
function _tt_cores(x::AbstractVector, q::Vector{Int})
    cores = Array{eltype(x),3}[]
    Ψ, χ = reshape(collect(x), 1, :), 1
    for j in eachindex(q)
        Ψ = reshape(Ψ, χ * q[j], :)
        if j == length(q)
            push!(cores, reshape(Ψ, χ, q[j], 1))
            break
        end
        F = svd(Ψ)
        k = max(1, count(>(1e-13 * F.S[1]), F.S))
        push!(cores, reshape(F.U[:, 1:k], χ, q[j], k))
        Ψ, χ = Diagonal(F.S[1:k]) * F.Vt[1:k, :], k
    end
    return cores
end

_links(cores) = [Index(size(cores[j], 3), "Link,l=$j") for j in 1:length(cores)-1]

"""
    mpo_from_dense(A, sites) -> MPO

Exact MPO of the dense matrix `A` in the `densemat` convention; each tensor
has its indices in the order (left link, s', s, right link).
"""
function mpo_from_dense(A::AbstractMatrix, sites)
    d = dim.(sites); L = length(sites)
    D = prod(d); size(A) == (D, D) || error("mpo_from_dense: size $(size(A)) for $D states")
    # digits of the row/column index, site 1 fastest, interleaved as (a_j, b_j) pairs
    digits_of(n) = let out = Int[]; for dj in d; push!(out, n % dj); n ÷= dj; end; out end
    x = zeros(eltype(A), D * D)
    q = d .^ 2
    for c in 0:D-1, r in 0:D-1
        a, b = digits_of(r), digits_of(c)
        i, stride = 0, 1
        for j in 1:L
            i += (a[j] + d[j] * b[j]) * stride; stride *= q[j]
        end
        x[i+1] = A[r+1, c+1]
    end
    cores = _tt_cores(x, q); links = _links(cores)
    tensors = map(1:L) do j
        C = reshape(cores[j], size(cores[j], 1), d[j], d[j], size(cores[j], 3))
        idx = Index[]; j > 1 && push!(idx, links[j-1]); push!(idx, sites[j]', sites[j]); j < L && push!(idx, links[j])
        ITensor(reshape(C, dim.(idx)...), idx...)
    end
    return MPO(tensors)
end

"""
    mps_from_dense(v, sites) -> MPS

Exact MPS of the vector `v` in the `densevec` convention.
"""
function mps_from_dense(v::AbstractVector, sites)
    cores = _tt_cores(v, dim.(sites)); links = _links(cores); L = length(sites)
    return MPS(map(1:L) do j
        idx = Index[]; j > 1 && push!(idx, links[j-1]); push!(idx, sites[j]); j < L && push!(idx, links[j])
        ITensor(reshape(cores[j], dim.(idx)...), idx...)
    end)
end

# A deterministic, non-symmetric, complex test matrix (distinguishes U W U† from
# U Wᵀ U†, which a real symmetric Hamiltonian cannot).
testmat(n) = ComplexF64[sin(i + 2j) + (i < j ? 0.3 : 0.0) + im * cos(3i - j) / 4 for i in 1:n, j in 1:n]
testvec(n) = ComplexF64[sin(1.3i) + im * cos(0.7i) / 3 for i in 1:n]

auxinfo(H, which) = let (s, side) = aux_site(H, which)
    (; position = findfirst(==(s), H.sites), dim = dim(s), side, tags = string(tags(s)))
end

# ── stdout capture (the code prints Info lines and progress) ──────────────────
function capture_stdout(f)
    path, io = mktemp()
    try
        val = try
            redirect_stdout(f, io)
        finally
            flush(io)
            close(io)
        end
        return val, read(path, String)
    finally
        rm(path; force=true)
    end
end

# ── Fixtures ──────────────────────────────────────────────────────────────────
# Built lazily on first use, each after Random.seed!(its own seed) with the
# global RNG state saved and restored around it and its stdout discarded, so a
# case sees the same RNG stream and prints the same text whichever case happens
# to build a fixture first (or whether cases before it were skipped).
#
# Only the chain ("chain_1d"), spin (add_spin!, pre and post) and BdG
# (add_superconductivity!, p-wave and s-wave) models come from the package's
# constructors. The square lattice, the sublattice models (honeycomb, kagome,
# SSH), the layer models and the exciton are hand-built exact dense operators
# with the constructors' site layout and index tags: the QTCI presets and the
# exciton contact term cost ~30 s of first-call compilation and the explicit
# sublattice builders ~13 s, while the QFT code under test only sees the MPO,
# the site list and the TBHamiltonian fields. It also keeps these tests about
# QFT_tk.jl: a change in a lattice builder does not fail them.
const FIXTURES = Dict{Symbol,Any}()
const FIXTURE_BUILDERS = Dict{Symbol,Tuple{Int,Function}}()
fixture!(f, name, seed) = (FIXTURE_BUILDERS[name] = (seed, f); name)

function fx(name::Symbol)
    haskey(FIXTURES, name) && return FIXTURES[name]
    seed, f = FIXTURE_BUILDERS[name]
    saved = copy(Random.default_rng())
    try
        Random.seed!(seed)
        FIXTURES[name] = first(capture_stdout(f))
    finally
        copy!(Random.default_rng(), saved)
    end
    return FIXTURES[name]
end

chain(L; kw...) = get_Hamiltonian("chain_1d", 1.0; L=L, kw...)

# Bit-reversal of n on L bits: converts the package's MSB-first linear index
# (site 1 = most significant bit) to the `densemat` index (site 1 fastest).
bitrev(n, L) = sum(((n >> (L - j)) & 1) << (j - 1) for j in 1:L)

"""
    square_dense(Lx, Ly; t=-1.0) -> Matrix{Float64}

Open-boundary nearest-neighbour square lattice on 2^Lx × 2^Ly sites, linear
index n = ix + iy·2^Lx encoded MSB-first on the L = Lx + Ly qubits (the 2D
encoding documented at the top of QFT_tk.jl), returned in the `densemat`
convention.
"""
function square_dense(Lx, Ly; t=-1.0)
    Nx, Ny, L = 2^Lx, 2^Ly, Lx + Ly
    A = zeros(Float64, Nx * Ny, Nx * Ny)
    for iy in 0:Ny-1, ix in 0:Nx-1
        n = ix + iy * Nx
        ix + 1 < Nx && (A[n+1, n+2] = A[n+2, n+1] = t)
        iy + 1 < Ny && (A[n+1, n+Nx+1] = A[n+Nx+1, n+1] = t)
    end
    p = [bitrev(n, L) + 1 for n in 0:Nx*Ny-1]   # p[n+1] = densemat index of n
    B = zeros(Float64, size(A))
    B[p, p] = A
    return B
end

"""
    exciton_dense(Hc, Hv, U) -> Matrix{Float64}

H_c ⊗ I_h − I_e ⊗ H_v + diag(U(e) δ_{e,h}) on the interleaved register
(e1, h1, e2, h2, ...), from single-particle matrices in the `densemat`
convention; the structure `exciton_hamiltonian` builds.
"""
function exciton_dense(Hc::AbstractMatrix, Hv::AbstractMatrix, U::AbstractVector)
    n = size(Hc, 1); L = round(Int, log2(n))
    ebits(d) = sum(((d >> (2i)) & 1) << i for i in 0:L-1)
    hbits(d) = sum(((d >> (2i + 1)) & 1) << i for i in 0:L-1)
    M = zeros(Float64, n^2, n^2)
    for d in 0:n^2-1, d2 in 0:n^2-1
        e, h, e2, h2 = ebits(d), hbits(d), ebits(d2), hbits(d2)
        v = (h2 == h ? Hc[e2+1, e+1] : 0.0) - (e2 == e ? Hv[h2+1, h+1] : 0.0)
        d2 == d && e == h && (v += U[e+1])
        M[d2+1, d+1] = v
    end
    return M
end

"""
    sublattice_dense(Lx, Ly, nsub, bonds) -> Matrix{Float64}

Hopping matrix on [position qubits (MSB-first n = ix + iy·2^Lx); sublattice
index (postpended, slowest)] in the `densemat` convention. Each bond
`(a, b, dx, dy, t)` couples sublattice `a` at (ix, iy) to sublattice `b` at
(ix+dx, iy+dy) with amplitude `t` (plus the Hermitian conjugate); bonds leaving
the open 2^Lx × 2^Ly grid are dropped.
"""
function sublattice_dense(Lx, Ly, nsub, bonds)
    Nx, Ny, L = 2^Lx, 2^Ly, Lx + Ly
    Np = Nx * Ny
    M = zeros(Float64, Np * nsub, Np * nsub)
    for iy in 0:Ny-1, ix in 0:Nx-1, (a, b, dx, dy, t) in bonds
        jx, jy = ix + dx, iy + dy
        (0 <= jx < Nx && 0 <= jy < Ny) || continue
        i = bitrev(ix + iy * Nx, L) + Np * (a - 1) + 1
        j = bitrev(jx + jy * Nx, L) + Np * (b - 1) + 1
        M[i, j] += t; M[j, i] += t
    end
    return M
end

const HONEYCOMB_BONDS = [(1, 2, 0, 0, -1.0), (1, 2, -1, 0, -1.0), (1, 2, 0, -1, -1.0)]
const KAGOME_BONDS = [(1, 2, 0, 0, -1.0), (1, 3, 0, 0, -1.0), (2, 3, 0, 0, -1.0),
                      (1, 2, -1, 0, -1.0), (1, 3, 0, -1, -1.0), (2, 3, 1, -1, -1.0)]
const SSH_BONDS = [(1, 2, 0, 0, -1.4), (2, 1, 1, 0, -0.6)]

# A position-register TBHamiltonian (from the chain builder) re-dressed with
# extra aux sites and a dense MPO; `kw` sets the aux fields.
function dressed(L, sites_of, M; kw...)
    H = chain(L)
    sites = sites_of(H.sites)
    return TBHamiltonian(H; sites=sites, mpo=mpo_from_dense(M, sites), kw...)
end

function exciton_fixture(L, U)
    Hc, Hv = chain(L), chain(L)
    sites = collect(Iterators.flatten(zip(Hc.sites, Hv.sites)))
    M = exciton_dense(densemat(Hc.mpo, Hc.sites), densemat(Hv.mpo, Hv.sites), U)
    return TBHamiltonian(Hc; sites=sites, mpo=mpo_from_dense(M, sites), scale=5.0, center=0.0)
end

fixture!(:chain3, 11) do; chain(3) end
fixture!(:chain4p, 12) do; chain(4; boundary=:periodic) end
fixture!(:chain4c, 13) do; TBHamiltonian(chain(4; boundary=:periodic); scale=3.0, center=0.4) end
fixture!(:square4, 14) do
    H = chain(4)
    TBHamiltonian(H; mpo=mpo_from_dense(square_dense(2, 2), H.sites), scale=4.4,
                  geometry=TB._square_geometry(4), Lx=2)
end
fixture!(:ssh3, 16) do
    sl = Index(2, "SSH")
    dressed(3, s -> [s; sl], sublattice_dense(3, 0, 2, SSH_BONDS);
            sublattice_s=sl, aux_side=:post, scale=2.0)
end
fixture!(:honeycomb4, 17) do
    sl = Index(2, "Honeycomb")
    dressed(4, s -> [s; sl], sublattice_dense(2, 2, 2, HONEYCOMB_BONDS);
            sublattice_s=sl, aux_side=:post, scale=3.2, geometry=TB._square_geometry(4), Lx=2)
end
fixture!(:kagome4, 18) do
    sl = Index(3, "Kagome")
    dressed(4, s -> [s; sl], sublattice_dense(2, 2, 3, KAGOME_BONDS);
            sublattice_s=sl, aux_side=:post, scale=4.5, geometry=TB._square_geometry(4), Lx=2)
end
fixture!(:spin3, 19) do
    H = chain(3); add_spin!(H); H.scale = 2.5; H
end
fixture!(:spinpost3, 20) do
    H = chain(3); add_spin!(H; position=:post); H.scale = 2.5; H
end
fixture!(:spinsq4, 21) do
    H = TBHamiltonian(fx(:square4)); add_spin!(H); H.scale = 4.4; H
end
fixture!(:bdg3, 22) do
    H = chain(3); add_superconductivity!(H, 0.3); H.scale = 3.0; H
end
fixture!(:bdgspin3, 23) do
    H = chain(3); add_spin!(H); add_superconductivity!(H, 0.2); H.scale = 3.0; H
end
# 1D bilayer chain: a prepended Qubit layer index (as bilayer_hamiltonian uses),
# intralayer chain hopping and interlayer hopping 0.3.
fixture!(:layer3, 24) do
    H = chain(3); layer = siteinds("Qubit", 1)[1]; sites = [layer; H.sites]
    Hp = densemat(H.mpo, H.sites)
    M = kron(Hp, Matrix(1.0I, 2, 2)) + kron(Matrix(0.3I, 8, 8), [0.0 1.0; 1.0 0.0])
    TBHamiltonian(H; sites=sites, mpo=mpo_from_dense(M, sites), layer_s=layer, scale=2.8)
end
# Honeycomb bilayer, AA-stacked: [layer (Qubit, prepended); position; sublattice
# (postpended)], the layout of bilayer_hamiltonian(:honeycomb, 2, 2; sublattice=true).
fixture!(:bilayer_hc, 25) do
    layer, sl = siteinds("Qubit", 1)[1], Index(2, "Honeycomb")
    M = kron(sublattice_dense(2, 2, 2, HONEYCOMB_BONDS), Matrix(1.0I, 2, 2)) +
        kron(Matrix(0.3I, 32, 32), [0.0 1.0; 1.0 0.0])
    dressed(4, s -> [layer; s; sl], M; layer_s=layer, sublattice_s=sl, aux_side=:pre,
            scale=3.5, geometry=TB._square_geometry(4), Lx=2)
end
fixture!(:exc2, 26) do; exciton_fixture(2, fill(-1.0, 4)) end
fixture!(:exc3, 27) do; exciton_fixture(3, [-1.0 - 0.1 * x for x in 1:8]) end
fixture!(:exc3_qft, 28) do; conjugate_by_qft_exciton(fx(:exc3), fx(:exc3).mpo) end
fixture!(:fib4, 29) do; fibonacci_hamiltonian(4; A=1.0, B=2.0) end
fixture!(:nogeom4, 30) do; TBHamiltonian(fx(:chain4p); geometry=nothing) end
fixture!(:sl_interior, 31) do
    H = fx(:honeycomb4); s = H.sites
    TBHamiltonian(H; sites=[s[1:2]; s[end]; s[3:end-1]])
end
fixture!(:sl_missing, 32) do
    TBHamiltonian(fx(:honeycomb4); sublattice_s=Index(2, "Stray"))
end

# ── Cases ──────────────────────────────────────────────────────────────────────
struct Case
    name::String
    seed::Int
    throws::Bool              # the case is meant to throw (the generator refuses other throws)
    requires::Vector{Symbol}  # TensorBinding names Tier 1 may delete; skip the case once gone
    run::Function             # () -> NamedTuple of plain data
end
const CASES = Case[]
case!(f, name; seed=DEFAULT_SEED, throws=false, requires=Symbol[]) =
    push!(CASES, Case(name, seed, throws, requires, f))

# Energies: rescaled (low-level get_bands) and physical (TBHamiltonian methods),
# both reaching outside the spectral window.
const WRES  = [-1.1, -0.8, -0.3, 0.0, 0.45, 0.9, 1.0]
const WPHYS = [-3.0, -1.5, -0.5, 0.0, 0.7, 1.9, 3.5]
const WEXC  = [-6.0, -3.0, -1.2, 0.0, 0.8, 2.5, 4.9]
const NB = 12   # Chebyshev moments for the band cases

# ---- conjugate_by_qft(W) --------------------------------------------------------
case!("qft_chain3_H") do
    H = fx(:chain3); (; A = densemat(conjugate_by_qft(H.mpo), H.sites))
end
case!("qft_chain4_identity") do
    H = fx(:chain4p); (; A = densemat(conjugate_by_qft(MPO(H.sites, "Id")), H.sites))
end
case!("qft_chain3_nonsymmetric") do
    s = fx(:chain3).sites
    (; A = densemat(conjugate_by_qft(mpo_from_dense(testmat(8), s)), s))
end
case!("qft_chain3_nonsymmetric_truncated_tol1e-2_maxdim2") do
    s = fx(:chain3).sites
    (; A = densemat(conjugate_by_qft(mpo_from_dense(testmat(8), s); tol=1e-2, maxdim=2), s))
end
case!("qft_square4_H") do
    H = fx(:square4); (; A = densemat(conjugate_by_qft(H.mpo), H.sites))
end

# ---- conjugate_by_qft(H, W) and the embeddings ----------------------------------
case!("qftH_chain3_noaux") do
    H = fx(:chain3); (; A = densemat(conjugate_by_qft(H, H.mpo), H.sites))
end
case!("qftH_spin3_pre") do
    H = fx(:spin3); (; A = densemat(conjugate_by_qft(H, H.mpo), H.sites))
end
case!("qftH_spinpost3") do
    H = fx(:spinpost3); (; A = densemat(conjugate_by_qft(H, H.mpo), H.sites))
end
case!("qftH_honeycomb4_post") do
    H = fx(:honeycomb4); (; A = densemat(conjugate_by_qft(H, H.mpo; maxdim=60), H.sites))
end
case!("qftH_bdgspin3_nonsymmetric") do
    H = fx(:bdgspin3)
    W = mpo_from_dense(testmat(32), H.sites)
    (; A = densemat(conjugate_by_qft(H, W), H.sites))
end
case!("qftH_bilayer_hc_pre_post_summary") do
    H = fx(:bilayer_hc); matsummary(densemat(conjugate_by_qft(H, H.mpo), H.sites))
end
case!("embed_identity_honeycomb4_post") do
    H = fx(:honeycomb4); ps = TB._pos_sites(H)
    (; A = densemat(TB._embed_in_full_sites(H, mpo_from_dense(testmat(16), ps)), H.sites))
end
case!("embed_identity_bdgspin3_pre_pre") do
    H = fx(:bdgspin3); ps = TB._pos_sites(H)
    (; A = densemat(TB._embed_in_full_sites(H, mpo_from_dense(testmat(8), ps)), H.sites))
end
case!("embed_identity_chain3_noaux") do
    H = fx(:chain3); ps = TB._pos_sites(H)
    (; A = densemat(TB._embed_in_full_sites(H, mpo_from_dense(testmat(8), ps)), H.sites))
end
case!("embed_displacement_spin3_pre") do
    H = fx(:spin3); ps = TB._pos_sites(H)
    (; A = densemat(TB._embed_displacement_in_full_sites(H, mpo_from_dense(testmat(8), ps)), H.sites))
end
case!("embed_displacement_kagome4_post_dim3") do
    H = fx(:kagome4); ps = TB._pos_sites(H)
    (; A = densemat(TB._embed_displacement_in_full_sites(H, mpo_from_dense(testmat(16), ps)), H.sites))
end
case!("embed_both_bilayer_hc_pre_post_summary") do
    H = fx(:bilayer_hc); ps = TB._pos_sites(H); W = mpo_from_dense(testmat(16), ps)
    (; identity = matsummary(densemat(TB._embed_in_full_sites(H, W), H.sites)),
       displacement = matsummary(densemat(TB._embed_displacement_in_full_sites(H, W), H.sites)))
end

# ---- conjugate_by_qft_exciton --------------------------------------------------------
case!("qftexc_exc2_H") do
    H = fx(:exc2); (; A = densemat(conjugate_by_qft_exciton(H, H.mpo), H.sites))
end
case!("qftexc_exc2_nonsymmetric") do
    H = fx(:exc2)
    (; A = densemat(conjugate_by_qft_exciton(H, mpo_from_dense(testmat(16), H.sites)), H.sites))
end
case!("qftexc_exc3_H_summary") do
    H = fx(:exc3); matsummary(densemat(fx(:exc3_qft), H.sites))
end
case!("qftexc_exc3_H_maxdim4_summary") do
    H = fx(:exc3); matsummary(densemat(conjugate_by_qft_exciton(H, H.mpo; tol=1e-6, maxdim=4), H.sites))
end
case!("qftexc_odd_site_count"; throws=true) do
    H = fx(:chain3); conjugate_by_qft_exciton(H, H.mpo)
end

# ---- legacy helpers scheduled for deletion ---------------------------------------
case!("projop_1DSL_chain3"; requires=[:projop_1DSL]) do
    H = fx(:chain3); W = mpo_from_dense(testmat(8), H.sites)
    (; SL1 = densemat(TB.projop_1DSL(W, H.sites, 3, 1), H.sites),
       SL2 = densemat(TB.projop_1DSL(W, H.sites, 3, 2), H.sites))
end
case!("projop_2DSL_square4"; requires=[:projop_2DSL]) do
    H = fx(:square4)
    (; SL1 = densemat(TB.projop_2DSL(H.mpo, H.sites, 2, 2, 1), H.sites),
       SL2 = densemat(TB.projop_2DSL(H.mpo, H.sites, 2, 2, 2), H.sites))
end
case!("sample_diag_chain4p"; requires=[:sample_diag]) do
    H = fx(:chain4p); Tk = conjugate_by_qft(H.mpo)
    (; full = TB.sample_diag(Tk, 0, 15), part = TB.sample_diag(Tk, 3, 6))
end
case!("project_spin_spin3"; requires=[:project_spin]) do
    H = fx(:spin3); ps = TB._pos_sites(H); W = mpo_from_dense(testmat(16), H.sites)
    (; up = densemat(TB.project_spin(W, H.spin_s, 1), ps),
       dn = densemat(TB.project_spin(W, H.spin_s, 2), ps))
end
case!("project_spin_nothing"; throws=true, requires=[:project_spin]) do
    TB.project_spin(fx(:spin3).mpo, nothing, 1)
end

# ---- _eval_diag_mps ------------------------------------------------------------------
case!("eval_diag_mps_real_L4") do
    s = siteinds("Qubit", 4); psi = mps_from_dense(real.(testvec(16)), s)
    (; values = [TB._eval_diag_mps(psi, x) for x in 0:15])
end
case!("eval_diag_mps_complex_L3_realpart") do
    s = siteinds("Qubit", 3); psi = mps_from_dense(testvec(8), s)
    (; values = [TB._eval_diag_mps(psi, x) for x in 0:7], dense = densevec(psi, s))
end
case!("eval_diag_mps_of_qft_diagonal_chain4p") do
    H = fx(:chain4p); A = TB.extract_diagonal_to_mps(conjugate_by_qft(H.mpo))
    (; values = [TB._eval_diag_mps(A, x) for x in 0:15])
end

# ---- _kpm_weight_matrix ----------------------------------------------------------------
const WK = [-1.2, -1.0, -0.999, -0.7, 0.0, 0.3, 0.999, 1.0, 1.5]
case!("kpm_weights_jackson_N12") do; (; W = TB._kpm_weight_matrix(12, WK)) end
case!("kpm_weights_jackson_N5_range") do; (; W = TB._kpm_weight_matrix(5, range(-0.9, 0.9; length=4))) end
case!("kpm_weights_lorentz_N12_default_lambda") do; (; W = TB._kpm_weight_matrix(12, WK; kernel=:lorentz)) end
case!("kpm_weights_lorentz_N12_lambda2.5") do; (; W = TB._kpm_weight_matrix(12, WK; kernel=:lorentz, lambda=2.5)) end
case!("kpm_weights_fejer_N7") do; (; W = TB._kpm_weight_matrix(7, WK; kernel=:fejer)) end
case!("kpm_weights_dirichlet_N7") do; (; W = TB._kpm_weight_matrix(7, WK; kernel=:dirichlet)) end
case!("kpm_weights_unknown_kernel"; throws=true) do; TB._kpm_weight_matrix(7, WK; kernel=:hodc) end

# ---- high-symmetry k-path helpers ----------------------------------------------------------
case!("hsk_points") do
    (; honeycomb = [hsk_honeycomb(Lx, Ly) for (Lx, Ly) in ((1, 1), (2, 2), (3, 3), (4, 3), (5, 2))],
       square     = [hsk_square(Lx, Ly) for (Lx, Ly) in ((1, 1), (2, 2), (3, 2))],
       triangular = [hsk_triangular(Lx, Ly) for (Lx, Ly) in ((1, 1), (2, 2), (4, 3))])
end
case!("hs_labels") do
    (; labels = [TB._hs_label(s) for s in (:G, :M, :K, :Kp, :X, :R, :A, :Y, :Gamma)])
end
case!("hsk_dispatch") do
    (; honeycomb = TB._hsk(:honeycomb, 3, 2), square = TB._hsk(:square, 3, 2),
       triangular = TB._hsk(:triangular, 3, 2))
end
case!("hsk_dispatch_unknown_lattice"; throws=true) do; TB._hsk(:kagome, 2, 2) end
case!("kpath_2d_default_npts") do
    kg, ticks = kpath_2d([(0, 0), (4, 2), (5, 3), (0, 0)], 3); (; kg, ticks)
end
case!("kpath_2d_npts3") do
    kg, ticks = kpath_2d([(0, 0), (3, 1), (2, 3)], 2; npts_per_segment=3); (; kg, ticks)
end
case!("kpath_2d_npts1") do
    kg, ticks = kpath_2d([(1, 0), (3, 3), (0, 2), (1, 0)], 2; npts_per_segment=1); (; kg, ticks)
end
case!("kpath_2d_single_vertex") do
    kg, ticks = kpath_2d([(2, 1)], 2); (; kg, ticks)
end
case!("kpath_setup_honeycomb_GMKpG") do
    kg, ticks, labels = kpath_setup(:honeycomb, 3, 3, [:G, :M, :Kp, :G]; npts_per_segment=4)
    (; kg, ticks, labels)
end
case!("kpath_setup_square_GXMG_default") do
    kg, ticks, labels = kpath_setup(:square, 2, 2, [:G, :X, :M, :G]); (; kg, ticks, labels)
end
case!("kpath_setup_triangular_GMKG") do
    kg, ticks, labels = kpath_setup(:triangular, 3, 2, [:G, :M, :K, :G]; npts_per_segment=2)
    (; kg, ticks, labels)
end
case!("kpath_setup_unknown_symbol"; throws=true) do; kpath_setup(:square, 2, 2, [:G, :Kp]) end

# ---- get_bands, low-level MPO method -----------------------------------------------------
lowbands(H, D; kw...) = (; Ak = get_bands(H.mpo, H.scale, H.center, H.sites, NB, D, WRES; kw...))

case!("bands_low_1d_default_grid") do; lowbands(fx(:chain4p), 1) end
case!("bands_low_1d_numx4_navg3_window_lorentz") do
    lowbands(fx(:chain4p), 1; num_x=4, num_avg=3, xmin=2, xmax=12, kernel=:lorentz, lambda=3.0)
end
case!("bands_low_1d_center_scale") do
    H = fx(:chain4p)
    (; Ak = get_bands(H.mpo, 3.0, 0.4, H.sites, NB, 1, WRES; num_x=5))
end
case!("bands_low_1d_k_groups_override") do
    lowbands(fx(:chain4p), 1; k_groups_override=[[0], [3, 4], [8], [15, 0]])
end
case!("bands_low_1d_tol_maxdim_cutoff") do
    lowbands(fx(:chain4p), 1; num_x=6, tol=1e-4, maxdim=3, cutoff=1e-6)
end
case!("bands_low_1d_sublattice_mask_both") do; lowbands(fx(:chain4p), 1; num_x=4, sublattice=true) end
case!("bands_low_1d_sublattice_mask_A") do; lowbands(fx(:chain4p), 1; num_x=4, sublattice=true, proj_sl=1) end
case!("bands_low_1d_sublattice_mask_B") do; lowbands(fx(:chain4p), 1; num_x=4, sublattice=true, proj_sl=2) end
case!("bands_low_2d_default_grid") do; lowbands(fx(:square4), 2) end
case!("bands_low_2d_numx3_navg2_numy3") do
    lowbands(fx(:square4), 2; num_x=3, num_avg=2, xmin=0, xmax=3, ymin=0, ymax=3, num_y=3)
end
# In 2D the centres are ilinspace(xmin, xmax, 2^Lx), so any window narrower
# than 2^Lx points trips ilinspace's assertion (pinned as it is today).
case!("bands_low_2d_narrow_window"; throws=true) do
    lowbands(fx(:square4), 2; num_x=3, xmin=1, xmax=3)
end
case!("bands_low_2d_numx_exceeds_grid") do; lowbands(fx(:square4), 2; num_x=9) end
case!("bands_low_2d_sublattice_mask_both") do; lowbands(fx(:square4), 2; num_x=4, sublattice=true) end
case!("bands_low_2d_sublattice_mask_B") do; lowbands(fx(:square4), 2; num_x=4, sublattice=true, proj_sl=2) end
case!("bands_low_spin_proj_sites1_fallback") do; lowbands(fx(:spin3), 1; num_x=4, spin_proj=true) end
case!("bands_low_spin_proj_down_explicit_index") do
    H = fx(:spin3); lowbands(H, 1; num_x=4, spin_proj=true, proj_s=2, spin_s_aux=H.spin_s)
end
case!("bands_low_nambu_proj_particle") do
    H = fx(:bdg3); lowbands(H, 1; num_x=4, nambu_proj=true, proj_nambu=1, nambu_s=H.nambu_s)
end
case!("bands_low_layer_proj_both") do
    H = fx(:layer3); lowbands(H, 1; num_x=4, layer_proj=true, layer_s=H.layer_s)
end
case!("bands_low_sublat_proj_post") do
    H = fx(:honeycomb4); lowbands(H, 2; num_x=4, sublat_proj=true, sublat_s=H.sublattice_s)
end
# Not pinned: sublat_side=:pre with the (postpended) honeycomb index makes
# project_aux contract a position tensor with the aux projector, and the
# process then dies with a segfault (exit 139) instead of raising an error.
# sublat_s given with sublat_proj=false: L_pos drops the aux site but the QFT is
# applied to the full 5-site MPO (pinned as it is today).
case!("bands_low_sublat_index_without_projection") do
    H = fx(:honeycomb4); lowbands(H, 2; num_x=4, sublat_s=H.sublattice_s)
end
case!("bands_low_printinfo") do; lowbands(fx(:chain4p), 1; num_x=3, printinfo=true) end
case!("bands_low_hodc_kernel"; throws=true) do; lowbands(fx(:chain4p), 1; num_x=3, kernel=:hodc) end

# ---- get_bands(H, Ncheb, D, ω) -----------------------------------------------------------
hbands(name, D; kw...) = (; Ak = get_bands(fx(name), NB, D, WPHYS; kw...))
hbands_nt(name, D; kw...) = let r = get_bands(fx(name), NB, D, WPHYS; kw...)
    (; Ak = r.Ak, ticks = r.ticks, labels = r.labels)
end

# The TBHamiltonian method's default num_x=60 exceeds 2^L for L < 6 (1D).
case!("bandsH_chain4p_default_numx60_exceeds_N"; throws=true) do; hbands(:chain4p, 1) end
case!("bandsH_chain4p_numx16_full_grid") do; hbands(:chain4p, 1; num_x=16) end
case!("bandsH_chain4p_numx5_navg2") do; hbands(:chain4p, 1; num_x=5, num_avg=2) end
case!("bandsH_chain4c_center") do; hbands(:chain4c, 1; num_x=5) end
case!("bandsH_chain4p_k_groups_override") do; hbands(:chain4p, 1; k_groups_override=[[1], [2, 14]]) end
case!("bandsH_square4_numx4") do; hbands(:square4, 2; num_x=4) end
case!("bandsH_square4_kpath_square") do
    hbands_nt(:square4, 2; kpath=[:G, :X, :M, :G], kpath_lattice=:square, num_x=2)
end
case!("bandsH_square4_kpath_square_Lx1") do
    hbands_nt(:square4, 2; kpath=[:G, :X, :M], kpath_lattice=:square, kpath_Lx=1, num_x=2)
end
case!("bandsH_square4_kpath_triangular") do
    hbands_nt(:square4, 2; kpath=[:G, :M, :K, :G], kpath_lattice=:triangular, num_x=2)
end
case!("bandsH_honeycomb4_kpath_honeycomb") do
    hbands_nt(:honeycomb4, 2; kpath=[:G, :M, :Kp, :G], kpath_lattice=:honeycomb, num_x=2)
end
case!("bandsH_kpath_without_lattice"; throws=true) do; hbands(:square4, 2; kpath=[:G, :X]) end
case!("bandsH_square4_sublattice_mask_both") do; hbands(:square4, 2; num_x=4, sublattice=true) end
case!("bandsH_square4_sublattice_mask_A") do; hbands(:square4, 2; num_x=4, sublattice=true, proj_sl=1) end
case!("bandsH_chain4p_sublattice_mask_B_1d") do; hbands(:chain4p, 1; num_x=4, sublattice=true, proj_sl=2) end
case!("bandsH_spin3_auto_both") do; hbands(:spin3, 1; num_x=4) end
case!("bandsH_spin3_up") do; hbands(:spin3, 1; num_x=4, proj_s=1) end
case!("bandsH_spin3_down_explicit_flag") do; hbands(:spin3, 1; num_x=4, spin_proj=true, proj_s=2) end
# Not pinned: get_bands on a spin index added with add_spin!(H; position=:post)
# projects the spin with side=:pre (hard-coded) and the process dies with a
# segfault (exit 139), like the sublat_side case above.
case!("bandsH_spinsq4_auto_2d") do; hbands(:spinsq4, 2; num_x=3) end
case!("bandsH_bdg3_auto_both") do; hbands(:bdg3, 1; num_x=4) end
case!("bandsH_bdg3_hole") do; hbands(:bdg3, 1; num_x=4, proj_nambu=2) end
case!("bandsH_bdgspin3_auto_all") do; hbands(:bdgspin3, 1; num_x=3) end
case!("bandsH_bdgspin3_particle_down") do; hbands(:bdgspin3, 1; num_x=3, proj_nambu=1, proj_s=2) end
case!("bandsH_layer3_auto") do; hbands(:layer3, 1; num_x=4) end
case!("bandsH_layer3_layer2") do; hbands(:layer3, 1; num_x=4, proj_layer=2) end
case!("bandsH_bilayer_hc_auto") do; hbands(:bilayer_hc, 2; num_x=3) end
case!("bandsH_bilayer_hc_layer1_sl2") do; hbands(:bilayer_hc, 2; num_x=3, proj_layer=1, proj_sl=2) end
case!("bandsH_honeycomb4_auto") do; hbands(:honeycomb4, 2; num_x=4) end
case!("bandsH_honeycomb4_sl1_navg2") do; hbands(:honeycomb4, 2; num_x=4, proj_sl=1, num_avg=2) end
case!("bandsH_kagome4_auto") do; hbands(:kagome4, 2; num_x=4) end
case!("bandsH_kagome4_sl3") do; hbands(:kagome4, 2; num_x=4, proj_sl=3) end
case!("bandsH_ssh3_auto_1d") do; hbands(:ssh3, 1; num_x=4) end
case!("bandsH_fibonacci_not_binary"; throws=true) do; get_bands(fx(:fib4), 4, 1, [0.0]) end

# ---- get_bands(H, Ncheb, ω): D from H.geometry ------------------------------------------
case!("bands3_chain4p_D_inferred_1") do; (; Ak = get_bands(fx(:chain4p), NB, WPHYS; num_x=5)) end
case!("bands3_honeycomb4_D_inferred_2") do; (; Ak = get_bands(fx(:honeycomb4), NB, WPHYS; num_x=3)) end
case!("bands3_square4_kpath_forwarded") do
    r = get_bands(fx(:square4), NB, WPHYS; kpath=[:G, :M], kpath_lattice=:square, num_x=3)
    (; Ak = r.Ak, ticks = r.ticks, labels = r.labels)
end
case!("bands3_no_geometry"; throws=true) do; get_bands(fx(:nogeom4), NB, WPHYS) end

# ---- get_exciton_bands --------------------------------------------------------------------
excb(; kw...) = (; A = get_exciton_bands(fx(:exc3_qft), fx(:exc3), NB, WEXC; kw...))

case!("excbands_default_all_Q") do; excb() end
case!("excbands_Q_list") do; excb(Q_list=[1, 4, 8]) end
case!("excbands_K_list_alias") do; excb(K_list=[2, 3]) end
case!("excbands_Q_groups") do; excb(Q_groups=[[1, 2], [5]]) end
case!("excbands_q_groups_alias") do; excb(q_groups=[[3, 7, 8]]) end
case!("excbands_K_groups_alias") do; excb(K_groups=[[6], [2, 4]]) end
case!("excbands_k_groups_alias_flat") do; excb(k_groups=[2, 5]) end
case!("excbands_num_q_navg_window") do; excb(num_q=3, num_avg=2, q_start=2, q_end=7) end
case!("excbands_num_k_k_window_aliases") do; excb(num_k=2, k_start=3, k_end=6) end
case!("excbands_lorentz") do; excb(Q_list=[1, 5], kernel=:lorentz, lambda=2.0) end
case!("excbands_hodc_default_eta") do; excb(Q_list=[2], kernel=:hodc) end
case!("excbands_hodc_eta_morder") do; excb(Q_list=[2], kernel=:hodc, eta=0.1, m_order=6) end
case!("excbands_maxdim_cutoff") do; excb(Q_list=[3], maxdim=2, cutoff=1e-4) end
case!("excbands_verbose") do; excb(num_q=5, verbose=true) end
case!("excbands_both_lists"; throws=true) do; excb(Q_list=[1], K_list=[2]) end
case!("excbands_two_groups"; throws=true) do; excb(Q_groups=[[1]], k_groups=[[2]]) end
case!("excbands_list_and_group"; throws=true) do; excb(Q_list=[1], q_groups=[[2]]) end
case!("excbands_num_q_zero"; throws=true) do; excb(num_q=0) end
case!("excbands_num_avg_zero"; throws=true) do; excb(num_avg=0) end
case!("excbands_bad_window"; throws=true) do; excb(q_start=5, q_end=4) end
case!("excbands_num_q_exceeds_window"; throws=true) do; excb(num_q=4, q_start=1, q_end=3) end
case!("excbands_Q_out_of_range"; throws=true) do; excb(Q_list=[9]) end
case!("excbands_empty_group"; throws=true) do; excb(Q_groups=[Int[]]) end
case!("excbands_not_exciton"; throws=true) do
    get_exciton_bands(fx(:chain4p).mpo, fx(:chain4p), NB, WEXC)
end
case!("excbands_HQFT_wrong_length"; throws=true) do
    get_exciton_bands(fx(:chain3).mpo, fx(:exc3), NB, WEXC)
end
case!("excbands_fibonacci_not_binary"; throws=true) do
    get_exciton_bands(fx(:fib4).mpo, fx(:fib4), NB, WEXC)
end

# ---- get_exciton_continuum ----------------------------------------------------------------
excc(; kw...) = (; A = get_exciton_continuum(fx(:exc3_qft), fx(:exc3), NB, WEXC; kw...))

case!("exccont_default_seed42") do; excc() end
case!("exccont_Q_list_Nsample3_seed7") do; excc(Q_list=[1, 3], N_sample=3, seed=7) end
case!("exccont_seed_nothing_global_rng"; seed=99) do; excc(Q_list=[2, 6], N_sample=2, seed=nothing) end
case!("exccont_k_list") do; excc(Q_list=[2], k_list=[1, 2, 5]) end
case!("exccont_k_list_Nsample0_ok") do; excc(Q_list=[4], k_list=[3], N_sample=0) end
case!("exccont_not_normalized") do; excc(Q_list=[1, 8], N_sample=2, normalize=false) end
case!("exccont_num_q_window") do; excc(num_q=3, q_start=2, q_end=8, N_sample=1) end
case!("exccont_lorentz") do; excc(Q_list=[5], N_sample=2, kernel=:lorentz, lambda=3.0) end
case!("exccont_hodc") do; excc(Q_list=[5], N_sample=2, kernel=:hodc, eta=0.05) end
case!("exccont_verbose") do; excc(Q_list=[3], N_sample=5, verbose=true, printinfo=true) end
case!("exccont_num_q_zero"; throws=true) do; excc(num_q=0) end
case!("exccont_bad_window"; throws=true) do; excc(q_start=6, q_end=2) end
case!("exccont_num_q_exceeds_window"; throws=true) do; excc(num_q=5, q_start=1, q_end=4) end
case!("exccont_Q_out_of_range"; throws=true) do; excc(Q_list=[0]) end
case!("exccont_empty_Q_list"; throws=true) do; excc(Q_list=Int[]) end
case!("exccont_empty_k_list"; throws=true) do; excc(Q_list=[1], k_list=Int[]) end
case!("exccont_k_out_of_range"; throws=true) do; excc(Q_list=[1], k_list=[9]) end
case!("exccont_Nsample_zero"; throws=true) do; excc(Q_list=[1], N_sample=0) end
case!("exccont_not_exciton"; throws=true) do
    get_exciton_continuum(fx(:chain4p).mpo, fx(:chain4p), NB, WEXC)
end

# ---- project_aux ------------------------------------------------------------------------
case!("project_aux_spin3_pre") do
    H = fx(:spin3); ps = TB._pos_sites(H); W = mpo_from_dense(testmat(16), H.sites)
    (; up = densemat(project_aux(W, H.spin_s, 1), ps), dn = densemat(project_aux(W, H.spin_s, 2; side=:pre), ps))
end
case!("project_aux_honeycomb4_post") do
    H = fx(:honeycomb4); ps = TB._pos_sites(H)
    (; A = densemat(project_aux(H.mpo, H.sublattice_s, 1; side=:post), ps),
       B = densemat(project_aux(H.mpo, H.sublattice_s, 2; side=:post), ps))
end
case!("project_aux_kagome4_post_sector3") do
    H = fx(:kagome4); ps = TB._pos_sites(H)
    (; C = densemat(project_aux(H.mpo, H.sublattice_s, 3; side=:post), ps))
end
case!("project_aux_bdgspin3_nambu_then_spin") do
    H = fx(:bdgspin3); W1 = project_aux(H.mpo, H.nambu_s, 2)
    (; after_nambu = densemat(W1, H.sites[2:end]),
       after_spin = densemat(project_aux(W1, H.spin_s, 1), TB._pos_sites(H)))
end
case!("project_aux_nothing_index"; throws=true) do; project_aux(fx(:chain3).mpo, nothing, 1) end

# ---- aux_site -------------------------------------------------------------------------
case!("aux_site_all_models") do
    (; spin3 = auxinfo(fx(:spin3), :spin), spinpost3 = auxinfo(fx(:spinpost3), :spin),
       bdg3 = auxinfo(fx(:bdg3), :nambu),
       bdgspin3 = auxinfo(fx(:bdgspin3), :nambu),
       layer3 = auxinfo(fx(:layer3), :layer),
       bilayer_hc = (auxinfo(fx(:bilayer_hc), :layer), auxinfo(fx(:bilayer_hc), :sublattice)),
       honeycomb4 = auxinfo(fx(:honeycomb4), :sublattice),
       kagome4 = auxinfo(fx(:kagome4), :sublattice), ssh3 = auxinfo(fx(:ssh3), :sublattice))
end
# BdG on a spinful model puts the spin index second ([nambu, spin, pos...]), so
# aux_site(H, :spin) refuses it as interior (get_bands reads H.spin_s instead).
case!("aux_site_bdgspin3_spin_interior"; throws=true) do; aux_site(fx(:bdgspin3), :spin) end
case!("aux_site_unknown_kind"; throws=true) do; aux_site(fx(:spin3), :orbital) end
case!("aux_site_missing_index"; throws=true) do; aux_site(fx(:chain3), :spin) end
case!("aux_site_interior_position"; throws=true) do; aux_site(fx(:sl_interior), :sublattice) end
case!("aux_site_index_not_in_sites"; throws=true) do; aux_site(fx(:sl_missing), :sublattice) end

# ---- _autoenable_proj -----------------------------------------------------------------
case!("autoenable_proj_all_models_flags_off") do
    names = (:chain3, :spin3, :bdg3, :bdgspin3, :layer3, :bilayer_hc, :honeycomb4, :kagome4, :exc3)
    (; flags = [TB._autoenable_proj(fx(n), false, false, false, false) for n in names])
end
case!("autoenable_proj_flags_already_on") do
    (; flags = [TB._autoenable_proj(fx(n), true, true, true, true) for n in (:chain3, :bdgspin3, :bilayer_hc)],
       mixed = TB._autoenable_proj(fx(:bdgspin3), true, false, false, false))
end

# ── Running and comparing ───────────────────────────────────────────────────────

"""
    run_case(case) -> NamedTuple

Seed the global RNG, run the case with stdout captured, and return its fields
plus `stdout` (everything it printed).
"""
function run_case(case::Case)
    Random.seed!(case.seed)
    value, out = capture_stdout(case.run)
    value isa NamedTuple || error("case $(case.name) returned a $(typeof(value)), not a NamedTuple")
    return merge(value, (; stdout = out))
end

available(case::Case) = all(n -> isdefined(TB, n), case.requires)

function message_prefix(err)
    hasfield(typeof(err), :msg) || return nothing
    msg = getfield(err, :msg)
    msg isa AbstractString || return nothing
    return String(first(first(split(msg, '\n')), MESSAGE_PREFIX_CHARS))
end

_isfloatlike(::Type{T}) where {T} = T <: Union{AbstractFloat, Complex{<:AbstractFloat}}
_close(a, e, tol) = (isnan(a) && isnan(e)) || a == e || abs(a - e) <= tol

"""
    mismatch(path, actual, expected) -> Union{Nothing,String}

`nothing` when `actual` matches the golden `expected` under the rules in the
file header, otherwise a message naming the first differing field.
"""
function mismatch(path::String, @nospecialize(a), @nospecialize(e))
    if e isa NamedTuple
        a isa NamedTuple || return "$path: got a $(typeof(a)), expected a NamedTuple"
        keys(a) == keys(e) || return "$path: fields $(keys(a)), expected $(keys(e))"
        for k in keys(e)
            m = mismatch("$path.$k", a[k], e[k]); m === nothing || return m
        end
        return nothing
    elseif e isa Tuple
        (a isa Tuple && length(a) == length(e)) || return "$path: got $(repr(a)), expected $(repr(e))"
        for i in eachindex(e)
            m = mismatch("$path[$i]", a[i], e[i]); m === nothing || return m
        end
        return nothing
    elseif e isa AbstractArray
        a isa AbstractArray || return "$path: got a $(typeof(a)), expected an array"
        size(a) == size(e) || return "$path: size $(size(a)), expected $(size(e))"
        eltype(a) == eltype(e) || return "$path: element type $(eltype(a)), expected $(eltype(e))"
        if _isfloatlike(eltype(e))
            tol = ATOL + RTOL * maximum(abs, e; init=0.0)
            i = findfirst(k -> !_close(a[k], e[k], tol), eachindex(e))
            i === nothing && return nothing
            return "$path: entry $(Tuple(CartesianIndices(e)[i])) is $(a[i]), expected $(e[i]) (tolerance $tol)"
        end
        for (i, k) in enumerate(eachindex(e))
            m = mismatch("$path[$i]", a[k], e[k]); m === nothing || return m
        end
        return nothing
    elseif _isfloatlike(typeof(e))
        typeof(a) == typeof(e) || return "$path: type $(typeof(a)), expected $(typeof(e))"
        _close(a, e, ATOL + RTOL * abs(e)) && return nothing
        return "$path: got $(repr(a)), expected $(repr(e))"
    else
        (typeof(a) == typeof(e) && isequal(a, e)) && return nothing
        return "$path: got $(repr(a)) ($(typeof(a))), expected $(repr(e)) ($(typeof(e)))"
    end
end

function check_case(case::Case, @nospecialize(golden))
    if golden.throws !== nothing
        try
            run_case(case)
        catch err
            if !(err isa golden.throws)
                @error "QFT output changed: different exception" case = case.name expected = golden.throws got = typeof(err)
                return false
            end
            got = message_prefix(err)
            got == golden.expected.message_prefix && return true
            @error "QFT output changed: different error message" case = case.name expected = golden.expected.message_prefix got
            return false
        end
        @error "QFT output changed: case no longer throws" case = case.name expected = golden.throws
        return false
    end
    actual = try
        run_case(case)
    catch err
        @error "QFT output changed: case now throws" case = case.name exception = (err, catch_backtrace())
        return false
    end
    m = mismatch(case.name, actual, golden.expected)
    m === nothing && return true
    @error "QFT output changed" detail = m
    return false
end

function run_tests(golden)
    @testset "QFT outputs are pinned" begin
        @test [c.name for c in CASES] == [g.name for g in golden]
        empty!(FIXTURES)
        nskip = 0
        for (case, g) in zip(CASES, golden)
            if !available(case)
                nskip += 1
                continue
            end
            @test check_case(case, g)
        end
        nskip > 0 && @info "golden_qft: skipped $nskip case(s) whose functions were removed"
    end
end

end # module QFTGoldenRunner

if !isdefined(@__MODULE__, :QFT_GOLDEN_GENERATOR)
    QFTGoldenRunner.run_tests(include(joinpath(@__DIR__, "data", "qft_golden.jl")))
end
