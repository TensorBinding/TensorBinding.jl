using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: tdvp_evolve, apply_mpo_to_mps, evolve_with_propagator, evolve_with_tdvp,
                     evolve_with_tdvp_timedep, compute_basis_overlaps, basis_amplitude,
                     phase_aligned_distance, rk4_step_dm_timedep, evolve_rk4_dm_timedep,
                     rk4_step_dm_nh, evolve_rk4_dm_nh, dm_expect, observables_trajectory,
                     timedep_observable_trajectory, purity, purity_trajectory, bond_current_x,
                     bond_current_x_trajectory, central_x_bond,
                     dmrg_gs, build_K, dmrg_spectral, local_weight,
                     get_green_krylov, haydock_cf, eval_haydock_cf, haydock_resolve_mpo,
                     exciton_hamiltonian, Exciton_Hamiltonian, build_interaction_op_exciton,
                     mpsexciton, mpsexcitonQ, mpsexcitonQTrace, mpsexcitonKQ, get_qpi,
                     to_binary_vector, binary_to_MPS, eval_mps, eval_mps_spatial, get_mps,
                     get_mpo, get_diagonal_mpo, extract_diagonal_to_mps, mps_to_diagonal_mpo,
                     mps2mpo, constant_mps, rms_error, build_shift_mpo, shift_mpo,
                     shift_pair_mpos, shift_adjoint_mpo, shift_hopping_mpo, fix_sites,
                     custom_mpo, fused_mpo, custom_mps, replace_sites, hadamard_mpo,
                     prepend_op, postpend_op, matrix_checker, get_matrix

# Characterization ("golden") tests for the dynamics area: time evolution
# (solvers/Timeev.jl, without build_tdvp_propagator_mpo / check_tdvp_vs_U_mpo),
# DMRG (solvers/DMRG.jl), the Krylov Green's function (solvers/Krylov.jl)
# and the Haydock recursion that Tier 1 moves there from physics/RPA_tk.jl, the
# exciton Hamiltonian and probes (physics/TwoParticle.jl), QPI
# (physics/QPI.jl), and the core/Utils.jl helpers that
# test/sampling_golden.jl does not cover.
#
# The tests pin what the code computes *today*, bugs included, so that moving,
# splitting or renaming files (docs/dev/REORGANISATION_TODO.md, Tier 1) cannot
# change an output unnoticed. Each case below is a small, seeded computation
# that returns a NamedTuple of plain values (MPS and MPO results are stored as
# dense vectors or matrices, or as a few scalar functionals). The expected
# values live in `test/data/dynamics_golden.jl`, written by
# `test/data/generate_dynamics_golden.jl` (see its header for how to rerun it).
#
# A failure here means an output changed. If the change is a regression, fix the
# code. If it is intentional, regenerate the data file in the same commit as the
# behaviour change and record it in the changelog.
#
# Comparison rules (`mismatch`):
#   * every recorded field must still be returned, with the same type for
#     scalars and the same shape and element type for arrays;
#   * floating-point values compare with isapprox(rtol=RTOL, atol=ATOL) (whole
#     arrays through their norm, non-finite entries exactly); integers, Bools,
#     Strings and Symbols compare with ==;
#   * a case recorded as throwing must still throw that exception type with the
#     same first MESSAGE_PREFIX_CHARS characters of its message;
#   * every case must have golden data and vice versa, and the number of cases
#     must equal EXPECTED_CASE_COUNT, so dropped cases are noticed;
#   * `Random.seed!(seed)` runs right before every case, so the QTCI pivots,
#     random DMRG starts and random-phase probes are reproducible.
#
# The runner below is shared with the generator, which includes this file with
# `DYNAMICS_GOLDEN_GENERATOR` defined so that only the module is loaded.

module DynamicsGoldenRunner

using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using Logging
const TB = TensorBinding

# Number of cases in CASES (and in test/data/dynamics_golden.jl). Update it by
# hand, in the same commit, when cases are added or removed on purpose.
const EXPECTED_CASE_COUNT = 114

const RTOL = 1e-10
const ATOL = 1e-12
const MESSAGE_PREFIX_CHARS = 60

# ── Dense views of MPS / MPO results ──────────────────────────────────────────
# Big-endian ordering, as in binary_to_MPS / _basis_state_mps: site 1 carries
# the most significant digit. MPO entries are M[i+1, j+1] = <i|M|j> with the
# primed (bra) leg as the row, matching get_matrix.

function _site_indices(M, n)
    N = length(M)
    N == 1 && return collect(inds(M[1]))
    n == 1 && return collect(uniqueinds(M[1], M[2]))
    n == N && return collect(uniqueinds(M[N], M[N-1]))
    return collect(uniqueinds(M[n], M[n-1], M[n+1]))
end

# Site tensor n as a plain array (left link, legs..., right link), with a
# dimension-1 axis standing in for a missing edge link. Contracting plain arrays
# (below) instead of ITensors keeps the test from compiling an NDTensors
# contraction for every tensor order.
function _site_array(M, n, legs)
    N = length(M)
    l = n > 1 ? commonind(M[n-1], M[n]) : nothing
    r = n < N ? commonind(M[n], M[n+1]) : nothing
    ix = Index[legs...]
    l === nothing || pushfirst!(ix, l)
    r === nothing || push!(ix, r)
    length(ix) == length(inds(M[n])) || error("dense: site $n has unexpected indices $(inds(M[n]))")
    A = Array(M[n], ix...)
    return reshape(A, (l === nothing ? 1 : dim(l)), dim.(legs)..., (r === nothing ? 1 : dim(r)))
end

"""
    dense(ψ::MPS) -> Vector
    dense(M::MPO) -> Matrix

Full vector / matrix of a small MPS or MPO in the big-endian basis order.
"""
function dense(ψ::MPS)
    acc = ones(Float64, 1, 1)                     # (basis so far, right link)
    for n in 1:length(ψ)
        s = only(_site_indices(ψ, n))
        A = _site_array(ψ, n, (s,))               # (l, s, r)
        l, d, r = size(A)
        B = reshape(acc * reshape(A, l, d * r), size(acc, 1), d, r)
        acc = reshape(permutedims(B, (2, 1, 3)), d * size(acc, 1), r)   # site n less significant
    end
    return vec(acc)
end

function dense(M::MPO)
    acc = ones(Float64, 1, 1, 1)                  # (rows so far, cols so far, right link)
    for n in 1:length(M)
        idx = _site_indices(M, n)
        bra = only(filter(i -> plev(i) == 1, idx))
        ket = only(filter(i -> plev(i) == 0, idx))
        noprime(bra) == ket || error("dense: site $n bra/ket legs differ")
        A = _site_array(M, n, (bra, ket))         # (l, bra, ket, r)
        l, db, dk, r = size(A)
        R, C = size(acc, 1), size(acc, 2)
        B = reshape(reshape(acc, R * C, l) * reshape(A, l, db * dk * r), R, C, db, dk, r)
        acc = reshape(permutedims(B, (3, 1, 4, 2, 5)), db * R, dk * C, r)
    end
    return acc[:, :, 1]
end

dense_all(states) = [dense(x) for x in states]
traj(states::Vector{MPS}) = permutedims(reduce(hcat, [dense(ψ) for ψ in states]))  # (nsteps+1) × D

quiet(f) = redirect_stdout(f, devnull)

# ── Fixtures (exact, no QTCI) ─────────────────────────────────────────────────
chain(L=3; t=1.0) = TB.get_Hamiltonian("chain_1d", t; L=L)

# Linear potential V(x) = c·x on the 1-indexed site x = 1 + Σ_i b_i 2^(L-i), as
# an exact bond-dimension-2 OpSum MPO.
function linpot(sites, c=0.1)
    L = length(sites)
    os = OpSum()
    os += c, "Id", 1
    for i in 1:L
        os += c * 2.0^(L - i), "Proj1", i
    end
    return MPO(os, sites)
end

# The same MPO with every site tensor's legs stored ket-first: (s, s', links...).
ket_first(M, s) = MPO([permute(M[n], s[n], s[n]', setdiff(inds(M[n]), [s[n], s[n]'])...)
                       for n in 1:length(M)])

normalized(ψ) = ψ / norm(ψ)
superpos(sites, ns, coeffs) =
    normalized(reduce(+, [c * TB.binary_to_MPS(n, length(sites), sites) for (n, c) in zip(ns, coeffs)]))

# A deterministic 3-site MPS with dim-4 physical legs for fused_mpo.
function fused_test_mps(nsites=3)
    s  = [Index(4, "Fused,n=$i") for i in 1:nsites]
    ls = [Index(2, "Link,l=$i") for i in 1:nsites-1]
    ts = ITensor[]
    for i in 1:nsites
        ix = i == 1 ? (s[i], ls[i]) : i == nsites ? (ls[i-1], s[i]) : (ls[i-1], s[i], ls[i])
        n  = prod(dim, ix)
        push!(ts, ITensor([sin(0.7k + i) + 0.1k for k in 1:n], ix...))
    end
    return MPS(ts), s
end

# ── Sampled functions ─────────────────────────────────────────────────────────
# QTCI specialises on the type of the function it samples, so every distinct
# anonymous function would compile TCI again (seconds each). The cases sample
# instances of these two callable structs instead, one compilation per type.

"`Profile{T}`: `p(n) = c0 + c1·n + c2·n² + c3·sin(w·n)`."
struct Profile{T<:Number}
    c0::T
    c1::T
    c2::T
    c3::T
    w::Float64
end
(p::Profile)(n) = p.c0 + p.c1 * n + p.c2 * n^2 + p.c3 * sin(p.w * n)
rprof(c0, c1=0.0, c2=0.0, c3=0.0, w=1.0) = Profile{Float64}(c0, c1, c2, c3, w)
cprof(c0, c1=0.0, c2=0.0, c3=0.0, w=1.0) = Profile{ComplexF64}(c0, c1, c2, c3, w)

"`Kernel{T}`: `k(i, j) = a·exp(-b|i-j|) + c·i` (+ `exp(iw(i-2j))/(1+i+j)` when complex)."
struct Kernel{T<:Number}
    a::T
    b::Float64
    c::T
    w::Float64
end
(k::Kernel{Float64})(i, j) = k.a * exp(-k.b * abs(i - j)) + k.c * i
(k::Kernel{ComplexF64})(i, j) =
    k.a * exp(-k.b * abs(i - j)) + k.c * i + exp(im * k.w * (i - 2j)) / (1 + i + j)

const KREAL = Kernel{Float64}(1.0, 1.0, 0.1, 0.0)
const KCPLX = Kernel{ComplexF64}(0.5 + 0.1im, 0.7, 0.05im, 0.4)

# ── Case registry ─────────────────────────────────────────────────────────────
struct Case
    group::String
    name::String
    seed::Int
    f::Function
end
const CASES = Case[]
case!(f, group, name, seed) = (push!(CASES, Case(group, name, seed, f)); nothing)

# ═════════════════════════════════════════════════════════════════════════════
# core/Utils.jl helpers
# ═════════════════════════════════════════════════════════════════════════════

case!("Utils", "utils_to_binary_vector", 101) do
    (; v5_4 = TB.to_binary_vector(5, 4), v0_3 = TB.to_binary_vector(0, 3),
       v7_3 = TB.to_binary_vector(7, 3), v6_5 = TB.to_binary_vector(6, 5),
       v9_2 = TB.to_binary_vector(9, 2))   # n ≥ 2^L: lpad does not truncate
end

case!("Utils", "utils_binary_to_MPS", 102) do
    s  = siteinds("Qubit", 3)
    s1 = siteinds("Qubit", 1)
    (; basis_L3 = reduce(hcat, [dense(TB.binary_to_MPS(n, 3, s)) for n in 0:7]),
       basis_L1 = reduce(hcat, [dense(TB.binary_to_MPS(n, 1, s1)) for n in 0:1]),
       maxlinkdim = maxlinkdim(TB.binary_to_MPS(5, 3, s)))
end

case!("Utils", "utils_get_mps_eval_mps_real", 103) do
    s = siteinds("Qubit", 3)
    A = TB.get_mps(3, s, rprof(0.5, 0.0, 0.1, 1.0))
    (; dense = dense(A), evals = [TB.eval_mps(A, n) for n in 0:7], maxlinkdim = maxlinkdim(A))
end

case!("Utils", "utils_get_mps_eval_mps_complex", 104) do
    s = siteinds("Qubit", 3)
    A = TB.get_mps(3, s, cprof(0.3, 0.5im, 0.0, 1.0 - 0.2im, 1.3); type=ComplexF64)
    (; dense = dense(A), evals = [TB.eval_mps(A, n) for n in 0:7])
end

case!("Utils", "utils_get_mps_L1", 105) do
    s1 = siteinds("Qubit", 1)
    (; real = dense(TB.get_mps(1, s1, rprof(2.0, 3.0))),
       complex = dense(TB.get_mps(1, s1, cprof(1.0, -2.0im); type=ComplexF64)))
end

case!("Utils", "utils_get_mpo_diagonal", 106) do
    s = siteinds("Qubit", 3)
    (; dense = dense(TB.get_mpo(3, s, rprof(1.0, 0.0, 0.25))))
end

case!("Utils", "utils_get_mpo_kernel", 107) do
    s = siteinds("Qubit", 3)
    (; dense = dense(TB.get_mpo(3, s, KREAL)))
end

case!("Utils", "utils_get_mpo_kernel_complex", 108) do
    s = siteinds("Qubit", 3)
    (; dense = dense(TB.get_mpo(3, s, KCPLX; type=ComplexF64)))
end

case!("Utils", "utils_get_diagonal_mpo", 109) do
    s = siteinds("Qubit", 3)
    (; dense = dense(TB.get_diagonal_mpo(3, s, rprof(0.0, -0.2, 0.01))))
end

case!("Utils", "utils_extract_diagonal_to_mps", 110) do
    H = chain(3)
    s = H.sites
    M  = +(H.mpo, linpot(s, 0.3); cutoff=1e-12)
    Mc = +(-im * H.mpo, linpot(s, 0.2 - 0.1im); cutoff=1e-12)
    K  = TB.get_mpo(3, s, KREAL)
    (; real = dense(TB.extract_diagonal_to_mps(M)),
       complex = dense(TB.extract_diagonal_to_mps(Mc)),
       kernel = dense(TB.extract_diagonal_to_mps(K)))
end

case!("Utils", "utils_mps_to_diagonal_mpo", 111) do
    s  = siteinds("Qubit", 3)
    s2 = siteinds("Qubit", 3)
    A  = TB.get_mps(3, s, rprof(1.0, -0.3, 0.05))
    (; same_sites = dense(TB.mps_to_diagonal_mpo(A, s)),
       new_sites  = dense(TB.mps_to_diagonal_mpo(A, s2)))
end

case!("Utils", "utils_mps_to_diagonal_mpo_single_site_throws", 112) do
    s1 = siteinds("Qubit", 1)
    TB.mps_to_diagonal_mpo(MPS([ITensor([1.0, 2.0], s1[1])]), s1)
end

case!("Utils", "utils_mps2mpo", 113) do
    s = siteinds("Qubit", 3)
    A = TB.get_mps(3, s, rprof(0.2, 0.1, -0.02, 0.8, 0.9))
    (; dense = dense(TB.mps2mpo(3, s, A)))
end

case!("Utils", "utils_constant_mps", 114) do
    s = siteinds("Qubit", 3)
    (; real = dense(TB.constant_mps(s, 2.5)),
       complex = dense(TB.constant_mps(s, 1.0 + 2.0im)),
       empty_length = length(TB.constant_mps(Index[], 1.0)))
end

case!("Utils", "utils_rms_error", 115) do
    s = siteinds("Qubit", 3)
    A = TB.get_mps(3, s, rprof(0.0, 0.0, 0.0, 1.0))
    B = TB.get_mps(3, s, rprof(0.0, 0.01, 0.0, 1.0))
    # rms_error calls inner(diff', diff), which ITensors accepts only through a
    # deprecated index-matching path and warns about once per session; the
    # warning is silenced here, the value is pinned.
    with_logger(NullLogger()) do
        (; rms = TB.rms_error(A, B), rms_self = TB.rms_error(A, A))
    end
end

case!("Utils", "utils_build_shift_mpo_positional", 116) do
    s = siteinds("Qubit", 3)
    (; q0_cyclic = dense(TB.build_shift_mpo(s, 0, true)),
       q1_cyclic = dense(TB.build_shift_mpo(s, 1, true)),
       q1_open   = dense(TB.build_shift_mpo(s, 1, false)),
       q3_cyclic = dense(TB.build_shift_mpo(s, 3, true)),
       q3_open   = dense(TB.build_shift_mpo(s, 3, false)),
       q5_open   = dense(TB.build_shift_mpo(s, 5, false)))
end

case!("Utils", "utils_build_shift_mpo_keyword", 117) do
    s = siteinds("Qubit", 3)
    # build_shift_mpo(sites, q::Integer) hits the keyword method (cyclic=false),
    # not the positional default cyclic=true.
    (; default = dense(TB.build_shift_mpo(s, 2)),
       cyclic  = dense(TB.build_shift_mpo(s, 2; cyclic=true)))
end

case!("Utils", "utils_shift_mpo", 118) do
    s = siteinds("Qubit", 3)
    (; q0 = dense(TB.shift_mpo(s, 0)),
       q2 = dense(TB.shift_mpo(s, 2)),
       qm1 = dense(TB.shift_mpo(s, -1)),
       qm3_cyclic = dense(TB.shift_mpo(s, -3; cyclic=true)))
end

case!("Utils", "utils_shift_pair_mpos", 119) do
    s = siteinds("Qubit", 3)
    K, Kd  = TB.shift_pair_mpos(s, 1; cyclic=true)
    K2, Kd2 = TB.shift_pair_mpos(s, 3)
    (; K_cyclic = dense(K), Kdag_cyclic = dense(Kd), K_open = dense(K2), Kdag_open = dense(Kd2))
end

case!("Utils", "utils_shift_adjoint_mpo", 120) do
    s = siteinds("Qubit", 3)
    M = TB.get_mpo(3, s, KCPLX; type=ComplexF64)
    (; dense = dense(TB.shift_adjoint_mpo(M)))
end

case!("Utils", "utils_shift_hopping_mpo", 121) do
    s   = siteinds("Qubit", 3)
    hop = linpot(s, 0.25)
    (; default = dense(TB.shift_hopping_mpo(hop, s, 1)),
       cyclic  = dense(TB.shift_hopping_mpo(hop, s, 2; cyclic=true)),
       maxdim  = dense(TB.shift_hopping_mpo(hop, s, 1; maxdim=2)),
       negative_q = dense(TB.shift_hopping_mpo(hop, s, -1)),
       apply_kwargs = dense(TB.shift_hopping_mpo(hop, s, 1; apply_kwargs=(; cutoff=1e-14))))
end

case!("Utils", "utils_fix_sites", 122) do
    s  = siteinds("Qubit", 3)
    s2 = siteinds("Qubit", 3)
    M  = TB.get_mpo(3, s, KCPLX; type=ComplexF64)
    Mc = copy(M)
    out = TB.fix_sites(Mc, s2)
    # Same MPO with every site tensor's legs stored ket-first (s, s', links...):
    # fix_sites reads siteinds(mpo)[i] positionally, so this pins how it treats
    # that storage order.
    Mt = ket_first(M, s)
    out_t = TB.fix_sites(Mt, s2)
    (; dense = dense(out), in_place = out === Mc,
       on_new_sites = all(n -> hasind(out[n], s2[n]) && hasind(out[n], s2[n]'), 1:3),
       dense_ket_first = dense(out_t))
end

case!("Utils", "utils_custom_mpo", 123) do
    s6 = siteinds("Qubit", 6)
    s3 = siteinds("Qubit", 3)
    A  = TB.get_mps(6, s6, rprof(0.0, 0.0, 0.01, 1.0, 0.3))
    (; dense = dense(TB.custom_mpo(A, s3)))
end

case!("Utils", "utils_fused_mpo", 124) do
    A, _ = fused_test_mps(3)
    s    = siteinds("Qubit", 3)
    (; dense = dense(TB.fused_mpo(A, s)))
end

case!("Utils", "utils_custom_mps", 125) do
    s  = siteinds("Qubit", 3)
    # A fixed tensor train (bond dims 1-2-3-1) rather than a QTCI result.
    cores = [reshape([sin(0.9k + r) for k in 1:prod(d)], d...)
             for (r, d) in enumerate(((1, 2, 2), (2, 2, 3), (3, 2, 1)))]
    tt = TB.TCI.TensorTrain(cores)
    (; dense = dense(TB.custom_mps(tt, s)))
end

case!("Utils", "utils_replace_sites", 126) do
    s  = siteinds("Qubit", 3)
    s2 = siteinds("Qubit", 3)
    M  = TB.get_mpo(3, s, KCPLX; type=ComplexF64)
    Mt = ket_first(M, s)
    (; dense = dense(TB.replace_sites(M, s2)), dense_ket_first = dense(TB.replace_sites(Mt, s2)))
end

case!("Utils", "utils_hadamard_mpo", 127) do
    H  = chain(3)
    s  = H.sites
    A  = +(H.mpo, linpot(s, 0.3); cutoff=1e-12)
    B  = TB.get_mpo(3, s, KREAL)
    s2 = siteinds("Qubit", 3)
    (; own_sites = dense(TB.hadamard_mpo(A, B, s)),
       new_sites = dense(TB.hadamard_mpo(A, B, s2)),
       truncated = dense(TB.hadamard_mpo(A, B, s; maxdim=1)))
end

case!("Utils", "utils_prepend_op", 128) do
    H  = chain(2)
    sl = Index(3, "Layer,l=0")
    sp = Index(2, "Spin")
    (; float = dense(TB.prepend_op(H.mpo, sl, [1.0 0.5 0.0; 0.5 2.0 0.0; 0.0 0.0 -1.0])),
       complex_int = dense(TB.prepend_op(H.mpo, sp, [0 -im; im 0])),
       int = dense(TB.prepend_op(H.mpo, sp, [1 2; 3 4])),
       any = dense(TB.prepend_op(H.mpo, sp, Any[1.0 0.0; 0.0 -1.0])),
       hop = dense(TB.prepend_op(H.mpo, sl, 1, 3)),
       proj = dense(TB.prepend_op(H.mpo, sl, 2)))
end

case!("Utils", "utils_postpend_op", 129) do
    H  = chain(2)
    sl = Index(3, "Layer,l=0")
    sp = Index(2, "Spin")
    (; float = dense(TB.postpend_op(H.mpo, sl, [1.0 0.5 0.0; 0.5 2.0 0.0; 0.0 0.0 -1.0])),
       complex_int = dense(TB.postpend_op(H.mpo, sp, [0 -im; im 0])),
       any = dense(TB.postpend_op(H.mpo, sp, Any[1.0 0.0; 0.0 -1.0])),
       hop = dense(TB.postpend_op(H.mpo, sl, 3, 1)),
       proj = dense(TB.postpend_op(H.mpo, sl, 1)))
end

case!("Utils", "utils_product_state_mps", 130) do
    s = [Index(2, "Qubit,Site,n=1"), Index(3, "Layer"), Index(2, "Qubit,Site,n=2")]
    (; mixed = dense(TB._product_state_mps(s, [2, 3, 1])),
       single = dense(TB._product_state_mps([Index(3, "Layer")], [2])))
end

case!("Utils", "utils_basis_state_mps", 131) do
    s = [Index(2, "Qubit,Site,n=1"), Index(3, "Layer"), Index(2, "Qubit,Site,n=2")]
    (; mixed = reduce(hcat, [dense(TB._basis_state_mps(k, s)) for k in 0:11]),
       single = dense(TB._basis_state_mps(2, [Index(3, "Layer")])))
end

case!("Utils", "utils_matrix_checker_get_matrix", 132) do
    s  = siteinds("Qubit", 3)
    M  = TB.get_mpo(3, s, KCPLX; type=ComplexF64)
    H  = chain(2)
    sl = Index(3, "Layer")
    Ml = TB.prepend_op(H.mpo, sl, [1.0 0.5 0.0; 0.5 2.0 0.0; 0.0 0.0 -1.0])
    (; get_matrix = TB.get_matrix(M, s), get_matrix_L = TB.get_matrix(M, 3, s),
       element = TB.matrix_checker(M, s, 2, 5), element_L = TB.matrix_checker(M, 3, s, 5, 2),
       mixed = TB.get_matrix(Ml, [sl; H.sites]))
end

case!("Utils", "utils_eval_mps_spatial", 133) do
    s = siteinds("Qubit", 4)
    A = TB.get_mps(4, s, rprof(1.0, 0.0, 0.02, 0.5, 0.4))
    r1 = TB.eval_mps_spatial(A)
    r2 = TB.eval_mps_spatial(A; num_x=4, num_avg=2)
    r3 = TB.eval_mps_spatial(A; num_x=4, box_half=1)
    r4 = TB.eval_mps_spatial(A; x_groups=[[1, 2], [5], [16, 9]])
    r5 = TB.eval_mps_spatial(A; num_x=3, x_start=3, x_end=12, box_half=1, Lx=2)
    (; default_values = r1.values, default_centers = r1.centers,
       avg_values = r2.values, avg_groups = r2.groups,
       box_values = r3.values, box_groups = r3.groups,
       groups_values = r4.values, groups_centers = r4.centers,
       window_values = r5.values, window_groups = r5.groups)
end

case!("Utils", "utils_eval_block_mps", 134) do
    s = siteinds("Qubit", 4)
    A = TB.get_mps(4, s, rprof(1.0, 0.0, 0.02, 0.5, 0.4))
    (; a1b1 = [TB._eval_block_mps(A, ix, iy, 1, 1, 2, 2) for ix in 0:1, iy in 0:1],
       a2b0 = [TB._eval_block_mps(A, ix, 0, 2, 0, 2, 2) for ix in 0:3],
       a0b2 = [TB._eval_block_mps(A, 0, iy, 0, 2, 2, 2) for iy in 0:3])
end

case!("Utils", "utils_sigma_ops", 135) do
    s = siteinds("Qubit", 1)[1]
    (; plus = Array(op("sigma_plus", s), s', s), minus = Array(op("sigma_minus", s), s', s))
end

# ═════════════════════════════════════════════════════════════════════════════
# solvers/DMRG.jl (+ _estimate_spectral_bounds, which Tier 1 moves there)
# ═════════════════════════════════════════════════════════════════════════════

case!("DMRG", "dmrg_gs_defaults", 201) do
    H = chain(3)
    E, ψ = TB.dmrg_gs(H.mpo, H.sites; outputlevel=0)
    (; E, probabilities = abs2.(dense(ψ)), norm = norm(ψ))
end

case!("DMRG", "dmrg_gs_scalar_kwargs", 202) do
    H = chain(3; t=0.7)
    E, ψ = TB.dmrg_gs(+(H.mpo, linpot(H.sites, 0.2); cutoff=1e-12), H.sites;
                      nsweeps=4, maxdim=4, noise=0.0, cutoff=1e-10, linkdim_init=2, outputlevel=0)
    (; E, probabilities = abs2.(dense(ψ)), maxlinkdim = maxlinkdim(ψ))
end

case!("DMRG", "dmrg_build_K", 203) do
    H = chain(3)
    (; K = dense(TB.build_K(H.mpo, H.sites, 0.3, 0.1)),
       K_truncated = dense(TB.build_K(H.mpo, H.sites, -0.5, 0.2; maxdim_K=2, cutoff_K=1e-6)))
end

case!("DMRG", "dmrg_spectral_random_start", 204) do
    H = chain(3)
    E, ψ = TB.dmrg_spectral(H.mpo, H.sites, 0.5, 0.2; nsweeps=4, maxdim=[4, 8], outputlevel=0)
    (; E, probabilities = abs2.(dense(ψ)))
end

case!("DMRG", "dmrg_spectral_warm_start", 205) do
    H  = chain(3)
    ψ0 = superpos(H.sites, [1, 2, 5], [1.0, 0.5, -0.3])
    E, ψ = TB.dmrg_spectral(H.mpo, H.sites, -1.2, 0.1; ψ0=ψ0, nsweeps=3, maxdim=8,
                            noise=0.0, outputlevel=0)
    (; E, probabilities = abs2.(dense(ψ)))
end

case!("DMRG", "dmrg_local_weight", 206) do
    s = siteinds("Qubit", 3)
    ψ = superpos(s, [0, 3, 6, 7], [0.3, -1.0, 0.6im, 0.2])
    (; weights = [TB.local_weight(ψ, i, 3, s) for i in 0:7])
end

case!("DMRG", "dmrg_estimate_spectral_bounds", 207) do
    H = chain(3)
    sc, ctr = quiet(() -> TB._estimate_spectral_bounds(+(H.mpo, linpot(H.sites, 0.2); cutoff=1e-12), H.sites))
    (; scale = sc, center = ctr)
end

# ═════════════════════════════════════════════════════════════════════════════
# solvers/Krylov.jl
# ═════════════════════════════════════════════════════════════════════════════

case!("Krylov", "krylov_vec_mps_from_mpo", 301) do
    H  = chain(3)
    s2 = siteinds("Qubit", 6)
    M  = +(H.mpo, linpot(H.sites, 0.3); cutoff=1e-12)
    (; identity = dense(TB._vec_mps_from_mpo(MPO(H.sites, "Id"), s2)),
       hamiltonian = dense(TB._vec_mps_from_mpo(M, s2)),
       truncated = dense(TB._vec_mps_from_mpo(M, s2; maxdim=1)))
end

case!("Krylov", "krylov_vec_mps_from_mpo_wrong_length_throws", 302) do
    H = chain(3)
    TB._vec_mps_from_mpo(H.mpo, siteinds("Qubit", 5))
end

case!("Krylov", "krylov_green_raw", 303) do
    H = chain(3)
    (; G = dense(TB.get_green_krylov(H.mpo, H.sites, 0.3; η=0.2, nsweeps=3, maxdim=32)))
end

case!("Krylov", "krylov_green_warm_start", 304) do
    H  = chain(3)
    x0 = (1 / (0.3 + 0.2im)) * MPO(H.sites, "Id")
    (; G = dense(TB.get_green_krylov(H.mpo, H.sites, 0.3; η=0.2, nsweeps=2, maxdim=16, x0_mpo=x0,
                                     verbose=true)))
end

case!("Krylov", "krylov_green_tbhamiltonian", 305) do
    H = chain(3; t=0.8)
    (; G = dense(TB.get_green_krylov(H, -0.4; η=0.1, nsweeps=2, maxdim=16,
                                     krylovdim=10, tol=1e-12, maxiter=50)))
end

case!("Krylov", "krylov_green_ishermitian_flag", 306) do
    H = chain(2)
    (; G = dense(TB.get_green_krylov(H.mpo, H.sites, 0.7; η=0.3, nsweeps=2, maxdim=16,
                                     ishermitian=true)))
end

# ── Haydock recursion (physics/RPA_tk.jl today; Tier 1 moves it to Krylov) ─────

const HAYDOCK_Z = ComplexF64[0.1 + 0.2im, -1.0 + 0.05im, 2.0 + 0.1im]

case!("Haydock", "haydock_identity_seed", 310) do
    H = chain(3)
    a, b, basis, norm0 = TB.haydock_cf(+(H.mpo, linpot(H.sites, 0.2); cutoff=1e-12),
                                       MPO(H.sites, "Id"), 5; maxdim=32)
    (; a, b, norm0, basis = dense_all(basis),
       cf = [TB.eval_haydock_cf(a, b, z) for z in HAYDOCK_Z],
       cf_truncated = [TB.eval_haydock_cf(a[1:3], b[1:3], z) for z in HAYDOCK_Z],
       resolve = dense(TB.haydock_resolve_mpo(a, b, basis, HAYDOCK_Z[1]; maxdim=32)))
end

# A non-symmetric seed |3><3|: haydock_cf's inner product tr(apply(dag(A), B)) is
# Σ conj(A_ij) B_ji, not the Frobenius Σ conj(A_ij) B_ij, so here the first
# residual H|3><3| gets "norm" <3|H|3> = 0 and the recursion stops after one step.
case!("Haydock", "haydock_projector_seed", 311) do
    H    = chain(3)
    ψ    = TB.binary_to_MPS(3, 3, H.sites)
    a, b, basis, norm0 = TB.haydock_cf(H.mpo, outer(ψ', ψ), 5; maxdim=32)
    (; a, b, norm0, nbasis = length(basis), basis = dense_all(basis),
       cf = [TB.eval_haydock_cf(a, b, z) for z in HAYDOCK_Z],
       resolve = dense(TB.haydock_resolve_mpo(a, b, basis, HAYDOCK_Z[1]; maxdim=32)))
end

case!("Haydock", "haydock_invariant_subspace", 312) do
    s  = siteinds("Qubit", 2)
    os = OpSum()
    os += 1.5, "X", 1
    Hx = MPO(os, s)
    a, b, basis, norm0 = TB.haydock_cf(Hx, MPO(s, "Id"), 4; verbose=true)
    (; a, b, norm0, nbasis = length(basis),
       cf = TB.eval_haydock_cf(a, b, 0.4 + 0.1im),
       resolve = dense(TB.haydock_resolve_mpo(a, b, basis, 0.4 + 0.1im)))
end

case!("Haydock", "haydock_single_step", 313) do
    H = chain(2)
    a, b, basis, norm0 = TB.haydock_cf(H.mpo, linpot(H.sites, 0.5), 1)
    (; a, b, norm0, cf = TB.eval_haydock_cf(a, b, 1.0 + 0.5im),
       resolve = dense(TB.haydock_resolve_mpo(a, b, basis, 1.0 + 0.5im)))
end

# ═════════════════════════════════════════════════════════════════════════════
# solvers/Timeev.jl (not build_tdvp_propagator_mpo / check_tdvp_vs_U_mpo)
# ═════════════════════════════════════════════════════════════════════════════

nh_mpo(H) = +(H.mpo, (-0.3im) * linpot(H.sites, 0.1); cutoff=1e-12)   # H − iΓ(x)
euler_propagator(H, dt) = +(MPO(H.sites, "Id"), (-im * dt) * H.mpo; cutoff=1e-12)
driven(H) = t -> +(H.mpo, (0.5 * sin(2t)) * linpot(H.sites, 0.1); cutoff=1e-12)
pure_dm(ψ) = outer(ψ', ψ)

case!("Timeev", "tdvp_evolve_normalized", 401) do
    H  = chain(3)
    ψ0 = TB.binary_to_MPS(2, 3, H.sites)
    (; psi = dense(TB.tdvp_evolve(-im * H.mpo, ψ0, 0.1)))
end

case!("Timeev", "tdvp_evolve_nh_normalize_flag", 402) do
    H  = chain(3)
    ψ0 = TB.binary_to_MPS(2, 3, H.sites)
    ψn = TB.tdvp_evolve(-im * nh_mpo(H), ψ0, 0.2)
    ψu = TB.tdvp_evolve(-im * nh_mpo(H), ψ0, 0.2; normalize=false, maxdim=4, cutoff=1e-12)
    (; normalized = dense(ψn), unnormalized = dense(ψu), norm_unnormalized = norm(ψu))
end

case!("Timeev", "tdvp_evolve_tbhamiltonian", 403) do
    H  = chain(3)
    ψ0 = TB.binary_to_MPS(5, 3, H.sites)
    (; psi = dense(TB.tdvp_evolve(H, ψ0, 0.1; maxdim=8)))
end

case!("Timeev", "tdvp_evolve_nsite1", 404) do
    H  = chain(3)
    ψ0 = superpos(H.sites, [2, 3, 5], [1.0, 0.5, -0.5])
    (; psi = dense(TB.tdvp_evolve(-im * H.mpo, ψ0, 0.1; nsite=1)))
end

case!("Timeev", "apply_mpo_to_mps", 405) do
    H  = chain(3)
    U  = euler_propagator(H, 0.1)
    ψ0 = TB.binary_to_MPS(2, 3, H.sites)
    ψu = TB.apply_mpo_to_mps(U, ψ0; normalize=false)
    (; normalized = dense(TB.apply_mpo_to_mps(U, ψ0)), unnormalized = dense(ψu),
       norm_unnormalized = norm(ψu))
end

case!("Timeev", "evolve_with_propagator", 406) do
    H  = chain(3)
    U  = euler_propagator(H, 0.1)
    ψ0 = TB.binary_to_MPS(2, 3, H.sites)
    sn = TB.evolve_with_propagator(U, ψ0, 3)
    su = TB.evolve_with_propagator(U, ψ0, 3; normalize_each_step=false)
    st = TB.evolve_with_propagator(U, superpos(H.sites, [2, 5], [1.0, 1.0]), 2; maxdim=1, cutoff=1e-12)
    (; normalized = traj(sn), unnormalized = traj(su), norms_unnormalized = norm.(su),
       truncated = traj(st), nstates = length(sn))
end

case!("Timeev", "evolve_with_tdvp_raw", 407) do
    H  = chain(3)
    ψ0 = TB.binary_to_MPS(2, 3, H.sites)
    st = TB.evolve_with_tdvp(-im * H.mpo, ψ0, 3, 0.1)
    (; states = traj(st), nstates = length(st))
end

case!("Timeev", "evolve_with_tdvp_nh_unnormalized", 408) do
    H  = chain(3)
    ψ0 = TB.binary_to_MPS(4, 3, H.sites)
    st = TB.evolve_with_tdvp(-im * nh_mpo(H), ψ0, 3, 0.1; normalize_each_step=false, maxdim=8)
    (; states = traj(st), norms = norm.(st))
end

case!("Timeev", "evolve_with_tdvp_tbhamiltonian", 409) do
    H  = chain(3; t=0.6)
    ψ0 = superpos(H.sites, [1, 6], [1.0, -1.0im])
    st = TB.evolve_with_tdvp(H, ψ0, 2, 0.05; maxdim=4, cutoff=1e-12)
    (; states = traj(st))
end

case!("Timeev", "evolve_with_tdvp_timedep", 410) do
    H  = chain(3)
    ψ0 = TB.binary_to_MPS(2, 3, H.sites)
    sn = TB.evolve_with_tdvp_timedep(driven(H), ψ0, 3, 0.1)
    su = TB.evolve_with_tdvp_timedep(t -> +(nh_mpo(H), (0.2t) * linpot(H.sites, 0.1); cutoff=1e-12),
                                     ψ0, 2, 0.1; normalize_each_step=false, krylovdim=5, tol=1e-12)
    (; normalized = traj(sn), unnormalized = traj(su), norms_unnormalized = norm.(su))
end

case!("Timeev", "compute_basis_overlaps", 411) do
    H  = chain(3)
    ψ0 = TB.binary_to_MPS(2, 3, H.sites)
    st = TB.evolve_with_propagator(euler_propagator(H, 0.1), ψ0, 2; normalize_each_step=false)
    r  = TB.compute_basis_overlaps(st, 3, H.sites)
    (; overlaps = r.overlaps, abs_overlaps = r.abs_overlaps,
       probabilities = r.probabilities, norms = r.norms)
end

case!("Timeev", "basis_amplitude", 412) do
    H = chain(3)
    ψ = TB.tdvp_evolve(-im * H.mpo, TB.binary_to_MPS(2, 3, H.sites), 0.3)
    (; amplitudes = [TB.basis_amplitude(ψ, n, 3, H.sites) for n in 0:7])
end

case!("Timeev", "phase_aligned_distance", 413) do
    H  = chain(3)
    ψ0 = TB.binary_to_MPS(2, 3, H.sites)
    ψ1 = TB.tdvp_evolve(-im * H.mpo, ψ0, 0.4)
    (; d01 = TB.phase_aligned_distance(ψ0, ψ1),
       d_scaled = TB.phase_aligned_distance(2.0 * ψ0, ψ1),
       d_zero = TB.phase_aligned_distance(ψ0, 0.0 * ψ1))
end

case!("Timeev", "rk4_step_dm_timedep", 414) do
    H  = chain(3)
    ρ0 = pure_dm(superpos(H.sites, [2, 3], [1.0, 1.0im]))
    Hoft = t -> +(H.mpo, (0.3t) * linpot(H.sites, 0.1); cutoff=1e-12)
    (; truncated = dense(TB.rk4_step_dm_timedep(Hoft, ρ0, 0.0, 0.05)),
       untruncated = dense(TB.rk4_step_dm_timedep(Hoft, ρ0, 0.2, 0.05; truncate_intermediates=false)),
       maxdim2 = dense(TB.rk4_step_dm_timedep(Hoft, ρ0, 0.0, 0.05; maxdim=2)))
end

case!("Timeev", "evolve_rk4_dm_timedep", 415) do
    H  = chain(3)
    ρ0 = pure_dm(superpos(H.sites, [2, 3], [1.0, 1.0im]))
    st = TB.evolve_rk4_dm_timedep(driven(H), ρ0, 2, 0.05; verbose=true)
    (; states = dense_all(st), traces = [tr(ρ) for ρ in st], nstates = length(st))
end

case!("Timeev", "rk4_step_dm_nh", 416) do
    H  = chain(3)
    ρ0 = pure_dm(superpos(H.sites, [2, 3], [1.0, 1.0im]))
    Hn = nh_mpo(H)
    (; truncated = dense(TB.rk4_step_dm_nh(_ -> Hn, ρ0, 0.0, 0.05)),
       untruncated = dense(TB.rk4_step_dm_nh(_ -> Hn, ρ0, 0.0, 0.05; truncate_intermediates=false)))
end

case!("Timeev", "evolve_rk4_dm_nh", 417) do
    H  = chain(3)
    ρ0 = pure_dm(superpos(H.sites, [2, 3], [1.0, 1.0im]))
    Hn = nh_mpo(H)
    st = TB.evolve_rk4_dm_nh(_ -> Hn, ρ0, 2, 0.1; verbose=true)
    (; states = dense_all(st), traces = [tr(ρ) for ρ in st],
       purities = TB.purity_trajectory(st), purity0 = TB.purity(ρ0))
end

case!("Timeev", "dm_observables", 418) do
    H  = chain(3)
    ρ0 = pure_dm(superpos(H.sites, [2, 3], [1.0, 1.0im]))
    st = TB.evolve_rk4_dm_timedep(driven(H), ρ0, 2, 0.05)
    X  = linpot(H.sites, 0.1)
    obs  = TB.observables_trajectory((E = H.mpo, X = X), st)
    obsd = TB.observables_trajectory(Dict("E" => H.mpo, "X" => X), st)
    (; expect_E0 = TB.dm_expect(H.mpo, ρ0), expect_X0 = TB.dm_expect(X, ρ0),
       nt_keys = sort!([string(k) for k in keys(obs)]),
       nt_E = obs[:E], nt_X = obs[:X],
       dict_keys = sort!(collect(keys(obsd))), dict_E = obsd["E"], dict_X = obsd["X"],
       timedep = TB.timedep_observable_trajectory(driven(H), st, 0.05))
end

case!("Timeev", "bond_current_x", 419) do
    H  = chain(3)
    ρ0 = pure_dm(superpos(H.sites, [2, 3, 4], [1.0, 1.0im, -0.5]))
    st = TB.evolve_rk4_dm_timedep(_ -> H.mpo, ρ0, 2, 0.05)
    (; currents = [TB.bond_current_x(ρ0, j, 1.0, 3, H.sites) for j in 0:6],
       current_complex_t = TB.bond_current_x(ρ0, 2, 0.5 + 0.1im, 3, H.sites),
       traj_scalar = TB.bond_current_x_trajectory(st, 2, 1.0, 3, H.sites),
       traj_callable = TB.bond_current_x_trajectory(st, 3, t -> 1.0 + t, 3, H.sites; dt=0.05))
end

case!("Timeev", "central_x_bond", 420) do
    (; L3 = TB.central_x_bond(3), L1 = TB.central_x_bond(1),
       L4_Nx4 = TB.central_x_bond(4; Nx=4), L5_Nx8 = TB.central_x_bond(5; Nx=8),
       L4_Nx2 = TB.central_x_bond(4; Nx=2))
end

# ═════════════════════════════════════════════════════════════════════════════
# physics/TwoParticle.jl
# ═════════════════════════════════════════════════════════════════════════════

exciton_fields(Hx) = (; dense = dense(Hx.mpo), L = Hx.L, N = Hx.N, nsites = length(Hx.sites),
                        scale = Hx.scale, center = Hx.center, Lx_is_nothing = Hx.Lx === nothing,
                        site_dims = dim.(Hx.sites))

case!("TwoParticle", "exciton_geometry_chain", 501) do
    Hx = TB.exciton_hamiltonian("chain_1d", 1.0, rprof(-2.0); L=2)
    merge(exciton_fields(Hx), (; geometry_1 = Hx.geometry(1), geometry_4 = Hx.geometry(4)))
end

case!("TwoParticle", "exciton_geometry_chain_onsite_scale", 502) do
    Hx = TB.exciton_hamiltonian("chain_1d", 1.0, rprof(-1.0, -0.25); L=2,
                                on_site = rprof(-3.125, 2.5, -0.5), scale = 3.0)
    exciton_fields(Hx)
end

case!("TwoParticle", "exciton_prebuilt_sectors", 503) do
    Hc = chain(2; t=1.0)
    Hv = chain(2; t=0.5)
    Hx = TB.exciton_hamiltonian(Hc, Hv, rprof(-1.5))
    merge(exciton_fields(Hx), (; sites_interleaved = Hx.sites == [Hc.sites[1], Hv.sites[1], Hc.sites[2], Hv.sites[2]]))
end

case!("TwoParticle", "exciton_square_2d", 504) do
    Hx = TB.exciton_hamiltonian("square_2d", 1.0, rprof(-1.0, 0.1); L=2, Lx=1, Ly=1)
    merge(exciton_fields(Hx), (; Lx = Hx.Lx))
end

case!("TwoParticle", "exciton_chain_L3_spectrum", 505) do
    Hx = TB.exciton_hamiltonian("chain_1d", 1.0, rprof(-2.0); L=3)
    M  = dense(Hx.mpo)
    (; eigenvalues = eigvals(Hermitian(M)), diagonal = diag(M), frobenius = norm(M),
       asymmetry = norm(M - transpose(M)), first_row = M[1, :], eltype = string(eltype(M)))
end

case!("TwoParticle", "exciton_nonbinary_space_throws", 506) do
    # Only the position-space type is checked, so a chain relabelled as a
    # Fibonacci space stands in for a real (slow to build) Fibonacci Hamiltonian.
    H  = chain(2)
    Hf = TB.TBHamiltonian(H; position_space = TB.FibonacciPositionSpace(MPO(H.sites, "Id")))
    TB.exciton_hamiltonian(Hf, Hf, rprof(1.0))
end

case!("TwoParticle", "exciton_mismatched_L_throws", 507) do
    TB.Exciton_Hamiltonian(chain(2), chain(3), rprof(1.0))
end

case!("TwoParticle", "exciton_low_level_builder", 508) do
    Hc = chain(2)
    Hv = chain(2; t=0.8)
    (; plain = dense(TB.Exciton_Hamiltonian(Hc, Hv, rprof(-1.0))),
       onsite = dense(TB.Exciton_Hamiltonian(Hc, Hv, rprof(-1.0); on_site = rprof(0.0, 0.3),
                                              maxbonddim_quantics = 4)))
end

case!("TwoParticle", "exciton_interaction_op", 509) do
    Hc = chain(2)
    Hv = chain(2)
    s_eh = collect(Iterators.flatten(zip(Hc.sites, Hv.sites)))
    (; dense = dense(TB.build_interaction_op_exciton(2, s_eh, rprof(1.0, 0.1))))
end

case!("TwoParticle", "exciton_mpsexciton", 510) do
    s = siteinds("Qubit", 4)
    (; pairs = reduce(hcat, [dense(TB.mpsexciton(xe, xh, s)) for xh in 1:4 for xe in 1:4]),
       bound = reduce(hcat, [dense(TB.mpsexciton(x, s)) for x in 1:4]))
end

case!("TwoParticle", "exciton_mpsexcitonQ", 511) do
    s2 = siteinds("Qubit", 4)
    s3 = siteinds("Qubit", 6)
    (; L2 = reduce(hcat, [dense(TB.mpsexcitonQ(Q, s2)) for Q in 1:4]),
       L3_Q6 = dense(TB.mpsexcitonQ(6, s3)), maxlinkdim_L3 = maxlinkdim(TB.mpsexcitonQ(6, s3)))
end

case!("TwoParticle", "exciton_mpsexcitonQTrace_global_rng", 512) do
    s2 = siteinds("Qubit", 4)
    (; L2 = reduce(hcat, [dense(TB.mpsexcitonQTrace(Q, s2)) for Q in 1:4]))
end

case!("TwoParticle", "exciton_mpsexcitonQTrace_explicit_rng", 513) do
    s3 = siteinds("Qubit", 6)
    rng = MersenneTwister(11)
    (; Q3 = dense(TB.mpsexcitonQTrace(3, s3; rng=rng)), Q8 = dense(TB.mpsexcitonQTrace(8, s3; rng=rng)))
end

case!("TwoParticle", "exciton_mpsexcitonKQ", 514) do
    s2 = siteinds("Qubit", 4)
    (; L2 = reduce(hcat, [dense(TB.mpsexcitonKQ(k, Q, s2)) for Q in 1:4 for k in 1:4]))
end

# Argument checks of the momentum probes: (function, bad argument) pairs.
for (fname, f) in (("mpsexcitonQ", (Q, s) -> TB.mpsexcitonQ(Q, s)),
                   ("mpsexcitonQTrace", (Q, s) -> TB.mpsexcitonQTrace(Q, s)),
                   ("mpsexcitonKQ", (Q, s) -> TB.mpsexcitonKQ(2, Q, s)))
    case!(() -> f(1, siteinds("Qubit", 3)), "TwoParticle", "exciton_$(fname)_odd_sites_throws", 520)
    case!(() -> f(1, [Index(3, "q3,n=$i") for i in 1:4]), "TwoParticle", "exciton_$(fname)_non_qubit_throws", 521)
    case!(() -> f(0, siteinds("Qubit", 4)), "TwoParticle", "exciton_$(fname)_Q_low_throws", 522)
    case!(() -> f(5, siteinds("Qubit", 4)), "TwoParticle", "exciton_$(fname)_Q_high_throws", 523)
end
case!(() -> TB.mpsexcitonKQ(5, 1, siteinds("Qubit", 4)), "TwoParticle", "exciton_mpsexcitonKQ_k_high_throws", 524)
case!(() -> TB.mpsexcitonKQ(0, 1, siteinds("Qubit", 4)), "TwoParticle", "exciton_mpsexcitonKQ_k_low_throws", 525)

# ═════════════════════════════════════════════════════════════════════════════
# physics/QPI.jl
# ═════════════════════════════════════════════════════════════════════════════

case!("QPI", "qpi_impurity_mpo", 601) do
    s = siteinds("Qubit", 3)
    (; x1 = dense(TB._impurity_mpo(1, 3, s, 0.7)), x4 = dense(TB._impurity_mpo(4, 3, s, 0.7)),
       x8 = dense(TB._impurity_mpo(8, 3, s, -1.3)))
end

case!("QPI", "qpi_chain_default", 602) do
    H = chain(3)
    (; qpi = TB.get_qpi(H, 20, [-0.5, 0.0, 0.8, 3.0]), scale_after = H.scale,
       central_index = TB.central_index(H))
end

case!("QPI", "qpi_chain_site_lorentz", 603) do
    H = chain(3)
    (; qpi = TB.get_qpi(H, 16, [-0.3, 0.4]; impurity_site=2, V=1.5, kernel=:lorentz, lambda=3.0,
                        maxdim=20, cutoff=1e-10))
end

case!("QPI", "qpi_chain_gaussian", 604) do
    H = chain(3)
    (; qpi = TB.get_qpi(H, 16, [0.0, 0.6]; impurity_mode=:gaussian, sigma=1.0, V=0.8))
end

case!("QPI", "qpi_chain_window", 605) do
    H = chain(3)
    (; qpi = TB.get_qpi(H, 16, [0.2]; impurity_site=3, window_fraction=0.8, window_sigma=1.0,
                        verbose=true))
end

case!("QPI", "qpi_square_window", 606) do
    H = TB.get_Hamiltonian("square_2d", 1.0; L=4, Lx=2, scale=4.5)
    (; qpi = TB.get_qpi(H, 12, [0.0, 0.5]; window_fraction=0.9, maxdim=40))
end

case!("QPI", "qpi_no_geometry_window_warns", 607) do
    H  = TB.TBHamiltonian(chain(3); geometry=nothing)
    logger = Test.TestLogger(min_level=Logging.Warn)
    q = with_logger(logger) do
        TB.get_qpi(H, 12, [0.1]; impurity_site=3, window_fraction=0.5)
    end
    (; qpi = q, warnings = [string(first(string(r.message), MESSAGE_PREFIX_CHARS)) for r in logger.logs])
end

case!("QPI", "qpi_lazy_scale", 608) do
    H = chain(3)
    H.scale = 0.0
    q = quiet(() -> TB.get_qpi(H, 12, [-0.2, 0.3]; impurity_site=5, V=0.5))
    (; qpi = q, scale_after = H.scale, center_after = H.center)
end

case!("QPI", "qpi_aux_dof_throws", 609) do
    H = chain(3)
    TB.add_spin!(H)
    TB.get_qpi(H, 8, [0.0])
end

case!(() -> TB.get_qpi(chain(3), 8, [0.0]; impurity_site=9), "QPI", "qpi_site_high_throws", 610)
case!(() -> TB.get_qpi(chain(3), 8, [0.0]; impurity_site=0), "QPI", "qpi_site_low_throws", 611)
case!(() -> TB.get_qpi(chain(3), 8, [0.0]; impurity_mode=:bogus), "QPI", "qpi_bad_mode_throws", 612)
case!("QPI", "qpi_gaussian_no_geometry_throws", 613) do
    TB.get_qpi(TB.TBHamiltonian(chain(3); geometry=nothing), 8, [0.0];
               impurity_site=2, impurity_mode=:gaussian)
end
case!("QPI", "qpi_default_site_no_geometry_throws", 614) do
    TB.get_qpi(TB.TBHamiltonian(chain(3); geometry=nothing), 8, [0.0])
end

# ═════════════════════════════════════════════════════════════════════════════
# Running and comparing
# ═════════════════════════════════════════════════════════════════════════════

"""
    run_case(case) -> (value, err)

Seed the global RNG with `case.seed` and run the case with stdout silenced.
Returns `(value, nothing)` or `(nothing, exception)`.
"""
function run_case(case::Case)
    Random.seed!(case.seed)
    try
        return (quiet(case.f), nothing)
    catch err
        return (nothing, err)
    end
end

"""
    message_prefix(err) -> Union{String,Nothing}

The first MESSAGE_PREFIX_CHARS characters of the first line of `err.msg`, or
`nothing` for an exception without a string message.
"""
function message_prefix(err)
    hasfield(typeof(err), :msg) || return nothing
    msg = getfield(err, :msg)
    msg isa AbstractString || return nothing
    return String(first(first(split(msg, '\n')), MESSAGE_PREFIX_CHARS))
end

function _approx(a::Number, e::Number)
    isequal(a, e) && return true
    return isapprox(a, e; rtol=RTOL, atol=ATOL)
end

function _approx(a::AbstractArray, e::AbstractArray)
    isequal(a, e) && return true
    fa = isfinite.(a)
    fa == isfinite.(e) || return false
    all(i -> fa[i] || isequal(a[i], e[i]), eachindex(a, e)) || return false
    return isapprox(a[fa], e[fa]; rtol=RTOL, atol=ATOL)
end

_is_float_like(T) = T <: Number && !(T <: Integer)

"""
    mismatch(actual, expected) -> Union{Nothing,String}

`nothing` when `actual` matches the golden `expected` value under the rules in
the file header, otherwise a description of the first difference.
"""
function mismatch(@nospecialize(actual), @nospecialize(expected))
    if expected isa NamedTuple
        actual isa NamedTuple || return "expected a NamedTuple, got $(typeof(actual))"
        for k in keys(expected)
            haskey(actual, k) || return "field $k missing"
            d = mismatch(actual[k], expected[k])
            d === nothing || return "field $k: $d"
        end
        return nothing
    elseif expected isa AbstractArray
        actual isa AbstractArray || return "expected an array, got $(typeof(actual))"
        size(actual) == size(expected) || return "size $(size(actual)) != expected $(size(expected))"
        eltype(actual) == eltype(expected) || return "eltype $(eltype(actual)) != expected $(eltype(expected))"
        T = eltype(expected)
        if _is_float_like(T)
            _approx(actual, expected) && return nothing
            i = argmax(abs.(ifelse.(isfinite.(expected), actual .- expected, 0)))
            return "values differ (rtol=$RTOL, atol=$ATOL); largest difference at $i: got $(actual[i]), expected $(expected[i])"
        elseif T <: AbstractArray || T === Any
            for i in eachindex(expected)
                d = mismatch(actual[i], expected[i])
                d === nothing || return "entry $i: $d"
            end
            return nothing
        else
            actual == expected && return nothing
            i = findfirst(k -> actual[k] != expected[k], eachindex(expected))
            return "entry $i: got $(repr(actual[i])), expected $(repr(expected[i]))"
        end
    elseif expected isa Number
        typeof(actual) == typeof(expected) || return "type $(typeof(actual)) != expected $(typeof(expected))"
        if _is_float_like(typeof(expected))
            _approx(actual, expected) && return nothing
        else
            actual == expected && return nothing
        end
        return "got $(repr(actual)), expected $(repr(expected))"
    else
        typeof(actual) == typeof(expected) || return "type $(typeof(actual)) != expected $(typeof(expected))"
        isequal(actual, expected) && return nothing
        return "got $(repr(actual)), expected $(repr(expected))"
    end
end

"""
    check_case(case, golden) -> Bool

`true` when the case still reproduces its golden entry; logs the difference
otherwise.
"""
function check_case(case::Case, @nospecialize(golden))
    if golden.seed != case.seed
        @error "Dynamics golden: seed changed; regenerate the data" case = case.name seed = case.seed golden = golden.seed
        return false
    end
    value, err = run_case(case)
    if golden.throws !== nothing
        if err === nothing
            @error "Dynamics output changed: case no longer throws" case = case.name expected = golden.throws
            return false
        end
        if !(err isa golden.throws)
            @error "Dynamics output changed: different exception" case = case.name expected = golden.throws got = typeof(err)
            return false
        end
        got = message_prefix(err)
        got == golden.expected.message_prefix && return true
        @error "Dynamics output changed: different error message" case = case.name expected = golden.expected.message_prefix got = got
        return false
    end
    if err !== nothing
        @error "Dynamics output changed: case now throws" case = case.name exception = (err, nothing)
        return false
    end
    d = mismatch(value, golden.expected)
    d === nothing && return true
    @error "Dynamics output changed" case = case.name detail = d
    return false
end

groups() = unique(c.group for c in CASES)

# A cold run is dominated by first-call compilation of the library code paths
# (TCI, OpSum, TDVP, DMRG, linsolve, KPM), a few minutes in all. For a quick
# check while moving one file, DYNAMICS_GOLDEN_GROUPS="Timeev,QPI" replays only
# those groups; the case-count and data-consistency checks still cover all
# cases. Leave it unset for a real check.
function selected_groups()
    spec = strip(get(ENV, "DYNAMICS_GOLDEN_GROUPS", ""))
    isempty(spec) && return groups()
    sel = [String(strip(g)) for g in split(spec, ',')]
    unknown = setdiff(sel, groups())
    isempty(unknown) || error("DYNAMICS_GOLDEN_GROUPS names unknown groups $unknown; known: $(groups())")
    @warn "DYNAMICS_GOLDEN_GROUPS is set: only $(join(sel, ", ")) are replayed; unset it for a full check"
    return sel
end

function run_tests(golden_entries)
    @testset "Dynamics outputs are pinned" begin
        names = [c.name for c in CASES]
        @test allunique(names)
        length(CASES) == EXPECTED_CASE_COUNT ||
            @error "Case count differs from EXPECTED_CASE_COUNT" got = length(CASES) expected = EXPECTED_CASE_COUNT
        @test length(CASES) == EXPECTED_CASE_COUNT
        golden = Dict(g.name => g for g in golden_entries)
        missing_data = setdiff(names, keys(golden))
        dropped      = setdiff(keys(golden), names)
        isempty(missing_data) || @error "Cases without golden data (regenerate)" cases = missing_data
        isempty(dropped)      || @error "Golden entries whose case was removed" cases = dropped
        @test isempty(missing_data)
        @test isempty(dropped)
        for group in selected_groups()
            @testset "$group" begin
                for case in CASES
                    case.group == group || continue
                    haskey(golden, case.name) || continue
                    @test check_case(case, golden[case.name])
                end
            end
        end
    end
end

end # module DynamicsGoldenRunner

if !isdefined(@__MODULE__, :DYNAMICS_GOLDEN_GENERATOR)
    DynamicsGoldenRunner.run_tests(include(joinpath(@__DIR__, "data", "dynamics_golden.jl")))
end
