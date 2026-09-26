# Generator for test/data/rpa_golden.jl, the pinned ("golden") outputs of
# src/physics/RPA_tk.jl, checked by test/golden_rpa.jl.
#
# Reference: the golden data was generated from commit 1a5548b (branch Anouar,
# "Run the third-round audit regression tests from runtests.jl"), before the
# Tier 1 code reorganisation (docs/dev/REORGANISATION_TODO.md). It pins what
# the RPA code does at that commit, remaining bugs included. It was regenerated
# at 4cf90f8, whose src/ computes the same RPA outputs, to add the keyword
# forwarding and default-keyword cases; every earlier case kept its values.
#
# Line 4 of the data file records the git tree hash of the working-tree src/,
# i.e. the src/ that the regenerating commit will contain.
#
# Regenerate ONLY when a behaviour change is intentional. Do it in the same
# commit as the change, review the diff of rpa_golden.jl case by case, and
# record the change in the changelog. A refactor that is meant to leave outputs
# alone must pass test/golden_rpa.jl against the existing data unchanged.
#
# How to run, from the repository root with the package environment:
#
#     JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. test/data/generate_rpa_golden.jl
#
# The output is deterministic: no timestamps, cases in a fixed order, a seed per
# case derived from its name, values written with `repr`. The script reloads the
# file it wrote and checks that every entry round-trips exactly.
#
# Never pinned, because the Tier 1 plan deleted them (in 28f611f): mps_kron,
# nsitelegs, merge_mps_to_mpo, convert_mpo, _swap_mpo, apply_interleave_swaps,
# get_Tnlists, get_bublle_expanded_from_Tn, build_bubble_mpo.
# get_bubble_mpo_haydock was on the same list but is kept (test/bugfix_rpa.jl
# exercises it), so it is pinned; drop its cases (and its EXPECTED_CASE_COUNTS
# entry) if it is ever deleted.
#
# Sizes are tiny on purpose (L <= 3 position qubits, Ncheb <= 8, maxdim 20): the
# test is a guard against changed outputs, not a physics check. The exceptions
# are the `*_defaults` cases, which call a driver with no keyword at all, so that
# its defaults (Ncheb = 150 or 50, maxdim = 200, η = 1e-3, P_method =
# :purification, …) are pinned too; they run on the smallest inputs (L = 2).
#
# The Dyson and Wynn drivers (get_rpa_susceptibility, get_rpa_susceptibility_wynn,
# get_magnon_susceptibility, get_magnon_susceptibility_wynn) hand their bubble
# keywords on to get_bubble_mpo. The `*_fwd_*` cases pin that forwarding: between
# them they set every forwarded keyword to a non-default value that changes the
# output (checked by dropping each keyword in turn when the cases were written),
# so a keyword the driver stops forwarding changes a pinned value. `verbose`
# changes no number; the runner pins it through `bubble_progress_printed`.
#   * FWD_KPM  : P_method=:kpm, ϵF, Ncheb, maxdim, cutoff, η, verbose;
#   * FWD_KRY  : GF_method=:krylov, krylov_nsweeps/maxdim/cutoff, and the McWeeny
#                path with a purify_tol that stops it early;
#   * FWD_SP2  : purify_method=:sp2 with a purify_maxdim and purify_maxiters that
#                bind.
#
# Behaviour pinned as it is at 1a5548b and worth knowing before regenerating
# (a fix of any of these changes the data on purpose):
#   * interleave_mpo embeds transpose(op) for site tensors stored (s', s), the
#     order op()/OpSum MPOs use: it maps siteinds(op, i) in storage order onto
#     (p, p'). Cases interleave_mpo_L2_n1 and ..._ketfirst give the same output
#     for mutually transposed operators;
#   * _get_density_matrix(:kpm) (get_density_from_Tn) returns the projector onto
#     the states ABOVE ϵF (tr(P·H) > 0), :purification the one below, so the
#     sign of every bubble flips with P_method;
#   * ϵF does not reach the :purification path (density_mcweeny and
#     density_mcweeny_ef03 are identical);
#   * the expansion Σ C[m,n] T_m(x) T_n(y) of chebyshev2d_gf_coeffs reproduces
#     f(x, y)/4 (it divides by (2N)^2), so every cheb2d bubble is 4x smaller
#     than the same Chebyshev formula with correctly normalised coefficients;
#   * wynn_epsilon returns 1e30 for the higher estimates once a sequence is
#     exactly converged (wynn_geometric_len7, wynn_constant_len5);
#   * haydock_cf measures with tr(conj(A)·B) instead of tr(A†B)
#     (haydock_cf_chain2_imaginary_hermitian_seed throws DomainError).

using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Random
const TB = TensorBinding

const RPA_GOLDEN_GENERATOR = true
include(joinpath(@__DIR__, "..", "golden_rpa.jl"))   # loads RPAGoldenRunner only
const R = RPAGoldenRunner

const OUTFILE = joinpath(@__DIR__, "rpa_golden.jl")

const CASES = Tuple{String,Symbol,Symbol,Tuple,NamedTuple}[]
add!(name, fn, setup, args=(), kwargs=NamedTuple()) = push!(CASES, (name, fn, setup, args, kwargs))
kw(; kwargs...) = values(kwargs)

# Stable seed per case (independent of case order and of Base.hash).
seed_of(name) = foldl((h, c) -> (31h + Int(c)) % 1_000_003, codeunits(name); init=7)

const ωs  = [0.3, 0.7]
const BKW = kw(P_method=:kpm, Ncheb=8, maxdim=20, η=0.1)        # get_bubble_mpo & co
const DKW = kw(rpa_nsweeps=2, rpa_maxdim=20)                   # Dyson linsolve
const CKW = kw(P_method=:kpm, Ncheb=6, maxdim=20, η=0.1)        # cheb2d family
const WKW = kw(K_max=2, maxdim_apply=20)                       # Wynn drivers
# _get_density_matrix takes positional arguments; the runner reads them by name.
dkw(; ϵF=0.0, P_method=:kpm, purify_method=:mcweeny) =
    kw(ϵF=ϵF, P_method=P_method, Ncheb=8, maxdim=20, cutoff=1e-8, purify_method=purify_method,
       purify_maxdim=40, purify_maxiters=30, purify_tol=1e-5, verbose=false)

# ══════════════════════════════════════════════════════════════════
# MPO kron / interleave plumbing (moves to core/Utils.jl)
# ══════════════════════════════════════════════════════════════════
add!("mpo_kron_q2_sub3", :mpo_kron, :kron)
add!("interleave_mpo_L2_n1", :interleave_mpo, :ileave2, (1,))
add!("interleave_mpo_L2_n0", :interleave_mpo, :ileave2, (0,))
# Same operator stored ket-first: interleave_mpo maps siteinds(op, i) in storage order.
add!("interleave_mpo_L2_n1_ketfirst", :interleave_mpo, :ileave2_ketfirst, (1,))
add!("interleave_mpo_L1_n0", :interleave_mpo, :ileave1, (0,))
add!("interleave_mpo_tb_qubit_sub3_A", :interleave_mpo_tb, :ileave_tb_het, (:A,))
add!("interleave_mpo_tb_qubits_A", :interleave_mpo_tb, :ileave_tb_q, (:A,))
add!("interleave_mpo_tb_qubits_B", :interleave_mpo_tb, :ileave_tb_q, (:B,))
add!("swap_every_other_legs_L4", :swap_every_other_legs, :swap4)
add!("collapse_mpo_pairs_L4", :collapse_mpo_pairs, :collapse4)
add!("collapse_mpo_pairs_qubit_sub3", :collapse_mpo_pairs, :collapse_het)
add!("rpa_pair_sites_qubits", :_rpa_pair_sites, :pair_q)
add!("rpa_pair_sites_qubits_sub3", :_rpa_pair_sites, :pair_het)
add!("build_heff_chain2_cplx_pair", :_build_heff, :chain2_cplx)

# ══════════════════════════════════════════════════════════════════
# Density matrices and the Dyson solve
# ══════════════════════════════════════════════════════════════════
add!("density_kpm_ef0", :_get_density_matrix, :chain3_cplx, (), dkw())
add!("density_kpm_ef03", :_get_density_matrix, :chain3_cplx, (), dkw(ϵF=0.3))
add!("density_mcweeny", :_get_density_matrix, :chain3_cplx, (), dkw(P_method=:purification))
# ϵF is not forwarded to the purification path.
add!("density_mcweeny_ef03", :_get_density_matrix, :chain3_cplx, (),
     dkw(ϵF=0.3, P_method=:purification))
add!("density_sp2", :_get_density_matrix, :chain3_cplx, (),
     dkw(P_method=:purification, purify_method=:sp2))
add!("density_cached_purification", :_get_density_matrix, :chain3_cached, (),
     dkw(P_method=:purification))
add!("density_bad_purify_method", :_get_density_matrix, :chain3_cplx, (),
     dkw(P_method=:purification, purify_method=:bogus))
add!("density_bad_P_method", :_get_density_matrix, :chain3_cplx, (), dkw(P_method=:bogus))
add!("rpa_from_bubble_diag_qubits", :rpa_from_bubble_diag, :dyson_q, (),
     kw(nsweeps=2, maxdim=20, cutoff=1e-8))
add!("rpa_from_bubble_diag_qubit_sub3", :rpa_from_bubble_diag, :dyson_het, (),
     kw(nsweeps=3, maxdim=20, cutoff=1e-10))

# ══════════════════════════════════════════════════════════════════
# Wynn ε and the Haydock recursion (moves to solvers/Krylov.jl)
# ══════════════════════════════════════════════════════════════════
add!("wynn_geometric_len7", :wynn_epsilon, :none, (cumsum([0.5^k for k in 0:6]),))
add!("wynn_complex_len6", :wynn_epsilon, :none, (cumsum([(0.6im)^k for k in 0:5]),))
add!("wynn_alternating_len5", :wynn_epsilon, :none, (cumsum([(-1.0)^k / (k + 1) for k in 0:4]),))
add!("wynn_constant_len5", :wynn_epsilon, :none, (fill(1.0, 5),))       # |d| < 1e-30 branch
add!("wynn_int_len5", :wynn_epsilon, :none, ([1, 2, 3, 4, 5],))
add!("wynn_len1", :wynn_epsilon, :none, ([0.5],))
add!("wynn_len2", :wynn_epsilon, :none, ([0.5, 0.75],))
add!("eval_haydock_cf_N3", :eval_haydock_cf, :none, ([0.1, -0.2, 0.3], [1.5, 0.4, 0.7], 0.2 + 0.05im))
add!("eval_haydock_cf_N1", :eval_haydock_cf, :none, ([0.4], [1.2], 0.1 + 0.2im))
add!("eval_haydock_cf_real_z", :eval_haydock_cf, :none, ([0.1, -0.3], [1.0, 0.5], 0.7))
add!("haydock_cf_chain2_N3", :haydock_cf, :hay_chain2, (3,), kw(maxdim=20, cutoff=1e-10))
# Seed = identity: the Krylov space of I under H closes (invariant-subspace exit).
add!("haydock_cf_chain2_identity_N6", :haydock_cf, :hay_chain2_id, (6,), kw(maxdim=20, cutoff=1e-10))
add!("haydock_cf_chain2_N1_defaults_verbose", :haydock_cf, :hay_chain2, (1,), kw(verbose=true))
# tr(apply(dag(A), B)) is tr(conj(A)·B), not the Frobenius product tr(A†B): for the
# Hermitian seed Y ⊗ I it is -4, so norm0 = sqrt(-4) throws.
add!("haydock_cf_chain2_imaginary_hermitian_seed", :haydock_cf, :hay_chain2_imag, (2,),
     kw(maxdim=20, cutoff=1e-10))
add!("haydock_resolve_N3", :haydock_resolve_mpo, :hay_chain2, (3, 3, 0.3 + 0.1im), kw(maxdim=20, cutoff=1e-10))
add!("haydock_resolve_N1", :haydock_resolve_mpo, :hay_chain2, (3, 1, -0.2 + 0.05im))
add!("haydock_bubble_chain3_kpm", :get_bubble_mpo_haydock, :chain3, (ωs,),
     kw(N_steps=4, P_method=:kpm, Ncheb=8, maxdim=20, η=0.1))
add!("haydock_bubble_cplx_pair_mcweeny_verbose", :get_bubble_mpo_haydock, :chain3_pair, ([0.3],),
     kw(N_steps=3, maxdim=20, η=0.1, verbose=true))

# ══════════════════════════════════════════════════════════════════
# k-space diagonal (moves to the QFT conjugation file)
# ══════════════════════════════════════════════════════════════════
add!("spect_k_chain3_cplx_H", :get_spect_k, :spect_H)
add!("spect_k_det_mpo", :get_spect_k, :spect_det)
add!("spect_k_det_mpo_truncated", :get_spect_k, :spect_det, (), kw(tol=1e-4, maxdim=2))

# ══════════════════════════════════════════════════════════════════
# get_bubble_mpo (physics/rpa/Bubble.jl)
# ══════════════════════════════════════════════════════════════════
add!("bubble_chain3_kpm_kpm", :get_bubble_mpo, :chain3, (0.3,), BKW)
add!("bubble_cplx_kpm_kpm", :get_bubble_mpo, :chain3_cplx, (0.3,), BKW)
add!("bubble_cplx_kpm_ef03", :get_bubble_mpo, :chain3_cplx, (0.3,), merge(BKW, kw(ϵF=0.3)))
add!("bubble_cplx_mcweeny", :get_bubble_mpo, :chain3_cplx, (0.3,), kw(Ncheb=8, maxdim=20, η=0.1))
add!("bubble_cplx_sp2", :get_bubble_mpo, :chain3_cplx, (0.3,),
     kw(Ncheb=8, maxdim=20, η=0.1, purify_method=:sp2))
add!("bubble_cached_purification", :get_bubble_mpo, :chain3_cached, (0.3,), kw(Ncheb=8, maxdim=20, η=0.1))
add!("bubble_cplx_krylov", :get_bubble_mpo, :chain3_cplx, (0.3,),
     merge(BKW, kw(GF_method=:krylov, krylov_nsweeps=2, krylov_maxdim=20)))
add!("bubble_pair_kpm_verbose", :get_bubble_mpo, :chain3_pair, (0.4,), merge(BKW, kw(verbose=true)))
add!("bubble_kagome_kpm", :get_bubble_mpo, :kagome2, (0.3,), BKW)
add!("bubble_spin2_ydir_kpm", :get_bubble_mpo, :spin2_y, (0.3,), BKW)
add!("bubble_bad_GF_method", :get_bubble_mpo, :chain3, (0.3,), merge(BKW, kw(GF_method=:bogus)))
add!("bubble_bad_P_method", :get_bubble_mpo, :chain3, (0.3,), merge(BKW, kw(P_method=:bogus)))
add!("bubble_L_mismatch", :get_bubble_mpo, :chain3_vs_chain2, (0.3,), BKW)

# ══════════════════════════════════════════════════════════════════
# Dyson drivers, magnon channel, spin projection (physics/rpa/Dyson.jl)
# ══════════════════════════════════════════════════════════════════
add!("rpa_charge_cplx", :get_rpa_susceptibility, :chain3_cplx, (0.3,), merge(BKW, DKW))
add!("rpa_charge_kagome", :get_rpa_susceptibility, :kagome2, (0.3,), merge(BKW, DKW))
add!("rpa_magnetic_spin3_z", :get_rpa_susceptibility, :spin3_z, (0.3,),
     merge(BKW, DKW, kw(mode=:magnetic)))
add!("rpa_bad_mode", :get_rpa_susceptibility, :chain3, (0.3,), merge(BKW, DKW, kw(mode=:bogus)))
add!("rpa_magnetic_spinless", :get_rpa_susceptibility, :chain3, (0.3,),
     merge(BKW, DKW, kw(mode=:magnetic)))
add!("magnon_bubble_spin3_z", :get_magnon_bubble, :spin3_z, (0.3,), BKW)
add!("magnon_bubble_spin3_z_post", :get_magnon_bubble, :spin3_z_post, (0.3,), BKW)
add!("magnon_bubble_spinless", :get_magnon_bubble, :chain3, (0.3,), BKW)
add!("magnon_chi_spin3_z", :get_magnon_susceptibility, :spin3_z, (0.3,), merge(BKW, DKW))
add!("magnon_chi_spinless", :get_magnon_susceptibility, :chain3, (0.3,), merge(BKW, DKW))
add!("wynn_from_bubbles_K4", :rpa_wynn_from_bubbles, :wynn_bubbles, (), kw(K_max=4, maxdim_apply=20))
add!("wynn_from_bubbles_K1_verbose", :rpa_wynn_from_bubbles, :wynn_bubbles, (),
     kw(K_max=1, maxdim_apply=20, cutoff_apply=1e-6, verbose=true))
add!("wynn_charge_cplx_K2", :get_rpa_susceptibility_wynn, :chain3_cplx, (ωs,), merge(BKW, WKW))
add!("wynn_charge_chain3_K3_verbose", :get_rpa_susceptibility_wynn, :chain3, ([0.4],),
     merge(BKW, kw(K_max=3, maxdim_apply=20, verbose=true)))
add!("wynn_magnetic_spin3_z", :get_rpa_susceptibility_wynn, :spin3_z, (ωs,),
     merge(BKW, WKW, kw(mode=:magnetic)))
add!("wynn_bad_mode", :get_rpa_susceptibility_wynn, :chain3, (ωs,), merge(BKW, WKW, kw(mode=:bogus)))
add!("wynn_magnetic_spinless", :get_rpa_susceptibility_wynn, :chain3, (ωs,),
     merge(BKW, WKW, kw(mode=:magnetic)))
add!("magnon_wynn_spin3_z", :get_magnon_susceptibility_wynn, :spin3_z, (ωs,), merge(BKW, WKW))
add!("magnon_wynn_spinless", :get_magnon_susceptibility_wynn, :chain3, (ωs,), merge(BKW, WKW))
add!("project_spin3_z_up", :_project_spin_sector, :spin3_z, (1,))
add!("project_spin3_z_dn", :_project_spin_sector, :spin3_z, (2,))
add!("project_spin3_z_post_up", :_project_spin_sector, :spin3_z_post, (1,))
add!("project_spinless", :_project_spin_sector, :chain3, (1,))

# ══════════════════════════════════════════════════════════════════
# Double Chebyshev family (physics/rpa/Cheb2D.jl)
# ══════════════════════════════════════════════════════════════════
add!("cheb2d_coeffs_asym_N5", :chebyshev2d_gf_coeffs, :none, (0.4, 2.2, 0.1, 2.5, -0.2, 0.3, 5))
add!("cheb2d_coeffs_sym_N7", :chebyshev2d_gf_coeffs, :none, (-0.7, 3.0, 0.0, 3.0, 0.0, 0.05, 7))
add!("jackson_kernel_N1", :_jackson_kernel, :none, (1,))
add!("jackson_kernel_N5", :_jackson_kernel, :none, (5,))
add!("jackson_kernel_N9", :_jackson_kernel, :none, (9,))
add!("weighted_mpo_sum_complex_skip_tiny", :_weighted_mpo_sum, :wsum, ([0.5 + 0.1im, 1e-15, -0.25],),
     kw(maxdim=10, cutoff=1e-12))
add!("weighted_mpo_sum_real_truncated", :_weighted_mpo_sum, :wsum_real, ([0.5, -0.25, 0.125],),
     kw(maxdim=2, cutoff=1e-8))
add!("weighted_mpo_sum_all_below_tol", :_weighted_mpo_sum, :wsum, ([1e-15, 1e-16, 0.0],),
     kw(maxdim=10, cutoff=1e-12))
add!("cheb2d_out_sites_chain3", :_cheb2d_out_sites, :chain3, ("fname",))
add!("cheb2d_out_sites_mismatch", :_cheb2d_out_sites, :spin2_vs_chain2, ("get_bubble_mpo_cheb2d",))
add!("cheb2d_position_sites_chain3", :_cheb2d_require_position_sites, :chain3, ("fname",))
add!("cheb2d_position_sites_spin", :_cheb2d_require_position_sites, :spin2_y, ("get_bubble_diag_cheb2d",))
add!("cheb2d_position_sites_kagome", :_cheb2d_require_position_sites, :kagome2, ("fname",))

add!("cheb2d_mpo_cplx", :get_bubble_mpo_cheb2d, :chain3_cplx, (ωs,), CKW)
add!("cheb2d_mpo_pair_mcweeny_verbose", :get_bubble_mpo_cheb2d, :chain3_pair, ([0.3],),
     kw(Ncheb=6, maxdim=20, η=0.1, verbose=true))
add!("cheb2d_mpo_cplx_coeff_tol", :get_bubble_mpo_cheb2d, :chain3_cplx, (ωs,), merge(CKW, kw(coeff_tol=0.12)))
add!("cheb2d_mpo_spin2_ydir", :get_bubble_mpo_cheb2d, :spin2_y, ([0.3],), CKW)
add!("cheb2d_mpo_site_mismatch", :get_bubble_mpo_cheb2d, :spin2_vs_chain2, ([0.3],), CKW)

add!("cheb2d_mpo_tucker_cplx", :get_bubble_mpo_cheb2d_tucker, :chain3_cplx, (ωs,), CKW)
add!("cheb2d_mpo_tucker_none_hosvd", :get_bubble_mpo_cheb2d_tucker, :chain3_cplx, (ωs,),
     merge(CKW, kw(kernel=:none, hooi_iters=0, tucker_tol=1e-14, tucker_maxrank=7)))
add!("cheb2d_mpo_tucker_maxrank2_verbose", :get_bubble_mpo_cheb2d_tucker, :chain3_cplx, (ωs,),
     merge(CKW, kw(tucker_maxrank=2, verbose=true)))
add!("cheb2d_mpo_tucker_pair", :get_bubble_mpo_cheb2d_tucker, :chain3_pair, ([0.3],), CKW)
add!("cheb2d_mpo_tucker_bad_kernel", :get_bubble_mpo_cheb2d_tucker, :chain3, ([0.3],),
     merge(CKW, kw(kernel=:bogus)))
add!("cheb2d_mpo_tucker_site_mismatch", :get_bubble_mpo_cheb2d_tucker, :spin2_vs_chain2, ([0.3],), CKW)

add!("cheb2d_diag_cplx", :get_bubble_diag_cheb2d, :chain3_cplx, (ωs,), CKW)
add!("cheb2d_diag_pair_verbose", :get_bubble_diag_cheb2d, :chain3_pair, ([0.3],), merge(CKW, kw(verbose=true)))
add!("cheb2d_diag_coeff_tol_qft_truncated", :get_bubble_diag_cheb2d, :chain3_cplx, (ωs,),
     merge(CKW, kw(coeff_tol=0.12, qft_tol=1e-6, qft_maxdim=4)))
add!("cheb2d_diag_chain3_mcweeny", :get_bubble_diag_cheb2d, :chain3, ([0.5],), kw(Ncheb=6, maxdim=20, η=0.1))
add!("cheb2d_diag_spin_refused", :get_bubble_diag_cheb2d, :spin2_y, ([0.3],), CKW)

add!("cheb2d_svd_cplx", :get_bubble_diag_cheb2d_svd, :chain3_cplx, (ωs,), CKW)
add!("cheb2d_svd_none", :get_bubble_diag_cheb2d_svd, :chain3_cplx, (ωs,), merge(CKW, kw(kernel=:none)))
add!("cheb2d_svd_maxrank1_verbose", :get_bubble_diag_cheb2d_svd, :chain3_cplx, (ωs,),
     merge(CKW, kw(svd_maxrank=1, svd_tol=1e-3, verbose=true)))
add!("cheb2d_svd_pair", :get_bubble_diag_cheb2d_svd, :chain3_pair, ([0.3],), CKW)
add!("cheb2d_svd_bad_kernel", :get_bubble_diag_cheb2d_svd, :chain3, ([0.3],), merge(CKW, kw(kernel=:bogus)))
add!("cheb2d_svd_spin_refused", :get_bubble_diag_cheb2d_svd, :spin2_y, ([0.3],), CKW)

add!("cheb2d_diag_tucker_cplx", :get_bubble_diag_cheb2d_tucker, :chain3_cplx, (ωs,), CKW)
add!("cheb2d_diag_tucker_none_hosvd", :get_bubble_diag_cheb2d_tucker, :chain3_cplx, (ωs,),
     merge(CKW, kw(kernel=:none, hooi_iters=0)))
add!("cheb2d_diag_tucker_maxrank2_coeff_tol_verbose", :get_bubble_diag_cheb2d_tucker, :chain3_cplx, (ωs,),
     merge(CKW, kw(tucker_maxrank=2, coeff_tol=1e-3, verbose=true)))
add!("cheb2d_diag_tucker_pair", :get_bubble_diag_cheb2d_tucker, :chain3_pair, ([0.3],), CKW)
add!("cheb2d_diag_tucker_bad_kernel", :get_bubble_diag_cheb2d_tucker, :chain3, ([0.3],),
     merge(CKW, kw(kernel=:bogus)))
add!("cheb2d_diag_tucker_spin_refused", :get_bubble_diag_cheb2d_tucker, :spin2_y, ([0.3],), CKW)

# ══════════════════════════════════════════════════════════════════
# Keyword forwarding of the Dyson and Wynn drivers to get_bubble_mpo
# ══════════════════════════════════════════════════════════════════
# See the header. Dropping a forwarded keyword fails every case that sets it,
# with one exception: maxdim=20 does not bind in the two magnon Krylov cases
# (it does in the magnon FWD_KPM and FWD_SP2 cases). Dropping purify_maxiters
# from a charge FWD_SP2 case makes the call throw an ArgumentError, which fails
# the case just as well. DKW's rpa_nsweeps and rpa_maxdim and WKW's maxdim_apply
# do not bind at this size; they are the drivers' own keywords, not forwarded.
const FWD_KPM = kw(P_method=:kpm, ϵF=0.3, Ncheb=8, maxdim=6, cutoff=1e-4, η=0.1, verbose=true)
const FWD_KRY = kw(GF_method=:krylov, krylov_nsweeps=2, krylov_maxdim=3, krylov_cutoff=1e-2,
                   purify_tol=1e-2, maxdim=20, η=0.1)
const FWD_SP2 = kw(purify_method=:sp2, purify_maxdim=3, purify_maxiters=3, Ncheb=8, maxdim=20, η=0.1)
for (fn, setup, label) in ((:get_rpa_susceptibility, :chain3_cplx, "rpa_charge_cplx"),
                           (:get_magnon_susceptibility, :spin3_z, "magnon_chi_spin3_z"))
    add!("$(label)_fwd_kpm_ef03_cutoff_verbose", fn, setup, (0.3,),
         merge(FWD_KPM, DKW, kw(rpa_cutoff=1e-4)))
    add!("$(label)_fwd_krylov_mcweeny_tol", fn, setup, (0.3,), merge(FWD_KRY, DKW))
    add!("$(label)_fwd_sp2_maxdim_maxiters", fn, setup, (0.3,), merge(FWD_SP2, DKW))
end
for (fn, setup, label) in ((:get_rpa_susceptibility_wynn, :chain3_cplx, "wynn_charge_cplx"),
                           (:get_magnon_susceptibility_wynn, :spin3_z, "magnon_wynn_spin3_z"))
    add!("$(label)_fwd_kpm_ef03_cutoff_verbose", fn, setup, ([0.3],),
         merge(FWD_KPM, WKW, kw(cutoff_apply=1e-4)))
    add!("$(label)_fwd_krylov_mcweeny_tol", fn, setup, ([0.3],), merge(FWD_KRY, WKW))
    add!("$(label)_fwd_sp2_maxdim_maxiters", fn, setup, ([0.3],), merge(FWD_SP2, WKW))
end

# ══════════════════════════════════════════════════════════════════
# Default keywords: every driver called with no keyword at all
# ══════════════════════════════════════════════════════════════════
add!("bubble_chain2_defaults", :get_bubble_mpo, :chain2, (0.3,))
add!("rpa_charge_chain2_defaults", :get_rpa_susceptibility, :chain2, (0.3,))
add!("wynn_charge_chain2_defaults", :get_rpa_susceptibility_wynn, :chain2, ([0.3],))
add!("magnon_bubble_spin2_z_defaults", :get_magnon_bubble, :spin2_z, (0.3,))
add!("magnon_chi_spin2_z_defaults", :get_magnon_susceptibility, :spin2_z, (0.3,))
add!("magnon_wynn_spin2_z_defaults", :get_magnon_susceptibility_wynn, :spin2_z, ([0.3],))
add!("rpa_from_bubble_diag_qubits_defaults", :rpa_from_bubble_diag, :dyson_q)
add!("wynn_from_bubbles_defaults", :rpa_wynn_from_bubbles, :wynn_bubbles)
add!("haydock_bubble_chain2_defaults", :get_bubble_mpo_haydock, :chain2, ([0.3],))
add!("cheb2d_mpo_chain2_defaults", :get_bubble_mpo_cheb2d, :chain2, ([0.3],))
add!("cheb2d_mpo_tucker_chain2_defaults", :get_bubble_mpo_cheb2d_tucker, :chain2, ([0.3],))
add!("cheb2d_diag_chain2_defaults", :get_bubble_diag_cheb2d, :chain2, ([0.3],))
add!("cheb2d_svd_chain2_defaults", :get_bubble_diag_cheb2d_svd, :chain2, ([0.3],))
add!("cheb2d_diag_tucker_chain2_defaults", :get_bubble_diag_cheb2d_tucker, :chain2, ([0.3],))

# ══════════════════════════════════════════════════════════════════
# Evaluation and output
# ══════════════════════════════════════════════════════════════════

function evaluate(cases)
    entries = NamedTuple[]
    for (name, fn, setup, args, kwargs) in cases
        seed = seed_of(name)
        t = time()
        result, err = try
            (R.run_case(fn, setup, seed, args, kwargs), nothing)
        catch e
            e isa InterruptException && rethrow()
            (nothing, e)
        end
        dt = round(time() - t; digits=2)
        if err === nothing
            println(rpad(name, 52), " ok      ", dt, " s")
            push!(entries, (; name, fn, setup, seed, args, kwargs, throws=nothing, expected=result))
        else
            prefix = R.message_prefix(err)
            shown = first(first(split(sprint(showerror, err), '\n')), 110)
            println(rpad(name, 52), " THROWS  ", shown, "  ", dt, " s")
            push!(entries, (; name, fn, setup, seed, args, kwargs, throws=typeof(err),
                            expected=(; message_prefix=prefix)))
        end
    end
    return entries
end

# The git tree hash of the working-tree src/, computed through a throwaway index.
function src_info()
    index = tempname()
    try
        repo = normpath(joinpath(@__DIR__, "..", ".."))
        tree = withenv("GIT_INDEX_FILE" => index) do
            run(`git -C $repo read-tree HEAD`)
            run(`git -C $repo add -A -- src`)
            readchomp(`git -C $repo write-tree --prefix=src/`)
        end
        return "src/ tree $(first(tree, 12))"
    catch
        return "an unknown src/ tree (not a git checkout)"
    finally
        rm(index; force=true)
    end
end

function write_golden(path, entries)
    source = src_info()
    open(path, "w") do io
        println(io, "# AUTO-GENERATED by test/data/generate_rpa_golden.jl -- do not edit by hand.")
        println(io, "#")
        println(io, "# Pinned outputs of src/physics/rpa/, checked by test/golden_rpa.jl.")
        println(io, "# Generated from $source")
        println(io, "# with Julia $VERSION, ITensors $(pkgversion(ITensors)), ITensorMPS $(pkgversion(ITensorMPS)).")
        println(io, "# Regenerate only for an intentional behaviour change; see the generator's header.")
        println(io, "#")
        println(io, "# Each entry is (name, fn, setup, seed, args, kwargs, throws, expected): `fn`")
        println(io, "# selects a call in RPAGoldenRunner.run_case, `setup` its non-literal inputs,")
        println(io, "# `throws` is the exception type a failing case must still raise (otherwise")
        println(io, "# `nothing`), and `expected` holds the record of the output or, for a throwing")
        println(io, "# case, the first $(R.MESSAGE_PREFIX_CHARS) characters of its message (`message_prefix`).")
        println(io, "#")
        println(io, "# $(length(entries)) cases, $(count(e -> e.throws !== nothing, entries)) of them pinned as throwing.")
        println(io)
        println(io, "RPA_GOLDEN_CASES = Any[]")
        for e in entries
            println(io)
            println(io, "push!(RPA_GOLDEN_CASES, (name = ", repr(e.name), ", fn = ", repr(e.fn),
                    ", setup = ", repr(e.setup), ", seed = ", repr(e.seed), ",")
            println(io, "    args = ", repr(e.args), ",")
            println(io, "    kwargs = ", repr(e.kwargs), ",")
            println(io, "    throws = ", repr(e.throws), ",")
            println(io, "    expected = ", repr(e.expected), "))")
        end
        println(io)
        println(io, "RPA_GOLDEN_CASES")
    end
end

t0 = time()
entries = evaluate(CASES)
println("evaluated $(length(entries)) cases in $(round(time() - t0; digits=1)) s")
write_golden(OUTFILE, entries)

# Round trip: every entry must read back exactly as it was computed.
loaded = include(OUTFILE)
length(loaded) == length(entries) || error("round trip: $(length(loaded)) entries read back, $(length(entries)) written")
for (a, b) in zip(loaded, entries)
    isequal(a, b) || error("round trip failed for case $(b.name)")
    typeof(a.expected) == typeof(b.expected) ||
        error("round trip changed the types of $(b.name).expected: $(typeof(a.expected)) != $(typeof(b.expected))")
end

counts = R.case_counts(entries)
println("Wrote $(length(entries)) cases to $OUTFILE ($(filesize(OUTFILE)) bytes)")
for fn in sort!(collect(keys(counts)))
    println("    :", rpad(string(fn) * "", 34), " => ", counts[fn], ",")
end
if counts != R.EXPECTED_CASE_COUNTS
    @warn "The case counts differ from EXPECTED_CASE_COUNTS in test/golden_rpa.jl. " *
          "If the change is intended, update that table in the same commit." got = counts expected = R.EXPECTED_CASE_COUNTS
end
