using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: get_Hamiltonian, add_spin!, nh_block_index, hermitized_hamiltonian,
                     hermitize, nh_kpm_scale, add_nh_onsite!, loss_profile_mpo, add_loss!,
                     nh_nonreciprocal_hopping_mpo, add_nh_nonreciprocal_hopping!,
                     add_nh_skin_hopping!, nh_block_source, nh_kpm_partials,
                     contract_nh_block, nh_preprocess_partials, nh_ones_mps,
                     nh_jackson_weights, nh_reconstruct_spectral_mps,
                     nh_spectral_function, nh_spectrum_grid

# Characterization ("golden") tests for the non-Hermitian toolkit,
# src/physics/NH_tk.jl.
#
# These tests pin what the NH code computes *today*, bugs included, so that the
# Tier 1 reorganisation (docs/dev/REORGANISATION_TODO.md: split NH_tk.jl into
# NH_model.jl and NH_KPM.jl, move helpers, delete dead code) cannot silently
# change an output. The expected values live in `test/data/nh_golden.jl`,
# written by `test/data/generate_nh_golden.jl` (see its header for the commit it
# was generated from and how to rerun it). Both files use the case list below.
#
# A failure here means an NH output changed. If the change is a regression,
# fix the code. If it is intentional, regenerate the data file in the same
# commit as the behaviour change and review the data diff case by case.
#
# What a case records: MPO outputs as dense matrices, MPS outputs as dense
# vectors (both in the big-endian basis of an explicit site list, first site =
# most significant bit), or, for the larger ones, a fingerprint (`fp`: size,
# norm, trace, a weighted sum and two entries); scalars as they are returned;
# errors as the exception type and the first MESSAGE_PREFIX_CHARS characters of
# the message (`caught`).
#
# Comparison rules (`mismatch`):
#   * floating-point scalars and arrays: same type / same size and element
#     type, and isapprox(rtol=RTOL, atol=ATOL) (norm-wise for arrays), so that a
#     Tier 2 reordering of the arithmetic does not trip the test;
#   * everything else (Int, Bool, Symbol, String, sizes, error records): equal
#     with the same type;
#   * every golden case must still exist and every case must have golden data,
#     except a `requires` case whose function is gone (see below);
#   * every function in MIN_CASES keeps at least its floor of compared cases,
#     so an empty or shrunken case list fails instead of passing vacuously.
#
# The "Truncation" section pins the forwarding of `maxdim` and `cutoff` through
# hermitize, the NH KPM routines and nh_spectrum_grid (its header lists the few
# truncations it cannot reach).
#
# Cases marked `requires = :name` exercise functions that the Tier 1 dead-code
# list slated for deletion (nh_imag_onsite_mpo, add_nh_imag_onsite!,
# add_nh_loss!, nh_reconstruct_spectral_mpo, nh_spectral_function_allsite_mpo;
# all deleted in 6d79888). They are compared while the function exists and
# skipped with an @info once it has been deleted, so the deletion does not need
# to touch this file. The generator leaves such a case out of regenerated data,
# and the test accepts the data with or without its record.
#
# Every case reseeds the global RNG (QTCI pivots in get_diagonal_mpo, DMRG
# start states in the scale estimate, the stochastic probes) with a seed
# derived from its name, so cases can be reordered or added freely.
#
# The runner below is shared with the generator, which includes this file with
# `NH_GOLDEN_GENERATOR` defined so that only the module is loaded.

module NHGolden

using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: get_Hamiltonian, add_spin!, nh_block_index, hermitized_hamiltonian,
                     hermitize, nh_kpm_scale, add_nh_onsite!, loss_profile_mpo, add_loss!,
                     nh_nonreciprocal_hopping_mpo, add_nh_nonreciprocal_hopping!,
                     add_nh_skin_hopping!, nh_block_source, nh_kpm_partials,
                     contract_nh_block, nh_preprocess_partials, nh_ones_mps,
                     nh_jackson_weights, nh_reconstruct_spectral_mps,
                     nh_spectral_function, nh_spectrum_grid
const TB = TensorBinding

const RTOL = 1e-10
const ATOL = 1e-12
const MESSAGE_PREFIX_CHARS = 60

const DATA_FILE = joinpath(@__DIR__, "data", "nh_golden.jl")

# ── Probes: package-independent views of MPO/MPS outputs ───────────────────────

"""
    dense_op(W, basis) -> Matrix

Dense matrix of the MPO `W` in the product basis of `basis` (a permutation of
the MPO's site indices), big-endian: `basis[1]` is the most significant bit.
"""
function dense_op(W::MPO, basis)
    T = ITensor(1.0)
    for i in 1:length(W)
        T *= W[i]
    end
    D = prod(dim, basis)
    return reshape(Array(T, reverse(prime.(basis))..., reverse(basis)...), D, D)
end

"""
    dense_vec(psi, basis) -> Vector

Dense vector of the MPS `psi` in the big-endian product basis of `basis`.
"""
function dense_vec(psi::MPS, basis)
    T = ITensor(1.0)
    for i in 1:length(psi)
        T *= psi[i]
    end
    return vec(Array(T, reverse(basis)...))
end

"""Position in `basis` of the (first) site index carried by each MPO tensor."""
chain_order(W::MPO, basis) = [findfirst(s -> hasind(W[i], s), basis) for i in 1:length(W)]

site_tags(sites) = [string(tags(s)) for s in sites]

_weight(i) = 1.0 + (i % 7) / 8 + i / 1024

"""
    fp(M) -> NamedTuple

Fingerprint of a dense array: size, norm, trace (matrices), a position-weighted
sum that moves when any single entry moves, and the first and last entries
(whose types pin the element type).
"""
function fp(M::AbstractArray)
    tr_ = ndims(M) == 2 ? sum(M[i, i] for i in 1:minimum(size(M))) : sum(M)
    return (size = size(M), norm = norm(M), trace = tr_,
            weighted = sum(M[i] * _weight(i) for i in eachindex(M)),
            first = M[begin], last = M[end])
end

"""Diagonal of a (diagonal) MPO and the norm of what lies off the diagonal."""
function diagrec(W::MPO, basis)
    d = dense_op(W, basis)
    return (diag = diag(d), offdiag_norm = norm(d - Diagonal(diag(d))))
end

function message_prefix(err)
    hasfield(typeof(err), :msg) || return ""
    msg = getfield(err, :msg)
    msg isa AbstractString || return ""
    return String(first(first(split(msg, '\n')), MESSAGE_PREFIX_CHARS))
end

"""
    caught(f) -> NamedTuple

Run `f()` and record the exception it throws (type name and message prefix),
or `error = "none"` when it returns normally.
"""
function caught(f)
    try
        f()
    catch err
        return (error = string(nameof(typeof(err))), message = message_prefix(err))
    end
    return (error = "none", message = "")
end

# ── Inputs ───────────────────────────────────────────────────────────────────

# Profiles handed to the QTCI builders. Every new function type re-specialises
# the QTCI stack, so the cases share these few (0-indexed n, ix, iy).
prof(n)       = 0.1 + 0.05 * n + 0.1 * (n % 2)   # real, 1D
cprof(n)      = prof(n) - 0.3im * n               # complex, 1D
prof2(ix, iy) = 0.05 + 0.1 * ix + 0.3 * iy        # real, 2D

# 8-site open chain, t = 1 (analytic MPO, no randomness).
chain(L::Int = 3) = get_Hamiltonian("chain_1d", 1.0; L = L)

# The chain with onsite loss -i prof(n), built by add_loss! (QTCI).
function lossy_chain(L::Int = 3)
    H = chain(L)
    add_loss!(H, prof)
    return H
end

lossy_nh(; z = 0.3 - 0.2im, kwargs...) = hermitize(lossy_chain(); z = z, kwargs...)

function spin_chain()
    H = chain()
    add_spin!(H)
    return H
end

# KPM settings shared by the online and reconstruction cases.
const NHALF  = 3      # Ncheb = 2 * NHALF = 6
const SCALE  = 4.0
const MAXDIM = 20

# ── Case registry ────────────────────────────────────────────────────────────

struct Case
    name     :: String
    requires :: Union{Nothing,Symbol}   # function that may be deleted in Tier 1
    run      :: Function
end

const CASES = Case[]
case(f, name::AbstractString; requires::Union{Nothing,Symbol} = nothing) =
    push!(CASES, Case(String(name), requires, f))

"""Stable per-case seed (independent of case order and Julia's `hash`)."""
case_seed(name::AbstractString) =
    foldl((h, c) -> (31 * h + Int(c)) % 2_147_483_647, codeunits(name); init = 17)

is_available(c::Case) = c.requires === nothing || isdefined(TensorBinding, c.requires)

"""Run one case with its seed, discarding what it prints (DMRG progress lines)."""
function run_case(c::Case)
    Random.seed!(case_seed(c.name))
    return redirect_stdout(devnull) do
        c.run()
    end
end

# ── Records shared by several cases ──────────────────────────────────────────

function hh_record(Hh, H)
    return (herm = dense_op(Hh.mpo, Hh.sites), chain = chain_order(Hh.mpo, Hh.sites),
            site_tags = site_tags(Hh.sites), L = Hh.L, N = Hh.N, nsites = length(Hh.sites),
            scale = Hh.scale, center = Hh.center, aux_side = Hh.aux_side, Lx = Hh.Lx,
            geometry_shared = Hh.geometry === H.geometry,
            position_space = string(nameof(typeof(Hh.position_space))),
            interaction_is_nothing = Hh.interaction_mpo === nothing,
            fock_is_nothing = Hh.fock_mpo === nothing,
            spin_is_nothing = Hh.spin_s === nothing)
end

function nh_record(NH)
    return (z = NH.z, placement = NH.block_placement, block_dim = dim(NH.block_s),
            block_tags = string(tags(NH.block_s)),
            block_pos = findfirst(==(NH.block_s), NH.hermitized.sites),
            shown = sprint(show, NH), hermitized = hh_record(NH.hermitized, NH.parent))
end

pos_sites(NH) = NH.parent.sites

grid_record(out) = (nout = length(out), x = collect(out[1]), y = collect(out[2]),
                    x_is_range = out[1] isa AbstractRange, Z = out[3])

# ═════════════════════════════════════════════════════════════════════════════
# Model building: struct, hermitization, add_nh_* and loss helpers
# ═════════════════════════════════════════════════════════════════════════════

case("inputs_chain_and_lossy_chain") do
    H0 = chain()
    H  = lossy_chain()
    (chain = dense_op(H0.mpo, H0.sites), chain_scale = H0.scale,
     lossy = dense_op(H.mpo, H.sites), lossy_scale = H.scale, lossy_center = H.center)
end

case("nh_block_index") do
    b = nh_block_index()
    (dim = dim(b), tags = string(tags(b)), plev = plev(b), fresh = b != nh_block_index())
end

case("hermitized_hamiltonian_post_z_minus_H") do
    H = lossy_chain()
    before = dense_op(H.mpo, H.sites)
    b = nh_block_index()
    Hh = hermitized_hamiltonian(H; z = 0.3 - 0.2im, block_s = b)
    merge(hh_record(Hh, H), (block_is_given = Hh.sites[end] == b,
                             parent_unchanged = dense_op(H.mpo, H.sites) == before))
end

case("hermitized_hamiltonian_pre_H_minus_z_scaled") do
    H = lossy_chain()
    Hh = hermitized_hamiltonian(H; z = -0.5 + 0.4im, block_placement = :pre,
                                convention = :H_minus_z, scale = 3.5)
    merge(hh_record(Hh, H), (block_first = hastags(Hh.sites[1], "NHBlock"),))
end

case("hermitized_hamiltonian_hermitian_parent_z0") do
    H = chain()
    hh_record(hermitized_hamiltonian(H), H)
end

case("hermitized_hamiltonian_errors") do
    H = chain()
    (bad_convention = caught(() -> hermitized_hamiltonian(H; convention = :H_plus_z)),
     bad_placement  = caught(() -> hermitized_hamiltonian(H; block_placement = :middle)))
end

case("hermitize_post") do
    H = lossy_chain()
    NH = hermitize(H; z = 0.3 - 0.2im)
    merge(nh_record(NH), (parent_is_input = NH.parent === H,
                          type = string(nameof(typeof(NH)))))
end

case("hermitize_pre_H_minus_z_scaled_integer_z") do
    nh_record(hermitize(lossy_chain(); z = 1, block_placement = :pre,
                        convention = :H_minus_z, scale = 2.5))
end

# hermitize(NH) inherits z and block_placement but not convention or scale.
case("hermitize_rebuild_from_nh") do
    H = lossy_chain()
    NH1 = hermitize(H; z = 0.2 + 0.1im, block_placement = :pre,
                    convention = :H_minus_z, scale = 2.0)
    NH2 = hermitize(NH1; z = -0.4im)
    NH3 = hermitize(NH1)
    NH4 = hermitize(NH1; convention = :H_minus_z, block_placement = :post, scale = 1.5)
    d1 = dense_op(NH1.hermitized.mpo, NH1.hermitized.sites)
    d3 = dense_op(NH3.hermitized.mpo, NH3.hermitized.sites)
    (nh2 = nh_record(NH2), nh2_new_block = NH2.block_s != NH1.block_s,
     nh2_parent_shared = NH2.parent === H,
     nh3_z = NH3.z, nh3_placement = NH3.block_placement, nh3_scale = NH3.hermitized.scale,
     nh3_minus_nh1 = norm(d3 - d1), nh3_plus_nh1 = norm(d3 + d1),
     nh4_placement = NH4.block_placement, nh4_scale = NH4.hermitized.scale,
     nh4 = fp(dense_op(NH4.hermitized.mpo, NH4.hermitized.sites)))
end

case("add_nh_onsite_number") do
    H = chain()
    ret = add_nh_onsite!(H, 0.2 - 0.1im)
    (dense = dense_op(H.mpo, H.sites), returns_input = ret === H,
     scale = H.scale, center = H.center)
end

case("add_nh_onsite_function") do
    H = chain()
    add_nh_onsite!(H, cprof)
    (dense = dense_op(H.mpo, H.sites), scale = H.scale)
end

case("add_nh_onsite_2d_function") do
    H = chain()
    add_nh_onsite!(H, prof2; Lx = 1)
    (dense = dense_op(H.mpo, H.sites),)
end

case("add_nh_onsite_real_type") do
    H = chain()
    add_nh_onsite!(H, prof; type = Float64, tol = 1e-10, maxdim = 50)
    (dense = dense_op(H.mpo, H.sites),)
end

case("add_nh_onsite_errors") do
    H = chain()
    Hs = spin_chain()
    H0 = chain()
    (no_Lx         = caught(() -> add_nh_onsite!(H, (ix, iy) -> 1.0)),
     bad_signature = caught(() -> add_nh_onsite!(H, "loss")),
     with_spin     = caught(() -> add_nh_onsite!(Hs, 0.1)),
     unchanged     = dense_op(H.mpo, H.sites) == dense_op(H0.mpo, H0.sites))
end

case("loss_profile_mpo") do
    H = chain()
    Hs = spin_chain()
    f = prof
    (full          = diagrec(loss_profile_mpo(H, f), H.sites),
     position      = diagrec(loss_profile_mpo(H, f; space = :position), H.sites),
     number        = diagrec(loss_profile_mpo(H, 0.3), H.sites),
     twod          = diagrec(loss_profile_mpo(H, prof2; Lx = 2), H.sites),
     complex_type  = diagrec(loss_profile_mpo(H, prof; type = ComplexF64), H.sites),
     spin_full     = diagrec(loss_profile_mpo(Hs, f), Hs.sites),
     spin_position = caught(() -> loss_profile_mpo(Hs, f; space = :position)),
     bad_space     = caught(() -> loss_profile_mpo(H, f; space = :bogus)),
     no_Lx         = caught(() -> loss_profile_mpo(H, (ix, iy) -> 1.0)),
     bad_signature = caught(() -> loss_profile_mpo(H, "loss")))
end

case("nh_imag_onsite_mpo"; requires = :nh_imag_onsite_mpo) do
    H = chain()
    f = prof
    (default   = diagrec(TB.nh_imag_onsite_mpo(H, f), H.sites),
     prefactor = diagrec(TB.nh_imag_onsite_mpo(H, f; prefactor = -0.5, space = :position), H.sites))
end

case("add_loss") do
    H1 = chain(); ret = add_loss!(H1, prof)
    H2 = chain(); add_loss!(H2, prof; coefficient = 0.5im, space = :position)
    H3 = chain(); add_loss!(H3, prof2; Lx = 1, coefficient = 1)
    Hs = spin_chain(); add_loss!(Hs, prof)
    (default = dense_op(H1.mpo, H1.sites), returns_input = ret === H1, scale_after = H1.scale,
     position_gain = dense_op(H2.mpo, H2.sites),
     twod_real_coefficient = dense_op(H3.mpo, H3.sites),
     spin_full = fp(dense_op(Hs.mpo, Hs.sites)))
end

case("add_nh_imag_onsite"; requires = :add_nh_imag_onsite!) do
    H1 = chain(); TB.add_nh_imag_onsite!(H1, prof)
    H2 = chain(); TB.add_nh_imag_onsite!(H2, prof; prefactor = 2.0, space = :position)
    (default = dense_op(H1.mpo, H1.sites), prefactor = dense_op(H2.mpo, H2.sites))
end

case("add_nh_loss"; requires = :add_nh_loss!) do
    H1 = chain(); TB.add_nh_loss!(H1, prof)
    H2 = chain(); TB.add_nh_loss!(H2, 0.25; prefactor = 1im)
    (default = dense_op(H1.mpo, H1.sites), gain_number = dense_op(H2.mpo, H2.sites))
end

case("nh_nonreciprocal_hopping_mpo") do
    H = chain()
    Hs = spin_chain()
    (numbers   = dense_op(nh_nonreciprocal_hopping_mpo(H, 1.2, 0.8), H.sites),
     functions = dense_op(nh_nonreciprocal_hopping_mpo(H, cprof, 0.5), H.sites),
     functions_backward = dense_op(nh_nonreciprocal_hopping_mpo(H, 0.5, cprof), H.sites),
     nn2       = dense_op(nh_nonreciprocal_hopping_mpo(H, 0.7, 0.3im; nn = 2), H.sites),
     real_type = dense_op(nh_nonreciprocal_hopping_mpo(H, prof, 1.0; type = Float64), H.sites),
     nn0           = caught(() -> nh_nonreciprocal_hopping_mpo(H, 1.0, 1.0; nn = 0)),
     bad_amplitude = caught(() -> nh_nonreciprocal_hopping_mpo(H, "t", 1.0)),
     with_spin     = caught(() -> nh_nonreciprocal_hopping_mpo(Hs, 1.0, 1.0)))
end

case("add_nh_nonreciprocal_hopping") do
    H1 = chain(); ret = add_nh_nonreciprocal_hopping!(H1, 0.3, -0.3)
    H2 = chain(); add_nh_nonreciprocal_hopping!(H2, cprof, 0.2; nn = 3)
    (numbers = dense_op(H1.mpo, H1.sites), returns_input = ret === H1, scale = H1.scale,
     function_nn3 = dense_op(H2.mpo, H2.sites))
end

case("add_nh_skin_hopping") do
    He = chain(); add_nh_skin_hopping!(He, 0.5, 0.2)
    Hl = chain(); add_nh_skin_hopping!(Hl, 0.5, 0.2; convention = :linear, nn = 2)
    Hc = chain(); add_nh_skin_hopping!(Hc, 1, 0.1im)
    (exp = dense_op(He.mpo, He.sites), linear_nn2 = dense_op(Hl.mpo, Hl.sites),
     complex_g = dense_op(Hc.mpo, Hc.sites),
     bad_convention = caught(() -> add_nh_skin_hopping!(chain(), 0.5, 0.2; convention = :cosh)))
end

# ═════════════════════════════════════════════════════════════════════════════
# NH KPM: scale, block source, partial recursion, reconstruction
# ═════════════════════════════════════════════════════════════════════════════

case("nh_kpm_scale") do
    H = lossy_chain()
    zs = (0.3 - 0.2im, -0.5, 1im)
    (explicit         = nh_kpm_scale(H, zs; scale = 2.0),
     explicit_integer = nh_kpm_scale(H, zs; scale = 3),
     estimated        = nh_kpm_scale(H, zs; maxdim = MAXDIM),
     estimated_zero_pre = nh_kpm_scale(H, zs; scale = 0.0, padding = 1.2, block_placement = :pre),
     generator        = nh_kpm_scale(H, (complex(x, 0.5) for x in (-1.0, 1.0));
                                     convention = :H_minus_z),
     parent_scale_after = H.scale,
     negative    = caught(() -> nh_kpm_scale(H, zs; scale = -1.0)),
     low_padding = caught(() -> nh_kpm_scale(H, zs; padding = 0.9)))
end

case("nh_resolve_scale") do
    NHs = lossy_nh(scale = 2.5)
    NH0 = lossy_nh()
    (stored   = TB._nh_resolve_scale(NHs),
     explicit = TB._nh_resolve_scale(NHs; scale = 3.0),
     zero_ignores_stored = TB._nh_resolve_scale(NHs; scale = 0.0, maxdim = MAXDIM),
     estimated = TB._nh_resolve_scale(NH0; nh_scale_padding = 1.3, maxdim = MAXDIM),
     negative  = caught(() -> TB._nh_resolve_scale(NHs; scale = -2.0)))
end

case("nh_block_source") do
    NHp = lossy_nh()
    NHq = lossy_nh(block_placement = :pre)
    sp, sq = NHp.hermitized.sites, NHq.hermitized.sites
    (post_default   = dense_op(nh_block_source(NHp), sp),
     post_12        = dense_op(nh_block_source(NHp; row = 1, col = 2), sp),
     pre_default    = dense_op(nh_block_source(NHq), sq),
     pre_via_index_22 = dense_op(nh_block_source(NHq.hermitized, NHq.block_s; row = 2, col = 2), sq))
end

case("nh_kpm_partials_hermitized") do
    NH = lossy_nh()
    s = NH.hermitized.sites
    P = nh_kpm_partials(NH.hermitized, NHALF; source = nh_block_source(NH), scale = SCALE,
                        maxdim = MAXDIM)
    (count = length(P), dense = [dense_op(p, s) for p in P])
end

case("nh_kpm_partials_nh_pre_source_12") do
    NH = lossy_nh(block_placement = :pre)
    s = NH.hermitized.sites
    P = nh_kpm_partials(NH, NHALF; source_row = 1, source_col = 2, scale = SCALE, maxdim = MAXDIM)
    (count = length(P), fps = [fp(dense_op(p, s)) for p in P])
end

case("nh_kpm_partials_nh_stored_scale_and_custom_source") do
    NH = lossy_nh(scale = 3.0)
    s = NH.hermitized.sites
    P  = nh_kpm_partials(NH, 2; maxdim = MAXDIM)
    Pr = nh_kpm_partials(NH.hermitized, 2; source = nh_block_source(NH), scale = 3.0,
                         maxdim = MAXDIM)
    Pc = nh_kpm_partials(NH, 2; source = MPO(s, "Id"), scale = SCALE, maxdim = MAXDIM)
    (count = length(P), fps = [fp(dense_op(p, s)) for p in P],
     stored_vs_explicit = maximum(norm(dense_op(P[k], s) - dense_op(Pr[k], s)) for k in eachindex(P)),
     custom_source = [fp(dense_op(p, s)) for p in Pc])
end

case("nh_kpm_partials_errors") do
    NH = lossy_nh()   # hermitized.scale == 0
    S = nh_block_source(NH)
    (unset_scale = caught(() -> nh_kpm_partials(NH.hermitized, 2; source = S)),
     zero_scale  = caught(() -> nh_kpm_partials(NH.hermitized, 2; source = S, scale = 0.0)))
end

case("contract_nh_block") do
    NHp = lossy_nh()
    NHq = lossy_nh(block_placement = :pre)
    pp, pq = pos_sites(NHp), pos_sites(NHq)
    Wp, Wq = NHp.hermitized.mpo, NHq.hermitized.mpo
    (post = [dense_op(contract_nh_block(Wp, NHp.block_s; row = r, col = c), pp)
             for (r, c) in ((1, 1), (1, 2), (2, 1), (2, 2))],
     post_default = dense_op(contract_nh_block(Wp, NHp.block_s), pp),
     pre_default  = dense_op(contract_nh_block(Wq, NHq.block_s), pq),
     pre_12       = dense_op(contract_nh_block(Wq, NHq.block_s; row = 1, col = 2), pq),
     post_length  = length(contract_nh_block(Wp, NHp.block_s)))
end

case("contract_nh_block_errors") do
    s = siteinds("Qubit", 3)
    (single_site  = caught(() -> contract_nh_block(MPO(s[1:1], "Id"), s[1])),
     middle_block = caught(() -> contract_nh_block(MPO(s, "Id"), s[2])))
end

case("nh_preprocess_partials") do
    NH = lossy_nh()
    p = pos_sites(NH)
    P = nh_kpm_partials(NH, 2; scale = SCALE, maxdim = MAXDIM)
    (default  = [dense_vec(d, p) for d in nh_preprocess_partials(P, NH.block_s)],
     block_11 = [dense_vec(d, p) for d in nh_preprocess_partials(P, NH.block_s; row = 1, col = 1)])
end

case("nh_ones_mps") do
    s = siteinds("Qubit", 3)
    mixed = [s[1], Index(3, "Mixed")]
    (three = dense_vec(nh_ones_mps(s), s), one = dense_vec(nh_ones_mps(s[1:1]), s[1:1]),
     mixed = dense_vec(nh_ones_mps(mixed), mixed), empty_length = length(nh_ones_mps(Index[])))
end

case("nh_jackson_weights") do
    (w1 = nh_jackson_weights(1), w6 = nh_jackson_weights(6), w9 = nh_jackson_weights(9))
end

case("nh_reconstruct_spectral_mps") do
    NH = lossy_nh()
    p = pos_sites(NH)
    P = nh_kpm_partials(NH, NHALF; scale = SCALE, maxdim = MAXDIM)
    A, dos = nh_reconstruct_spectral_mps(P, NHALF, NH.block_s; maxdim = MAXDIM)
    A11, dos11 = nh_reconstruct_spectral_mps(P, NHALF, NH.block_s; row = 1, col = 1)
    A2, dos2 = nh_reconstruct_spectral_mps(P, 2, NH.block_s)   # first 4 of 6 partials
    (A = dense_vec(A, p), dos = dos, A11 = dense_vec(A11, p), dos11 = dos11,
     A_n2 = dense_vec(A2, p), dos_n2 = dos2,
     too_few = caught(() -> nh_reconstruct_spectral_mps(P[1:2], NHALF, NH.block_s)))
end

case("nh_reconstruct_spectral_mpo"; requires = :nh_reconstruct_spectral_mpo) do
    NH = lossy_nh()
    p, s = pos_sites(NH), NH.hermitized.sites
    P = nh_kpm_partials(NH, NHALF; scale = SCALE, maxdim = MAXDIM)
    ldos, dos, rot = TB.nh_reconstruct_spectral_mpo(P, NHALF, NH; maxdim = 40)
    ldos2, dos2, rot2 = TB.nh_reconstruct_spectral_mpo(P, NHALF, NH; maxdim = 40,
                                                       rotate_row = 2, rotate_col = 1,
                                                       diag_block = 2)
    (ldos = dense_vec(ldos, p), dos = dos, rotated = fp(dense_op(rot, s)),
     ldos_21_2 = dense_vec(ldos2, p), dos_21_2 = dos2, rotated_21_2 = fp(dense_op(rot2, s)),
     too_few = caught(() -> TB.nh_reconstruct_spectral_mpo(P[1:2], NHALF, NH)))
end

case("nh_spectral_function_explicit_scale") do
    NH = lossy_nh()
    A, dos, P = nh_spectral_function(NH, NHALF; scale = SCALE, maxdim = MAXDIM)
    (A = dense_vec(A, pos_sites(NH)), dos = dos, count = length(P),
     last_partial = fp(dense_op(P[end], NH.hermitized.sites)))
end

case("nh_spectral_function_pre_blocks_12") do
    NH = lossy_nh(block_placement = :pre)
    A, dos, P = nh_spectral_function(NH, NHALF; scale = SCALE, maxdim = MAXDIM,
                                     source_row = 1, source_col = 2,
                                     block_row = 1, block_col = 2)
    (A = dense_vec(A, pos_sites(NH)), dos = dos, count = length(P))
end

case("nh_spectral_function_estimated_scale") do
    NH = lossy_nh()
    A, dos, P = nh_spectral_function(NH, NHALF; nh_scale_padding = 1.1, maxdim = MAXDIM)
    (A = dense_vec(A, pos_sites(NH)), dos = dos, count = length(P))
end

case("nh_spectral_function_allsite_mpo"; requires = :nh_spectral_function_allsite_mpo) do
    NH = lossy_nh()
    ldos, dos, rot, P = TB.nh_spectral_function_allsite_mpo(NH, NHALF; scale = SCALE, maxdim = 40)
    (ldos = dense_vec(ldos, pos_sites(NH)), dos = dos,
     rotated = fp(dense_op(rot, NH.hermitized.sites)), count = length(P))
end

# ═════════════════════════════════════════════════════════════════════════════
# NH KPM: online probes and the complex-energy grid
# ═════════════════════════════════════════════════════════════════════════════

case("nh_kpm_probe_mps") do
    NHp = lossy_nh()
    NHq = lossy_nh(block_placement = :pre)
    sp, sq = NHp.hermitized.sites, NHq.hermitized.sites
    (post_b1_r5 = dense_vec(TB._nh_kpm_probe_mps(sp, NHp.block_s, 1, 5), sp),
     post_b2_r0 = dense_vec(TB._nh_kpm_probe_mps(sp, NHp.block_s, 2, 0), sp),
     pre_b1_r5  = dense_vec(TB._nh_kpm_probe_mps(sq, NHq.block_s, 1, 5), sq),
     pre_b2_r6  = dense_vec(TB._nh_kpm_probe_mps(sq, NHq.block_s, 2, 6), sq))
end

case("nh_kpm_mps_ldos") do
    NHp = lossy_nh()
    NHq = lossy_nh(block_placement = :pre, z = -0.6 + 0.1im)
    (post_r0 = TB._nh_kpm_mps_ldos(NHp, NHALF, 0; scale = SCALE, maxdim = MAXDIM),
     post_r5 = TB._nh_kpm_mps_ldos(NHp, NHALF, 5; scale = SCALE, maxdim = MAXDIM),
     pre_r5  = TB._nh_kpm_mps_ldos(NHq, NHALF, 5; scale = SCALE, maxdim = MAXDIM),
     default_kwargs = TB._nh_kpm_mps_ldos(NHp, 2, 3; scale = SCALE))
end

case("nh_scalar_online") do
    NH = lossy_nh()
    (default   = TB._nh_scalar_online(NH, NHALF; scale = SCALE, maxdim = MAXDIM),
     blocks_12 = TB._nh_scalar_online(NH, NHALF; scale = SCALE, maxdim = MAXDIM,
                                      source_row = 1, source_col = 2,
                                      block_row = 1, block_col = 2),
     pre       = TB._nh_scalar_online(lossy_nh(block_placement = :pre), NHALF;
                                      scale = SCALE, maxdim = MAXDIM),
     stored_scale = TB._nh_scalar_online(lossy_nh(scale = SCALE), NHALF; maxdim = MAXDIM))
end

case("nh_diag_online") do
    NH = lossy_nh()
    NHq = lossy_nh(block_placement = :pre)
    A, dos = TB._nh_diag_online(NH, NHALF; scale = SCALE, maxdim = MAXDIM)
    Aq, dosq = TB._nh_diag_online(NHq, NHALF; scale = SCALE, maxdim = MAXDIM,
                                  block_row = 1, block_col = 1)
    (A = dense_vec(A, pos_sites(NH)), dos = dos,
     A_pre_11 = dense_vec(Aq, pos_sites(NHq)), dos_pre_11 = dosq)
end

case("nh_random_probes") do
    NH = lossy_nh()
    s = NH.hermitized.sites
    ket, bra = TB._nh_random_probes(s, NH.block_s, 1, 2)
    (ket = dense_vec(ket, s), bra = dense_vec(bra, s))
end

case("nh_stochastic_online") do
    (post_two = TB._nh_stochastic_online(lossy_nh(), NHALF; scale = SCALE, n_random = 2,
                                         maxdim = MAXDIM),
     pre_one  = TB._nh_stochastic_online(lossy_nh(block_placement = :pre), 2; scale = SCALE,
                                         n_random = 1, maxdim = MAXDIM))
end

case("nh_spectrum_grid_scalar_default_mode") do
    H = lossy_chain()
    out = nh_spectrum_grid(H, (-1.0, 1.0), 2, (-0.5, 0.5), 2, NHALF; scale = SCALE, maxdim = MAXDIM)
    merge(grid_record(out), (parent_scale_after = H.scale,))
end

case("nh_spectrum_grid_diag") do
    out = nh_spectrum_grid(lossy_chain(), (-1.0, 1.0), 2, (-0.5, 0.5), 2, NHALF;
                           scale = SCALE, maxdim = MAXDIM, mode = :diag)
    merge(grid_record(out), (Z_spatial = out[4],))
end

case("nh_spectrum_grid_mps") do
    out = nh_spectrum_grid(lossy_chain(), (-1.0, 1.0), 3, (0.3, 0.3), 1, NHALF;
                           scale = SCALE, maxdim = MAXDIM, mode = :mps, probe_site = 5)
    grid_record(out)
end

case("nh_spectrum_grid_stochastic") do
    out = nh_spectrum_grid(lossy_chain(), (-1.0, 1.0), 2, (-0.2, -0.2), 1, NHALF;
                           scale = SCALE, maxdim = MAXDIM, mode = :stochastic, n_random = 2)
    grid_record(out)
end

case("nh_spectrum_grid_estimated_pre_H_minus_z") do
    out = nh_spectrum_grid(lossy_chain(), (0.5, 0.5), 1, (-0.4, 0.4), 2, NHALF;
                           nh_scale_padding = 1.2, block_placement = :pre,
                           convention = :H_minus_z, maxdim = MAXDIM)
    grid_record(out)
end

case("nh_spectrum_grid_bad_mode") do
    (bad_mode = caught(() -> nh_spectrum_grid(chain(), (0.0, 1.0), 2, (0.0, 1.0), 2, 2;
                                              scale = SCALE, mode = :exact)),)
end

# The chain cases above never go through QTCI for the hopping; this one starts
# from the (QTCI-built) AAH preset and lets the scale come from DMRG.
case("aah_lossy_spectral_function") do
    H = get_Hamiltonian("aah", (V = 1.0, phi = 0.3, t = 1.0); L = 3)
    aah = dense_op(H.mpo, H.sites)
    add_loss!(H, prof)
    NH = hermitize(H; z = 0.1 + 0.2im)
    A, dos, P = nh_spectral_function(NH, NHALF; maxdim = MAXDIM)
    (aah = aah, lossy = dense_op(H.mpo, H.sites), A = dense_vec(A, H.sites), dos = dos,
     count = length(P))
end

# ═════════════════════════════════════════════════════════════════════════════
# Truncation: maxdim / cutoff forwarding
# ═════════════════════════════════════════════════════════════════════════════
#
# None of the cases above truncates: on the 8-site chain every bond fits in
# MAXDIM = 20 (and in the default maxdim = 100 / 200), so a caller that stopped
# forwarding `maxdim` or `cutoff` would go unnoticed. The cases below use a
# 16-site chain with complex non-reciprocal hopping and loss, whose hermitized
# MPO has bonds [4, 6, 6, 2]. Each function is run once with a small maxdim
# (cutoff left at its default) and once with TRUNC_CUTOFF = 1e-3 (maxdim left
# at its default), so the two keywords are pinned separately:
#   * TRUNC_MAXDIM = 4 truncates the hermitized MPO and the MPO Chebyshev steps;
#   * TRUNC_MAXDIM_MPS = 3 serves the MPS-only recursions (_nh_kpm_mps_ldos,
#     _nh_stochastic_online, modes :mps and :stochastic), whose bonds never
#     exceed 4 on 16 sites;
#   * the spectral MPS A(r, z) has bonds of at most 4 on 16 sites as well, so
#     the maxdim that nh_spectral_function forwards to the reconstruction, and
#     the one _nh_diag_online uses to accumulate A, get two 64-site cases
#     (bonds up to 8) with TRUNC_MAXDIM_WIDE = 6.
# MPO outputs are fingerprinted (32×32 dense matrices would bloat the data
# file) together with their link dimensions, which show the truncation.
#
# The inputs and maxdims were chosen for conditioning. An odd maxdim on the MPO
# recursion (3 on 16 sites; 5 and 7 on 64 sites), or 2 on the plain lossy
# chain, made the results sensitive to rounding: 1e-13 relative noise on the
# parent MPO moved truncated KPM outputs by far more than the comparison
# tolerance (up to tens of percent), as when the cut splits a degenerate pair
# of singular values. With the values below the same noise moves no recorded
# value by more than 5% of the tolerance.
#
# Checked by scratch mutation of a copy of NH_tk.jl: dropping every
# `maxdim=maxdim` outside the model builders fails all of these cases and none
# of the cases above. Dropping one `maxdim=maxdim` or `cutoff=cutoff` at a
# time in hermitize, the NH KPM routines or nh_spectrum_grid (the functions
# slated for deletion aside) fails at least one of these cases, except where
# the keyword cannot bind here: `apply(S, ·)` (S has bond dimension 1), maxdim
# on A applied to a single-site probe (at most three basis states), cutoff on
# the sum that hermitized_hamiltonian truncates again right after, and cutoff
# on the final `- p_{k-2}` sum of _nh_kpm_mps_ldos.

const TRUNC_L      = 4       # 16-site chain
const TRUNC_Z      = 0.3 - 0.2im
const TRUNC_MAXDIM = 4
const TRUNC_MAXDIM_MPS = 3
const TRUNC_CUTOFF = 1e-3
const TRUNC_L_WIDE = 6       # 64-site chain
const SCALE_WIDE   = 40.0    # above its spectral radius (about 21.7)
const TRUNC_MAXDIM_WIDE = 6

function trunc_chain(L::Int = TRUNC_L)
    H = chain(L)
    add_nh_nonreciprocal_hopping!(H, cprof, 0.2)
    add_loss!(H, prof)
    return H
end

trunc_nh(; L::Int = TRUNC_L, z = TRUNC_Z, kwargs...) = hermitize(trunc_chain(L); z = z, kwargs...)

mpo_record(W::MPO, basis) = (fp = fp(dense_op(W, basis)), linkdims = linkdims(W))
partials_record(P, basis) = (fps = [fp(dense_op(p, basis)) for p in P],
                             maxlinkdims = [maxlinkdim(p) for p in P])

case("hermitized_hamiltonian_truncated") do
    H = trunc_chain()
    Hf = hermitized_hamiltonian(H; z = TRUNC_Z)
    Hm = hermitized_hamiltonian(H; z = TRUNC_Z, maxdim = TRUNC_MAXDIM)
    Hc = hermitized_hamiltonian(H; z = TRUNC_Z, cutoff = TRUNC_CUTOFF)
    Hp = hermitized_hamiltonian(H; z = TRUNC_Z, maxdim = TRUNC_MAXDIM, block_placement = :pre)
    Hz = hermitized_hamiltonian(H; z = TRUNC_Z, cutoff = TRUNC_CUTOFF, convention = :H_minus_z)
    (full = mpo_record(Hf.mpo, Hf.sites), maxdim = mpo_record(Hm.mpo, Hm.sites),
     cutoff = mpo_record(Hc.mpo, Hc.sites), pre_maxdim = mpo_record(Hp.mpo, Hp.sites),
     H_minus_z_cutoff = mpo_record(Hz.mpo, Hz.sites))
end

case("hermitize_truncated") do
    H = trunc_chain()
    NH = hermitize(H; z = TRUNC_Z)
    rec(N) = mpo_record(N.hermitized.mpo, N.hermitized.sites)
    (maxdim         = rec(hermitize(H; z = TRUNC_Z, maxdim = TRUNC_MAXDIM)),
     cutoff         = rec(hermitize(H; z = TRUNC_Z, cutoff = TRUNC_CUTOFF)),
     rebuild_maxdim = rec(hermitize(NH; maxdim = TRUNC_MAXDIM)),
     rebuild_cutoff = rec(hermitize(NH; cutoff = TRUNC_CUTOFF)))
end

case("nh_kpm_scale_truncated") do
    H = trunc_chain()
    zs = (TRUNC_Z, -0.5)
    (full   = nh_kpm_scale(H, zs),
     maxdim = nh_kpm_scale(H, zs; maxdim = TRUNC_MAXDIM),
     cutoff = nh_kpm_scale(H, zs; cutoff = TRUNC_CUTOFF))
end

case("nh_resolve_scale_truncated") do
    NH = trunc_nh()
    (maxdim = TB._nh_resolve_scale(NH; maxdim = TRUNC_MAXDIM),
     cutoff = TB._nh_resolve_scale(NH; cutoff = TRUNC_CUTOFF))
end

case("nh_kpm_partials_truncated") do
    NH = trunc_nh()
    Hh, S = NH.hermitized, nh_block_source(NH)
    s = Hh.sites
    (hermitized_maxdim = partials_record(nh_kpm_partials(Hh, NHALF; source = S, scale = SCALE,
                                                         maxdim = TRUNC_MAXDIM), s),
     hermitized_cutoff = partials_record(nh_kpm_partials(Hh, NHALF; source = S, scale = SCALE,
                                                         cutoff = TRUNC_CUTOFF), s),
     nh_maxdim = partials_record(nh_kpm_partials(NH, NHALF; scale = SCALE, maxdim = TRUNC_MAXDIM), s),
     nh_cutoff = partials_record(nh_kpm_partials(NH, NHALF; scale = SCALE, cutoff = TRUNC_CUTOFF), s),
     nh_estimated_maxdim = partials_record(nh_kpm_partials(NH, 2; maxdim = TRUNC_MAXDIM), s),
     nh_estimated_cutoff = partials_record(nh_kpm_partials(NH, 2; cutoff = TRUNC_CUTOFF), s))
end

case("nh_reconstruct_spectral_mps_truncated") do
    NH = trunc_nh()
    p = pos_sites(NH)
    P = nh_kpm_partials(NH, NHALF; scale = SCALE)
    Af, dosf = nh_reconstruct_spectral_mps(P, NHALF, NH.block_s)
    Am, dosm = nh_reconstruct_spectral_mps(P, NHALF, NH.block_s; maxdim = 2)
    (full = (A = dense_vec(Af, p), linkdims = linkdims(Af), dos = dosf),
     maxdim_2 = (A = dense_vec(Am, p), linkdims = linkdims(Am), dos = dosm))
end

case("nh_spectral_function_truncated") do
    NH = trunc_nh()
    p, s = pos_sites(NH), NH.hermitized.sites
    rec((A, dos, P)) = (A = dense_vec(A, p), linkdims = linkdims(A), dos = dos,
                        count = length(P), last_partial = fp(dense_op(P[end], s)))
    (maxdim = rec(nh_spectral_function(NH, NHALF; scale = SCALE, maxdim = TRUNC_MAXDIM)),
     cutoff = rec(nh_spectral_function(NH, NHALF; scale = SCALE, cutoff = TRUNC_CUTOFF)))
end

case("nh_kpm_mps_ldos_truncated") do
    NH = trunc_nh()
    (maxdim = TB._nh_kpm_mps_ldos(NH, NHALF, 5; scale = SCALE, maxdim = TRUNC_MAXDIM_MPS),
     cutoff = TB._nh_kpm_mps_ldos(NH, NHALF, 5; scale = SCALE, cutoff = TRUNC_CUTOFF))
end

case("nh_scalar_online_truncated") do
    NH = trunc_nh()
    (maxdim = TB._nh_scalar_online(NH, NHALF; scale = SCALE, maxdim = TRUNC_MAXDIM),
     cutoff = TB._nh_scalar_online(NH, NHALF; scale = SCALE, cutoff = TRUNC_CUTOFF),
     estimated_maxdim = TB._nh_scalar_online(NH, 2; maxdim = TRUNC_MAXDIM),
     estimated_cutoff = TB._nh_scalar_online(NH, 2; cutoff = TRUNC_CUTOFF))
end

case("nh_diag_online_truncated") do
    NH = trunc_nh()
    p = pos_sites(NH)
    rec((A, dos)) = (A = dense_vec(A, p), linkdims = linkdims(A), dos = dos)
    (maxdim = rec(TB._nh_diag_online(NH, NHALF; scale = SCALE, maxdim = TRUNC_MAXDIM)),
     cutoff = rec(TB._nh_diag_online(NH, NHALF; scale = SCALE, cutoff = TRUNC_CUTOFF)),
     estimated_maxdim = rec(TB._nh_diag_online(NH, 2; maxdim = TRUNC_MAXDIM)),
     estimated_cutoff = rec(TB._nh_diag_online(NH, 2; cutoff = TRUNC_CUTOFF)))
end

case("nh_stochastic_online_truncated") do
    NH = trunc_nh()
    (maxdim = TB._nh_stochastic_online(NH, NHALF; scale = SCALE, n_random = 1,
                                       maxdim = TRUNC_MAXDIM_MPS),
     cutoff = TB._nh_stochastic_online(NH, NHALF; scale = SCALE, n_random = 1,
                                       cutoff = TRUNC_CUTOFF),
     estimated_maxdim = TB._nh_stochastic_online(NH, 2; n_random = 1, maxdim = TRUNC_MAXDIM_MPS),
     estimated_cutoff = TB._nh_stochastic_online(NH, 2; n_random = 1, cutoff = TRUNC_CUTOFF))
end

trunc_grid(; kwargs...) =
    nh_spectrum_grid(trunc_chain(), (-0.5, 0.5), 2, (-0.2, -0.2), 1, NHALF; kwargs...)

case("nh_spectrum_grid_truncated_scalar") do
    (maxdim = grid_record(trunc_grid(scale = SCALE, maxdim = TRUNC_MAXDIM)),
     cutoff = grid_record(trunc_grid(scale = SCALE, cutoff = TRUNC_CUTOFF)),
     estimated_maxdim = grid_record(trunc_grid(maxdim = TRUNC_MAXDIM)),
     estimated_cutoff = grid_record(trunc_grid(cutoff = TRUNC_CUTOFF)))
end

case("nh_spectrum_grid_truncated_diag") do
    rec(out) = merge(grid_record(out), (Z_spatial = out[4],))
    (maxdim = rec(trunc_grid(scale = SCALE, maxdim = TRUNC_MAXDIM, mode = :diag)),
     cutoff = rec(trunc_grid(scale = SCALE, cutoff = TRUNC_CUTOFF, mode = :diag)))
end

case("nh_spectrum_grid_truncated_mps") do
    (maxdim = grid_record(trunc_grid(scale = SCALE, maxdim = TRUNC_MAXDIM_MPS, mode = :mps,
                                     probe_site = 5)),
     cutoff = grid_record(trunc_grid(scale = SCALE, cutoff = TRUNC_CUTOFF, mode = :mps,
                                     probe_site = 5)))
end

case("nh_spectrum_grid_truncated_stochastic") do
    (maxdim = grid_record(trunc_grid(scale = SCALE, maxdim = TRUNC_MAXDIM_MPS, mode = :stochastic,
                                     n_random = 1)),
     cutoff = grid_record(trunc_grid(scale = SCALE, cutoff = TRUNC_CUTOFF, mode = :stochastic,
                                     n_random = 1)))
end

# 64 sites: see TRUNC_MAXDIM_WIDE in the section header.
case("nh_spectral_function_truncated_L6") do
    NH = trunc_nh(L = TRUNC_L_WIDE)
    A, dos, P = nh_spectral_function(NH, NHALF; scale = SCALE_WIDE, maxdim = TRUNC_MAXDIM_WIDE)
    (A = dense_vec(A, pos_sites(NH)), linkdims = linkdims(A), dos = dos, count = length(P))
end

case("nh_diag_online_truncated_L6") do
    NH = trunc_nh(L = TRUNC_L_WIDE)
    A, dos = TB._nh_diag_online(NH, NHALF; scale = SCALE_WIDE, maxdim = TRUNC_MAXDIM_WIDE)
    (A = dense_vec(A, pos_sites(NH)), linkdims = linkdims(A), dos = dos)
end

# ── Comparison ───────────────────────────────────────────────────────────────

const FloatLike = Union{AbstractFloat, Complex{<:AbstractFloat}}

"""
    mismatch(actual, expected) -> Union{Nothing,String}

`nothing` when `actual` matches the golden `expected` (see the rules in the
header), otherwise a short description of the first difference.
"""
function mismatch(@nospecialize(a), @nospecialize(e))
    if e isa NamedTuple
        a isa NamedTuple || return "expected a NamedTuple, got $(typeof(a))"
        keys(a) == keys(e) || return "fields $(keys(a)) != expected $(keys(e))"
        for k in keys(e)
            m = mismatch(a[k], e[k])
            m === nothing || return "$k: $m"
        end
        return nothing
    elseif e isa Tuple
        (a isa Tuple && length(a) == length(e)) || return "got $(repr(a)), expected $(repr(e))"
        for i in eachindex(e)
            m = mismatch(a[i], e[i])
            m === nothing || return "[$i]: $m"
        end
        return nothing
    elseif e isa AbstractArray
        a isa AbstractArray || return "expected an array, got $(typeof(a))"
        size(a) == size(e) || return "size $(size(a)) != expected $(size(e))"
        eltype(a) == eltype(e) || return "eltype $(eltype(a)) != expected $(eltype(e))"
        if eltype(e) <: FloatLike
            isapprox(a, e; rtol = RTOL, atol = ATOL, nans = true) && return nothing
            i = argmax(abs.(a .- e))
            return "values differ: norm(diff) = $(norm(a - e)), norm(expected) = $(norm(e)), " *
                   "largest at $(Tuple(CartesianIndices(e)[i])): got $(repr(a[i])), expected $(repr(e[i]))"
        end
        for i in eachindex(e)
            m = mismatch(a[i], e[i])
            m === nothing || return "[$i]: $m"
        end
        return nothing
    elseif e isa FloatLike
        typeof(a) == typeof(e) || return "type $(typeof(a)) != expected $(typeof(e))"
        isapprox(a, e; rtol = RTOL, atol = ATOL, nans = true) && return nothing
        return "got $(repr(a)), expected $(repr(e))"
    else
        (typeof(a) == typeof(e) && isequal(a, e)) && return nothing
        return "got $(repr(a))::$(typeof(a)), expected $(repr(e))::$(typeof(e))"
    end
end

"""
    load_golden(path=DATA_FILE) -> Vector{Pair{String,NamedTuple}}

Evaluate the golden data file inside this module.
"""
load_golden(path::AbstractString = DATA_FILE) = Base.include(@__MODULE__, path)

# Minimum number of compared cases per function. A case counts for a function
# when its name is the function's name (without a leading `_` or trailing `!`)
# or starts with that name and `_`; cases marked `requires` never count, so
# their deletion in Tier 1 leaves the floors alone. Without the floors an empty
# case list (or data regenerated from one) would pass every check vacuously.
# The floors are the counts when the table was written: raise one when adding
# cases for its function; lower one only when retiring a case on purpose.
const MIN_CASES = [
    "nh_block_index"               => 1,
    "hermitized_hamiltonian"       => 5,
    "hermitize"                    => 4,
    "add_nh_onsite"                => 5,
    "loss_profile_mpo"             => 1,
    "add_loss"                     => 1,
    "nh_nonreciprocal_hopping_mpo" => 1,
    "add_nh_nonreciprocal_hopping" => 1,
    "add_nh_skin_hopping"          => 1,
    "nh_kpm_scale"                 => 2,
    "nh_resolve_scale"             => 2,
    "nh_block_source"              => 1,
    "nh_kpm_partials"              => 5,
    "contract_nh_block"            => 2,
    "nh_preprocess_partials"       => 1,
    "nh_ones_mps"                  => 1,
    "nh_jackson_weights"           => 1,
    "nh_reconstruct_spectral_mps"  => 2,
    "nh_spectral_function"         => 5,
    "nh_kpm_probe_mps"             => 1,
    "nh_kpm_mps_ldos"              => 2,
    "nh_scalar_online"             => 2,
    "nh_diag_online"               => 3,
    "nh_random_probes"             => 1,
    "nh_stochastic_online"         => 2,
    "nh_spectrum_grid"             => 10,
]

counts_for(fname, names) = count(n -> n == fname || startswith(n, fname * "_"), names)

function run_tests(golden = load_golden())
    expected = Dict(golden)
    @testset "NH outputs are pinned" begin
        @test allunique(first.(golden))
        @test allunique(c.name for c in CASES)
        # A skipped case may keep its record (data generated while its function
        # existed) or not (data regenerated since); every other case needs one.
        missing_data  = [c.name for c in CASES if is_available(c) && !haskey(expected, c.name)]
        missing_cases = setdiff(first.(golden), [c.name for c in CASES])
        isempty(missing_data) || @error "NH cases without golden data (regenerate?)" missing_data
        isempty(missing_cases) || @error "Golden NH cases no longer in the case list" missing_cases
        @test isempty(missing_data)
        @test isempty(missing_cases)
        compared = [c.name for c in CASES if c.requires === nothing && haskey(expected, c.name)]
        for (fname, nmin) in MIN_CASES
            n = counts_for(fname, compared)
            n >= nmin || @error "Too few NH golden cases for $fname" found = n minimum = nmin
            @test n >= nmin
        end
        for c in CASES
            if !is_available(c)
                @info "NH golden case skipped: TensorBinding.$(c.requires) no longer exists" case = c.name
                continue
            end
            haskey(expected, c.name) || continue
            actual, err, bt = try
                (run_case(c), nothing, nothing)
            catch e
                (nothing, e, catch_backtrace())
            end
            if err !== nothing
                @error "NH golden case now throws" case = c.name exception = (err, bt)
                @test err === nothing
                continue
            end
            m = mismatch(actual, expected[c.name])
            m === nothing || @error "NH output changed" case = c.name detail = m
            @test m === nothing
        end
    end
end

end # module NHGolden

if !isdefined(@__MODULE__, :NH_GOLDEN_GENERATOR)
    NHGolden.run_tests()
end
