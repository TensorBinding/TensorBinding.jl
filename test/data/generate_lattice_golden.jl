# Generator for test/data/lattice_golden.jl, the pinned ("golden") outputs of
# the lattice builders, checked by test/golden_lattice.jl.
#
# Reference: first generated on top of commit 1a5548b (branch Anouar, "Run the
# third-round audit regression tests from runtests.jl"), before the Tier 1
# code reorganisation (docs/dev/REORGANISATION_TODO.md). The data pins what the
# lattice code does at that commit, remaining bugs included.
#
# Line 4 of the data file records the git tree hash of the working-tree src/,
# i.e. the src/ that the regenerating commit will contain. It does not move
# when only tests or docs change.
#
# Regenerate ONLY when a behaviour change is intentional. Do it in the same
# commit as the change, review the diff of lattice_golden.jl case by case, and
# record the change in the changelog. A refactor that is meant to leave outputs
# alone must pass test/golden_lattice.jl against the existing data unchanged.
#
# How to run, from the repository root with the package environment:
#
#     JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. test/data/generate_lattice_golden.jl
#
# All cases run in one process. Each case reseeds the RNGs with its own seed
# (derived from its name by `name_seed`, stored in the data file), so the output
# depends neither on the order in which the test replays the cases nor on which
# other cases exist. Values are written with
# `repr`; the script reloads the file it wrote and checks that every entry
# round-trips exactly.
#
# Sizes are the smallest that exercise each branch: 1D chains of 2^3 sites, 2D
# grids of 2^2 x 2^1 cells (asymmetric, so an Lx/Ly swap is visible), 2^2 x 2^2
# where a builder needs it. Out of scope on purpose (other work is changing
# them): get_Hamiltonian("haldane"), "chernhex"/H2DChernhex,
# haldane_hoppingf/chirality, and the default scale of "chern8" and
# "qc2dsquare" (their cases pass an explicit `scale=`).
#
# Error messages are pinned by their first 60 characters. One of them,
# build_hamiltonian's "Model 'uniform' is 1D; call build_hamiltonian(model, L; -",
# ends in the mojibake that the Tier 1 item "re-save 2Dlattice_tk.jl as UTF-8"
# restores; when that lands, regenerate and check that the data diff touches only
# that message_prefix.

using TensorBinding
const TB = TensorBinding

const LATTICE_GOLDEN_GENERATOR = true
include(joinpath(@__DIR__, "..", "golden_lattice.jl"))   # loads LatticeGoldenRunner only
const R = LatticeGoldenRunner

const OUTFILE = joinpath(@__DIR__, "lattice_golden.jl")

const CASES = Tuple{String,Symbol,NamedTuple}[]
add!(name, builder, spec) = push!(CASES, (name, builder, spec))
none = NamedTuple()

# Exceptions that indicate a broken generator rather than pinned behaviour.
const GENERATOR_BUGS = (UndefVarError, MethodError, KeyError,
                        (isdefined(Base, :FieldError) ? (getfield(Base, :FieldError),) : ())...)

# ═════════════════════════════════════════════════════════════════════════════
# lattice/2Dlattice_tk.jl
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. binary shift MPOs ──────────────────────────────────────────────────────
add!("generate_kin_u_L3", :shift_primitive, (fn = :generate_kin_u, L = 3, num_site = 8))
add!("generate_kin_d_L3", :shift_primitive, (fn = :generate_kin_d, L = 3, num_site = 8))
add!("generate_kin_u_L3_num_site_mismatch", :shift_primitive, (fn = :generate_kin_u, L = 3, num_site = 16))

# ── 5. row / column / checkerboard masks ──────────────────────────────────────
for (Lx, Ly) in ((2, 1), (1, 2)), which in (:xplus, :xplain)
    add!("row_break_Lx$(Lx)_Ly$(Ly)_$which", :mask, (fn = :_row_break_mpo, Lx, Ly, kw = (; which)))
end
add!("row_break_Lx2_Ly1_bad_which", :mask, (fn = :_row_break_mpo, Lx = 2, Ly = 1, kw = (which = :middle,)))
for keep in (:even, :odd)
    add!("row_select_Lx2_Ly1_$keep", :mask, (fn = :_row_select_mpo, Lx = 2, Ly = 1, kw = (; keep)))
    add!("col_select_Lx2_Ly1_$keep", :mask, (fn = :_col_select_mpo, Lx = 2, Ly = 1, kw = (; keep)))
end
add!("row_select_Lx2_Ly1_bad_keep", :mask, (fn = :_row_select_mpo, Lx = 2, Ly = 1, kw = (keep = :none,)))
add!("col_select_Lx2_Ly1_bad_keep", :mask, (fn = :_col_select_mpo, Lx = 2, Ly = 1, kw = (keep = :none,)))
add!("row_checker_Lx2_Ly1", :mask, (fn = :_row_checker_mpo, Lx = 2, Ly = 1, kw = none))
add!("row_checker_Lx1_Ly2", :mask, (fn = :_row_checker_mpo, Lx = 1, Ly = 2, kw = none))

# ── 2-4. legacy square / triangle / honeycomb hopping builders ─────────────────
add!("intrachain_hopping_Lx2_Ly1_default", :legacy_hopping, (fn = :intrachain_hopping, Lx = 2, Ly = 1, kw = none))
add!("intrachain_hopping_Lx2_Ly1_testhop_tcomplex", :legacy_hopping,
     (fn = :intrachain_hopping, Lx = 2, Ly = 1, hopping = :test, kw = (t = 0.5 + 0.2im,)))
add!("interchain_hopping_square_Lx2_Ly1_default", :legacy_hopping,
     (fn = :interchain_hopping_square, Lx = 2, Ly = 1, kw = none))
add!("interchain_hopping_square_Lx2_Ly1_testhop_tcomplex", :legacy_hopping,
     (fn = :interchain_hopping_square, Lx = 2, Ly = 1, hopping = :test, kw = (t = 0.7im,)))
for fn in (:interchain_hopping_square_2nd_plus, :interchain_hopping_square_2nd_minus)
    add!("$(fn)_Lx2_Ly1_default", :legacy_hopping, (; fn, Lx = 2, Ly = 1, kw = none))
    add!("$(fn)_Lx2_Ly1_testhop_t2complex", :legacy_hopping,
         (; fn, Lx = 2, Ly = 1, hopping = :test, kw = (t2 = 0.3 - 0.1im,)))
end
add!("interchain_hopping_triangle_Lx2_Ly1", :legacy_hopping, (fn = :interchain_hopping_triangle, Lx = 2, Ly = 1, kw = none))
add!("interchain_hopping_honeycomb_Lx2_Ly1", :legacy_hopping, (fn = :interchain_hopping_honeycomb, Lx = 2, Ly = 1, kw = none))
for fn in (:skeleton, :odd_template, :even_template, :odd_skeleton, :even_skeleton)
    add!("$(fn)_Lx2_Ly1", :legacy_mask, (; fn, Lx = 2, Ly = 1, kw = none))
end

# ── 6. NNN 2D kinetic builders (complex, position-dependent hopping profile) ──
add!("kineticintra2DNNN_Lx2_Ly1_nn1", :kinetic2d, (fn = :kineticintra2DNNN, Lx = 2, Ly = 1, hopping = :test, nn = 1, kw = none))
add!("kineticintra2DNNN_Lx2_Ly1_nn2", :kinetic2d, (fn = :kineticintra2DNNN, Lx = 2, Ly = 1, hopping = :test, nn = 2, kw = none))
add!("kineticintra2DNNN_Lx2_Ly1_nn1_apply_kwargs", :kinetic2d,
     (fn = :kineticintra2DNNN, Lx = 2, Ly = 1, hopping = :test, nn = 1,
      kw = (apply_kwargs = (cutoff = 1e-10, maxdim = 20),)))
add!("kineticintra2DNNN_Lx2_Ly1_nn0", :kinetic2d, (fn = :kineticintra2DNNN, Lx = 2, Ly = 1, hopping = :test, nn = 0, kw = none))
add!("kineticinterNNNSWNE_Lx2_Ly1_nn5", :kinetic2d, (fn = :kineticinterNNNSWNE, Lx = 2, Ly = 1, hopping = :test, nn = 5, kw = none))
add!("kineticinterNNNSENW_Lx2_Ly1_nn3", :kinetic2d, (fn = :kineticinterNNNSENW, Lx = 2, Ly = 1, hopping = :test, nn = 3, kw = none))
add!("kineticinterNNNtriSWNE_Lx2_Ly1_nn5", :kinetic2d, (fn = :kineticinterNNNtriSWNE, Lx = 2, Ly = 1, hopping = :test, nn = 5, kw = none))
add!("kineticinterNNNtriSENW_Lx2_Ly1_nn3", :kinetic2d, (fn = :kineticinterNNNtriSENW, Lx = 2, Ly = 1, hopping = :test, nn = 3, kw = none))
add!("kineticinterNNNtri_bravais_diag_Lx2_Ly1", :kinetic2d,
     (fn = :kineticinterNNNtri_bravais_diag, Lx = 2, Ly = 1, hopping = :test, nn = nothing, kw = none))
add!("kineticintra2DNNhex_Lx2_Ly1_nn1", :kinetic2d, (fn = :kineticintra2DNNhex, Lx = 2, Ly = 1, hopping = :test, nn = 1, kw = none))
add!("kineticintra2DNNhex_Lx2_Ly1_nn1_realhop", :kinetic2d, (fn = :kineticintra2DNNhex, Lx = 2, Ly = 1, hopping = :testre, nn = 1, kw = none))

# ── 7. preset model Hamiltonians (QTCI hopping profiles) ─────────────────────
add!("HUniform_L3_default", :preset1d, (fn = :HUniform, args = (3, 1.0), kw = none))
add!("HUniform_L3_v0.3_nn2", :preset1d, (fn = :HUniform, args = (3, 0.8), kw = (v = 0.3, nn = 2)))
add!("HUniform_L3_v0", :preset1d, (fn = :HUniform, args = (3, 1.0), kw = (v = 0.0,)))
add!("HSSH_L3_default", :preset1d, (fn = :HSSH, args = (3, 1.0, 0.3), kw = none))
add!("HSSH_L3_nn2", :preset1d, (fn = :HSSH, args = (3, 1.0, 0.3), kw = (nn = 2,)))
add!("HAAH_L3_default", :preset1d, (fn = :HAAH, args = (3, 0.5, 0.3, 1.0), kw = none))
add!("HAAH_L3_b0.4", :preset1d, (fn = :HAAH, args = (3, 0.5, 0.3, 1.0), kw = (b = 0.4,)))
for fn in (:HUniform2Dsquare, :HUniform2Dhex, :HUniform2Dtri, :HUniform2Dtri_bravais)
    add!("$(fn)_Lx2_Ly1", :preset2d, (; fn, args = (2, 1, 1.0), kw = none))
end
add!("HChern8_Lx2_Ly2_default", :preset2d, (fn = :HChern8, args = (2, 2, 1.0, 1.0), kw = none))
add!("HChern8_Lx2_Ly1_a1.5_t2", :preset2d, (fn = :HChern8, args = (2, 1, 0.5, 1.0), kw = (a = 1.5, t2 = 0.3)))
add!("HQC2Dsquare_Lx2_Ly2_default", :preset2d, (fn = :HQC2Dsquare, args = (2, 2), kw = none))
add!("HQC2Dsquare_Lx2_Ly1_t0.8", :preset2d, (fn = :HQC2Dsquare, args = (2, 1, 0.8), kw = none))

# ── 8. sublattice lattices: positions and Hamiltonians ────────────────────────
for fn in (:kagome_positions, :lieb_positions, :honeycomb_sublattice_positions, :dice_positions),
    (Lx, Ly) in ((2, 1), (1, 2))
    add!("$(fn)_Lx$(Lx)_Ly$(Ly)", :sublattice_positions, (; fn, Lx, Ly))
end
add!("kagome_hamiltonian_Lx2_Ly1_default", :sublattice_hamiltonian, (fn = :kagome_hamiltonian, args = (2, 1), kw = none))
add!("kagome_hamiltonian_Lx2_Ly1_anisotropic", :sublattice_hamiltonian,
     (fn = :kagome_hamiltonian, args = (2, 1, 1.0), kw = (t_AB = 1.0, t_AC = 0.8, t_BC = 0.6)))
add!("kagome_hamiltonian_Lx2_Ly1_maxdim4", :sublattice_hamiltonian, (fn = :kagome_hamiltonian, args = (2, 1), kw = (maxdim = 4,)))
add!("kagome_hamiltonian_Lx1_Ly1_complex_tAB", :sublattice_hamiltonian,
     (fn = :kagome_hamiltonian, args = (1, 1), kw = (t_AB = 0.5im,)))
add!("lieb_hamiltonian_Lx2_Ly1_default", :sublattice_hamiltonian, (fn = :lieb_hamiltonian, args = (2, 1), kw = none))
add!("lieb_hamiltonian_Lx2_Ly1_anisotropic", :sublattice_hamiltonian,
     (fn = :lieb_hamiltonian, args = (2, 1), kw = (t_AB = 1.0, t_AC = 0.5)))
add!("honeycomb_sublattice_hamiltonian_Lx2_Ly1_default", :sublattice_hamiltonian,
     (fn = :honeycomb_sublattice_hamiltonian, args = (2, 1), kw = none))
add!("honeycomb_sublattice_hamiltonian_Lx2_Ly1_tcomplex", :sublattice_hamiltonian,
     (fn = :honeycomb_sublattice_hamiltonian, args = (2, 1, 0.7 + 0.3im), kw = none))
add!("honeycomb_nnn_hamiltonian_Lx2_Ly1_default", :sublattice_hamiltonian,
     (fn = :honeycomb_nnn_hamiltonian, args = (2, 1), kw = none))
add!("honeycomb_nnn_hamiltonian_Lx2_Ly1_t2real", :sublattice_hamiltonian,
     (fn = :honeycomb_nnn_hamiltonian, args = (2, 1, 1.0, 0.1), kw = none))
add!("honeycomb_nnn_hamiltonian_Lx2_Ly1_t2imag", :sublattice_hamiltonian,
     (fn = :honeycomb_nnn_hamiltonian, args = (2, 1, 1.0, 0.1im), kw = none))
add!("dice_hamiltonian_Lx2_Ly1_default", :sublattice_hamiltonian, (fn = :dice_hamiltonian, args = (2, 1), kw = none))
add!("dice_hamiltonian_Lx2_Ly1_anisotropic", :sublattice_hamiltonian,
     (fn = :dice_hamiltonian, args = (2, 1), kw = (t_AB = 1.0, t_AC = 0.7)))
add!("ssh_sublattice_hamiltonian_L3_default", :sublattice_hamiltonian,
     (fn = :ssh_sublattice_hamiltonian, args = (3,), kw = none))
add!("ssh_sublattice_hamiltonian_L3_t0.8_d-0.3", :sublattice_hamiltonian,
     (fn = :ssh_sublattice_hamiltonian, args = (3, 0.8, -0.3), kw = none))

# ── 9. registry and build_hamiltonian ─────────────────────────────────────────
for (tag, s) in (("empty", ""), ("two_floats", "t=1.0, d=0.5"),
                 ("all_types", "a=true b=FALSE n=-3 x=.5 y=1e-3 z=+2 s=abc"),
                 ("separators", "t=1.0,,  v=2\tw=1.5e+2"), ("value_with_equals", "k=v=w"),
                 ("bad_token", "t=1.0 bad"), ("empty_value", "k="))
    add!("parse_param_string_$tag", :parse_param_string, (; s))
end
add!("model_registry_without_chernhex", :model_registry, (skip = ("chernhex",),))
add!("build_hamiltonian_uniform_L3_mparams", :build_hamiltonian, (model = "uniform", dims = (3,), kw = (mparams = "t=1.0",)))
add!("build_hamiltonian_ssh_L3_dict", :build_hamiltonian,
     (model = "ssh", dims = (3,), kw = (mparam_dict = Dict{Symbol,Any}(:t => 1.0, :d => 0.3),)))
add!("build_hamiltonian_aah_L3_mparams_extra_b", :build_hamiltonian,
     (model = "aah", dims = (3,), kw = (mparams = "V=0.5, phi=0.2, t=1.0 b=0.4",)))
add!("build_hamiltonian_SSH_uppercase_nn2", :build_hamiltonian, (model = "SSH", dims = (3,), kw = (mparams = "t=1.0, d=0.2, nn=2",)))
add!("build_hamiltonian_uniform_dict_overrides_string", :build_hamiltonian,
     (model = "uniform", dims = (3,), kw = (mparams = "t=1.0", mparam_dict = Dict{Symbol,Any}(:t => 0.5, :v => 0.2))))
# A long unknown name keeps the pinned 60-character message prefix clear of the
# list of known models (which other work may change).
add!("build_hamiltonian_unknown_1d", :build_hamiltonian, (model = "no_such_model_used_as_a_golden_probe", dims = (3,), kw = none))
add!("build_hamiltonian_unknown_2d", :build_hamiltonian, (model = "no_such_model_used_as_a_golden_probe", dims = (2, 1), kw = none))
add!("build_hamiltonian_2d_model_1d_call", :build_hamiltonian, (model = "square_2d", dims = (3,), kw = (mparams = "t=1.0",)))
add!("build_hamiltonian_ssh_missing_d", :build_hamiltonian, (model = "ssh", dims = (3,), kw = (mparams = "t=1.0",)))
for model in ("square_2d", "hex_2d", "triangular_2d", "triangular_bravais", "qc2dsquare")
    add!("build_hamiltonian_$(model)_Lx2_Ly1", :build_hamiltonian, (; model, dims = (2, 1), kw = (mparams = "t=1.0",)))
end
add!("build_hamiltonian_chern8_Lx2_Ly1", :build_hamiltonian, (model = "chern8", dims = (2, 1), kw = (mparams = "V=0.5, t=1.0",)))
add!("build_hamiltonian_1d_model_2d_call", :build_hamiltonian, (model = "uniform", dims = (2, 1), kw = (mparams = "t=1.0",)))
add!("build_hamiltonian_chern8_missing_t", :build_hamiltonian, (model = "chern8", dims = (2, 1), kw = (mparams = "V=1.0",)))

# ── 10. geometry helpers for spatial LDOS plots ───────────────────────────────
for lattice in (:honeycomb, :kagome, :lieb, :dice)
    add!("geom_positions_$(lattice)_Lx2_Ly1", :geom_positions, (fn = :_geom_positions, lattice, Lx = 2, Ly = 1))
    for fn in (:_nsublat, :_geom_n_sub)
        add!("$(fn)_$lattice", :geom_counts, (; fn, lattice))
    end
end

# ═════════════════════════════════════════════════════════════════════════════
# core/TBSystem.jl: get_Hamiltonian and the geometry helpers
# ═════════════════════════════════════════════════════════════════════════════

gh(name, geometry, params, kw; extra...) =
    add!("get_Hamiltonian_$name", :get_hamiltonian, (; geometry, params, kw, extra...))

gh("chain_1d_L3", "chain_1d", 1.0, (L = 3,))
gh("chain_1d_L3_periodic", "chain_1d", 0.7, (L = 3, boundary = :periodic))
gh("chain_1d_L3_bc_string", "chain_1d", 1.0, (L = 3, bc = "periodic"))
gh("chain_1d_L3_scale_maxdim2", "chain_1d", 1.0, (L = 3, scale = 3.0, maxdim = 2))
gh("chain_1d_L3_ref_sites_ignored", "chain_1d", 1.0, (L = 3,); ref_sites = true)
gh("custom_L3_real_matrix_geometry", "custom", :custom_nn, (L = 3, scale = 2.0, type = Float64);
   custom_geometry = (kind = :matrix, Lx = 2, Ly = 1))
gh("custom_L3_complex_function_geometry", "custom", :custom_cplx, (L = 3, scale = 2.5);
   custom_geometry = (kind = :function, Lx = 2, Ly = 1))
gh("custom_L3_no_geometry", "custom", :custom_nn, (L = 3, scale = 2.0))
gh("custom_L3_missing_scale", "custom", :custom_nn, (L = 3,))
gh("kagome_L3_default_split", "kagome", 1.0, (L = 3,))
gh("kagome_L3_Lx2_Ly1", "kagome", 0.8, (L = 3, Lx = 2, Ly = 1))
gh("kagome_L3_namedtuple_scale", "kagome", (t = 0.9,), (L = 3, Lx = 2, Ly = 1, scale = 4.0))
gh("lieb_L3_Lx2_Ly1", "lieb", 1.0, (L = 3, Lx = 2, Ly = 1))
gh("lieb_L3_dict", "lieb", Dict(:t => 0.6), (L = 3, Lx = 2, Ly = 1))
gh("honeycomb_L3_default_split", "honeycomb", 1.0, (L = 3,))
gh("honeycomb_L3_Lx2_Ly1_ref_sites_ignored", "honeycomb", 1.0, (L = 3, Lx = 2, Ly = 1); ref_sites = true)
gh("honeycomb_nnn_L3_namedtuple", "honeycomb_nnn", (t = 1.0, t2 = 0.15), (L = 3, Lx = 2, Ly = 1))
gh("honeycomb_nnn_L3_dict", "honeycomb_nnn", Dict(:t => 1.0, :t2 => 0.1), (L = 3, Lx = 2, Ly = 1))
gh("dice_L3_Lx2_Ly1", "dice", 1.0, (L = 3, Lx = 2, Ly = 1))
gh("ssh_sublattice_L3_number", "ssh_sublattice", 1.0, (L = 3,))
gh("ssh_sublattice_L3_namedtuple", "ssh_sublattice", (t = 1.0, d = 0.3), (L = 3,))
gh("ssh_sublattice_L3_dict_scale", "ssh_sublattice", Dict(:t => 0.8, :d => -0.2), (L = 3, scale = 2.5))
gh("uniform_L3_number", "uniform", 1.0, (L = 3,))
gh("uniform_L3_namedtuple_v", "uniform", (t = 0.8, v = 0.2), (L = 3,))
gh("uniform_L3_dict_nn2", "uniform", Dict(:t => 1.0, :nn => 2), (L = 3,))
gh("ssh_L3_namedtuple", "ssh", (t = 1.0, d = 0.3), (L = 3,))
gh("ssh_L3_number_missing_d", "ssh", 1.0, (L = 3,))
gh("ssh_L3_ref_sites", "ssh", (t = 1.0, d = 0.3), (L = 3,); ref_sites = true)
gh("aah_L3_namedtuple", "aah", (V = 0.5, phi = 0.2, t = 1.0), (L = 3,))
gh("aah_L3_number_missing_params", "aah", 0.5, (L = 3,))
gh("square_2d_L3_default_split", "square_2d", 1.0, (L = 3,))
gh("square_2d_L3_Lx2", "square_2d", 1.0, (L = 3, Lx = 2))
gh("square_2d_L3_Lx2_Ly1_scale", "square_2d", 0.9, (L = 3, Lx = 2, Ly = 1, scale = 4.0))
gh("square_2d_L3_Lx2_ref_sites", "square_2d", 1.0, (L = 3, Lx = 2); ref_sites = true)
gh("hex_2d_L3_Lx2", "hex_2d", 1.0, (L = 3, Lx = 2))
gh("triangular_2d_L3_Lx2", "triangular_2d", 1.0, (L = 3, Lx = 2))
gh("triangular_bravais_L3_Lx2", "triangular_bravais", 1.0, (L = 3, Lx = 2))
gh("chern8_L3_Lx2_explicit_scale", "chern8", (V = 0.5, t = 1.0), (L = 3, Lx = 2, scale = 5.0))
gh("qc2dsquare_L4_Lx2_explicit_scale", "qc2dsquare", 1.0, (L = 4, Lx = 2, scale = 5.0))
gh("fibonacci_L4", "fibonacci", (A = 1.0, B = 2.0), (L = 4,))
gh("metallic_mean_L3_m2", "metallic_mean", (A = 1.0, B = 2.0), (L = 3, m = 2))
gh("kbonacci_L4_k3", "kbonacci", (A = 0.64, B = 0.8, C = 1.0), (L = 4, k = 3))
gh("fibonacci_ref_sites_rejected", "fibonacci", (A = 1.0, B = 2.0), (L = 4,); ref_sites = true)
gh("unknown_geometry", "no_such_geometry_used_as_a_golden_probe", 1.0, (L = 3,))   # prefix stops before the list

# ── geometry closures, _preset_geometry, _estimate_scale, *_positions ─────────
add!("chain_geometry_n4", :geometry_closure, (fn = :_chain_geometry, Nx = nothing, n = 4))
for fn in (:_square_geometry, :_tri_geometry, :_tri_bravais_geometry, :_hex_geometry), (Nx, n) in ((4, 16), (2, 8))
    add!("$(fn)_Nx$(Nx)_n$n", :geometry_closure, (; fn, Nx, n))
end
for geometry in ("uniform", "ssh", "aah", "chain_1d")
    add!("preset_geometry_$geometry", :preset_geometry, (; geometry, Nx = nothing, n = 4))
end
for geometry in ("square_2d", "hex_2d", "triangular_2d", "triangular_bravais")
    add!("preset_geometry_$(geometry)_Nx4", :preset_geometry, (; geometry, Nx = 4, n = 8))
end
for geometry in ("kagome", "custom", "fibonacci")
    add!("preset_geometry_$(geometry)_none", :preset_geometry, (; geometry, Nx = 4, n = 8))
end
for (tag, geometry, params) in (("chain_1d_number", "chain_1d", 1.0), ("chain_1d_namedtuple_negative", "chain_1d", (t = -0.5,)),
                                ("ssh_dict", "ssh", Dict(:t => 0.7)), ("aah_namedtuple", "aah", (t = 1.0, V = 0.5)),
                                ("aah_number", "aah", 2.0), ("uniform", "uniform", 2.0),
                                ("square_2d", "square_2d", 1.0), ("hex_2d", "hex_2d", 1.0),
                                ("triangular_2d", "triangular_2d", 1.0), ("triangular_bravais", "triangular_bravais", 1.0),
                                ("kagome_fallback", "kagome", 1.0), ("unknown_fallback", "foo", 3.0),
                                ("namedtuple_without_t", "square_2d", (V = 2.0,)), ("dict_without_t", "square_2d", Dict(:V => 2.0)),
                                ("dict_with_t", "hex_2d", Dict(:t => -2.0)), ("complex_t", "square_2d", 0.6 + 0.8im))
    add!("estimate_scale_$tag", :estimate_scale, (; geometry, params))
end
for fn in (:honeycomb_positions, :square_positions, :triangular_positions, :triangular_bravais_positions)
    add!("$(fn)_L4_default_Lx", :positions, (; fn, L = 4, kw = none))
    add!("$(fn)_L3_Lx$(fn === :square_positions ? 1 : 2)", :positions,
         (; fn, L = 3, kw = (Lx = fn === :square_positions ? 1 : 2,)))
end
add!("central_index_square_closure_Nx4", :central_index, (kind = :closure, fn = :_square_geometry, Nx = 4, N = 16))
add!("central_index_H_square_2d_L4", :central_index,
     (kind = :H, base = (via = :get_hamiltonian, geometry = "square_2d", params = 1.0, kw = (L = 4, Lx = 2))))
add!("central_index_H_without_geometry", :central_index,
     (kind = :H, base = (via = :bilayer, lattice = :square, Lx = 1, Ly = 1, kw = none)))

# ═════════════════════════════════════════════════════════════════════════════
# lattice/Twisted_tk.jl and lattice/Bilayer_tk.jl
# ═════════════════════════════════════════════════════════════════════════════

add!("lattice_positions_square_Lx2_Ly1", :lattice_positions, (lattice = :square, Lx = 2, Ly = 1, kw = none))
add!("lattice_positions_triangular_Lx2_Ly1", :lattice_positions, (lattice = :triangular, Lx = 2, Ly = 1, kw = none))
add!("lattice_positions_honeycomb_Lx2_Ly1", :lattice_positions, (lattice = :honeycomb, Lx = 2, Ly = 1, kw = none))
add!("lattice_positions_square_Lx2_Ly1_rot30", :lattice_positions, (lattice = :square, Lx = 2, Ly = 1, kw = (angle_deg = 30.0,)))
add!("lattice_positions_honeycomb_Lx1_Ly2_rot-12.5", :lattice_positions, (lattice = :honeycomb, Lx = 1, Ly = 2, kw = (angle_deg = -12.5,)))
add!("lattice_positions_unknown", :lattice_positions, (lattice = :foo, Lx = 1, Ly = 1, kw = none))
for lattice in (:square, :triangular, :honeycomb)
    add!("monolayer_hamiltonian_$(lattice)_Lx2_Ly1", :monolayer, (; lattice, Lx = 2, Ly = 1, kw = (t = 0.9,)))
end
add!("monolayer_hamiltonian_unknown", :monolayer, (lattice = :foo, Lx = 1, Ly = 1, kw = none))
add!("prepend_layer_projector_3layers_k2", :layer_ops, (fn = :prepend_layer_projector, nlayers = 3, levels = (2,)))
add!("prepend_layer_hopping_3layers_1_3", :layer_ops, (fn = :prepend_layer_hopping, nlayers = 3, levels = (1, 3)))
add!("postpend_layer_projector_3layers_k2", :layer_ops, (fn = :postpend_layer_projector, nlayers = 3, levels = (2,)))
add!("postpend_layer_hopping_3layers_3_1", :layer_ops, (fn = :postpend_layer_hopping, nlayers = 3, levels = (3, 1)))
add!("twisted_bilayer_square_Lx1_Ly2_10deg", :twisted, (fn = :twisted_bilayer_hamiltonian, args = (:square, 1, 2, 10.0), kw = none))
add!("twisted_bilayer_honeycomb_Lx1_Ly1_kwargs", :twisted,
     (fn = :twisted_bilayer_hamiltonian, args = (:honeycomb, 1, 1, 21.8),
      kw = (t_intra = 0.9, t_inter = 0.2, α_decay = 0.5, tol = 1e-8)))
add!("twisted_multilayer_triangular_Lx1_Ly1_3layers", :twisted,
     (fn = :twisted_multilayer_hamiltonian, args = (:triangular, 1, 1, [0.0, 5.0, 10.0]), kw = none))
add!("twisted_multilayer_one_layer", :twisted, (fn = :twisted_multilayer_hamiltonian, args = (:square, 1, 1, [0.0]), kw = none))

add!("interlayer_mpo_square_AA", :interlayer_mpo, (lattice = :square, stacking = :AA, Lx = 2, Ly = 1, kw = (t_inter = 0.3,)))
add!("interlayer_mpo_honeycomb_Bernal", :interlayer_mpo, (lattice = :honeycomb, stacking = :Bernal, Lx = 2, Ly = 1, kw = (t_inter = 0.4,)))
add!("interlayer_mpo_honeycomb_Bernal_complex", :interlayer_mpo,
     (lattice = :honeycomb, stacking = :Bernal, Lx = 2, Ly = 1, kw = (t_inter = 0.2 + 0.1im,)))
add!("interlayer_mpo_square_Bernal", :interlayer_mpo, (lattice = :square, stacking = :Bernal, Lx = 2, Ly = 1, kw = none))
add!("interlayer_mpo_unknown_stacking", :interlayer_mpo, (lattice = :honeycomb, stacking = :ABC, Lx = 2, Ly = 1, kw = none))
add!("bilayer_square_Lx2_Ly1_default", :bilayer, (lattice = :square, Lx = 2, Ly = 1, kw = none))
add!("bilayer_honeycomb_Lx2_Ly1_Bernal", :bilayer,
     (lattice = :honeycomb, Lx = 2, Ly = 1, kw = (stacking = :Bernal, t_intra = 0.9, t_inter = 0.2)))
add!("bilayer_triangular_Lx1_Ly1_AA", :bilayer, (lattice = :triangular, Lx = 1, Ly = 1, kw = (t_inter = 0.5,)))
add!("bilayer_square_Lx1_Ly1_AA_complex_tinter", :bilayer, (lattice = :square, Lx = 1, Ly = 1, kw = (t_inter = 0.3im,)))
add!("bilayer_honeycomb_Lx1_Ly1_sublattice_AA", :bilayer, (lattice = :honeycomb, Lx = 1, Ly = 1, kw = (sublattice = true,)))
add!("bilayer_honeycomb_Lx1_Ly1_sublattice_Bernal", :bilayer,
     (lattice = :honeycomb, Lx = 1, Ly = 1, kw = (sublattice = true, stacking = :Bernal, t_inter = 0.25)))
add!("bilayer_square_sublattice_rejected", :bilayer, (lattice = :square, Lx = 1, Ly = 1, kw = (sublattice = true,)))
add!("multilayer_square_Lx1_Ly1_3layers", :multilayer, (lattice = :square, Lx = 1, Ly = 1, n_layers = 3, kw = none))
add!("multilayer_honeycomb_Lx1_Ly1_3layers_Bernal", :multilayer,
     (lattice = :honeycomb, Lx = 1, Ly = 1, n_layers = 3, kw = (stacking = :Bernal,)))
add!("multilayer_honeycomb_Lx1_Ly1_3layers_sublattice_Bernal", :multilayer,
     (lattice = :honeycomb, Lx = 1, Ly = 1, n_layers = 3, kw = (sublattice = true, stacking = :Bernal)))
add!("multilayer_one_layer", :multilayer, (lattice = :square, Lx = 1, Ly = 1, n_layers = 1, kw = none))

# ═════════════════════════════════════════════════════════════════════════════
# lattice/Flake_tk.jl
# ═════════════════════════════════════════════════════════════════════════════

const XS = [-0.5, 0.25, 1.0, 1.75, 2.5]
const YS = [-0.5, 0.5, 1.5, 2.5]
disk     = (:disk, 1.0, 1.0, 1.2)
rect     = (:rect, 1.0, 1.0, 2.0, 1.0)
half     = (:halfplane, 1.0, 2.0, 0.5)
annulus  = (:annulus, 1.0, 1.0, 0.5, 1.5)
triangle = (:polygon, [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)])
for (tag, sdf) in (("disk", disk), ("rect", rect), ("halfplane", half), ("annulus", annulus),
                   ("convex_polygon", triangle), ("union", (:union, disk, rect)),
                   ("intersect", (:intersect, disk, half)), ("subtract", (:subtract, rect, (:disk, 1.0, 1.0, 0.4))),
                   ("nested", (:union, (:intersect, triangle, half), annulus)),
                   ("polygon_two_vertices", (:polygon, [(0.0, 0.0), (1.0, 0.0)])))
    add!("sdf_$tag", :sdf, (; sdf, dim = 2, xs = XS, ys = YS))
end
add!("sdf_interval", :sdf, (sdf = (:interval, 0.0, 2.0), dim = 1, xs = XS, ys = nothing))

sq4 = (via = :get_hamiltonian, geometry = "square_2d", params = 1.0, kw = (L = 4, Lx = 2))
add!("mask_hamiltonian_square_L4_disk_default", :mask_hamiltonian, (base = sq4, sdf = (:disk, 1.5, 1.5, 1.2), kw = none))
add!("mask_hamiltonian_square_L4_disk_sharp", :mask_hamiltonian,
     (base = sq4, sdf = (:disk, 1.5, 1.5, 1.2), kw = (sigma = 0.15, tol = 1e-10, maxdim = 50)))
add!("mask_hamiltonian_triangular_bravais_L4_polygon", :mask_hamiltonian,
     (base = (via = :get_hamiltonian, geometry = "triangular_bravais", params = 1.0, kw = (L = 4, Lx = 2)),
      sdf = (:polygon, [(0.2, 0.1), (3.5, 0.1), (1.5, 2.8)]), kw = none))
add!("mask_hamiltonian_kagome_L2", :mask_hamiltonian,
     (base = (via = :get_hamiltonian, geometry = "kagome", params = 1.0, kw = (L = 2,)), sdf = (:disk, 0.5, 0.5, 1.0), kw = none))
add!("mask_hamiltonian_bilayer_rejected", :mask_hamiltonian,
     (base = (via = :bilayer, lattice = :square, Lx = 1, Ly = 1, kw = none), sdf = (:disk, 0.5, 0.5, 1.0), kw = none))
add!("mask_hamiltonian_no_geometry_rejected", :mask_hamiltonian,
     (base = (via = :no_geometry, of = sq4), sdf = (:disk, 0.5, 0.5, 1.0), kw = none))

# ═════════════════════════════════════════════════════════════════════════════
# lattice/TJunction_tk.jl
# ═════════════════════════════════════════════════════════════════════════════

chain3 = (via = :get_hamiltonian, geometry = "chain_1d", params = 1.0, kw = (L = 3,))
add!("tjunction_index", :tjunction_parts, (fn = :tjunction_index,))
add!("tjunction_positions_N4_j0", :tjunction_parts, (fn = :tjunction_positions, N = 4, junction_site = 0))
add!("tjunction_positions_N4_j2", :tjunction_parts, (fn = :tjunction_positions, N = 4, junction_site = 2))
add!("site_projector_mpo_L3_n5", :tjunction_parts, (fn = :_site_projector_mpo, L = 3, n = 5))
add!("site_projector_mpo_L3_n8_out_of_range", :tjunction_parts, (fn = :_site_projector_mpo, L = 3, n = 8))
add!("add_tjunction_chain_L3_default", :add_tjunction, (base = chain3, t_j = 0.5, kw = none))
add!("add_tjunction_chain_L3_right_end", :add_tjunction, (base = chain3, t_j = 0.5, kw = (junction_site = 7,)))
add!("add_tjunction_chain_L3_float_coupling", :add_tjunction,
     (base = chain3, t_j = 0.5, kw = (coupling = [0.0 0.6 0.0; 0.6 0.0 0.4; 0.0 0.4 0.0], junction_site = 3)))
add!("add_tjunction_chain_L3_int_coupling", :add_tjunction, (base = chain3, t_j = 0.5, kw = (coupling = [0 1 0; 1 0 1; 0 1 0],)))
add!("add_tjunction_junction_out_of_range", :add_tjunction, (base = chain3, t_j = 0.5, kw = (junction_site = 8,)))
add!("add_tjunction_on_sublattice_rejected", :add_tjunction,
     (base = (via = :get_hamiltonian, geometry = "kagome", params = 1.0, kw = (L = 2,)), t_j = 0.5, kw = none))
add!("tjunction_hamiltonian_L3_default", :tjunction_hamiltonian, (args = (3, 1.0, 0.5), kw = none))
add!("tjunction_hamiltonian_L3_j3", :tjunction_hamiltonian, (args = (3, 1.0, 0.5), kw = (junction_site = 3,)))
add!("tjunction_hamiltonian_L3_coupling", :tjunction_hamiltonian,
     (args = (3, 1.0, 0.5), kw = (coupling = [0.0 0.7 0.0; 0.7 0.0 0.7; 0.0 0.7 0.0],)))
add!("tjunction_hamiltonian_L3_periodic", :tjunction_hamiltonian, (args = (3, 0.8, 0.4), kw = (boundary = :periodic,)))
add!("tjunction_hamiltonian_negative_junction", :tjunction_hamiltonian, (args = (3, 1.0, 0.5), kw = (junction_site = -1,)))
add!("tjunction_lattice_Lx1_Ly1_L2_default", :tjunction_lattice, (args = (1, 1, 2, 1.0, 0.5, 0.8), kw = none))
add!("tjunction_lattice_Lx1_Ly1_L2_j1_periodic", :tjunction_lattice,
     (args = (1, 1, 2, 1.0, 0.5, 0.3), kw = (junction_site = 1, boundary = :periodic)))
add!("tjunction_lattice_Lx2_Ly1_L1_coupling", :tjunction_lattice,
     (args = (2, 1, 1, 1.0, 0.5, 0.8), kw = (coupling = [0.0 0.9 0.2; 0.9 0.0 0.5; 0.2 0.5 0.0],)))

# ═════════════════════════════════════════════════════════════════════════════
# lattice/NNNeighbor_tk.jl
# ═════════════════════════════════════════════════════════════════════════════

for (dx, dy) in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (4, 0), (0, 2))
    add!("shift_mpo_Lx2_Ly1_dx$(dx)_dy$(dy)", :shift_mpo_nnn, (; dx, dy, Lx = 2, Ly = 1, kw = none))
end

gh_base(geometry, params, kw) = (via = :get_hamiltonian, geometry, params, kw)
sq_L4  = gh_base("square_2d", 1.0, (L = 4, Lx = 2))
tri_L4 = gh_base("triangular_bravais", 1.0, (L = 4, Lx = 2))
hc_L4  = gh_base("honeycomb", 1.0, (L = 4, Lx = 2, Ly = 2))
hex_L4 = gh_base("hex_2d", 1.0, (L = 4, Lx = 2))
kag_L2 = gh_base("kagome", 1.0, (L = 2, Lx = 1, Ly = 1))
lieb_L2 = gh_base("lieb", 1.0, (L = 2, Lx = 1, Ly = 1))
dice_L2 = gh_base("dice", 1.0, (L = 2, Lx = 1, Ly = 1))
bil_sq = (via = :bilayer, lattice = :square, Lx = 1, Ly = 1, kw = none)
bil_hc_sub = (via = :bilayer, lattice = :honeycomb, Lx = 1, Ly = 1, kw = (sublattice = true,))
sq_nogeom = (via = :no_geometry, of = sq_L4)

for nn in 1:3
    add!("shell_disps_square_L4_nn$nn", :shell_disps, (base = sq_L4, nn, Lx = 2, Ly = 2))
    add!("shell_disps_honeycomb_L4_nn$nn", :shell_disps, (base = hc_L4, nn, Lx = 2, Ly = 2))
end
add!("shell_disps_triangular_bravais_L4_nn1", :shell_disps, (base = tri_L4, nn = 1, Lx = 2, Ly = 2))
add!("shell_disps_triangular_bravais_L4_nn2", :shell_disps, (base = tri_L4, nn = 2, Lx = 2, Ly = 2))
add!("shell_disps_hex_2d_L4_nn1", :shell_disps, (base = hex_L4, nn = 1, Lx = 2, Ly = 2))
add!("shell_disps_triangular_2d_L4_nn1", :shell_disps,
     (base = gh_base("triangular_2d", 1.0, (L = 4, Lx = 2)), nn = 1, Lx = 2, Ly = 2))
add!("shell_disps_kagome_L2_nn1", :shell_disps, (base = kag_L2, nn = 1, Lx = 1, Ly = 1))
add!("shell_disps_lieb_L2_nn1", :shell_disps, (base = lieb_L2, nn = 1, Lx = 1, Ly = 1))
add!("shell_disps_dice_L2_nn1", :shell_disps, (base = dice_L2, nn = 1, Lx = 1, Ly = 1))
add!("shell_disps_without_geometry", :shell_disps, (base = bil_sq, nn = 1, Lx = 1, Ly = 1))

ah(name, base, f, kw) = add!("add_hopping_2D_$name", :add_hopping_2D, (; base, f, kw))
ah("square_L4_scalar_nn1", sq_L4, 0.1, (Lx = 2, Ly = 2, nn = 1))
ah("square_L4_scalar_nn2", sq_L4, 0.05, (Lx = 2, Ly = 2, nn = 2))
ah("square_L4_imag_nn3", sq_L4, 0.1im, (Lx = 2, Ly = 2, nn = 3))
ah("square_L4_dir_nn1", sq_L4, :amp_dir, (Lx = 2, Ly = 2, nn = 1))
ah("square_L4_pos1d_nn1", sq_L4, :amp_pos1d, (Lx = 2, Ly = 2, nn = 1))
ah("square_L4_pos2d_nn1", sq_L4, :amp_pos2d, (Lx = 2, Ly = 2, nn = 1))
ah("square_L4_pos1d_dir_nn2", sq_L4, :amp_pos1d_dir, (Lx = 2, Ly = 2, nn = 2))
ah("square_L4_pos2d_dir_nn1", sq_L4, :amp_pos2d_dir, (Lx = 2, Ly = 2, nn = 1))
ah("square_L4_scalar_nn1_maxdim4_tol1e-6", sq_L4, 0.1, (Lx = 2, Ly = 2, nn = 1, maxdim = 4, tol = 1e-6))
ah("honeycomb_L4_imag_nn2", hc_L4, 0.1im, (Lx = 2, Ly = 2, nn = 2))
ah("kagome_L2_dir_nn1", kag_L2, :amp_dir, (Lx = 1, Ly = 1, nn = 1))
ah("triangular_bravais_L4_pos2d_dir_nn1", tri_L4, :amp_pos2d_dir, (Lx = 2, Ly = 2, nn = 1))
ah("hex_2d_L4_scalar_nn1", hex_L4, 0.1, (Lx = 2, Ly = 2, nn = 1))
ah("square_L4_bad_signature", sq_L4, :amp_bad, (Lx = 2, Ly = 2, nn = 1))
ah("square_L2_shell_too_wide", gh_base("square_2d", 1.0, (L = 2, Lx = 1)), 0.1, (Lx = 1, Ly = 1, nn = 3))
ah("square_L4_layer_without_layers", sq_L4, 0.1, (Lx = 2, Ly = 2, nn = 1, layer = 1))
ah("bilayer_square_lattice_all_layers", bil_sq, 0.1, (Lx = 1, Ly = 1, nn = 1, lattice = :square))
ah("bilayer_square_lattice_layer1", bil_sq, 0.1, (Lx = 1, Ly = 1, nn = 1, lattice = :square, layer = 1))
ah("bilayer_square_lattice_layer2_pos2d", bil_sq, :amp_pos2d, (Lx = 1, Ly = 1, nn = 1, lattice = :square, layer = [2]))
ah("bilayer_square_geometry_matrix", bil_sq, 0.1,
   (Lx = 1, Ly = 1, nn = 2, geometry = (kind = :matrix, rows = :square, Lx = 1, Ly = 1, nrows = 4)))
ah("bilayer_square_missing_lattice", bil_sq, 0.1, (Lx = 1, Ly = 1, nn = 1))
ah("bilayer_square_layer_out_of_range", bil_sq, 0.1, (Lx = 1, Ly = 1, nn = 1, lattice = :square, layer = 3))
ah("bilayer_honeycomb_sublattice_nn2", bil_hc_sub, 0.05, (Lx = 1, Ly = 1, nn = 2))
ah("no_geometry_square_lattice_kw", sq_nogeom, 0.1, (Lx = 2, Ly = 2, nn = 1, lattice = :square))
ah("no_geometry_square_geometry_function", sq_nogeom, 0.1,
   (Lx = 2, Ly = 2, nn = 1, geometry = (kind = :function, rows = :square, Lx = 2, Ly = 2, nrows = 16)))
ah("no_geometry_square_geometry_matrix_wrong_rows", sq_nogeom, 0.1,
   (Lx = 2, Ly = 2, nn = 1, geometry = (kind = :matrix, rows = :square, Lx = 2, Ly = 2, nrows = 15)))

for (tag, layer) in (("all", nothing), ("int", 2), ("vector", [1, 3]), ("range", 2:3), ("out_of_range", 4))
    add!("resolve_layer_selection_$tag", :resolve_layer_selection, (nlayers = 3, layer))
end

# ═════════════════════════════════════════════════════════════════════════════
# Evaluation and output
# ═════════════════════════════════════════════════════════════════════════════

# A case's seed is a fixed function of its name (not of its position), so that
# adding or removing a case does not reseed, and so change, any other case.
name_seed(name::AbstractString) =
    foldl((h, c) -> (31 * h + Int(c)) % 2_147_483_647, codeunits(name); init = 17)

function evaluate_all(cases)
    names = first.(cases)
    allunique(names) || error("duplicate case names: $(unique(filter(n -> count(==(n), names) > 1, names)))")
    entries = NamedTuple[]
    notes = String[]
    for (name, builder, spec) in cases
        seed = name_seed(name)
        t = time()
        result, err = R.evaluate(builder, spec, seed)
        dt = time() - t
        dt > 2 && push!(notes, "slow ($(round(dt; digits=1)) s): $name")
        if err !== nothing
            any(T -> err isa T, GENERATOR_BUGS) &&
                error("case $name: $(typeof(err)) looks like a generator bug: $(sprint(showerror, err))")
            push!(notes, "throws $(typeof(err)): $name -- $(first(sprint(showerror, err), 120))")
            push!(entries, (; name, builder, seed, throws = typeof(err), spec,
                              expected = (; message_prefix = R.message_prefix(err))))
        else
            push!(entries, (; name, builder, seed, throws = nothing, spec, expected = R.encode(result)))
        end
    end
    return entries, notes
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
        println(io, "# AUTO-GENERATED by test/data/generate_lattice_golden.jl -- do not edit by hand.")
        println(io, "#")
        println(io, "# Pinned outputs of the lattice builders, checked by test/golden_lattice.jl.")
        println(io, "# Generated from $source")
        println(io, "# with Julia $VERSION. Regenerate only for an intentional behaviour change;")
        println(io, "# see the generator's header.")
        println(io, "#")
        println(io, "# Each entry is (name, builder, seed, throws, spec, expected): `builder` selects")
        println(io, "# a call in LatticeGoldenRunner.run_case, `seed` reseeds the RNGs before it,")
        println(io, "# `throws` is the exception type a failing case must still raise (otherwise")
        println(io, "# `nothing`), and `expected` holds every returned field or, for a throwing")
        println(io, "# case, the first $(R.MESSAGE_PREFIX_CHARS) characters of its message (`message_prefix`).")
        println(io, "# Matrices with more than $(R.DENSE_MAX_LENGTH) entries are stored as sparse records")
        println(io, "# (`__sparse__ = true`: size, element type, column-major linear indices and")
        println(io, "# values of the entries with magnitude >= $(R.DROP_BELOW)).")
        println(io, "#")
        println(io, "# $(length(entries)) cases, $(count(e -> e.throws !== nothing, entries)) of them pinned as throwing.")
        println(io)
        println(io, "LATTICE_GOLDEN_CASES = Any[]")
        for e in entries
            println(io)
            println(io, "push!(LATTICE_GOLDEN_CASES, (name = ", repr(e.name), ", builder = ", repr(e.builder),
                    ", seed = ", repr(e.seed), ", throws = ", repr(e.throws), ",")
            println(io, "    spec = ", repr(e.spec), ",")
            println(io, "    expected = ", repr(e.expected), "))")
        end
        println(io)
        println(io, "LATTICE_GOLDEN_CASES")
    end
end

t_start = time()
entries, notes = evaluate_all(CASES)
foreach(println, notes)
write_golden(OUTFILE, entries)

# Round trip: every entry must read back exactly as it was computed.
loaded = include(OUTFILE)
length(loaded) == length(entries) || error("round trip: $(length(loaded)) entries read back, $(length(entries)) written")
for (a, b) in zip(loaded, entries)
    isequal(a, b) || error("round trip failed for case $(b.name)")
    b.throws === nothing && (R.mismatch(R.decode(a.expected), R.decode(b.expected)) === nothing ||
                             error("round trip changed case $(b.name)"))
end

by_builder = R.case_counts(entries)
println("Wrote $(length(entries)) cases to $OUTFILE ($(filesize(OUTFILE)) bytes) in $(round(time() - t_start; digits=1)) s")
for builder in sort!(collect(keys(by_builder)))
    println("    ", rpad(repr(builder), 26), "=> ", by_builder[builder], ",")
end
if by_builder != R.EXPECTED_CASE_COUNTS
    @warn "The case counts differ from EXPECTED_CASE_COUNTS in test/golden_lattice.jl. " *
          "If the change is intended, update that table in the same commit." got = by_builder expected = R.EXPECTED_CASE_COUNTS
end
