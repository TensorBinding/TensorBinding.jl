# Generator for test/data/sampling_golden.jl, the pinned ("golden") outputs of
# every sampling planner, checked by test/sampling_golden.jl.
#
# Reference: the golden data was first generated from commit c87275a
# ("Add a keyword copy constructor TBHamiltonian(H; field=value, ...)"), before
# the code-organisation refactor (docs/dev/REORGANISATION_TODO.md). It was
# regenerated on top of 1ffb7aa (branch Anouar, after the 2026-09 bug round):
# every case was byte-identical except the ten `_exciton_block_groups` cases,
# which were dropped because 552ff3f removed that unreachable helper. The data
# pins what the planners do after the bug round, remaining bugs included.
#
# Line 4 of the data file records the git tree hash of the working-tree src/,
# i.e. the src/ that the regenerating commit will contain. It does not move when
# only tests or docs change, so such a rerun reproduces the file byte for byte.
#
# Regenerate ONLY when a behaviour change is intentional. Do it in the same
# commit as the change, review the diff of sampling_golden.jl case by case, and
# record the change in the changelog. A refactor that is meant to leave outputs
# alone must pass test/sampling_golden.jl against the existing data unchanged.
#
# How to run, from the repository root with the package environment:
#
#     JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. test/data/generate_sampling_golden.jl
#
# The output is deterministic: no timestamps, cases in a fixed order, values
# written with `repr`. The script reloads the file it wrote and checks that
# every entry round-trips exactly.
#
# Case names say where each parameter set comes from:
#   manuscript_*, notebook_*, test_*  parameter sets used by tracked scripts,
#                                     notebooks and tests (the values the entry
#                                     point forwards to the planner);
#   hypothetical_*                    a tracked parameter set run through a
#                                     planner its script does not use today;
#   all other names                   a systematic grid over every branch.

using TensorBinding
const TB = TensorBinding

const SAMPLING_GOLDEN_GENERATOR = true
include(joinpath(@__DIR__, "..", "sampling_golden.jl"))   # loads SamplingGoldenRunner only

const OUTFILE = joinpath(@__DIR__, "sampling_golden.jl")

# Cases with more than DIGEST_CASE_INTS integers store each field holding more
# than DIGEST_FIELD_INTS integers as a digest (see SamplingGoldenRunner.digest).
const DIGEST_CASE_INTS  = 100_000
const DIGEST_FIELD_INTS = 10_000

const CASES = Tuple{String,Symbol,Tuple,NamedTuple}[]
add!(name, planner, args, kwargs=NamedTuple()) = push!(CASES, (name, planner, args, kwargs))
kw(; kwargs...) = values(kwargs)
wtag(xs, xe) = xs === nothing ? "wdefault" : "w$(xs)-$(xe)"
window_kw(xs, xe) = xs === nothing ? NamedTuple() : (; x_start = xs, x_end = xe)

# ═════════════════════════════════════════════════════════════════════════════
# (a) Parameter sets used by tracked scripts, notebooks and tests
# ═════════════════════════════════════════════════════════════════════════════

# ── spatial_sampling_plan: nonequilibrium manuscript scripts ─────────────────
# get_nh_density_trajectory_gpu / get_state_amplitude_trajectory_gpu forward
# (Lbits; grid=false, reduce, num_x, num_avg, x_start, x_end, x_groups).
add!("manuscript_nhdens_L20_block_nx128", :spatial_sampling_plan, (20,),
     kw(grid=false, reduce=:block, num_x=128, num_avg=1, x_start=1, x_end=2^20, x_groups=nothing))
add!("manuscript_nhdens_noargs_L20_block_nx1024", :spatial_sampling_plan, (20,),
     kw(grid=false, reduce=:block, num_x=1024, num_avg=1, x_start=1, x_end=2^20, x_groups=nothing))
add!("manuscript_nhstate_small_L20_point_nx256_w524161-524416", :spatial_sampling_plan, (20,),
     kw(grid=false, reduce=:point, num_x=256, num_avg=1, x_start=524161, x_end=524416, x_groups=nothing))
add!("manuscript_nhstate_big_L20_point_nx128_navg5", :spatial_sampling_plan, (20,),
     kw(grid=false, reduce=:point, num_x=128, num_avg=5, x_start=1, x_end=2^20, x_groups=nothing))

# ── spatial_sampling_plan: single-particle LDOS map (get_ldos_spatial_gpu) ────
add!("manuscript_ldos_Lx12_Ly12_block_nx64_ny64_average", :spatial_sampling_plan, (24,),
     kw(Lx=12, grid=false, reduce=:block, n_sub=2, num_x=64, num_y=64, num_avg=1,
        x_start=1, x_end=2^24, xwin=nothing, ywin=nothing, x_groups=nothing,
        box_half=0, sublattice=:average))
# The script's no-ARGS defaults (Lx=Ly=4, num_x=num_y=32, block) throw today.
add!("manuscript_ldos_noargs_Lx4_Ly4_block_nx32_ny32", :spatial_sampling_plan, (8,),
     kw(Lx=4, grid=false, reduce=:block, n_sub=2, num_x=32, num_y=32, num_avg=1,
        x_start=1, x_end=2^8, xwin=nothing, ywin=nothing, x_groups=nothing,
        box_half=0, sublattice=:average))

# ── Hubbard SCF magnetization (get_scf_magnetization_gpu -> _tb_spatial_plan_gpu) ─
add!("manuscript_hubbard_mag_Lx12_Ly12_grid_nx70_ny70_bh2", :spatial_sampling_plan, (24,),
     kw(Lx=12, grid=true, reduce=:point, num_x=70, num_y=70, num_avg=1,
        x_start=1, x_end=2^24, x_groups=nothing, box_half=2))
# The same call through _tb_spatial_plan_gpu is pinned only at the no-ARGS size
# below (and in the tb_spatial_plan_gpu_* grid), to keep the data file small.
add!("manuscript_hubbard_mag_noargs_Lx3_grid_nx8_ny8", :spatial_sampling_plan, (6,),
     kw(Lx=3, grid=true, num_x=8, num_y=8, box_half=0, x_end=64))
add!("manuscript_hubbard_mag_noargs_Lx3_grid_nx8_ny8_via_tb_spatial_plan_gpu",
     :tb_spatial_plan_gpu, (6,),
     kw(num_x=8, num_y=8, num_avg=1, x_start=1, x_end=64, x_groups=nothing,
        box_half=0, reduce=:point, Lx=3))

# ── Exciton LDOS: the script builds X_groups inline with the planner's 1D point
#    formula; these are the planner calls it is element-for-element equal to. ──
add!("manuscript_exciton_ldos_L20_nx100_navg6", :spatial_sampling_plan, (20,),
     kw(num_x=100, num_avg=6, x_start=1, x_end=2^20))
add!("manuscript_exciton_ldos_sh_L10_nx64", :spatial_sampling_plan, (10,),
     kw(num_x=64, num_avg=1, x_start=1, x_end=2^10))
add!("manuscript_exciton_ldos_noargs_L5_nx32", :spatial_sampling_plan, (5,),
     kw(num_x=32, num_avg=1, x_start=1, x_end=32))

# ── Chern map: the script samples inline; this planner call gives the same centres.
#    (The .sh variant, Lx=Ly=14 with 70x70, has no tracked result and is left out
#    to keep the data file small.) ──
add!("manuscript_chern_Lx12_Ly12_grid_nx100_ny100", :spatial_sampling_plan, (24,),
     kw(Lx=12, grid=true, num_x=100, num_y=100))

# ── What get_ldos_spatial_mps_gpu's automatic plan would give for tracked
#    parameter sets (no tracked script calls it today). ──
add!("hypothetical_gpu_mps_auto_nhstate_big_N2^20_nx128_navg5", :gpu_mps_auto, (2^20,),
     kw(num_x=128, num_avg=5, x_start=1, x_end=2^20))
add!("hypothetical_gpu_mps_auto_exciton_ldos_N2^20_nx100_navg6", :gpu_mps_auto, (2^20,),
     kw(num_x=100, num_avg=6, x_start=1, x_end=2^20))
add!("hypothetical_gpu_mps_auto_exciton_ldos_sh_N1024_nx64", :gpu_mps_auto, (1024,),
     kw(num_x=64, num_avg=1, x_start=1, x_end=1024))
add!("hypothetical_gpu_mps_auto_nhstate_small_N2^20_nx256_w524161-524416", :gpu_mps_auto, (2^20,),
     kw(num_x=256, num_avg=1, x_start=524161, x_end=524416))

# ── Notebooks: get_ldos_spatial forwards the full keyword set ────────────────
ldos_kw(L, n_sub, num_x, x_end) =
    kw(Lx=L ÷ 2, grid=false, reduce=:point, n_sub=n_sub, num_x=num_x, num_y=nothing,
       num_avg=1, x_start=1, x_end=x_end, xwin=nothing, ywin=nothing, x_groups=nothing,
       box_half=0, sublattice=:auto)
add!("notebook_getting_started_chain_L6_ldos_nx64", :spatial_sampling_plan, (6,), ldos_kw(6, 1, 64, 64))
add!("notebook_aux_ldos_case0_chain_L10_nx8", :spatial_sampling_plan, (10,), ldos_kw(10, 1, 8, 1024))
add!("notebook_aux_ldos_case1_spinful_L5_nx8", :spatial_sampling_plan, (5,), ldos_kw(5, 1, 8, 32))
add!("notebook_aux_ldos_case2_bdg_L4_nx16", :spatial_sampling_plan, (4,), ldos_kw(4, 1, 16, 16))
add!("notebook_aux_ldos_case3_rashba_L5_nx16", :spatial_sampling_plan, (5,), ldos_kw(5, 1, 16, 32))
add!("notebook_aux_ldos_case4_honeycomb_L6_nsub2_nx64", :spatial_sampling_plan, (6,), ldos_kw(6, 2, 64, 64))
add!("notebook_aux_ldos_kagome_lieb_dice_L6_nsub3_nx64", :spatial_sampling_plan, (6,), ldos_kw(6, 3, 64, 64))
add!("notebook_tjunction_L4_nsub3_nx16", :spatial_sampling_plan, (4,), ldos_kw(4, 3, 16, 16))

# ── Tests: get_ldos_spatial calls in test/*.jl ───────────────────────────────
let H = TB.fibonacci_hamiltonian(4; A=1.0, B=2.0)
    perm = TB.site_permutation(H; ordering=:conumber)
    add!("test_fibonacci_L4_conumber_ldos_x_groups", :spatial_sampling_plan, (4,),
         merge(ldos_kw(4, 1, 8, H.N), (; x_groups = [[p] for p in perm])))
end
add!("test_metallic_mean_m2_L3_N17_ldos", :spatial_sampling_plan, (3,), ldos_kw(3, 1, 17, 17))
add!("test_kbonacci_k3_L5_N24_ldos", :spatial_sampling_plan, (5,), ldos_kw(5, 1, 24, 24))
add!("test_gpu_mps_ldos_cpu_ref_fibonacci_L4_x_groups", :spatial_sampling_plan, (4,),
     merge(ldos_kw(4, 1, 8, 8), (; x_groups = [[1, 2], [4], [7, 8]])))
add!("test_gpu_mps_ldos_cpu_ref_chain_L3_x_groups", :spatial_sampling_plan, (3,),
     merge(ldos_kw(3, 1, 8, 8), (; x_groups = [[1], [3, 4]])))
# get_ldos_spatial_mps_gpu group normalisation in test/gpu_mps_ldos.jl (N = 8)
add!("test_gpu_mps_ldos_groups_int_vector", :gpu_mps_auto, (8,), kw(x_groups=[1]))
add!("test_gpu_mps_ldos_groups_empty", :gpu_mps_auto, (8,), kw(x_groups=Vector{Vector{Int}}()))
add!("test_gpu_mps_ldos_groups_out_of_range", :gpu_mps_auto, (8,), kw(x_groups=[[0]]))
add!("test_gpu_mps_ldos_groups_fibonacci", :gpu_mps_auto, (8,), kw(x_groups=[[1, 2], [4], [7, 8]]))
add!("test_gpu_mps_ldos_groups_chain", :gpu_mps_auto, (8,), kw(x_groups=[[1], [3, 4]]))

# ── kspace_sampling_plan / kpath_setup: bands scripts, notebooks, tests ──────
kspace_kw(num_x; override=nothing) =
    kw(num_x=num_x, num_y=10, num_avg=1, xmin=0, xmax=nothing, ymin=0, ymax=nothing,
       k_groups_override=override)
const G_X_M_G  = [:G, :X, :M, :G]
const G_M_Kp_G = [:G, :M, :Kp, :G]
function add_kpath!(name, lattice, Lx, Ly, path, npts; L_pos=nothing)
    add!("$(name)_kpath_setup", :kpath_setup, (lattice, Lx, Ly, path), kw(npts_per_segment=npts))
    if L_pos !== nothing
        kg = TB.kpath_setup(lattice, Lx, Ly, path; npts_per_segment=npts)[1]
        add!("$(name)_kspace_override", :kspace_sampling_plan, (L_pos, 2),
             kspace_kw(npts; override=kg))
    end
end
add_kpath!("manuscript_hubbard_bands_Lx12_Ly12_GXMG_n50", :square, 12, 12, G_X_M_G, 50; L_pos=24)
add_kpath!("manuscript_hubbard_bands_noargs_Lx3_Ly3_GXMG_n32", :square, 3, 3, G_X_M_G, 32)
add_kpath!("manuscript_bands_Lx12_Ly12_GMKpG_n50", :honeycomb, 12, 12, G_M_Kp_G, 50; L_pos=23)
add_kpath!("manuscript_bands_sh_Lx14_Ly14_GMKpG_n50", :honeycomb, 14, 14, G_M_Kp_G, 50; L_pos=27)
add_kpath!("manuscript_bands_noargs_Lx4_Ly4_GMKpG_n50", :honeycomb, 4, 4, G_M_Kp_G, 50)
add_kpath!("notebook_momentum_honeycomb_kagome_dice_bilayer_Lx4_GMKpG_n20", :honeycomb, 4, 4, G_M_Kp_G, 20; L_pos=8)
add_kpath!("notebook_momentum_lieb_square_bilayer_Lx4_GXMG_n20", :square, 4, 4, G_X_M_G, 20; L_pos=8)
add!("notebook_getting_started_bands_L6_nx64", :kspace_sampling_plan, (6, 1), kspace_kw(64))
add!("notebook_momentum_chain_L5_nx32", :kspace_sampling_plan, (5, 1), kspace_kw(32))
add!("notebook_momentum_ssh_L6_nx64", :kspace_sampling_plan, (6, 1), kspace_kw(64))
add!("test_runtests_bands_chain_L4_nx4", :kspace_sampling_plan, (4, 1), kspace_kw(4))

# ── fibonacci_ldos_sampling_plan: test/fibonacci_sampling.jl ─────────────────
add!("test_fibonacci_sampling_L4_nx3_navg2", :fibonacci_ldos_sampling_plan, (4,), kw(num_x=3, num_avg=2))
add!("test_fibonacci_sampling_L10_depth2_nx3_navg3_uncentered", :fibonacci_ldos_sampling_plan, (10,),
     kw(depth=2, num_x=3, num_avg=3, centered=false))
add!("test_fibonacci_sampling_L4_depth0_nx100_navg100", :fibonacci_ldos_sampling_plan, (4,),
     kw(depth=0, num_x=100, num_avg=100))
for orientation in (:standard, :reversed)
    add!("test_fibonacci_sampling_L8_depth1_nx7_navg4_$(orientation)", :fibonacci_ldos_sampling_plan, (8,),
         kw(depth=1, num_x=7, num_avg=4, orientation=orientation, alignment=:atomic, centered=true, origin=0))
end
add!("test_fibonacci_sampling_L8_depth0_nx7_navg4", :fibonacci_ldos_sampling_plan, (8,), kw(depth=0, num_x=7, num_avg=4))
add!("test_fibonacci_sampling_L8_depth0_nx7_navg4_raw", :fibonacci_ldos_sampling_plan, (8,),
     kw(depth=0, num_x=7, num_avg=4, alignment=:raw))
add!("test_fibonacci_sampling_L8_depth0_nx7_navg4_origin3", :fibonacci_ldos_sampling_plan, (8,),
     kw(depth=0, num_x=7, num_avg=4, origin=3))
add!("test_fibonacci_sampling_L43_depth3_nx100_navg5", :fibonacci_ldos_sampling_plan, (43,),
     kw(depth=3, num_x=100, num_avg=5))
add!("test_fibonacci_sampling_L43_depth12_nx100_navg5", :fibonacci_ldos_sampling_plan, (43,),
     kw(depth=12, num_x=100, num_avg=5))
# The TBHamiltonian overload only checks the position space and forwards H.L;
# test/fibonacci_sampling.jl already asserts that it equals this L=2 plan and
# that a binary chain throws. It is not repeated here because building the
# Hamiltonians would add ~20 s of compilation to a standalone run.
add!("test_fibonacci_sampling_L2_nx2_navg1", :fibonacci_ldos_sampling_plan, (2,), kw(num_x=2, num_avg=1))
for (tag, kwargs) in (("nx0", kw(num_x=0)), ("navg0", kw(num_avg=0)),
                      ("orientation_sideways", kw(orientation=:sideways)),
                      ("alignment_molecular", kw(alignment=:molecular)),
                      ("depth1", kw(depth=1)))
    add!("test_fibonacci_sampling_L4_error_$(tag)", :fibonacci_ldos_sampling_plan, (4,), kwargs)
end
add!("test_fibonacci_sampling_L8_error_depth1_raw", :fibonacci_ldos_sampling_plan, (8,), kw(depth=1, alignment=:raw))
add!("test_fibonacci_sampling_L8_error_depth1_origin1", :fibonacci_ldos_sampling_plan, (8,), kw(depth=1, origin=1))

# ═════════════════════════════════════════════════════════════════════════════
# (b) Systematic grid over every branch
# ═════════════════════════════════════════════════════════════════════════════

# ── spatial_sampling_plan, 1D :point ─────────────────────────────────────────
# Windows: the default, one whose length (2^L - 2) is not a multiple of most
# num_x, a short one (length 3, so num_x > window gives empty groups today) and a
# tail window of length 5.
for L in (3, 4, 5)
    N = 2^L
    for (xs, xe) in ((nothing, nothing), (2, N - 1), (3, 5), (N - 4, N))
        for num_x in (0, 1, 3, 4, 5, 7, 8, 16), num_avg in (1, 2, 3, 4)
            add!("point1d_L$(L)_nx$(num_x)_navg$(num_avg)_$(wtag(xs, xe))", :spatial_sampling_plan, (L,),
                 merge(kw(num_x=num_x, num_avg=num_avg), window_kw(xs, xe)))
        end
    end
end
# Degenerate 1D windows and counts
add!("point1d_L3_nx0_empty_window_5-4", :spatial_sampling_plan, (3,), kw(num_x=0, x_start=5, x_end=4))
add!("point1d_L3_nx3_empty_window_5-4", :spatial_sampling_plan, (3,), kw(num_x=3, x_start=5, x_end=4))
add!("point1d_L3_nx-2", :spatial_sampling_plan, (3,), kw(num_x=-2))
add!("point1d_L3_nx4_navg0", :spatial_sampling_plan, (3,), kw(num_x=4, num_avg=0))
add!("point1d_L3_nx4_navg-1", :spatial_sampling_plan, (3,), kw(num_x=4, num_avg=-1))
add!("point1d_L3_nx4_x_end_beyond_2^L", :spatial_sampling_plan, (3,), kw(num_x=4, x_end=12))
add!("point1d_L3_nx4_x_start0", :spatial_sampling_plan, (3,), kw(num_x=4, x_start=0))
add!("point1d_L3_nx4_navg8", :spatial_sampling_plan, (3,), kw(num_x=4, num_avg=8))

# ── 1D :point with box_half (applies only when Lx is given and num_avg <= 1) ──
for box_half in (1, 2), num_x in (4, 16), num_avg in (1, 2)
    add!("point1d_box_L4_Lx2_bh$(box_half)_nx$(num_x)_navg$(num_avg)", :spatial_sampling_plan, (4,),
         kw(Lx=2, num_x=num_x, num_avg=num_avg, box_half=box_half))
end
add!("point1d_box_L5_Lx2_bh1_nx8_w3-30", :spatial_sampling_plan, (5,),
     kw(Lx=2, num_x=8, box_half=1, x_start=3, x_end=30))
add!("point1d_box_L5_Lx3_bh1_nx5", :spatial_sampling_plan, (5,), kw(Lx=3, num_x=5, box_half=1))
add!("point1d_box_L4_noLx_bh1_nx4", :spatial_sampling_plan, (4,), kw(num_x=4, box_half=1))

# ── 2D :point grid ───────────────────────────────────────────────────────────
# (L, Lx, xwin, ywin): the windows are 0-indexed unit-cell ranges inside the system.
const GRID_GEOMS = ((4, 2, (1, 3), (0, 2)),
                    (5, 2, (1, 2), (2, 6)),
                    (6, 3, (2, 6), (1, 7)))
nytag(ny) = ny === nothing ? "nynothing" : "ny$(ny)"
for (L, Lx, xwin, ywin) in GRID_GEOMS
    for windowed in (false, true)
        for num_x in (0, 1, 3, 4), num_y in (nothing, 0, 2, 5), num_avg in (1, 3)
            wkw = windowed ? (; xwin, ywin) : NamedTuple()
            add!("grid2d_L$(L)_Lx$(Lx)_nx$(num_x)_$(nytag(num_y))_navg$(num_avg)" *
                 (windowed ? "_xwin$(xwin[1])-$(xwin[2])_ywin$(ywin[1])-$(ywin[2])" : ""),
                 :spatial_sampling_plan, (L,),
                 merge(kw(Lx=Lx, grid=true, num_x=num_x, num_y=num_y, num_avg=num_avg), wkw))
        end
    end
end
for num_x in (0, 3), num_y in (nothing, 2), num_avg in (1, 2)
    add!("grid2d_L5_Lx2_nx$(num_x)_$(nytag(num_y))_navg$(num_avg)_xwin1-2_only", :spatial_sampling_plan, (5,),
         kw(Lx=2, grid=true, num_x=num_x, num_y=num_y, num_avg=num_avg, xwin=(1, 2)))
    add!("grid2d_L5_Lx2_nx$(num_x)_$(nytag(num_y))_navg$(num_avg)_ywin2-6_only", :spatial_sampling_plan, (5,),
         kw(Lx=2, grid=true, num_x=num_x, num_y=num_y, num_avg=num_avg, ywin=(2, 6)))
end
add!("grid2d_L4_Lx2_nx2_xwin_beyond_system_2-5", :spatial_sampling_plan, (4,),
     kw(Lx=2, grid=true, num_x=2, xwin=(2, 5)))
add!("grid2d_L4_Lx2_nx2_navg2_xwin_beyond_system_2-5", :spatial_sampling_plan, (4,),
     kw(Lx=2, grid=true, num_x=2, num_avg=2, xwin=(2, 5)))
add!("grid2d_L4_Lx2_nx2_inverted_xwin_3-1", :spatial_sampling_plan, (4,),
     kw(Lx=2, grid=true, num_x=2, xwin=(3, 1)))
add!("grid2d_L4_Lx0_nx0", :spatial_sampling_plan, (4,), kw(Lx=0, grid=true, num_x=0))
add!("grid2d_L4_Lx4_nx0", :spatial_sampling_plan, (4,), kw(Lx=4, grid=true, num_x=0))
add!("grid2d_L4_noLx_error", :spatial_sampling_plan, (4,), kw(grid=true, num_x=2))

# ── 2D :point grid with box_half ─────────────────────────────────────────────
for (L, Lx, xwin, ywin) in GRID_GEOMS[1:2], box_half in (1, 2), num_x in (2, 4),
    num_avg in (1, 2), windowed in (false, true)
    wkw = windowed ? (; xwin, ywin) : NamedTuple()
    add!("grid2d_box_L$(L)_Lx$(Lx)_bh$(box_half)_nx$(num_x)_navg$(num_avg)" * (windowed ? "_windowed" : ""),
         :spatial_sampling_plan, (L,),
         merge(kw(Lx=Lx, grid=true, num_x=num_x, num_avg=num_avg, box_half=box_half), wkw))
end

# ── reduce=:block, 1D (no Lx) ────────────────────────────────────────────────
for L in (3, 4, 5), num_x in (1, 2, 4, 8, 16, 32, 0, 3, 6, -4)
    add!("block1d_L$(L)_nx$(num_x)", :spatial_sampling_plan, (L,), kw(reduce=:block, num_x=num_x))
end
add!("block1d_L4_nx4_nsub3", :spatial_sampling_plan, (4,), kw(reduce=:block, num_x=4, n_sub=3))
add!("block1d_L4_nx4_nsub2_resolve", :spatial_sampling_plan, (4,),
     kw(reduce=:block, num_x=4, n_sub=2, sublattice=:resolve))
add!("block1d_L4_nx4_nsub0", :spatial_sampling_plan, (4,), kw(reduce=:block, num_x=4, n_sub=0))
add!("block1d_L4_nx4_ignores_window_navg_groups", :spatial_sampling_plan, (4,),
     kw(reduce=:block, num_x=4, x_start=5, x_end=9, num_avg=3, x_groups=[1, 2]))

# ── reduce=:block, 2D (Lx given) ─────────────────────────────────────────────
for (L, Lx) in ((4, 2), (5, 2), (5, 3)), num_x in (1, 2, 4, 8, 3, 0),
    num_y in (nothing, 1, 2, 4, 8, 3, 0)
    add!("block2d_L$(L)_Lx$(Lx)_nx$(num_x)_$(nytag(num_y))", :spatial_sampling_plan, (L,),
         kw(Lx=Lx, reduce=:block, num_x=num_x, num_y=num_y))
end
for sublattice in (:auto, :resolve, :average), n_sub in (1, 2)
    add!("block2d_L4_Lx2_nx2_ny2_nsub$(n_sub)_$(sublattice)", :spatial_sampling_plan, (4,),
         kw(Lx=2, reduce=:block, num_x=2, num_y=2, n_sub=n_sub, sublattice=sublattice))
end
add!("block2d_L4_Lx2_nx2_ny2_nsub0", :spatial_sampling_plan, (4,), kw(Lx=2, reduce=:block, num_x=2, num_y=2, n_sub=0))
add!("block2d_L4_Lx2_nx2_ny2_grid_true", :spatial_sampling_plan, (4,), kw(Lx=2, grid=true, reduce=:block, num_x=2, num_y=2))
add!("block2d_L4_Lx2_nx2_ny2_ignores_box_window", :spatial_sampling_plan, (4,),
     kw(Lx=2, reduce=:block, num_x=2, num_y=2, box_half=1, xwin=(1, 2), num_avg=2))
add!("block2d_L4_Lx0_nx1", :spatial_sampling_plan, (4,), kw(Lx=0, reduce=:block, num_x=1, num_y=4))

# ── x_groups override ────────────────────────────────────────────────────────
const XGROUPS = (("ints", [3, 1, 7]),
                 ("vectors", [[1, 2], [5], [8, 6, 7]]),
                 ("ranges", [1:2, 4:4]),
                 ("empty_ints", Int[]),
                 ("empty_vectors", Vector{Int}[]),
                 ("duplicates", [[1, 1, 2]]),
                 ("unsorted_group", [[5, 3]]),
                 ("tuple", (2, 4)),
                 ("out_of_range", [0, 17]))
for (tag, groups) in XGROUPS
    add!("xgroups_L4_$(tag)", :spatial_sampling_plan, (4,), kw(x_groups=groups))
    add!("xgroups_L4_$(tag)_Lx2_bh1", :spatial_sampling_plan, (4,), kw(Lx=2, x_groups=groups, box_half=1))
end
add!("xgroups_L4_vectors_Lx2_bh1_navg2", :spatial_sampling_plan, (4,),
     kw(Lx=2, x_groups=[[1, 2], [5]], box_half=1, num_avg=2))
add!("xgroups_L4_vectors_grid_true", :spatial_sampling_plan, (4,),
     kw(Lx=2, grid=true, num_x=2, x_groups=[[1, 2], [5]]))
add!("xgroups_L4_vectors_ignores_window", :spatial_sampling_plan, (4,),
     kw(x_groups=[[1, 2], [5]], num_x=3, num_avg=2, x_start=4, x_end=9))
for sublattice in (:auto, :resolve, :average), box_half in (0, 1)
    add!("xgroups_L4_nsub2_$(sublattice)_bh$(box_half)", :spatial_sampling_plan, (4,),
         kw(Lx=2, n_sub=2, x_groups=[1, 2], box_half=box_half, sublattice=sublattice))
end

# ── n_sub > 1 with :point (sublattice resolve / average decision) ────────────
const NSUB_LAYOUTS = (("1d_full", kw(num_x=0)),
                      ("1d_nx8", kw(num_x=8)),
                      ("1d_full_bh1", kw(Lx=2, num_x=0, box_half=1)),
                      ("grid_full", kw(Lx=2, grid=true, num_x=0)),
                      ("grid_nx2", kw(Lx=2, grid=true, num_x=2)),
                      ("grid_nx4_ny2", kw(Lx=2, grid=true, num_x=4, num_y=2)),
                      ("grid_full_navg2", kw(Lx=2, grid=true, num_x=0, num_avg=2)),
                      ("xgroups", kw(x_groups=[1, 2])))
for n_sub in (2, 3), sublattice in (:auto, :resolve, :average), (tag, layout) in NSUB_LAYOUTS
    add!("nsub$(n_sub)_L4_$(tag)_$(sublattice)", :spatial_sampling_plan, (4,),
         merge(layout, kw(n_sub=n_sub, sublattice=sublattice)))
end
add!("nsub1_L4_1d_full_resolve", :spatial_sampling_plan, (4,), kw(num_x=0, n_sub=1, sublattice=:resolve))
add!("nsub0_L4_1d_full", :spatial_sampling_plan, (4,), kw(num_x=0, n_sub=0))
add!("error_L4_sublattice_bogus", :spatial_sampling_plan, (4,), kw(num_x=4, sublattice=:bogus))
add!("error_L4_reduce_bogus", :spatial_sampling_plan, (4,), kw(num_x=4, reduce=:bogus))
add!("defaults_L3", :spatial_sampling_plan, (3,))

# ── eval_mps_spatial: the planner calls it makes (src/core/Utils.jl) ─────────
# eval_mps_spatial itself is not called: evaluating even a 4-qubit MPS adds
# ~8 s of ITensor compilation to a standalone run. It forwards
# (L; Lx = box_half > 0 && Lx === nothing ? L ÷ 2 : Lx, num_x, num_avg,
#  x_start, x_end, x_groups, box_half) with num_x and x_end defaulting to 2^L.
emps_kw(L; num_x=2^L, num_avg=1, x_start=1, x_end=2^L, x_groups=nothing, box_half=0, Lx=nothing) =
    kw(Lx=(box_half > 0 && Lx === nothing) ? L ÷ 2 : Lx, num_x=num_x, num_avg=num_avg,
       x_start=x_start, x_end=x_end, x_groups=x_groups, box_half=box_half)
add!("eval_mps_spatial_forwarded_L4_defaults", :spatial_sampling_plan, (4,), emps_kw(4))
add!("eval_mps_spatial_forwarded_L4_nx4_navg2", :spatial_sampling_plan, (4,), emps_kw(4; num_x=4, num_avg=2))
add!("eval_mps_spatial_forwarded_L4_nx3_w3-10", :spatial_sampling_plan, (4,), emps_kw(4; num_x=3, x_start=3, x_end=10))
add!("eval_mps_spatial_forwarded_L4_x_groups", :spatial_sampling_plan, (4,), emps_kw(4; x_groups=[[1, 2], [9]]))
add!("eval_mps_spatial_forwarded_L4_nx4_bh1_default_Lx", :spatial_sampling_plan, (4,), emps_kw(4; num_x=4, box_half=1))
add!("eval_mps_spatial_forwarded_L4_nx4_bh1_Lx1", :spatial_sampling_plan, (4,), emps_kw(4; num_x=4, box_half=1, Lx=1))

# ── _tb_spatial_plan_gpu (grid = x_groups === nothing, Lx defaults to L ÷ 2) ─
add!("tb_spatial_plan_gpu_L4_defaults", :tb_spatial_plan_gpu, (4,))
add!("tb_spatial_plan_gpu_L4_nx2", :tb_spatial_plan_gpu, (4,), kw(num_x=2))
add!("tb_spatial_plan_gpu_L4_nx2_ny4", :tb_spatial_plan_gpu, (4,), kw(num_x=2, num_y=4))
add!("tb_spatial_plan_gpu_L4_x_groups", :tb_spatial_plan_gpu, (4,), kw(x_groups=[1, 5]))
add!("tb_spatial_plan_gpu_L4_block_nx2_ny2", :tb_spatial_plan_gpu, (4,), kw(reduce=:block, num_x=2, num_y=2))
add!("tb_spatial_plan_gpu_L4_nx2_bh1", :tb_spatial_plan_gpu, (4,), kw(num_x=2, box_half=1))
add!("tb_spatial_plan_gpu_L4_nx2_Lx1", :tb_spatial_plan_gpu, (4,), kw(num_x=2, Lx=1))
add!("tb_spatial_plan_gpu_L4_nx2_navg2", :tb_spatial_plan_gpu, (4,), kw(num_x=2, num_avg=2))
add!("tb_spatial_plan_gpu_L5_defaults", :tb_spatial_plan_gpu, (5,))
# Odd L with a sub-sampled grid, so that the default Lx = div(L, 2) (and not
# cld(L, 2)) is pinned; the full-grid defaults above cannot tell them apart.
add!("tb_spatial_plan_gpu_L5_nx2", :tb_spatial_plan_gpu, (5,), kw(num_x=2))
add!("tb_spatial_plan_gpu_L5_nx2_ny4", :tb_spatial_plan_gpu, (5,), kw(num_x=2, num_y=4))
add!("tb_spatial_plan_gpu_L5_block_nx2_ny2", :tb_spatial_plan_gpu, (5,), kw(reduce=:block, num_x=2, num_y=2))
add!("tb_spatial_plan_gpu_L7_nx4_bh1", :tb_spatial_plan_gpu, (7,), kw(num_x=4, box_half=1))

# ── get_ldos_spatial_mps_gpu automatic plan (copied inline code) ─────────────
for N in (8, 16, 17, 24)
    for (xs, xe) in ((nothing, nothing), (3, N - 2))
        for num_x in (1, 3, 4, 5, 7, 8, 16, 17), num_avg in (1, 2, 3, 4)
            add!("gpu_mps_auto_N$(N)_nx$(num_x)_navg$(num_avg)_$(wtag(xs, xe))", :gpu_mps_auto, (N,),
                 merge(kw(num_x=num_x, num_avg=num_avg), window_kw(xs, xe)))
        end
    end
end
add!("gpu_mps_auto_N8_defaults", :gpu_mps_auto, (8,))
add!("gpu_mps_auto_N150_defaults", :gpu_mps_auto, (150,))
add!("gpu_mps_auto_N16_nx4_navg16", :gpu_mps_auto, (16,), kw(num_x=4, num_avg=16))
for (tag, kwargs) in (("nx0", kw(num_x=0)), ("navg0", kw(num_avg=0)),
                      ("x_start0", kw(num_x=4, x_start=0)), ("x_end_beyond_N", kw(num_x=4, x_end=17)),
                      ("x_start_after_x_end", kw(num_x=1, x_start=6, x_end=5)))
    add!("gpu_mps_auto_N16_error_$(tag)", :gpu_mps_auto, (16,), kwargs)
end
for (tag, groups) in (("ints", [3, 1]), ("vectors", [[1, 2], [5]]), ("ranges", [1:2, 4:4]),
                      ("empty_group", [Int[]]), ("beyond_N", [17]))
    add!("gpu_mps_auto_N16_x_groups_$(tag)", :gpu_mps_auto, (16,), kw(x_groups=groups))
end

# ── fibonacci_ldos_sampling_plan ─────────────────────────────────────────────
# Every convention combination at three (num_x, num_avg) settings; depth=1 with
# alignment=:raw is pinned as throwing.
for L in (5, 6, 8), depth in (0, 1), orientation in (:standard, :reversed),
    alignment in (:atomic, :raw), centered in (true, false),
    (num_x, num_avg) in ((3, 1), (3, 5), (7, 2))
    add!("fibonacci_L$(L)_depth$(depth)_$(orientation)_$(alignment)_" *
         (centered ? "centered" : "uncentered") * "_nx$(num_x)_navg$(num_avg)",
         :fibonacci_ldos_sampling_plan, (L,),
         kw(depth=depth, num_x=num_x, num_avg=num_avg, orientation=orientation,
            alignment=alignment, centered=centered))
end
for L in (5, 6, 8), depth in (0, 1), num_avg in (1, 3)
    add!("fibonacci_L$(L)_depth$(depth)_nx100_navg$(num_avg)", :fibonacci_ldos_sampling_plan, (L,),
         kw(depth=depth, num_x=100, num_avg=num_avg))
end
for orientation in (:standard, :reversed), alignment in (:atomic, :raw)
    add!("fibonacci_L6_depth0_$(orientation)_$(alignment)_origin2_nx3_navg3", :fibonacci_ldos_sampling_plan, (6,),
         kw(depth=0, num_x=3, num_avg=3, orientation=orientation, alignment=alignment, origin=2))
end
add!("fibonacci_L8_depth2", :fibonacci_ldos_sampling_plan, (8,), kw(depth=2, num_x=3, num_avg=2))
add!("fibonacci_L8_depth3_error", :fibonacci_ldos_sampling_plan, (8,), kw(depth=3, num_x=3))
add!("fibonacci_L5_defaults", :fibonacci_ldos_sampling_plan, (5,))

# ── kspace_sampling_plan ─────────────────────────────────────────────────────
for L_pos in (3, 4), (xmin, xmax) in ((0, nothing), (2, 9)), num_x in (1, 2, 3, 5, 8, 16, 17),
    num_avg in (1, 2, 3, 4)
    add!("kspace1d_L$(L_pos)_nx$(num_x)_navg$(num_avg)_x$(xmin)-$(something(xmax, "max"))",
         :kspace_sampling_plan, (L_pos, 1), kw(num_x=num_x, num_avg=num_avg, xmin=xmin, xmax=xmax))
end
add!("kspace1d_L3_nx0_error", :kspace_sampling_plan, (3, 1), kw(num_x=0))
for L_pos in (4, 5), num_x in (1, 2, 3, 4, 8), num_y in (1, 10), num_avg in (1, 3),
    (xmin, xmax, ymin, ymax) in ((0, nothing, 0, nothing), (0, 2, 0, 2), (1, nothing, 1, nothing))
    add!("kspace2d_L$(L_pos)_nx$(num_x)_ny$(num_y)_navg$(num_avg)_x$(xmin)-$(something(xmax, "max"))" *
         "_y$(ymin)-$(something(ymax, "max"))",
         :kspace_sampling_plan, (L_pos, 2),
         kw(num_x=num_x, num_y=num_y, num_avg=num_avg, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))
end
add!("kspace_D3_error", :kspace_sampling_plan, (4, 3), kw(num_x=2))
add!("kspace1d_L4_override", :kspace_sampling_plan, (4, 1), kw(num_x=3, k_groups_override=[[0], [5, 6]]))
add!("kspace2d_L4_override_empty", :kspace_sampling_plan, (4, 2), kw(num_x=3, k_groups_override=Vector{Int}[]))

# ── ilinspace ────────────────────────────────────────────────────────────────
for (xmin, xmax, num_x) in ((0, 15, 1), (0, 15, 2), (0, 15, 4), (0, 15, 5), (0, 15, 15), (0, 15, 16),
                            (0, 15, 17), (0, 15, 0), (3, 10, 1), (3, 10, 3), (3, 10, 8), (3, 10, 9),
                            (-2, 2, 3), (0, 0, 1), (5, 5, 1), (0, 1, 2), (0, 6, 4), (0, 7, 3))
    add!("ilinspace_$(xmin)_$(xmax)_$(num_x)", :ilinspace, (xmin, xmax, num_x))
end

# ── kpath_setup (kpath_2d + hsk_* high-symmetry points) ──────────────────────
for (lattice, path) in ((:honeycomb, G_M_Kp_G), (:honeycomb, [:G, :K, :M, :G]),
                        (:square, G_X_M_G), (:triangular, [:G, :M, :K, :G])),
    (Lx, Ly) in ((3, 3), (4, 4), (4, 3), (2, 5)), npts in (5, 20)
    add!("kpath_$(lattice)_$(join(path))_Lx$(Lx)_Ly$(Ly)_n$(npts)", :kpath_setup, (lattice, Lx, Ly, path),
         kw(npts_per_segment=npts))
end
add!("kpath_square_single_point_G", :kpath_setup, (:square, 3, 3, [:G]), kw(npts_per_segment=5))
add!("kpath_square_GXMG_n1", :kpath_setup, (:square, 3, 3, G_X_M_G), kw(npts_per_segment=1))
add!("kpath_unknown_lattice_error", :kpath_setup, (:kagome, 3, 3, [:G, :M]), kw(npts_per_segment=5))

# ═════════════════════════════════════════════════════════════════════════════
# Evaluate, write, and verify the round trip
# ═════════════════════════════════════════════════════════════════════════════

count_ints(::Integer)       = 1
count_ints(::AbstractRange) = 2
count_ints(x::AbstractVector) = sum(count_ints, x; init=0)
count_ints(x::Union{Tuple,NamedTuple}) = sum(count_ints, values(x); init=0)
count_ints(_) = 0

function compact(expected::NamedTuple)
    count_ints(expected) > DIGEST_CASE_INTS || return expected, Symbol[]
    digested = Symbol[]
    vals = map(keys(expected)) do field
        v = expected[field]
        if v isa AbstractVector && !(v isa AbstractRange) && count_ints(v) > DIGEST_FIELD_INTS
            push!(digested, field)
            return SamplingGoldenRunner.digest(v)
        end
        return v
    end
    return NamedTuple{keys(expected)}(vals), digested
end

# An exception of one of these types means the case itself is malformed (a
# misspelt keyword, a wrong argument type), not a behaviour worth pinning.
const GENERATOR_BUGS = (MethodError, UndefVarError, UndefKeywordError, TypeError)

function evaluate(cases)
    names = first.(cases)
    allunique(names) || error("duplicate case names: $(unique(filter(n -> count(==(n), names) > 1, names)))")
    entries = NamedTuple[]
    notes = String[]
    for (name, planner, args, kwargs) in cases
        result, throws, prefix = try
            (SamplingGoldenRunner.run_case(planner, args, kwargs), nothing, nothing)
        catch err
            any(T -> err isa T, GENERATOR_BUGS) && rethrow()
            push!(notes, "throws $(typeof(err)): $name -- $(first(sprint(showerror, err), 120))")
            (nothing, typeof(err), SamplingGoldenRunner.message_prefix(err))
        end
        if result === nothing
            expected = (; message_prefix = prefix)
        else
            expected, digested = compact(result)
            isempty(digested) || push!(notes, "digested $(join(digested, ", ")): $name")
        end
        push!(entries, (; name, planner, args, throws, kwargs, expected))
    end
    return entries, notes
end

# The git tree hash of the working-tree src/, computed through a throwaway index.
# Data are regenerated in the same commit as the behaviour change, so this is the
# src/ of that commit; a commit id could not be recorded inside its own commit.
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

function write_golden(path, entries, notes)
    source = src_info()
    open(path, "w") do io
        println(io, "# AUTO-GENERATED by test/data/generate_sampling_golden.jl -- do not edit by hand.")
        println(io, "#")
        println(io, "# Pinned outputs of the sampling planners, checked by test/sampling_golden.jl.")
        println(io, "# Generated from $source")
        println(io, "# with Julia $VERSION. Regenerate only for an intentional behaviour change;")
        println(io, "# see the generator's header.")
        println(io, "#")
        println(io, "# Each entry is (name, planner, args, throws, kwargs, expected): `planner`")
        println(io, "# selects a call in SamplingGoldenRunner.run_case, `throws` is the exception")
        println(io, "# type a failing case must still raise (otherwise `nothing`), and `expected`")
        println(io, "# holds every field the call returned or, for a throwing case, the first")
        println(io, "# $(SamplingGoldenRunner.MESSAGE_PREFIX_CHARS) characters of its message (`message_prefix`, `nothing` when")
        println(io, "# the exception has no message).")
        println(io, "#")
        println(io, "# Cases with more than $DIGEST_CASE_INTS integers store each field holding more")
        println(io, "# than $DIGEST_FIELD_INTS integers as a digest instead (length, first/last")
        println(io, "# $(SamplingGoldenRunner.DIGEST_EDGE) entries, sum and position-weighted sum):")
        digest_notes = filter(startswith("digested"), notes)
        isempty(digest_notes) && println(io, "#   (none)")
        for note in digest_notes
            println(io, "#   ", note)
        end
        println(io, "#")
        println(io, "# $(length(entries)) cases, $(count(e -> e.throws !== nothing, entries)) of them pinned as throwing.")
        println(io)
        # Vector{Any}: Base's push!(::Vector{Any}, @nospecialize x) is compiled
        # once, not once per entry type, which roughly halves the load time.
        println(io, "SAMPLING_GOLDEN_CASES = Any[]")
        for e in entries
            println(io)
            println(io, "push!(SAMPLING_GOLDEN_CASES, (name = ", repr(e.name), ", planner = ", repr(e.planner),
                    ", args = ", repr(e.args), ", throws = ", repr(e.throws), ",")
            println(io, "    kwargs = ", repr(e.kwargs), ",")
            println(io, "    expected = ", repr(e.expected), "))")
        end
        println(io)
        println(io, "SAMPLING_GOLDEN_CASES")
    end
end

# The :gpu_mps_auto cases run a copy of get_ldos_spatial_mps_gpu's inline plan;
# refuse to pin a copy that no longer matches the source.
SamplingGoldenRunner.check_inline_plan_copy() ||
    error("the copy of get_ldos_spatial_mps_gpu's inline plan in test/sampling_golden.jl is stale; update it first")

entries, notes = evaluate(CASES)
foreach(println, notes)
write_golden(OUTFILE, entries, notes)

# Round trip: every entry must read back exactly as it was computed, with the
# same leaf element types that test/sampling_golden.jl compares.
loaded = include(OUTFILE)
length(loaded) == length(entries) || error("round trip: $(length(loaded)) entries read back, $(length(entries)) written")
for (a, b) in zip(loaded, entries)
    a == b || error("round trip failed for case $(b.name)")
    b.throws === nothing || continue
    for field in keys(b.expected)
        SamplingGoldenRunner.is_digest(b.expected[field]) && continue
        SamplingGoldenRunner.leaftype(a.expected[field]) == SamplingGoldenRunner.leaftype(b.expected[field]) ||
            error("round trip changed the element type of $(b.name).$field: " *
                  "$(SamplingGoldenRunner.leaftype(a.expected[field])) != $(SamplingGoldenRunner.leaftype(b.expected[field]))")
    end
end

by_planner = SamplingGoldenRunner.case_counts(entries)
println("Wrote $(length(entries)) cases to $OUTFILE ($(filesize(OUTFILE)) bytes)")
for planner in sort!(collect(keys(by_planner)))
    println("  ", rpad(planner, 34), by_planner[planner])
end
if by_planner != SamplingGoldenRunner.EXPECTED_CASE_COUNTS
    @warn "The case counts differ from EXPECTED_CASE_COUNTS in test/sampling_golden.jl. " *
          "If the change is intended, update that table in the same commit." got = by_planner expected = SamplingGoldenRunner.EXPECTED_CASE_COUNTS
end
