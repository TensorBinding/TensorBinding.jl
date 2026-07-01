#!/usr/bin/env julia
# APSOS_nhdos_gpu.jl
#
# GPU deterministic diagonal-trace non-Hermitian DOS / spectral-weight grid for
# an open AAH chain with spatially modulated loss. This is the production-script
# version of the local_testing_4 notebook panel (a).
#
# Output (OUTDIR):
#   NHDOS_tileXXXX_ofYYYY_<tag>.csv -- columns: ix, iy, Re(z), Im(z), weight
#   xgrid_<tag>.csv                 -- sampled Re(z) values
#   ygrid_<tag>.csv                 -- sampled Im(z) values
#
# If num_tiles=1, also writes the legacy full-grid files:
#   NHDOS_<tag>.csv          -- columns: Re(z), Im(z), weight
#   NHDOS_matrix_<tag>.csv   -- matrix with rows Im(z), columns Re(z)
#
# Usage:
#   julia --project=@. APSOS_nhdos_gpu.jl \
#     L t gamma0 loss_b loss_phase scale Nh maxdim nx ny \
#     xmin xmax ymin ymax legacy_n_random legacy_seed cutoff OUTDIR tile_id num_tiles \
#     [aah_V aah_phi aah_b loss_harmonics nh_scale_pad nh_bound_check_stride]
#
# Use scale=auto, none, nothing, dmrg, 0, or 0.0 to let TensorBinding choose
# a universal NH KPM scale from the zero-shift hermitized norm plus all grid z.
# A positive numeric scale keeps explicit-scale behavior.
#
# tile_id is 0-based. Defaults are intentionally small enough for a quick GPU
# smoke test, but the Slurm script below sets production-oriented values.

using CUDA   # must be first -- enables the NDTensors GPU backend
using Dates, LinearAlgebra, DelimitedFiles
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "../../../src/TensorBinding.jl"))
using .TensorBinding
include(joinpath(@__DIR__, "APSOS_NH_loss.jl"))

# -- command-line arguments ---------------------------------------------------
L            = length(ARGS) >= 1  ? parse(Int,     ARGS[1])  : 7
t            = length(ARGS) >= 2  ? parse(Float64, ARGS[2])  : 1.0
gamma0       = length(ARGS) >= 3  ? parse(Float64, ARGS[3])  : 1.0
golden_b     = (1 + sqrt(5.0)) / 2
comm_loss_b  = apsos_commensurate_loss_b()
loss_b       = length(ARGS) >= 4  ? parse(Float64, ARGS[4])  : comm_loss_b
loss_phase   = length(ARGS) >= 5  ? parse(Float64, ARGS[5])  : 0.0
scale_token   = length(ARGS) >= 6  ? lowercase(strip(ARGS[6])) : "auto"
auto_scale    = scale_token in ("auto", "none", "nothing", "dmrg", "0", "0.0")
scale_arg     = auto_scale ? nothing : parse(Float64, ARGS[6])
Nh           = length(ARGS) >= 7  ? parse(Int,     ARGS[7])  : 10
maxdim       = length(ARGS) >= 8  ? parse(Int,     ARGS[8])  : 100
nx           = length(ARGS) >= 9  ? parse(Int,     ARGS[9])  : 40
ny           = length(ARGS) >= 10 ? parse(Int,     ARGS[10]) : 40
xmin         = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : -2.25
xmax         = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) :  2.25
ymin         = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : -2.25
ymax         = length(ARGS) >= 14 ? parse(Float64, ARGS[14]) :  0.25
legacy_n_random = length(ARGS) >= 15 ? parse(Int,     ARGS[15]) : 0
legacy_seed     = length(ARGS) >= 16 ? parse(Int,     ARGS[16]) : 42
cutoff       = length(ARGS) >= 17 ? parse(Float64, ARGS[17]) : 1e-4
OUTDIR       = length(ARGS) >= 18 ? ARGS[18]                 : "outputs_nhdos_gpu"
tile_id      = length(ARGS) >= 19 ? parse(Int,     ARGS[19]) : 0
num_tiles    = length(ARGS) >= 20 ? parse(Int,     ARGS[20]) : 1
aah_V        = length(ARGS) >= 21 ? parse(Float64, ARGS[21]) : 0.5
aah_phi      = length(ARGS) >= 22 ? parse(Float64, ARGS[22]) : 0.0
aah_b        = length(ARGS) >= 23 ? parse(Float64, ARGS[23]) : golden_b
loss_harmonics = length(ARGS) >= 24 ? parse(Int, ARGS[24]) : 15
nh_scale_pad = length(ARGS) >= 25 ? parse(Float64, ARGS[25]) : 1.25
nh_bound_check_stride = length(ARGS) >= 26 ? parse(Int, ARGS[26]) : 2

mkpath(OUTDIR)

num_tiles >= 1 || error("num_tiles must be >= 1, got $num_tiles")
0 <= tile_id < num_tiles || error("tile_id=$tile_id must satisfy 0 <= tile_id < num_tiles=$num_tiles")
nh_bound_check_stride >= 0 || error("nh_bound_check_stride must be >= 0, got $nh_bound_check_stride")

N_sites = 2^L
loss_profile, loss_period_sites = apsos_cusped_loss_profile(N_sites, gamma0, loss_b;
    loss_phase=loss_phase,
    loss_harmonics=loss_harmonics)
loss_values = [loss_profile(n) for n in 0:(N_sites - 1)]
loss_bound = maximum(abs, loss_values)

println("[$(now())] APSOS_nhdos_gpu starting")
println("[info] GPU device: $(CUDA.name(CUDA.device()))")
println("[info] L=$L N=$N_sites")
println("[info] AAH open chain: t=$t  V=$aah_V  phi=$aah_phi  b=$aah_b")
println("[info] loss: gamma0=$gamma0  loss_b=$loss_b  period_sites=$loss_period_sites  loss_phase=$loss_phase  harmonics=$loss_harmonics")
println("[info] scale request=$(auto_scale ? "auto/NH-bound" : string(scale_arg))  nh_scale_pad=$nh_scale_pad  bound_check_stride=$nh_bound_check_stride  Nh=$Nh (recurrence order $(2 * Nh))  maxdim=$maxdim  cutoff=$cutoff")
println("[info] method: deterministic diagonal trace via GPU online MPO-MPO recurrence")
println("[info] GPU contraction dtype: ComplexF64")
println("[info] grid: Re(z)=[$xmin, $xmax] nx=$nx  Im(z)=[$ymin, $ymax] ny=$ny")
println("[info] legacy stochastic args ignored: n_random=$legacy_n_random  seed=$legacy_seed")
println("[info] tile_id=$tile_id / num_tiles=$num_tiles")

# -- build non-Hermitian modulated-loss AAH chain on CPU ----------------------
println("[info] Building open AAH chain...")
aah_params = (V=aah_V, phi=aah_phi, t=t,
              b=aah_b, tol_quantics=cutoff, maxbonddim_quantics=maxdim)
@time H = TensorBinding.get_Hamiltonian("aah", aah_params;
    L=L,
    scale=scale_arg,
    tol=cutoff,
    maxdim=maxdim)

println("[info] Adding loss profile -i*gamma(n)...")
@time TensorBinding.add_loss!(H, loss_profile; maxdim=maxdim, tol=cutoff)
println("[info] $H")
println("[info] MPO bond dim: $(ITensorMPS.maxlinkdim(H.mpo))")

# -- full-grid scale and tile coordinates ------------------------------------
xgrid = collect(range(xmin, xmax; length=nx))
ygrid = collect(range(ymin, ymax; length=ny))
full_z_points = ComplexF64[ComplexF64(x, y) for x in xgrid for y in ygrid]
z_radius = maximum(abs, full_z_points)

println("[info] Estimating universal NH KPM scale from zero-shift hermitized norm and full z-grid...")
@time nh_scale = TensorBinding.nh_kpm_scale(H, full_z_points;
    scale=scale_arg,
    padding=nh_scale_pad,
    maxdim=maxdim,
    cutoff=cutoff,
    printinfo=true)
println("[info] NH KPM scale=$nh_scale")
nh_parent_norm_bound = auto_scale ? nh_scale / nh_scale_pad - z_radius : NaN
if auto_scale
    worst_ratio = (nh_parent_norm_bound + z_radius) / nh_scale
    println("[check] global Chebyshev safety: parent_norm_bound=$nh_parent_norm_bound  z_radius=$z_radius  worst_bound/scale=$worst_ratio  ok=$(worst_ratio <= 1.0)")
else
    println("[check] global Chebyshev safety: manual scale=$nh_scale; automatic parent-norm bound not available in script diagnostics.")
end

# -- GPU warm-up --------------------------------------------------------------
println("[info] GPU warm-up...")
warmup_s = siteinds("S=1/2", 4)
warmup_H = MPO(warmup_s, "Id")
@time CUDA.cu(warmup_H)
CUDA.synchronize()
println("[info] Warm-up complete.")

# -- NH deterministic diagonal-trace spectral grid tile -----------------------
total_points = nx * ny
point_ids = collect((tile_id + 1):num_tiles:total_points)

z_points = Vector{ComplexF64}(undef, length(point_ids))
ix_list  = Vector{Int}(undef, length(point_ids))
iy_list  = Vector{Int}(undef, length(point_ids))

for (j, pid) in enumerate(point_ids)
    ix = div(pid - 1, ny) + 1
    iy = mod(pid - 1, ny) + 1
    ix_list[j] = ix
    iy_list[j] = iy
    z_points[j] = ComplexF64(xgrid[ix], ygrid[iy])
end

println("[info] Running GPU NH deterministic diagonal-trace spectral tile...")
println("[info] tile owns $(length(point_ids)) / $total_points z-points")
if auto_scale && nh_bound_check_stride > 0
    println("[check] tile Chebyshev safety diagnostics every $nh_bound_check_stride owned z-points:")
    for j in 1:nh_bound_check_stride:length(z_points)
        z = z_points[j]
        local_bound = nh_parent_norm_bound + abs(z)
        local_ratio = local_bound / nh_scale
        println("[check] z[$j/$(length(z_points))] global_id=$(point_ids[j]) Re=$(real(z)) Im=$(imag(z)) |z|=$(abs(z)) bound/scale=$local_ratio ok=$(local_ratio <= 1.0)")
    end
    if !isempty(z_points) && (length(z_points) - 1) % nh_bound_check_stride != 0
        j = length(z_points)
        z = z_points[j]
        local_bound = nh_parent_norm_bound + abs(z)
        local_ratio = local_bound / nh_scale
        println("[check] z[$j/$(length(z_points))] global_id=$(point_ids[j]) Re=$(real(z)) Im=$(imag(z)) |z|=$(abs(z)) bound/scale=$local_ratio ok=$(local_ratio <= 1.0)")
    end
elseif nh_bound_check_stride == 0
    println("[check] tile Chebyshev safety diagnostics disabled by bound_check_stride=0")
else
    println("[check] tile Chebyshev safety diagnostics skipped for manual scale request.")
end
@time dos_values = TensorBinding.get_nh_dos_points_diag_trace_gpu(H, z_points, Nh;
    scale     = nh_scale,
    point_ids = point_ids,
    maxdim    = maxdim,
    cutoff    = cutoff,
    dtype     = ComplexF64,
    printinfo = true)

println("[info] tile values length: $(length(dos_values))")

# -- save ---------------------------------------------------------------------
tag = "L$(L)_g$(apsos_tagnum(gamma0))_ph$(apsos_tagnum(loss_phase))_lh$(loss_harmonics)_Nc$(2 * Nh)_nx$(nx)_ny$(ny)"

tile_str = lpad(string(tile_id), 4, '0')
ntile_str = lpad(string(num_tiles), 4, '0')

tile_long = Matrix{Float64}(undef, length(point_ids), 5)
for j in eachindex(point_ids)
    tile_long[j, 1] = ix_list[j]
    tile_long[j, 2] = iy_list[j]
    tile_long[j, 3] = real(z_points[j])
    tile_long[j, 4] = imag(z_points[j])
    tile_long[j, 5] = dos_values[j]
end

tile_file = joinpath(OUTDIR, "NHDOS_tile$(tile_str)_of$(ntile_str)_$(tag).csv")
x_file    = joinpath(OUTDIR, "xgrid_$(tag).csv")
y_file    = joinpath(OUTDIR, "ygrid_$(tag).csv")
params_file = joinpath(OUTDIR, "params_$(tag).csv")
params_rows = Any[
    "L" L;
    "N_sites" N_sites;
    "t" t;
    "aah_V" aah_V;
    "aah_phi" aah_phi;
    "aah_b" aah_b;
    "gamma0" gamma0;
    "loss_b" loss_b;
    "loss_phase" loss_phase;
    "loss_harmonics" loss_harmonics;
    "loss_period_sites" loss_period_sites;
    "scale_request" scale_token;
    "scale_mode" (auto_scale ? "auto_nh_bound" : "manual");
    "scale" nh_scale;
    "center" 0.0;
    "nh_scale" nh_scale;
    "nh_scale_pad" nh_scale_pad;
    "nh_scale_mode" "zero_shift_bound";
    "nh_parent_norm_bound" nh_parent_norm_bound;
    "nh_bound_check_stride" nh_bound_check_stride;
    "z_radius" z_radius;
    "loss_bound" loss_bound;
    "Nh" Nh;
    "recurrence_order" 2 * Nh;
    "maxdim" maxdim;
    "cutoff" cutoff;
    "nx" nx;
    "ny" ny;
    "xmin" xmin;
    "xmax" xmax;
    "ymin" ymin;
    "ymax" ymax;
    "tile_id" tile_id;
    "num_tiles" num_tiles;
]

writedlm(tile_file, tile_long, ',')
writedlm(x_file, collect(xgrid), ',')
writedlm(y_file, collect(ygrid), ',')
writedlm(params_file, params_rows, ',')

println("[$(now())] saved $(tile_file)")
println("[$(now())] saved $(x_file)")
println("[$(now())] saved $(y_file)")
println("[$(now())] saved $(params_file)")

if num_tiles == 1
    Z = fill(NaN, ny, nx)
    for j in eachindex(point_ids)
        Z[iy_list[j], ix_list[j]] = dos_values[j]
    end

    long = Matrix{Float64}(undef, nx * ny, 3)
    for (ix, x) in enumerate(xgrid), (iy, y) in enumerate(ygrid)
        row = (ix - 1) * ny + iy
        long[row, 1] = x
        long[row, 2] = y
        long[row, 3] = Z[iy, ix]
    end

    long_file = joinpath(OUTDIR, "NHDOS_$(tag).csv")
    mat_file  = joinpath(OUTDIR, "NHDOS_matrix_$(tag).csv")
    writedlm(long_file, long, ',')
    writedlm(mat_file, Z, ',')

    println("[$(now())] saved $(long_file)")
    println("[$(now())] saved $(mat_file)")
end
println("[$(now())] Done.")
