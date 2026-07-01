#!/usr/bin/env julia
# APSOS_ldos_gpu.jl
#
# GPU-accelerated spatial LDOS A(r,ω) on a 2D sampling grid.
# Runs the full Chebyshev recurrence once on GPU and samples all positions
# in a single pass — no array jobs needed.
#
# Output (all files in OUTDIR):
#   ldos_<tag>.csv     — (Nω × n_cols) spectral weight, row = ω, col = position
#   omega_<tag>.csv    — (Nω × 1) omega grid
#   positions_<tag>.csv — (n_cols × 3) with columns ix, iy, sublat
#                         (sublat 1=A, 2=B when resolved; 0 = unit-cell average)
#
# `reduce` selects the sampling procedure (see get_ldos_spatial_gpu /
# spatial_sampling_plan):
#   block — partition into num_x×num_y blocks (powers of two) and INTEGRATE each
#           block (trace out within-block bits). Gap-free: a thin in-gap edge /
#           domain-wall network on a gapped background CANNOT fall between pixels.
#           The right tool for a large-scale map of edge states. (default)
#   point — read the LDOS AT num_x×num_y cells (optionally box_half-averaged).
#           Cheap, but a coarse grid ALIASES thin features between samples.
#
# `sublattice` controls the geometry-aware sublattice handling:
#   average — one value per unit cell, sublattice traced out (clean large-scale map)
#   resolve — separate A/B columns per unit cell (atomic detail)
#   auto    — average at large scale, resolve at full unit-cell resolution
#
# Usage:
#   julia --project=@. APSOS_ldos_gpu.jl \
#     Lx Ly t t2 M phi Ncheb maxdim Nomega Emin Emax num_x num_y box_half cutoff \
#     reduce sublattice OUTDIR gpu_type
#
# Defaults (all optional):
#   Lx=4  Ly=4  t=1.0  t2=0.2  M=0.1  phi=1.5707963
#   Ncheb=500  maxdim=200  Nomega=300  Emin=-4.0  Emax=4.0
#   num_x=32  num_y=32  box_half=0  cutoff=1e-4  reduce=block  sublattice=average
#   OUTDIR=outputs_ldos_gpu  gpu_type=ComplexF64
#   (reduce=block requires num_x, num_y powers of two)

using CUDA   # must be first — enables the NDTensors GPU backend
using Dates, LinearAlgebra, DelimitedFiles
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "../../../src/TensorBinding.jl"))
using .TensorBinding
include(joinpath(@__DIR__, "APSOS_Haldane.jl"))

function parse_gpu_complex_type(s)
    key = lowercase(String(s))
    key in ("complexf32", "f32", "float32", "single") && return ComplexF32
    key in ("complexf64", "f64", "float64", "double") && return ComplexF64
    error("Unsupported gpu_type=$s. Use ComplexF32 or ComplexF64.")
end

# ── command-line arguments ────────────────────────────────────────────────────
Lx     = length(ARGS) >= 1  ? parse(Int,     ARGS[1])  : 4
Ly     = length(ARGS) >= 2  ? parse(Int,     ARGS[2])  : 4
t      = length(ARGS) >= 3  ? parse(Float64, ARGS[3])  : 1.0
t2     = length(ARGS) >= 4  ? parse(Float64, ARGS[4])  : 0.2
M      = length(ARGS) >= 5  ? parse(Float64, ARGS[5])  : 0.1
phi    = length(ARGS) >= 6  ? parse(Float64, ARGS[6])  : π/2
Ncheb  = length(ARGS) >= 7  ? parse(Int,     ARGS[7])  : 500
maxdim = length(ARGS) >= 8  ? parse(Int,     ARGS[8])  : 200
Nomega = length(ARGS) >= 9  ? parse(Int,     ARGS[9])  : 300
Emin   = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : -4.0
Emax   = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) :  4.0
num_x    = length(ARGS) >= 12 ? parse(Int,  ARGS[12]) : 32   # sample cols (x)
num_y    = length(ARGS) >= 13 ? parse(Int,  ARGS[13]) : 32   # sample rows (y)
box_half = length(ARGS) >= 14 ? parse(Int,    ARGS[14]) : 0      # 0 = no box avg; 2 → 5×5 box (point mode)
cutoff   = length(ARGS) >= 15 ? parse(Float64, ARGS[15]) : 1e-4
reduce_mode = length(ARGS) >= 16 ? Symbol(ARGS[16])    : :block    # point | block
sublattice = length(ARGS) >= 17 ? Symbol(ARGS[17])     : :average  # average | resolve | auto
OUTDIR   = length(ARGS) >= 18 ? ARGS[18]                 : "outputs_ldos_gpu"
gpu_type_arg = length(ARGS) >= 19 ? ARGS[19]             : "ComplexF64"
gpu_type = parse_gpu_complex_type(gpu_type_arg)
reduce_mode in (:point, :block) ||
    error("reduce must be point or block; got $reduce_mode")
sublattice in (:average, :resolve, :auto) ||
    error("sublattice must be average, resolve, or auto; got $sublattice")

Nx = 2^Lx
Ny = 2^Ly
omegalist = range(Emin, Emax; length=Nomega)
mkpath(OUTDIR)

println("[$(now())] APSOS_ldos_gpu starting")
println("[info] GPU device: $(CUDA.name(CUDA.device()))")
println("[info] Lx=$Lx Ly=$Ly  t=$t t2=$t2 M=$M phi=$phi")
println("[info] Ncheb=$Ncheb  maxdim=$maxdim  Nomega=$Nomega  E=[$Emin, $Emax]  cutoff=$cutoff")
println("[info] GPU tensor type: $gpu_type")
println("[info] reduce=$reduce_mode  grid: $(num_x)×$(num_y)  box_half=$box_half  sublattice=$sublattice")

# ── build Hamiltonian (CPU) ───────────────────────────────────────────────────
println("[info] Building Hamiltonian...")
@time H = build_APSOS_hamiltonian(Lx, Ly, t, t2, M, phi; maxdim=maxdim)
println("[info] $H")

# ── GPU warm-up ───────────────────────────────────────────────────────────────
println("[info] GPU warm-up...")
warmup_s = siteinds("S=1/2", 4)
warmup_H = MPO(warmup_s, "Id")
@time CUDA.cu(warmup_H)
CUDA.synchronize()
println("[info] Warm-up complete.")

# ── build sampling positions ──────────────────────────────────────────────────
# Two procedures (see get_ldos_spatial_gpu / spatial_sampling_plan):
#   :point — read the LDOS AT num_x×num_y evenly-spaced cells (optionally box-
#            averaged). Coarse grids ALIAS thin features (edge/domain-wall
#            channels can fall between samples).
#   :block — partition into num_x×num_y blocks (powers of two) and INTEGRATE each
#            block by tracing out the within-block bits. Gap-free, so a thin edge
#            network on a gapped background cannot be missed; the right tool for a
#            large-scale map of in-gap LDOS. `positions` are the block centres.
if reduce_mode === :block
    (ispow2(num_x) && ispow2(num_y)) ||
        error("reduce=block requires num_x and num_y to be powers of two (got $num_x, $num_y)")
    Wx = Nx ÷ num_x          # block width in x (unit cells)
    Wy = Ny ÷ num_y
    positions = [(ixp * Wx + Wx ÷ 2, iyp * Wy + Wy ÷ 2)
                 for iyp in 0:(num_y - 1) for ixp in 0:(num_x - 1)]   # block centres
else
    ix_samples = round.(Int, range(0, Nx - 1; length=num_x))
    iy_samples = round.(Int, range(0, Ny - 1; length=num_y))
    positions  = [(ix, iy) for iy in iy_samples for ix in ix_samples]
end

n_pos = length(positions)
ncols_expected = sublattice === :average ? n_pos : 2 * n_pos
println("[info] $(n_pos) pixels ($reduce_mode), sublattice=$sublattice → $(ncols_expected) spatial columns")

# ── LDOS (GPU) ────────────────────────────────────────────────────────────────
# sublattice=:average traces out A/B into one value per pixel (clean map);
# :resolve keeps per-atom columns. reduce=:block integrates each block.
println("[info] Computing LDOS (GPU)...")
@time ldos = if reduce_mode === :block
    TensorBinding.get_ldos_spatial_gpu(H, Ncheb, omegalist;
        reduce     = :block,
        num_x      = num_x, num_y = num_y,
        sublattice = sublattice,
        maxdim     = maxdim,
        cutoff     = cutoff,
        type       = gpu_type,
        printinfo  = true)
else
    uc_indices = [ix + iy * Nx + 1 for (ix, iy) in positions]
    x_groups   = [[uc] for uc in uc_indices]
    TensorBinding.get_ldos_spatial_gpu(H, Ncheb, omegalist;
        x_groups   = x_groups,
        box_half   = box_half,
        sublattice = sublattice,
        maxdim     = maxdim,
        cutoff     = cutoff,
        type       = gpu_type,
        printinfo  = true)
end

println("[info] ldos shape: $(size(ldos))")

# ── save ──────────────────────────────────────────────────────────────────────
type_tag = replace(gpu_type_arg, "Complex" => "C")
tag = "Lx$(Lx)_Ly$(Ly)_t2$(t2)_M$(M)_Nc$(Ncheb)_mdim$(maxdim)_nx$(num_x)_ny$(num_y)_bh$(box_half)_cut$(cutoff)_$(reduce_mode)_sl$(sublattice)_typ$(type_tag)"

# Spectral weight matrix (Nω × n_spatial_cols)
ldos_file = joinpath(OUTDIR, "ldos_$(tag).csv")
writedlm(ldos_file, ldos, ',')

# Omega grid
omega_file = joinpath(OUTDIR, "omega_$(tag).csv")
writedlm(omega_file, collect(omegalist), ',')

# Position table: each column of Ak maps to a row here (ix, iy, sublat).
# n_sub = 2 when resolved (sublat = 1,2), 1 when averaged (sublat = 0 → UC mean).
n_sub = size(ldos, 2) ÷ n_pos
pos_rows = Matrix{Int}(undef, n_pos * n_sub, 3)
for (i, (ix, iy)) in enumerate(positions)
    for s in 1:n_sub
        row = (i - 1) * n_sub + s
        pos_rows[row, 1] = ix
        pos_rows[row, 2] = iy
        pos_rows[row, 3] = n_sub == 1 ? 0 : s
    end
end
pos_file = joinpath(OUTDIR, "positions_$(tag).csv")
writedlm(pos_file, pos_rows, ',')

println("[$(now())] saved $(ldos_file)")
println("[$(now())] saved $(omega_file)")
println("[$(now())] saved $(pos_file)")
