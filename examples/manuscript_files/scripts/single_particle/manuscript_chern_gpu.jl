#!/usr/bin/env julia
# APSOS_chern_gpu.jl
#
# GPU-accelerated local Chern marker C(r) over a 2D sampling grid.
# The projector (McWeeny) and C1–C4 MPO assembly all run on GPU in F32.
# Evaluation of C_at(uc) uses GPU inner products.
#
# Output: CSV with columns  ix, iy, C_real
#
# Usage:
#   julia --project=@. APSOS_chern_gpu.jl \
#     Lx Ly t t2 M phi maxdim l Lambda num_x num_y cutoff OUTDIR

using CUDA
using Dates, LinearAlgebra, DelimitedFiles
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "../../../src/TensorBinding.jl"))
using .TensorBinding
include(joinpath(@__DIR__, "APSOS_Haldane.jl"))

# ── command-line arguments ────────────────────────────────────────────────────
Lx      = length(ARGS) >= 1  ? parse(Int,     ARGS[1])  : 4
Ly      = length(ARGS) >= 2  ? parse(Int,     ARGS[2])  : 4
t       = length(ARGS) >= 3  ? parse(Float64, ARGS[3])  : 1.0
t2      = length(ARGS) >= 4  ? parse(Float64, ARGS[4])  : 0.2
M       = length(ARGS) >= 5  ? parse(Float64, ARGS[5])  : 0.1
phi     = length(ARGS) >= 6  ? parse(Float64, ARGS[6])  : π/2
maxdim  = length(ARGS) >= 7  ? parse(Int,     ARGS[7])  : 200
l_param = length(ARGS) >= 8  ? parse(Int,     ARGS[8])  : 4
Lambda  = length(ARGS) >= 9  ? parse(Float64, ARGS[9])  : 10.0
num_x   = length(ARGS) >= 10 ? parse(Int,     ARGS[10]) : 8
num_y   = length(ARGS) >= 11 ? parse(Int,     ARGS[11]) : 8
cutoff  = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : 1e-8
OUTDIR  = length(ARGS) >= 13 ? ARGS[13]                 : "outputs_chern_gpu"

Nx = 2^Lx
Ny = 2^Ly
mkpath(OUTDIR)

println("[$(now())] APSOS_chern_gpu starting")
println("[info] GPU device: $(CUDA.name(CUDA.device()))")
println("[info] Lx=$Lx Ly=$Ly  t=$t t2=$t2 M=$M phi=$phi")
println("[info] maxdim=$maxdim  l=$l_param  Lambda=$Lambda  cutoff=$cutoff")
println("[info] grid: $(num_x)x$(num_y)")

# ── build Hamiltonian ─────────────────────────────────────────────────────────
println("[info] Building Hamiltonian...")
@time H = build_APSOS_hamiltonian(Lx, Ly, t, t2, M, phi; maxdim=maxdim)
println("[info] $H")

# ── GPU warm-up ───────────────────────────────────────────────────────────────
println("[info] GPU warm-up...")
warmup_s = siteinds("S=1/2", 4)
warmup_H = MPO(warmup_s, "Id")
@time begin
    wH_gpu = CUDA.cu(warmup_H)
    apply(wH_gpu, wH_gpu)
    CUDA.synchronize()
end
println("[info] Warm-up done.")

# ── Chern marker (GPU) ────────────────────────────────────────────────────────
println("[info] Building Chern marker (GPU McWeeny + C1–C4)...")
@time C_at = TensorBinding.get_C_gpu(H;
    method    = :mcweeny,
    maxdim    = maxdim,
    cutoff    = cutoff,
    l         = l_param,
    Lambda    = Lambda,
    printinfo = true)
println("[info] Chern operator ready.")

# ── 2D sampling grid ─────────────────────────────────────────────────────────
ix_samples = round.(Int, range(0, Nx - 1; length=num_x))
iy_samples = round.(Int, range(0, Ny - 1; length=num_y))
positions  = [(ix, iy) for iy in iy_samples for ix in ix_samples]
uc_indices = [ix + iy * Nx + 1 for (ix, iy) in positions]

# ── evaluate ─────────────────────────────────────────────────────────────────
println("[info] Evaluating Chern marker at $(length(positions)) positions...")
rows = Matrix{Float64}(undef, length(positions), 3)
@time for (k, (uc, (ix, iy))) in enumerate(zip(uc_indices, positions))
    rows[k, 1] = ix
    rows[k, 2] = iy
    rows[k, 3] = real(C_at(uc))
end

# ── save ──────────────────────────────────────────────────────────────────────
tag   = "Lx$(Lx)_Ly$(Ly)_t2$(t2)_M$(M)_mdim$(maxdim)_l$(l_param)_nx$(num_x)_ny$(num_y)_cut$(cutoff)"
fname = joinpath(OUTDIR, "Chern_$(tag).csv")
writedlm(fname, rows, ',')
println("[$(now())] saved $(fname)")
