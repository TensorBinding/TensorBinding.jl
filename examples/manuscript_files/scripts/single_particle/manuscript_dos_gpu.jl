#!/usr/bin/env julia
# APSOS_dos_gpu.jl
#
# GPU-accelerated stochastic DOS via MPS Chebyshev KPM.
# Ham MPO and each sample MPS run fully on GPU; moments are scalars on CPU.
# Unlike the LDOS script (one MPO pass, many positions), this loops over
# N_sample independent MPS recursions — no array jobs needed for a single run.
#
# Output (OUTDIR):
#   DOS_<tag>.csv   — two columns: omega, DOS   (normalize=true → per-state)
#
# Usage:
#   julia --project=@. APSOS_dos_gpu.jl \
#     Lx Ly t t2 M phi Ncheb maxdim Nomega Emin Emax N_sample seed OUTDIR
#
# Defaults (all optional, fall back to small test values):
#   Lx=4  Ly=4  t=1.0  t2=0.2  M=0.1  phi=1.5707963
#   Ncheb=200  maxdim=200  Nomega=300  Emin=-4.0  Emax=4.0
#   N_sample=100  seed=42  OUTDIR=outputs_dos_gpu

using CUDA   # must be first — enables the NDTensors GPU backend
using Dates, LinearAlgebra, DelimitedFiles
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "../../../src/TensorBinding.jl"))
using .TensorBinding
include(joinpath(@__DIR__, "APSOS_Haldane.jl"))

# ── command-line arguments ────────────────────────────────────────────────────
Lx       = length(ARGS) >= 1  ? parse(Int,     ARGS[1])  : 4
Ly       = length(ARGS) >= 2  ? parse(Int,     ARGS[2])  : 4
t        = length(ARGS) >= 3  ? parse(Float64, ARGS[3])  : 1.0
t2       = length(ARGS) >= 4  ? parse(Float64, ARGS[4])  : 0.2
M        = length(ARGS) >= 5  ? parse(Float64, ARGS[5])  : 0.1
phi      = length(ARGS) >= 6  ? parse(Float64, ARGS[6])  : π/2
Ncheb    = length(ARGS) >= 7  ? parse(Int,     ARGS[7])  : 200
maxdim   = length(ARGS) >= 8  ? parse(Int,     ARGS[8])  : 200
Nomega   = length(ARGS) >= 9  ? parse(Int,     ARGS[9])  : 300
Emin     = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : -4.0
Emax     = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) :  4.0
N_sample = length(ARGS) >= 12 ? parse(Int,     ARGS[12]) : 100
seed     = length(ARGS) >= 13 ? parse(Int,     ARGS[13]) : 42
cutoff   = length(ARGS) >= 14 ? parse(Float64, ARGS[14]) : 1e-4
OUTDIR   = length(ARGS) >= 15 ? ARGS[15]                 : "outputs_dos_gpu"

omegalist = range(Emin, Emax; length=Nomega)
mkpath(OUTDIR)

println("[$(now())] APSOS_dos_gpu starting")
println("[info] GPU device: $(CUDA.name(CUDA.device()))")
println("[info] Lx=$Lx Ly=$Ly  t=$t t2=$t2 M=$M phi=$phi")
println("[info] Ncheb=$Ncheb  maxdim=$maxdim  Nomega=$Nomega  E=[$Emin, $Emax]")
println("[info] N_sample=$N_sample  seed=$seed  cutoff=$cutoff")

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

# ── stochastic DOS (GPU) ──────────────────────────────────────────────────────
println("[info] Running stochastic KPM ($N_sample samples)...")
@time dos = TensorBinding.get_dos_stochastic_gpu(H, Ncheb, omegalist;
    N_sample  = N_sample,
    seed      = seed,
    normalize = true,
    maxdim    = maxdim,
    cutoff    = cutoff,
    printinfo = true)

println("[info] dos shape: $(size(dos))")

# ── save ──────────────────────────────────────────────────────────────────────
tag   = "Lx$(Lx)_Ly$(Ly)_t2$(t2)_M$(M)_Nc$(Ncheb)_mdim$(maxdim)_Ns$(N_sample)_seed$(seed)_cut$(cutoff)"
fname = joinpath(OUTDIR, "DOS_$(tag).csv")
writedlm(fname, hcat(collect(omegalist), dos), ',')
println("[$(now())] saved $(fname)")
