#!/usr/bin/env julia
# APSOS_bands_gpu.jl
#
# GPU-accelerated band structure A(k,ω) along a high-symmetry k-path.
# Requires a CUDA-capable GPU.  Load CUDA before anything else so the
# NDTensors GPU backend is available when TensorBinding initialises.
#
# GPU handles: entire Chebyshev MPO recurrence, QFT sandwich per step.
# CPU handles: k-path setup, KPM weights, sublattice projection (one site
#              contraction), scalar accumulation.
#
# Output:
#   Ak_<params>.csv    — (Nω × Nk) spectral weight matrix, row = ω, col = k
#   meta_<params>.csv  — omega grid + tick positions + tick labels (Nω × 3)
#
# Usage:
#   julia --project=. APSOS_bands_gpu.jl \
#     Lx Ly t t2 M phi Ncheb maxdim Nomega Emin Emax num_k cutoff OUTDIR gpu_type
#
# Defaults (all optional):
#   Lx=4  Ly=4  t=1.0  t2=0.2  M=0.1  phi=1.5707963
#   Ncheb=500  maxdim=200  Nomega=500  Emin=-4.0  Emax=4.0  num_k=50
#   cutoff=1e-4  OUTDIR=outputs_bands_gpu  gpu_type=ComplexF64

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
Nomega = length(ARGS) >= 9  ? parse(Int,     ARGS[9])  : 500
Emin   = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : -4.0
Emax   = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) :  4.0
num_k  = length(ARGS) >= 12 ? parse(Int,     ARGS[12]) : 50   # k-points per segment
cutoff = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : 1e-4
OUTDIR = length(ARGS) >= 14 ? ARGS[14]                 : "outputs_bands_gpu"
gpu_type_arg = length(ARGS) >= 15 ? ARGS[15]             : "ComplexF64"
gpu_type = parse_gpu_complex_type(gpu_type_arg)

omegalist = range(Emin, Emax; length=Nomega)
mkpath(OUTDIR)

println("[$(now())] APSOS_bands_gpu starting")
println("[info] GPU device: $(CUDA.name(CUDA.device()))")
println("[info] Lx=$Lx Ly=$Ly  t=$t t2=$t2 M=$M phi=$phi")
println("[info] Ncheb=$Ncheb  maxdim=$maxdim  Nomega=$Nomega  num_k=$num_k  E=[$Emin, $Emax]  cutoff=$cutoff")
println("[info] GPU tensor type: $gpu_type")

# ── build Hamiltonian (CPU) ───────────────────────────────────────────────────
println("[info] Building Hamiltonian...")
@time H = build_APSOS_hamiltonian(Lx, Ly, t, t2, M, phi; maxdim=maxdim)
println("[info] $H")

# ── GPU warm-up (eliminates JIT latency from timing) ─────────────────────────
println("[info] GPU warm-up...")
warmup_s = siteinds("S=1/2", 4)
warmup_H = MPO(warmup_s, "Id")
@time CUDA.cu(warmup_H)
CUDA.synchronize()
println("[info] Warm-up complete.")

# ── band structure (GPU) ──────────────────────────────────────────────────────
println("[info] Computing bands along G-M-K'-G  (GPU)...")
@time res = TensorBinding.get_bands_gpu(H, Ncheb, omegalist;
    kpath         = [:G, :M, :Kp, :G],
    kpath_lattice = :honeycomb,
    num_x         = num_k,
    maxdim        = maxdim,
    cutoff        = cutoff,
    type          = gpu_type,
    printinfo     = true)

println("[info] Ak shape: $(size(res.Ak))")

# ── save ──────────────────────────────────────────────────────────────────────
type_tag = gpu_type === ComplexF64 ? "CF64" : "CF32"
tag = "Lx$(Lx)_Ly$(Ly)_t2$(t2)_M$(M)_Nc$(Ncheb)_mdim$(maxdim)_nk$(num_k)_cut$(cutoff)_typ$(type_tag)"

ak_file = joinpath(OUTDIR, "Ak_$(tag).csv")
writedlm(ak_file, res.Ak, ',')

Ntick = length(res.ticks)
meta  = Matrix{Any}(undef, Nomega, 3)
for i in 1:Nomega
    meta[i, 1] = i <= Ntick ? res.ticks[i]  : ""
    meta[i, 2] = i <= Ntick ? res.labels[i] : ""
    meta[i, 3] = collect(omegalist)[i]
end
meta_file = joinpath(OUTDIR, "meta_$(tag).csv")
writedlm(meta_file, meta, ',')

println("[$(now())] saved $(ak_file)")
println("[$(now())] saved $(meta_file)")
