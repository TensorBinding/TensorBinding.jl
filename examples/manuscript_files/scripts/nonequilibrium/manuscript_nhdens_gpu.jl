#!/usr/bin/env julia
# APSOS_nhdens_gpu.jl
#
# GPU non-Hermitian density-matrix evolution for the modulated-loss open AAH chain.
# This is the production-script version of panel (b) in APSOS_NH_testing:
# diagonal density <x|rho(t)|x> under d rho/dt = -i(H rho - rho Hdagger).
#
# Output (OUTDIR):
#   NHDENS_<tag>.csv        -- matrix: rows sampled positions, columns times
#   times_<tag>.csv         -- sampled times
#   positions_<tag>.csv     -- columns: center_1based, x_0based
#   maxlinkdim_<tag>.csv    -- sampled density-MPO maxlinkdim vs time
#
# Usage:
#   julia --project=@. APSOS_nhdens_gpu.jl \
#     L t gamma0 loss_b loss_phase scale nsteps dt sample_every \
#     maxdim cutoff purif_maxiter purif_tol num_x num_avg reduce OUTDIR \
#     [aah_V aah_phi aah_b loss_harmonics]

using CUDA   # must be first -- enables the NDTensors GPU backend
using Dates, LinearAlgebra, DelimitedFiles
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "../../../src/TensorBinding.jl"))
using .TensorBinding
include(joinpath(@__DIR__, "APSOS_NH_loss.jl"))

# -- command-line arguments ---------------------------------------------------
L              = length(ARGS) >= 1  ? parse(Int,     ARGS[1])  : 20
t              = length(ARGS) >= 2  ? parse(Float64, ARGS[2])  : 1.0
gamma0         = length(ARGS) >= 3  ? parse(Float64, ARGS[3])  : 1.0
golden_b       = (1 + sqrt(5.0)) / 2
comm_loss_b    = apsos_commensurate_loss_b()
loss_b         = length(ARGS) >= 4  ? parse(Float64, ARGS[4])  : comm_loss_b
loss_phase     = length(ARGS) >= 5  ? parse(Float64, ARGS[5])  : 0.0
scale          = length(ARGS) >= 6  ? parse(Float64, ARGS[6])  : 4.5
nsteps         = length(ARGS) >= 7  ? parse(Int,     ARGS[7])  : 120
dt             = length(ARGS) >= 8  ? parse(Float64, ARGS[8])  : 0.125
sample_every   = length(ARGS) >= 9  ? parse(Int,     ARGS[9])  : 1
maxdim         = length(ARGS) >= 10 ? parse(Int,     ARGS[10]) : 100
cutoff         = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : 1e-4
purif_maxiter  = length(ARGS) >= 12 ? parse(Int,     ARGS[12]) : 30
purif_tol      = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : 1e-6
num_x          = length(ARGS) >= 14 ? parse(Int,     ARGS[14]) : 1024
num_avg        = length(ARGS) >= 15 ? parse(Int,     ARGS[15]) : 1
valid_reduce_args = ("point", ":point", "block", ":block")
reduce_arg     = (length(ARGS) >= 16 && ARGS[16] in valid_reduce_args) ? ARGS[16] : "block"
reduce         = Symbol(replace(reduce_arg, ":" => ""))
OUTDIR         = length(ARGS) >= 17 ? ARGS[17] :
                 (length(ARGS) >= 16 && !(ARGS[16] in valid_reduce_args) ? ARGS[16] : "outputs_nhdens_gpu")
aah_V          = length(ARGS) >= 18 ? parse(Float64, ARGS[18]) : 0.5
aah_phi        = length(ARGS) >= 19 ? parse(Float64, ARGS[19]) : 0.0
aah_b          = length(ARGS) >= 20 ? parse(Float64, ARGS[20]) : golden_b
loss_harmonics = length(ARGS) >= 21 ? parse(Int, ARGS[21]) : 15

mkpath(OUTDIR)

N_sites = 2^L
loss_profile, loss_period_sites = apsos_cusped_loss_profile(N_sites, gamma0, loss_b;
    loss_phase=loss_phase,
    loss_harmonics=loss_harmonics)

println("[$(now())] APSOS_nhdens_gpu starting")
println("[info] GPU device: $(CUDA.name(CUDA.device()))")
println("[info] L=$L N=$N_sites")
println("[info] AAH open chain: t=$t  V=$aah_V  phi=$aah_phi  b=$aah_b")
println("[info] loss: gamma0=$gamma0  loss_b=$loss_b  period_sites=$loss_period_sites  loss_phase=$loss_phase  harmonics=$loss_harmonics")
println("[info] scale=$scale  nsteps=$nsteps  dt=$dt  sample_every=$sample_every")
println("[info] maxdim=$maxdim  cutoff=$cutoff  purif_maxiter=$purif_maxiter  purif_tol=$purif_tol")
println("[info] density sampling: reduce=$reduce  num_x=$num_x  num_avg=$num_avg")

# -- build Hamiltonians on CPU ------------------------------------------------
println("[info] Building Hermitian open AAH chain...")
aah_params = (V=aah_V, phi=aah_phi, t=t,
              b=aah_b, tol_quantics=cutoff, maxbonddim_quantics=maxdim)
@time H_herm = TensorBinding.get_Hamiltonian("aah", aah_params;
    L=L,
    scale=scale,
    tol=cutoff,
    maxdim=maxdim)
H_herm.scale = scale
H_herm.center = 0.0
println("[info] H_herm: $H_herm")

println("[info] Building NH modulated-loss chain...")
H_nh = deepcopy(H_herm)
@time TensorBinding.add_loss!(H_nh, loss_profile; maxdim=maxdim, tol=cutoff)
println("[info] H_nh: $H_nh")
println("[info] NH MPO bond dim: $(ITensorMPS.maxlinkdim(H_nh.mpo))")

# -- GPU warm-up --------------------------------------------------------------
println("[info] GPU warm-up...")
warmup_s = siteinds("S=1/2", 4)
warmup_H = MPO(warmup_s, "Id")
@time CUDA.cu(warmup_H)
CUDA.synchronize()
println("[info] Warm-up complete.")

# -- initial half-filled density matrix --------------------------------------
println("[info] GPU McWeeny purification for rho0...")
@time rho0_gpu = TensorBinding._mcweeny_purify_gpu(H_herm;
    fermi    = 0.0,
    maxdim   = maxdim,
    cutoff   = cutoff,
    maxiters = purif_maxiter,
    tol      = purif_tol,
    return_gpu = true)
CUDA.synchronize()
println("[info] rho0 maxlinkdim=$(ITensorMPS.maxlinkdim(rho0_gpu))")

# -- NH RK4 density evolution -------------------------------------------------
println("[info] Running GPU NH density evolution...")
@time res = TensorBinding.get_nh_density_trajectory_gpu(H_nh, rho0_gpu;
    nsteps       = nsteps,
    dt           = dt,
    sample_every = sample_every,
    num_x        = num_x,
    num_avg      = num_avg,
    reduce       = reduce,
    maxdim       = maxdim,
    cutoff       = cutoff,
    printinfo    = true)
CUDA.synchronize()

println("[info] density shape: $(size(res.density))")

# -- save ---------------------------------------------------------------------
tag = "L$(L)_g$(apsos_tagnum(gamma0))_ph$(apsos_tagnum(loss_phase))_lh$(loss_harmonics)_nx$(num_x)"

density_file = joinpath(OUTDIR, "NHDENS_$(tag).csv")
times_file   = joinpath(OUTDIR, "times_$(tag).csv")
pos_file     = joinpath(OUTDIR, "positions_$(tag).csv")
mlink_file   = joinpath(OUTDIR, "maxlinkdim_$(tag).csv")
params_file  = joinpath(OUTDIR, "params_$(tag).csv")
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
    "scale" scale;
    "nsteps" nsteps;
    "dt" dt;
    "sample_every" sample_every;
    "maxdim" maxdim;
    "cutoff" cutoff;
    "purif_maxiter" purif_maxiter;
    "purif_tol" purif_tol;
    "num_x" num_x;
    "num_avg" num_avg;
    "reduce" string(reduce);
]

writedlm(density_file, res.density, ',')
writedlm(times_file, res.times, ',')
writedlm(pos_file, hcat(res.centers, res.centers .- 1), ',')
writedlm(mlink_file, hcat(res.times, res.maxlinkdims), ',')
writedlm(params_file, params_rows, ',')

println("[$(now())] saved $(density_file)")
println("[$(now())] saved $(times_file)")
println("[$(now())] saved $(pos_file)")
println("[$(now())] saved $(mlink_file)")
println("[$(now())] saved $(params_file)")
println("[$(now())] Done.")
