#!/usr/bin/env julia
# APSOS_nhstate_gpu.jl
#
# GPU TDVP propagation of a comb single-particle state for the modulated-loss
# open AAH chain. This is the production-script version of panel (d) in
# APSOS_NH_testing:
#   no-loss reference comb, lossy comb.
#
# Output (OUTDIR):
#   NHSTATE_<run>_<tag>.csv        -- matrix: rows sampled positions, columns times
#   times_<tag>.csv                -- sampled times
#   positions_<tag>.csv            -- columns: center_1based, x_0based
#   norms_<run>_<tag>.csv          -- columns: time, ||psi(t)||
#   maxlinkdim_<run>_<tag>.csv     -- columns: time, state-MPS maxlinkdim
#   packets_<tag>.csv              -- run label and comb/window metadata
#   init_sites_<tag>.csv           -- 0-based sites occupied by the initial paired comb
#
# Usage:
#   julia --project=@. APSOS_nhstate_gpu.jl \
#     L t gamma0 loss_b loss_phase nsteps dt sample_every \
#     maxdim cutoff num_x num_avg reduce component normalize_each_step \
#     init_stride sample_window OUTDIR [aah_V aah_phi aah_b loss_harmonics]
#
# The default initial state is a normalized paired comb with support on
# (|n> + |n+1>) / sqrt(2) for n = 0, init_stride, 2*init_stride, ... .
# The sampled output window is centered in the chain and has `sample_window` sites.

using CUDA   # must be first -- enables the NDTensors GPU backend
using Dates, LinearAlgebra, DelimitedFiles
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "../../../src/TensorBinding.jl"))
using .TensorBinding
include(joinpath(@__DIR__, "APSOS_NH_loss.jl"))

function parse_bool_arg(x)
    s = lowercase(String(x))
    s in ("1", "true", "t", "yes", "y") && return true
    s in ("0", "false", "f", "no", "n") && return false
    error("Cannot parse boolean argument '$x'. Use true/false or 1/0.")
end

# -- command-line arguments ---------------------------------------------------
L                   = length(ARGS) >= 1  ? parse(Int,     ARGS[1])  : 20
t                   = length(ARGS) >= 2  ? parse(Float64, ARGS[2])  : 1.0
gamma0              = length(ARGS) >= 3  ? parse(Float64, ARGS[3])  : 1.0
golden_b            = (1 + sqrt(5.0)) / 2
comm_loss_b         = apsos_commensurate_loss_b()
loss_b              = length(ARGS) >= 4  ? parse(Float64, ARGS[4])  : comm_loss_b
loss_phase          = length(ARGS) >= 5  ? parse(Float64, ARGS[5])  : 0.0
nsteps              = length(ARGS) >= 6  ? parse(Int,     ARGS[6])  : 600
dt                  = length(ARGS) >= 7  ? parse(Float64, ARGS[7])  : 0.025
sample_every        = length(ARGS) >= 8  ? parse(Int,     ARGS[8])  : 1
maxdim              = length(ARGS) >= 9  ? parse(Int,     ARGS[9])  : 100
cutoff              = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 1e-6
num_x               = length(ARGS) >= 11 ? parse(Int,     ARGS[11]) : 256
num_avg             = length(ARGS) >= 12 ? parse(Int,     ARGS[12]) : 1
reduce_arg          = length(ARGS) >= 13 ? ARGS[13]                 : "point"
component_arg       = length(ARGS) >= 14 ? ARGS[14]                 : "real"
pointavg_arg        = length(ARGS) >= 15 ? ARGS[15]                 : "complex"
normalize_each_step = length(ARGS) >= 16 ? parse_bool_arg(ARGS[16]) : false
init_stride         = length(ARGS) >= 17 ? parse(Int,     ARGS[17]) : 64
sample_window       = length(ARGS) >= 18 ? parse(Int,     ARGS[18]) : 256
OUTDIR              = length(ARGS) >= 19 ? ARGS[19]                 : "outputs_nhstate_gpu"
aah_V               = length(ARGS) >= 20 ? parse(Float64, ARGS[20]) : 0.5
aah_phi             = length(ARGS) >= 21 ? parse(Float64, ARGS[21]) : 0.0
aah_b               = length(ARGS) >= 22 ? parse(Float64, ARGS[22]) : golden_b
loss_harmonics      = length(ARGS) >= 23 ? parse(Int, ARGS[23]) : 15

reduce    = Symbol(replace(reduce_arg,    ":" => ""))
component = Symbol(replace(component_arg, ":" => ""))
pointavg  = Symbol(replace(pointavg_arg,  ":" => ""))
reduce    in (:point, :block)                         || error("reduce must be point or block, got $reduce")
component in (:real, :imag, :abs, :abs2, :probability) || error("component must be real, imag, abs, abs2, or probability, got $component")
pointavg  in (:complex, :abs, :abs2)                  || error("pointavg must be complex, abs, or abs2, got $pointavg")

mkpath(OUTDIR)

N_sites = 2^L
init_stride > 0 || error("init_stride must be positive, got $init_stride")
sample_window > 0 || error("sample_window must be positive, got $sample_window")
sample_window <= N_sites || error("sample_window=$sample_window exceeds N=$N_sites")
sample_start0 = div(N_sites - sample_window, 2)
sample_end0   = sample_start0 + sample_window - 1
x_start_sample = sample_start0 + 1
x_end_sample   = sample_end0 + 1
reduce == :block && sample_window < N_sites &&
    error("central finite-window sampling requires reduce=point; got reduce=:block with sample_window=$sample_window")
loss_profile, loss_period_sites = apsos_cusped_loss_profile(N_sites, gamma0, loss_b;
    loss_phase=loss_phase,
    loss_harmonics=loss_harmonics)

println("[$(now())] APSOS_nhstate_gpu starting")
println("[info] GPU device: $(CUDA.name(CUDA.device()))")
println("[info] L=$L N=$N_sites")
println("[info] AAH open chain: t=$t  V=$aah_V  phi=$aah_phi  b=$aah_b")
println("[info] loss: gamma0=$gamma0  loss_b=$loss_b  period_sites=$loss_period_sites  loss_phase=$loss_phase  harmonics=$loss_harmonics")
println("[info] nsteps=$nsteps  dt=$dt  sample_every=$sample_every")
println("[info] maxdim=$maxdim  cutoff=$cutoff  normalize_each_step=$normalize_each_step")
println("[info] sampling: reduce=$reduce  num_x=$num_x  num_avg=$num_avg  component=$component")
println("[info] initial paired-comb stride=$init_stride")
println("[info] sample window: x0=$sample_start0:$sample_end0  width=$sample_window")

# -- build Hamiltonians on CPU ------------------------------------------------
println("[info] Building Hermitian open AAH chain...")
aah_params = (V=aah_V, phi=aah_phi, t=t,
              b=aah_b, tol_quantics=cutoff, maxbonddim_quantics=maxdim)
@time H_herm = TensorBinding.get_Hamiltonian("aah", aah_params;
    L=L,
    tol=cutoff,
    maxdim=maxdim)
println("[info] H_herm: $H_herm")
println("[info] Hermitian MPO bond dim: $(ITensorMPS.maxlinkdim(H_herm.mpo))")

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

# -- initial paired-comb state ------------------------------------------------
function stride_comb_state(H, stride::Int; offset0::Int = 0)
    stride_bits = round(Int, log2(stride))
    2^stride_bits == stride ||
        error("init_stride=$stride is not a power of two; exact product comb requires stride=2^p.")
    stride_bits <= H.L ||
        error("init_stride=$stride exceeds the system size N=$(H.N).")
    offset0 < stride ||
        error("comb offset0=$offset0 must satisfy offset0 < init_stride=$stride.")

    amp = 1 / sqrt(2.0)
    tensors = Vector{ITensor}(undef, H.L)
    for j in 1:H.L
        s = H.sites[j]
        T = ITensor(Float64, s)
        if j <= H.L - stride_bits
            T[s => 1] = amp
            T[s => 2] = amp
        else
            bit = (offset0 >> (H.L - j)) & 1
            T[s => 1] = bit == 0 ? 1.0 : 0.0
            T[s => 2] = bit == 1 ? 1.0 : 0.0
        end
        tensors[j] = T
    end
    return MPS(tensors)
end

# Explicitly build the requested state from two old-style combs:
#   sum_q (|q*stride> + |q*stride - 1>) / sqrt(2 * num_teeth).
function stride_pair_comb_state(H, stride::Int; init_cutoff::Real = 1e-12)
    stride >= 2 ||
        error("paired comb requires init_stride >= 2 to avoid overlapping |n> + |n-1> teeth.")
    comb0 = stride_comb_state(H, stride; offset0=0)
    comb1 = stride_comb_state(H, stride; offset0=-1)
    return normalize(+(comb0, comb1; cutoff=init_cutoff))
end

psi_comb = stride_pair_comb_state(H_herm, init_stride)
init_teeth = collect(0:init_stride:(N_sites - 1))
init_sites = sort!(vcat(init_teeth, init_teeth .+ 1))

runs = [
    (; label="noloss", H=H_herm, psi=psi_comb, hkind="hermitian"),
    (; label="loss",   H=H_nh,   psi=psi_comb, hkind="nh_loss"),
]

function tagnum(x)
    s = string(round(Float64(x); sigdigits=5))
    return replace(s, "." => "p", "-" => "m", "+" => "p")
end

tag = "L$(L)_g$(apsos_tagnum(gamma0))_ph$(apsos_tagnum(loss_phase))_lh$(loss_harmonics)_c$(init_stride)_w$(sample_window)"

times_written = false
positions_written = false
packet_rows = Matrix{Any}(undef, length(runs), 6)
cmax_noloss = NaN

for (irun, run) in enumerate(runs)
    global times_written, positions_written, cmax_noloss

    println("[info] Running GPU TDVP state trajectory: $(run.label)")
    res_ref = Ref{Any}()
    @time begin
        res_ref[] = TensorBinding.get_state_amplitude_trajectory_gpu(run.H, run.psi;
            nsteps=nsteps,
            dt=dt,
            sample_every=sample_every,
            num_x=num_x,
            num_avg=num_avg,
            reduce=reduce,
            x_start=x_start_sample,
            x_end=x_end_sample,
            component=component,
            pointavg=pointavg,
            normalize_each_step=normalize_each_step,
            maxdim=maxdim,
            cutoff=cutoff,
            printinfo=true)
        CUDA.synchronize()
    end
    res = res_ref[]
    println("[info] $(run.label) amplitude shape: $(size(res.amplitude))")
    println("[info] $(run.label) norm range: $(extrema(res.norms))")

    amp_file   = joinpath(OUTDIR, "NHSTATE_$(run.label)_$(tag).csv")
    norm_file  = joinpath(OUTDIR, "norms_$(run.label)_$(tag).csv")
    mlink_file = joinpath(OUTDIR, "maxlinkdim_$(run.label)_$(tag).csv")

    writedlm(amp_file, res.amplitude, ',')
    writedlm(norm_file, hcat(res.times, res.norms), ',')
    writedlm(mlink_file, hcat(res.times, res.maxlinkdims), ',')

    println("[$(now())] saved $(amp_file)")
    println("[$(now())] saved $(norm_file)")
    println("[$(now())] saved $(mlink_file)")

    if !times_written
        times_file = joinpath(OUTDIR, "times_$(tag).csv")
        writedlm(times_file, res.times, ',')
        println("[$(now())] saved $(times_file)")
        times_written = true
    end

    if !positions_written
        pos_file = joinpath(OUTDIR, "positions_$(tag).csv")
        writedlm(pos_file, hcat(res.centers, res.centers .- 1), ',')
        println("[$(now())] saved $(pos_file)")
        positions_written = true
    end

    run.label == "noloss" && (cmax_noloss = maximum(abs, res.amplitude))
    packet_rows[irun, :] = Any[run.label, init_stride, length(init_sites),
                                sample_start0, sample_end0, run.hkind]
end

packet_file = joinpath(OUTDIR, "packets_$(tag).csv")
cmax_file = joinpath(OUTDIR, "cmax_$(tag).csv")
init_file = joinpath(OUTDIR, "init_sites_$(tag).csv")
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
    "loss_period_sites" loss_period_sites;
    "loss_harmonics" loss_harmonics;
    "nsteps" nsteps;
    "dt" dt;
    "sample_every" sample_every;
    "maxdim" maxdim;
    "cutoff" cutoff;
    "num_x" num_x;
    "num_avg" num_avg;
    "reduce" string(reduce);
    "component" string(component);
    "normalize_each_step" normalize_each_step;
    "init_stride" init_stride;
    "sample_window" sample_window;
    "sample_start0" sample_start0;
    "sample_end0" sample_end0;
]
writedlm(packet_file, packet_rows, ',')
writedlm(cmax_file, ["cmax_noloss" cmax_noloss], ',')
writedlm(init_file, init_sites, ',')
writedlm(params_file, params_rows, ',')
println("[$(now())] saved $(packet_file)")
println("[$(now())] saved $(cmax_file)")
println("[$(now())] saved $(init_file)")
println("[$(now())] saved $(params_file)")
println("[$(now())] Done.")
