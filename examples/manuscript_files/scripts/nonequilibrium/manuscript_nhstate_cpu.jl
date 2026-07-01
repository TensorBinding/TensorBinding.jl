#!/usr/bin/env julia
# APSOS_nhstate_cpu.jl
#
# CPU TDVP propagation of localized single-particle states for the modulated-loss
# open AAH chain -- small-scale counterpart of APSOS_nhstate_gpu.jl.
# Scaffolded from panel (d) of APSOS_NH_testing.ipynb.
#
# Three runs: no-loss reference packet, low-loss packet, high-loss packet.
# Uses TensorBinding.evolve_with_tdvp(H::TBHamiltonian, ...) which internally
# applies the generator -im*H.mpo, matching the GPU convention.
#
# Output (OUTDIR):
#   NHSTATE_<run>_<tag>.csv        -- matrix: rows sampled positions, columns times
#                                     (abs amplitudes |<x|psi(t)>|)
#   norms_<run>_<tag>.csv          -- columns: time, ||psi(t)||
#   maxlinkdim_<run>_<tag>.csv     -- columns: time, MPS maxlinkdim
#   times_<tag>.csv                -- sampled times
#   positions_<tag>.csv            -- columns: center_1based, x_0based
#   packets_<tag>.csv              -- run label and packet basis sites
#   cmax_<tag>.csv                 -- max |<x|psi(t)>| from no-loss run
#
# Usage:
#   julia --project=@. APSOS_nhstate_cpu.jl \
#     L t gamma0 loss_b loss_phase nsteps dt sample_every \
#     maxdim cutoff normalize_each_step \
#     low_center high_center OUTDIR [aah_V aah_phi aah_b loss_harmonics]
#
# All arguments are optional; defaults give a small-scale T=15 run at L=7.

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
L                   = length(ARGS) >= 1  ? parse(Int,     ARGS[1])  : 7
t_hop               = length(ARGS) >= 2  ? parse(Float64, ARGS[2])  : 1.0
gamma0              = length(ARGS) >= 3  ? parse(Float64, ARGS[3])  : 1.0
golden_b            = (1 + sqrt(5.0)) / 2
comm_loss_b         = apsos_commensurate_loss_b()
loss_b              = length(ARGS) >= 4  ? parse(Float64, ARGS[4])  : comm_loss_b
loss_phase          = length(ARGS) >= 5  ? parse(Float64, ARGS[5])  : 0.0
nsteps              = length(ARGS) >= 6  ? parse(Int,     ARGS[6])  : 600
dt                  = length(ARGS) >= 7  ? parse(Float64, ARGS[7])  : 0.025
sample_every        = length(ARGS) >= 8  ? parse(Int,     ARGS[8])  : 1
maxdim              = length(ARGS) >= 9  ? parse(Int,     ARGS[9])  : 50
cutoff              = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 1e-10
normalize_each_step = length(ARGS) >= 11 ? parse_bool_arg(ARGS[11]) : false
low_center_arg      = length(ARGS) >= 12 ? parse(Int,     ARGS[12]) : -1
high_center_arg     = length(ARGS) >= 13 ? parse(Int,     ARGS[13]) : -1
OUTDIR              = length(ARGS) >= 14 ? ARGS[14]                 : "results/nonequilibrium"
aah_V               = length(ARGS) >= 15 ? parse(Float64, ARGS[15]) : 0.5
aah_phi             = length(ARGS) >= 16 ? parse(Float64, ARGS[16]) : 0.0
aah_b               = length(ARGS) >= 17 ? parse(Float64, ARGS[17]) : golden_b
loss_harmonics      = length(ARGS) >= 18 ? parse(Int, ARGS[18]) : 15

mkpath(OUTDIR)

N_sites = 2^L
low_center  = low_center_arg  < 0 ? div(N_sites, 2) : mod(low_center_arg,  N_sites)
high_center = high_center_arg < 0 ? div(N_sites, 4) : mod(high_center_arg, N_sites)
loss_profile, loss_period_sites = apsos_cusped_loss_profile(N_sites, gamma0, loss_b;
    loss_phase=loss_phase,
    loss_harmonics=loss_harmonics)

println("[$(now())] APSOS_nhstate_cpu starting")
println("[info] L=$L  N=$N_sites")
println("[info] AAH open chain: t=$t_hop  V=$aah_V  phi=$aah_phi  b=$aah_b")
println("[info] loss: gamma0=$gamma0  loss_b=$loss_b  period_sites=$loss_period_sites  loss_phase=$loss_phase  harmonics=$loss_harmonics")
println("[info] nsteps=$nsteps  dt=$dt  sample_every=$sample_every  T=$(nsteps*dt)")
println("[info] maxdim=$maxdim  cutoff=$cutoff  normalize_each_step=$normalize_each_step")
println("[info] packet centers, 0-based: low=$low_center  high=$high_center")
println("[info] OUTDIR=$OUTDIR")

# -- build Hamiltonians -------------------------------------------------------
println("[info] Building Hermitian open AAH chain...")
aah_params = (V=aah_V, phi=aah_phi, t=t_hop,
              b=aah_b, tol_quantics=cutoff, maxbonddim_quantics=maxdim)
@time H_herm = TensorBinding.get_Hamiltonian("aah", aah_params;
    L=L,
    tol=cutoff,
    maxdim=maxdim)
println("[info] H_herm bond dim: $(ITensorMPS.maxlinkdim(H_herm.mpo))")

println("[info] Building NH modulated-loss chain...")
H_nh = deepcopy(H_herm)
@time TensorBinding.add_loss!(H_nh, loss_profile; maxdim=maxdim, tol=cutoff)
println("[info] H_nh bond dim: $(ITensorMPS.maxlinkdim(H_nh.mpo))")

# -- initial two-site packets -------------------------------------------------
function two_site_packet(H, center0::Int; init_cutoff::Real = 1e-12)
    N = H.N
    right0 = clamp(center0, 0, N - 1)
    left0  = clamp(center0 - 1, 0, N - 1)
    left0 == right0 && (right0 = clamp(center0 + 1, 0, N - 1))
    psi = normalize(+(
        TensorBinding.binary_to_MPS(left0,  L, H.sites),
        TensorBinding.binary_to_MPS(right0, L, H.sites);
        cutoff = init_cutoff))
    return psi, left0, right0
end

psi_low,  low_left,  low_right  = two_site_packet(H_herm, low_center)
psi_high, high_left, high_right = two_site_packet(H_herm, high_center)

runs = [
    (; label="noloss",   H=H_herm, psi=psi_low,  center=low_center,  left=low_left,  right=low_right),
    (; label="lowloss",  H=H_nh,   psi=psi_low,  center=low_center,  left=low_left,  right=low_right),
    (; label="highloss", H=H_nh,   psi=psi_high, center=high_center, left=high_left, right=high_right),
]

tag = "cpu_L$(L)_g$(apsos_tagnum(gamma0))_ph$(apsos_tagnum(loss_phase))_lh$(loss_harmonics)"

# -- shared output arrays (written once) --------------------------------------
# Sampled time indices: 0, sample_every, 2*sample_every, ... up to nsteps
sampled_steps = 0:sample_every:nsteps          # 0-based step indices
n_times       = length(sampled_steps)
times_vec     = Float64.(sampled_steps) .* dt   # physical times

# All N sites (0-based): positions file columns are (1-based center, 0-based x)
x_0based  = collect(0:N_sites-1)
x_1based  = x_0based .+ 1
positions_mat = hcat(x_1based, x_0based)        # (N_sites, 2)

times_written     = false
positions_written = false
packet_rows       = Matrix{Any}(undef, length(runs), 5)
cmax_noloss       = NaN

# -- precompute all basis states once (shared across runs) --------------------
println("[info] Precomputing $N_sites basis MPS...")
@time basis_states = [TensorBinding.binary_to_MPS(i - 1, L, H_herm.sites) for i in 1:N_sites]

# -- main loop ----------------------------------------------------------------
for (irun, run) in enumerate(runs)
    global times_written, positions_written, cmax_noloss

    println("[$(now())] Running CPU TDVP: $(run.label)")

    # TDVP trajectory — evolve_with_tdvp(H::TBHamiltonian, ...) uses -im*H.mpo
    all_states = Vector{MPS}(undef, nsteps + 1)
    @time all_states = TensorBinding.evolve_with_tdvp(
        run.H, run.psi, nsteps, Float64(dt);
        normalize_each_step = normalize_each_step,
        maxdim = maxdim,
        cutoff = cutoff,
        outputlevel = 0,
    )

    # Subsample states and compute overlaps + norms
    amp_mat  = Matrix{Float64}(undef, N_sites, n_times)   # (n_pos, n_times)
    norm_vec = Vector{Float64}(undef, n_times)
    mld_vec  = Vector{Int}(undef, n_times)

    for (it, step) in enumerate(sampled_steps)
        psi = all_states[step + 1]   # step 0 → index 1
        nrm2 = real(inner(psi, psi))
        norm_vec[it] = sqrt(max(nrm2, 0.0))
        mld_vec[it]  = ITensorMPS.maxlinkdim(psi)
        for ix in 1:N_sites
            amp_mat[ix, it] = abs(inner(basis_states[ix], psi))
        end
    end

    amp_file   = joinpath(OUTDIR, "NHSTATE_$(run.label)_$(tag).csv")
    norm_file  = joinpath(OUTDIR, "norms_$(run.label)_$(tag).csv")
    mlink_file = joinpath(OUTDIR, "maxlinkdim_$(run.label)_$(tag).csv")

    writedlm(amp_file,   amp_mat,                      ',')
    writedlm(norm_file,  hcat(times_vec, norm_vec),    ',')
    writedlm(mlink_file, hcat(times_vec, mld_vec),     ',')

    println("[$(now())] saved $(amp_file)  shape=$(size(amp_mat))")
    println("[$(now())] saved $(norm_file)")
    println("[$(now())] saved $(mlink_file)")
    println("[info] $(run.label) norm range: [$(minimum(norm_vec)), $(maximum(norm_vec))]")

    if !times_written
        times_file = joinpath(OUTDIR, "times_$(tag).csv")
        writedlm(times_file, times_vec, ',')
        println("[$(now())] saved $(times_file)")
        times_written = true
    end

    if !positions_written
        pos_file = joinpath(OUTDIR, "positions_$(tag).csv")
        writedlm(pos_file, positions_mat, ',')
        println("[$(now())] saved $(pos_file)")
        positions_written = true
    end

    run.label == "noloss" && (cmax_noloss = maximum(amp_mat))
    packet_rows[irun, :] = Any[run.label, run.center, run.left, run.right,
                                run.H === H_herm ? "hermitian" : "nh_loss"]
end

packet_file = joinpath(OUTDIR, "packets_$(tag).csv")
cmax_file   = joinpath(OUTDIR, "cmax_$(tag).csv")
params_file = joinpath(OUTDIR, "params_$(tag).csv")
params_rows = Any[
    "L" L;
    "N_sites" N_sites;
    "t_hop" t_hop;
    "aah_V" aah_V;
    "aah_phi" aah_phi;
    "aah_b" aah_b;
    "gamma0" gamma0;
    "loss_b" loss_b;
    "loss_phase" loss_phase;
    "loss_harmonics" loss_harmonics;
    "loss_period_sites" loss_period_sites;
    "nsteps" nsteps;
    "dt" dt;
    "sample_every" sample_every;
    "maxdim" maxdim;
    "cutoff" cutoff;
    "normalize_each_step" normalize_each_step;
    "low_center" low_center;
    "high_center" high_center;
]
writedlm(packet_file, packet_rows, ',')
writedlm(cmax_file, ["cmax_noloss" cmax_noloss], ',')
writedlm(params_file, params_rows, ',')
println("[$(now())] saved $(packet_file)")
println("[$(now())] saved $(cmax_file)")
println("[$(now())] saved $(params_file)")
println("[$(now())] Done.  T=$(nsteps*dt)  cmax_noloss=$cmax_noloss")
