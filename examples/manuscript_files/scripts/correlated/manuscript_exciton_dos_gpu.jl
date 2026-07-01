#!/usr/bin/env julia
# APSOS_exciton_dos_gpu.jl
#
# GPU-accelerated stochastic exciton DOS via MPS Chebyshev KPM with the HODC
# (Higher-Order Delta Chebyshev) reconstruction kernel.
#
# The exciton Hamiltonian lives on a 2L-site interleaved electron–hole space
# (H = H_c ⊗ I − I ⊗ H_v + U). The on-site potential V(x) = V0·(1 + 0.2·cos(k1·x))
# is a uniform V0 with a 20% cosine ripple of period N/4 (type-I confinement).
# get_dos_stochastic_gpu samples N_sample random basis states of the full
# Hilbert space (scattering continuum) plus N_bound random |x,x> states (bound
# sector). This production script saves the unweighted sampled signal
# (dos_weighting=:sample), so the bound contribution is visible instead of being
# suppressed by the continuum phase-space factor. The HODC kernel gives sharper
# spectral features than Jackson at fixed Ncheb.
#
# Output (OUTDIR):
#   excitonDOS_<tag>.csv   — two columns: omega, sample-weighted DOS
#
# Usage:
#   julia --project=@. APSOS_exciton_dos_gpu.jl \
#     L t U V0 scale Ncheb maxdim Nomega Emin Emax N_sample N_bound seed eta m_order cutoff domainwall OUTDIR eta_shift
#
# Defaults (all optional, fall back to small test values):
#   L=5  t=-1.0  U=6.0  V0=1.5  scale=10.0
#   Ncheb=100  maxdim=100  Nomega=200  Emin=-10.0  Emax=10.0
#   N_sample=30  N_bound=20  seed=42  eta=0.0(→1/(Ncheb+1-eta_shift))  m_order=6  cutoff=1e-4
#   domainwall=false (0/1 or true/false)  OUTDIR=outputs_exciton_dos_gpu

using CUDA   # must be first — enables the NDTensors GPU backend
using Dates, LinearAlgebra, DelimitedFiles
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "../../../src/TensorBinding.jl"))
using .TensorBinding

# ── command-line arguments ────────────────────────────────────────────────────
L        = length(ARGS) >= 1  ? parse(Int,     ARGS[1])  : 5
t        = length(ARGS) >= 2  ? parse(Float64, ARGS[2])  : -1.0
U        = length(ARGS) >= 3  ? parse(Float64, ARGS[3])  : 6.0
V0       = length(ARGS) >= 4  ? parse(Float64, ARGS[4])  : 1.5
scale    = length(ARGS) >= 5  ? parse(Float64, ARGS[5])  : 10.0
Ncheb    = length(ARGS) >= 6  ? parse(Int,     ARGS[6])  : 100
maxdim   = length(ARGS) >= 7  ? parse(Int,     ARGS[7])  : 100
Nomega   = length(ARGS) >= 8  ? parse(Int,     ARGS[8])  : 200
Emin     = length(ARGS) >= 9  ? parse(Float64, ARGS[9])  : -10.0
Emax     = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) :  10.0
N_sample = length(ARGS) >= 11 ? parse(Int,     ARGS[11]) : 30
N_bound  = length(ARGS) >= 12 ? parse(Int,     ARGS[12]) : 20
seed     = length(ARGS) >= 13 ? parse(Int,     ARGS[13]) : 42
eta      = length(ARGS) >= 14 ? parse(Float64, ARGS[14]) : 0.0   # 0 → 1/(Ncheb+1)
m_order  = length(ARGS) >= 15 ? parse(Int,     ARGS[15]) : 6
cutoff   = length(ARGS) >= 16 ? parse(Float64, ARGS[16]) : 1e-4
domainwall = length(ARGS) >= 17 ? lowercase(ARGS[17]) in ("1", "true", "yes") : false
OUTDIR   = length(ARGS) >= 18 ? ARGS[18]                 : "outputs_exciton_dos_gpu"
eta_shift = length(ARGS) >= 19 ? parse(Float64, ARGS[19]) : 0.0

omegalist = range(Emin, Emax; length=Nomega)
mkpath(OUTDIR)

function effective_hodc_eta(Ncheb::Int, eta::Float64, eta_shift::Float64)
    eta_shift >= 0.0 || error("eta_shift=$eta_shift must be nonnegative")
    if eta != 0.0
        eta_shift == 0.0 ||
            @warn "eta_shift=$eta_shift ignored because explicit eta=$eta was provided"
        return eta
    end
    denom = Ncheb + 1 - eta_shift
    denom > 0.0 ||
        error("eta_shift=$eta_shift must be smaller than Ncheb+1=$(Ncheb + 1)")
    return 1 / denom
end

eta_eff = effective_hodc_eta(Ncheb, eta, eta_shift)

println("[$(now())] APSOS_exciton_dos_gpu starting")
println("[info] GPU device: $(CUDA.name(CUDA.device()))")
println("[info] L=$L  t=$t  U=$U  V0=$V0  scale=$scale  domainwall=$domainwall")
println("[info] Ncheb=$Ncheb  maxdim=$maxdim  Nomega=$Nomega  E=[$Emin, $Emax]")
println("[info] N_sample=$N_sample  N_bound=$N_bound  seed=$seed")
println("[info] kernel=:hodc  eta=$eta  eta_shift=$eta_shift  eta_eff=$eta_eff  m_order=$m_order  cutoff=$cutoff")
println("[info] dos_weighting=:sample  normalize=false")
println("[info] output split: continuum x_e!=x_h and bound x_e=x_h")

# ── build exciton Hamiltonian (CPU) ───────────────────────────────────────────
# V(x) = V0·(1 + 0.1·(cos(k1·x) + cos(k2·x))): two incommensurate cosines that
# produce a mini-band structure in the DOS (b1 small scale, b2 large scale,
# N = 2^L sites; x is 1-indexed). With domainwall=true the large-scale wavevector
# k2 is ramped smoothly across x = N/2 (tanh kink). V enters +V on the electron
# and −V on the valence sector, so after H_c − H_v both carriers feel +V (type-I).
N_sites = 2^L
function Vx(x; V0=V0, N=N_sites, domainwall=domainwall)
    b1 = 3*sqrt(5)/2        # small incommensurate scale → mini-band structure in the DOS
    k1 = 2*pi/b1
    b2 = sqrt(3)*(N)/10     # large incommensurate scale
    k2 = 2*pi/b2
    if domainwall == true
        Xdw = N/2
        W   = 1/sqrt(N)
        k2 *= (1 + 0.3*tanh((x + 0.5 - Xdw)/W))
    end
    return V0*(1 + 0.1*(cos(k1*x) + cos(k2*x)))
end
println("[info] Building exciton Hamiltonian...")
@time H_exc = TensorBinding.exciton_hamiltonian("chain_1d", t, x -> U;
                                                L       = L,
                                                on_site = Vx,
                                                scale   = scale,
                                                maxdim  = maxdim)
println("[info] $H_exc")

# ── GPU warm-up ───────────────────────────────────────────────────────────────
println("[info] GPU warm-up...")
warmup_s = siteinds("S=1/2", 4)
warmup_H = MPO(warmup_s, "Id")
@time CUDA.cu(warmup_H)
CUDA.synchronize()
println("[info] Warm-up complete.")

# ── stochastic exciton DOS (GPU, HODC kernel) ─────────────────────────────────
println("[info] Running continuum stochastic KPM ($N_sample samples, x_e != x_h)...")
@time dos_continuum = TensorBinding.get_dos_stochastic_gpu(H_exc, Ncheb, omegalist;
    N_sample  = N_sample,
    N_bound   = 0,
    seed      = seed,
    kernel    = :hodc,
    eta       = eta_eff,
    m_order   = m_order,
    normalize = false,
    dos_weighting = :sample,
    continuum_only = true,
    maxdim    = maxdim,
    cutoff    = cutoff,
    printinfo = true)

println("[info] continuum dos shape: $(size(dos_continuum))")

bound_seed = seed + 1
println("[info] Running bound stochastic KPM ($N_bound samples, x_e = x_h; seed=$bound_seed)...")
@time dos_bound = TensorBinding.get_dos_stochastic_gpu(H_exc, Ncheb, omegalist;
    N_sample  = 0,
    N_bound   = N_bound,
    seed      = bound_seed,
    kernel    = :hodc,
    eta       = eta_eff,
    m_order   = m_order,
    normalize = false,
    dos_weighting = :sample,
    maxdim    = maxdim,
    cutoff    = cutoff,
    printinfo = true)

dos_total = dos_continuum .+ dos_bound
println("[info] bound dos shape: $(size(dos_bound))")

# ── save ──────────────────────────────────────────────────────────────────────
tag = "L$(L)_t$(t)_U$(U)_V0$(V0)_Nc$(Ncheb)_mdim$(maxdim)_Ns$(N_sample)_Nb$(N_bound)_m$(m_order)_eta$(round(eta_eff; sigdigits=4))_seed$(seed)_wSplitSample"

continuum_file = joinpath(OUTDIR, "excitonDOS_continuum_$(tag).csv")
bound_file     = joinpath(OUTDIR, "excitonDOS_bound_$(tag).csv")
total_file     = joinpath(OUTDIR, "excitonDOS_$(tag).csv")

writedlm(continuum_file, hcat(collect(omegalist), dos_continuum), ',')
writedlm(bound_file,     hcat(collect(omegalist), dos_bound), ',')
writedlm(total_file,     hcat(collect(omegalist), dos_total), ',')

println("[$(now())] saved $(continuum_file)")
println("[$(now())] saved $(bound_file)")
println("[$(now())] saved $(total_file)")
