#!/usr/bin/env julia
# APSOS_exciton_ldos_gpu.jl
#
# GPU-accelerated spatial exciton LDOS A(X,ω) = ⟨X,X|δ(ω−H)|X,X⟩ via MPS Chebyshev
# KPM with the HODC (Higher-Order Delta Chebyshev) reconstruction kernel.
#
# The exciton Hamiltonian lives on a 2L-site interleaved electron–hole space
# (H = H_c ⊗ I − I ⊗ H_v + U). The on-site potential V(x) = V0·(1 + 0.1·(cos(k1·x)
# + cos(k2·x))) is two incommensurate cosines (mini-band structure in the DOS),
# with an optional tanh domain wall on the large-scale wavevector at x = N/2.
# get_exciton_ldos_spatial_gpu runs one GPU Chebyshev recursion per bound exciton
# position X (electron = hole = X) and reconstructs A(X,ω) for that |X,X⟩ probe.
#
# Output (all files in OUTDIR):
#   exciton_ldos_<tag>.csv  — (Nω × num_x) spectral weight, row = ω, col = X
#   omega_<tag>.csv         — (Nω × 1) omega grid
#   positions_<tag>.csv     — (num_x × 1) sampled exciton positions X (1-indexed)
#
# Usage:
#   julia --project=@. APSOS_exciton_ldos_gpu.jl \
#     L t U V0 scale Ncheb maxdim Nomega Emin Emax num_x eta m_order cutoff domainwall OUTDIR \
#     num_avg x_start x_end eta_shift
#
# Defaults (all optional, fall back to small test values):
#   L=5  t=-1.0  U=6.0  V0=1.5  scale=10.0
#   Ncheb=100  maxdim=100  Nomega=200  Emin=-10.0  Emax=10.0
#   num_x=32  eta=0.0(→1/(Ncheb+1-eta_shift))  m_order=6  cutoff=1e-4
#   domainwall=false (0/1 or true/false)  OUTDIR=outputs_exciton_ldos_gpu
#   num_avg=1  x_start=1  x_end=2^L

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
num_x    = length(ARGS) >= 11 ? parse(Int,     ARGS[11]) : 32   # sampled exciton positions
eta      = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : 0.0  # 0 → 1/(Ncheb+1)
m_order  = length(ARGS) >= 13 ? parse(Int,     ARGS[13]) : 6
cutoff   = length(ARGS) >= 14 ? parse(Float64, ARGS[14]) : 1e-4
domainwall = length(ARGS) >= 15 ? lowercase(ARGS[15]) in ("1", "true", "yes") : false
OUTDIR   = length(ARGS) >= 16 ? ARGS[16]                 : "outputs_exciton_ldos_gpu"
num_avg  = length(ARGS) >= 17 ? parse(Int,     ARGS[17]) : 1

N_sites = 2^L
x_start = length(ARGS) >= 18 ? parse(Int,     ARGS[18]) : 1
x_end   = length(ARGS) >= 19 ? parse(Int,     ARGS[19]) : N_sites
eta_shift = length(ARGS) >= 20 ? parse(Float64, ARGS[20]) : 0.0

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

println("[$(now())] APSOS_exciton_ldos_gpu starting")
println("[info] GPU device: $(CUDA.name(CUDA.device()))")
println("[info] L=$L  t=$t  U=$U  V0=$V0  scale=$scale  domainwall=$domainwall")
println("[info] Ncheb=$Ncheb  maxdim=$maxdim  Nomega=$Nomega  E=[$Emin, $Emax]")
println("[info] num_x=$num_x  num_avg=$num_avg  x_range=[$x_start, $x_end]")
println("[info] kernel=:hodc  eta=$eta  eta_shift=$eta_shift  eta_eff=$eta_eff  m_order=$m_order  cutoff=$cutoff")

# ── build exciton Hamiltonian (CPU) ───────────────────────────────────────────
# V(x) = V0·(1 + 0.1·(cos(k1·x) + cos(k2·x))): two incommensurate cosines that
# produce a mini-band structure in the DOS (b1 small scale, b2 large scale,
# N = 2^L sites; x is 1-indexed). With domainwall=true the large-scale wavevector
# k2 is ramped smoothly across x = N/2 (tanh kink). V enters +V on the electron
# and −V on the valence sector, so after H_c − H_v both carriers feel +V (type-I).
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

# ── sampled exciton positions X (electron = hole = X) ─────────────────────────
function build_exciton_x_groups(N::Int; num_x::Int, num_avg::Int, x_start::Int, x_end::Int)
    num_x > 0 || error("num_x must be positive")
    num_avg > 0 || error("num_avg must be positive")
    1 <= x_start <= x_end <= N || error("expected 1 <= x_start <= x_end <= N")
    window = x_end - x_start + 1
    num_x <= window || error("num_x=$num_x exceeds sampling window length $window")
    dx = div(window, num_x)
    dx_sub = max(1, div(dx, num_avg))
    return [[x_start + (i - 1) * dx + k * dx_sub
             for k in 0:num_avg-1
             if x_start + (i - 1) * dx + k * dx_sub <= x_end]
            for i in 1:num_x]
end

X_groups = build_exciton_x_groups(H_exc.N;
                                  num_x=num_x,
                                  num_avg=num_avg,
                                  x_start=x_start,
                                  x_end=x_end)
X_centers = first.(X_groups)
println("[info] $(length(X_groups)) coarse exciton positions in [$x_start, $x_end]")
println("[info] spatial averaging: total probes=$(sum(length, X_groups))")

# ── spatial exciton LDOS (GPU, HODC kernel) ───────────────────────────────────
println("[info] Computing exciton LDOS (GPU)...")
@time ldos = TensorBinding.get_exciton_ldos_spatial_gpu(H_exc, Ncheb, omegalist;
    X_groups  = X_groups,
    kernel    = :hodc,
    eta       = eta_eff,
    m_order   = m_order,
    maxdim    = maxdim,
    cutoff    = cutoff,
    printinfo = true)

println("[info] ldos shape: $(size(ldos))")

# ── save ──────────────────────────────────────────────────────────────────────
tag = "L$(L)_t$(t)_U$(U)_V0$(V0)_Nc$(Ncheb)_mdim$(maxdim)_nx$(num_x)_navg$(num_avg)_x$(x_start)-$(x_end)_m$(m_order)_eta$(round(eta_eff; sigdigits=4))_dw$(domainwall)"

ldos_file = joinpath(OUTDIR, "exciton_ldos_$(tag).csv")
writedlm(ldos_file, ldos, ',')

omega_file = joinpath(OUTDIR, "omega_$(tag).csv")
writedlm(omega_file, collect(omegalist), ',')

pos_file = joinpath(OUTDIR, "positions_$(tag).csv")
writedlm(pos_file, X_centers, ',')

println("[$(now())] saved $(ldos_file)")
println("[$(now())] saved $(omega_file)")
println("[$(now())] saved $(pos_file)")
