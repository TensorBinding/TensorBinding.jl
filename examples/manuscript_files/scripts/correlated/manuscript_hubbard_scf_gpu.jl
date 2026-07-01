#!/usr/bin/env julia
# APSOS_hubbard_scf_gpu.jl
#
# GPU-accelerated collinear magnetic mean-field SCF for the half-filled
# single-band Hubbard model on a 2^Lx x 2^Ly square lattice with spatially
# modulated nearest-neighbor hopping:
#
#     t(ix) = t0 * (1 + A*cos(2*pi*ix/(Nx/4)))
#
# The per-iteration spin-resolved density matrices are obtained by McWeeny
# purification on GPU in F32. Mean-field MPO assembly and density mixing stay on
# CPU. The loop is seeded with a 2D Neel density to break spin symmetry.
#
# Output (all files in OUTDIR):
#   hubbard_mag_<tag>.csv  : rows ix, iy, n_up, n_dn, m_loc=(n_up-n_dn)/2
#   scf_history_<tag>.csv  : rows iter, rms_error
#   summary_<tag>.csv      : m_stag, converged, iterations, t_amp
#   Ak_<tag>.csv           : spin-summed mean-field band spectral weight
#   meta_<tag>.csv         : k-path ticks, labels, omega grid
#
# Usage:
#   julia --project=@. APSOS_hubbard_scf_gpu.jl \
#     Lx Ly t t_amp U scale maxiters mixing tol maxdim cutoff purif_maxiter \
#     bands_Ncheb bands_Nomega bands_Emin bands_Emax num_k \
#     num_x num_y box_half OUTDIR bands_gpu_type mag_reduce
#
# Defaults:
#   Lx=3 Ly=3 t=1.0 t_amp=0.5 U=5.5 scale=8.0
#   maxiters=60 mixing=0.3 tol=1e-3 maxdim=100 cutoff=1e-5 purif_maxiter=30
#   bands_Ncheb=50 bands_Nomega=300 bands_Emin=-6 bands_Emax=6 num_k=32
#   num_x=0 num_y=0 -> full Nx x Ny grid, box_half=0, OUTDIR=outputs_hubbard_scf_gpu
#   bands_gpu_type=ComplexF64
#   mag_reduce=point   # or block; block requires power-of-two num_x,num_y

using CUDA   # must be first: enables the NDTensors GPU backend
using Dates, LinearAlgebra, DelimitedFiles
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "../../../src/TensorBinding.jl"))
using .TensorBinding

function parse_gpu_complex_type(s)
    key = lowercase(String(s))
    key in ("complexf32", "f32", "float32", "single") && return ComplexF32
    key in ("complexf64", "f64", "float64", "double") && return ComplexF64
    error("Unsupported bands_gpu_type=$s. Use ComplexF32 or ComplexF64.")
end

function parse_reduce_mode(s)
    key = Symbol(lowercase(String(s)))
    key in (:point, :block) && return key
    error("Unsupported mag_reduce=$s. Use point or block.")
end

# Command-line arguments
Lx            = length(ARGS) >= 1  ? parse(Int,     ARGS[1])  : 3
Ly            = length(ARGS) >= 2  ? parse(Int,     ARGS[2])  : 3
t             = length(ARGS) >= 3  ? parse(Float64, ARGS[3])  : 1.0
t_amp         = length(ARGS) >= 4  ? parse(Float64, ARGS[4])  : 0.5
U             = length(ARGS) >= 5  ? parse(Float64, ARGS[5])  : 5.5
scale         = length(ARGS) >= 6  ? parse(Float64, ARGS[6])  : 8.0
maxiters      = length(ARGS) >= 7  ? parse(Int,     ARGS[7])  : 60
mixing        = length(ARGS) >= 8  ? parse(Float64, ARGS[8])  : 0.3
tol           = length(ARGS) >= 9  ? parse(Float64, ARGS[9])  : 1e-3
maxdim        = length(ARGS) >= 10 ? parse(Int,     ARGS[10]) : 100
cutoff        = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : 1e-5
purif_maxiter = length(ARGS) >= 12 ? parse(Int,     ARGS[12]) : 30
bands_Ncheb   = length(ARGS) >= 13 ? parse(Int,     ARGS[13]) : 50
bands_Nomega  = length(ARGS) >= 14 ? parse(Int,     ARGS[14]) : 300
bands_Emin    = length(ARGS) >= 15 ? parse(Float64, ARGS[15]) : -6.0
bands_Emax    = length(ARGS) >= 16 ? parse(Float64, ARGS[16]) : 6.0
num_k         = length(ARGS) >= 17 ? parse(Int,     ARGS[17]) : 32
num_x         = length(ARGS) >= 18 ? parse(Int,     ARGS[18]) : 0
num_y         = length(ARGS) >= 19 ? parse(Int,     ARGS[19]) : num_x
box_half      = length(ARGS) >= 20 ? parse(Int,     ARGS[20]) : 0
OUTDIR        = length(ARGS) >= 21 ? ARGS[21]                 : "outputs_hubbard_scf_gpu"
bands_gpu_type_arg = length(ARGS) >= 22 ? ARGS[22]             : "ComplexF64"
bands_gpu_type = parse_gpu_complex_type(bands_gpu_type_arg)
mag_reduce_arg = length(ARGS) >= 23 ? ARGS[23]                 : "point"
mag_reduce = parse_reduce_mode(mag_reduce_arg)

Nx, Ny = 2^Lx, 2^Ly
N      = Nx * Ny
num_x <= 0 && (num_x = Nx)
num_y <= 0 && (num_y = Ny)
bands_omega = range(bands_Emin, bands_Emax; length = bands_Nomega)
mkpath(OUTDIR)

println("[$(now())] APSOS_hubbard_scf_gpu starting")
println("[info] GPU device: $(CUDA.name(CUDA.device()))")
println("[info] Lx=$Lx Ly=$Ly  ($(Nx)x$(Ny) = $N sites)  t0=$t  t_amp=$t_amp  U=$U")
println("[info] scale=$scale  maxiters=$maxiters  mixing=$mixing  tol=$tol")
println("[info] maxdim=$maxdim  cutoff=$cutoff  purif_maxiter=$purif_maxiter")
println("[info] bands: Ncheb=$bands_Ncheb  Nomega=$bands_Nomega  E=[$bands_Emin, $bands_Emax]  num_k=$num_k  gpu_type=$bands_gpu_type")
println("[info] sampling: reduce=$mag_reduce  num_x=$num_x  num_y=$num_y  total=$(num_x*num_y)  box_half=$box_half")

# Build spinless square-lattice Hamiltonian with modulated NN hopping.
println("[info] Building modulated square-lattice Hamiltonian...")
hop_period = Nx / 4
tmod(ix, iy) = t * (1 + t_amp * cos(2*pi * ix / hop_period))
@time begin
    H_hub = TensorBinding.get_Hamiltonian("square_2d", 0.0001;
                                          L = Lx + Ly, Lx = Lx, Ly = Ly,
                                          scale = 4.0)
    TensorBinding.add_hopping_2D!(H_hub, tmod; Lx = Lx, Ly = Ly, nn = 1,
                                  maxdim = maxdim, tol = cutoff)
end
sites_pos = copy(H_hub.sites)

_, rho_up0 = TensorBinding.initial_guess_Neel_up(Lx, Ly, sites_pos)
_, rho_dn0 = TensorBinding.initial_guess_Neel_dn(Lx, Ly, sites_pos)

TensorBinding.add_spin!(H_hub)
TensorBinding.add_interaction!(H_hub, U)
println("[info] $H_hub")

# GPU warm-up
println("[info] GPU warm-up...")
warmup_s = siteinds("S=1/2", 4)
warmup_H = MPO(warmup_s, "Id")
@time begin
    wH_gpu = CUDA.cu(warmup_H)
    apply(wH_gpu, wH_gpu)
    CUDA.synchronize()
end
println("[info] Warm-up done.")

# Magnetic Hubbard SCF
println("[info] Running magnetic Hubbard SCF (GPU)...")
@time res = TensorBinding.scf_magnetic_hubbard_gpu(H_hub, U;
    initial_up    = rho_up0,
    initial_dn    = rho_dn0,
    background     = 0.5,
    Nel_up         = div(N, 2),
    Nel_dn         = div(N, 2),
    fermi          = 0.0,
    scale          = scale,
    max_scf_iter   = maxiters,
    scf_tol        = tol,
    mix            = mixing,
    maxdim         = maxdim,
    cutoff         = cutoff,
    purif_maxiter  = purif_maxiter,
    purif_tol      = 1e-5,
    verbose        = true)
println("[info] converged=$(res.converged)  iterations=$(res.iterations)  rms=$(res.rms_error)")

# Sample local moments on GPU and extract scalars.
println("[info] Sampling magnetization (GPU scalar extraction)...")
@time mag = TensorBinding.get_scf_magnetization_gpu(res;
    num_x = num_x,
    num_y = num_y,
    box_half = box_half,
    reduce = mag_reduce,
    Lx = Lx)
n_up  = mag.n_up
n_dn  = mag.n_dn
m_loc = (n_up .- n_dn) ./ 2
xs    = [(c - 1) % Nx for c in mag.centers]
ys    = [div(c - 1, Nx) for c in mag.centers]
stagg = [(-1)^(xs[k] + ys[k]) for k in eachindex(xs)]
m_stag = sum(stagg .* m_loc) / length(m_loc)

println("[info] sampled $(length(m_loc)) grid points ($(num_x)x$(num_y), reduce=$mag_reduce, box_half=$box_half, stride=$(mag.stride_x)x$(mag.stride_y))")
println("[info] <n_up>=$(round(sum(n_up)/length(n_up); digits=4))  <n_dn>=$(round(sum(n_dn)/length(n_dn); digits=4))")
println("[info] staggered magnetization m_s=$(round(m_stag; digits=5))  max|m_i|=$(round(maximum(abs.(m_loc)); digits=5))")

# Compute spin-summed mean-field bands as an explicit separate GPU pass.
println("[info] Computing converged mean-field bands (GPU)...")
@time bands = TensorBinding.get_scf_bands_gpu(res, bands_Ncheb, bands_omega;
    kpath = [:G, :X, :M, :G],
    kpath_lattice = :square,
    kpath_Lx = Lx,
    num_x = num_k,
    maxdim = maxdim,
    cutoff = cutoff,
    type = bands_gpu_type,
    printinfo = true)
println("[info] bands shape=$(size(bands.Ak))")

# Save
tag = "Lx$(Lx)_Ly$(Ly)_t$(t)_A$(t_amp)_U$(U)_mdim$(maxdim)_mix$(mixing)"

site_rows = hcat(xs, ys, real.(n_up), real.(n_dn), real.(m_loc))
site_file = joinpath(OUTDIR, "hubbard_mag_$(tag).csv")
writedlm(site_file, site_rows, ',')

hist_rows = hcat([h.iter for h in res.history], [h.rms_error for h in res.history])
hist_file = joinpath(OUTDIR, "scf_history_$(tag).csv")
writedlm(hist_file, hist_rows, ',')

summary_file = joinpath(OUTDIR, "summary_$(tag).csv")
writedlm(summary_file,
         ["m_stag" m_stag;
          "converged" Int(res.converged);
         "iterations" res.iterations;
         "t_amp" t_amp;
           "num_x" num_x;
           "num_y" num_y;
           "box_half" box_half;
           "mag_reduce" string(mag_reduce);
           "mag_stride_x" mag.stride_x;
           "mag_stride_y" mag.stride_y;
           "bands_Ncheb" bands_Ncheb;
           "num_k" num_k;
           "bands_gpu_type" string(bands_gpu_type)],
         ',')

bands_type_tag = bands_gpu_type === ComplexF64 ? "CF64" : "CF32"
ak_file = joinpath(OUTDIR, "Ak_$(tag)_Nc$(bands_Ncheb)_nk$(num_k)_typ$(bands_type_tag).csv")
writedlm(ak_file, bands.Ak, ',')

Ntick = isnothing(bands.ticks) ? 0 : length(bands.ticks)
meta = Matrix{Any}(undef, bands_Nomega, 3)
for i in 1:bands_Nomega
    meta[i, 1] = i <= Ntick ? bands.ticks[i] : ""
    meta[i, 2] = i <= Ntick ? bands.labels[i] : ""
    meta[i, 3] = collect(bands_omega)[i]
end
meta_file = joinpath(OUTDIR, "meta_$(tag)_Nc$(bands_Ncheb)_nk$(num_k)_typ$(bands_type_tag).csv")
writedlm(meta_file, meta, ',')

println("[$(now())] saved $(site_file)")
println("[$(now())] saved $(hist_file)")
println("[$(now())] saved $(summary_file)")
println("[$(now())] saved $(ak_file)")
println("[$(now())] saved $(meta_file)")
