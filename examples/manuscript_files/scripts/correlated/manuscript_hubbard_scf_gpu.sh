#!/bin/bash
#SBATCH -J APSOS-hubbard-scf-gpu
#SBATCH -o logs/hubbard_scf_gpu_%j.out
#SBATCH -e logs/hubbard_scf_gpu_%j.err
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-18g-ia,gpu-h200-141g-ellis,gpu-h200-35g-ia-ellis,gpu-h200-35g-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00

# GPU magnetic Hubbard SCF on a 2^Lx x 2^Ly square lattice with cosine-modulated
# nearest-neighbor hopping. Each SCF iteration
# purifies the two spin-resolved density matrices (McWeeny) on GPU; the mean-field
# MPO / Hartree assembly and density mixing stay on CPU. Seeded with a 2D Néel
# density to break the spin symmetry in the antiferromagnetic channel.

set -euo pipefail

# ── parameters ───────────────────────────────────────────────────────────────
Lx=12
Ly=12
t=1.0
t_amp=0.5
U=5.5
scale=8.0        # purification half-bandwidth for the modulated mean-field run
maxiters=80
mixing=0.3
tol=5e-3
maxdim=150
cutoff=1e-6      # ComplexF32 may NaN below ~1e-5 (warned, not enforced)
purif_maxiter=30
bands_Ncheb=100
bands_Nomega=300
bands_Emin=-7.0
bands_Emax=7.0
num_k=50
bands_gpu_type=ComplexF64
num_x=70         # sampled columns for the moment map (0 -> all Nx columns)
num_y=70         # sampled rows for the moment map (0 -> all Ny rows)
box_half=2       # 2D neighborhood half-width; >0 smooths but washes out Néel
mag_reduce=point # point or block; block requires power-of-two num_x/num_y and ignores box_half
OUTDIR="outputs_hubbard_scf_gpu"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# ── environment ───────────────────────────────────────────────────────────────
module load triton/2024.1-gcc
module load cuda/12.2.1
module load julia/1.10.3

export JULIA_DEPOT_PATH=/scratch/work/moustaa1/julia_depot
export JULIA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/${OUTDIR}"

cd "${SCRIPT_DIR}"

echo "[$(date)] Starting APSOS_hubbard_scf_gpu on $(hostname)"
echo "[info] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[info] Lx=$Lx Ly=$Ly  t=$t  t_amp=$t_amp  U=$U  scale=$scale"
echo "[info] maxiters=$maxiters  mixing=$mixing  tol=$tol  maxdim=$maxdim  cutoff=$cutoff"
echo "[info] bands: Ncheb=$bands_Ncheb  Nomega=$bands_Nomega  E=[$bands_Emin,$bands_Emax]  num_k=$num_k  gpu_type=$bands_gpu_type"
echo "[info] sampling: reduce=$mag_reduce  num_x=$num_x  num_y=$num_y  total=$((num_x * num_y))  box_half=$box_half"

args=(
    "$Lx" "$Ly" "$t" "$t_amp" "$U" "$scale"
    "$maxiters" "$mixing" "$tol"
    "$maxdim" "$cutoff" "$purif_maxiter"
    "$bands_Ncheb" "$bands_Nomega" "$bands_Emin" "$bands_Emax" "$num_k"
    "$num_x" "$num_y" "$box_half"
    "$OUTDIR" "$bands_gpu_type" "$mag_reduce"
)

julia --project=@. APSOS_hubbard_scf_gpu.jl "${args[@]}"

echo "[$(date)] Done."
