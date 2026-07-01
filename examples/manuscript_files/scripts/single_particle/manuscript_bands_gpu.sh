#!/bin/bash
#SBATCH -J APSOS-bands-gpu
#SBATCH -o logs/bands_gpu_%j.out
#SBATCH -e logs/bands_gpu_%j.err
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-18g-ia,gpu-h200-141g-ellis,gpu-h200-35g-ia-ellis,gpu-h200-35g-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=08:00:00

# No array: the bands computation requires the full Chebyshev recurrence
# regardless of how many k-points are sampled — parallelising over k-points
# would replicate that cost in every task without reducing wall time.
# All GPU speedup comes from running the MPO algebra on a single GPU.

set -euo pipefail

# ── parameters ───────────────────────────────────────────────────────────────
Lx=14
Ly=14
t=1.0
t2=0.2
M=0.1
phi=1.5707963267948966   # π/2
Ncheb=500
maxdim=200
Nomega=500
Emin=-4.0
Emax=4.0
num_k=50
cutoff=1e-4
gpu_type=ComplexF64
OUTDIR="outputs_bands_gpu"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# ── environment ───────────────────────────────────────────────────────────────
module load triton/2024.1-gcc
module load cuda/12.2.1
module load julia/1.10.3

export JULIA_DEPOT_PATH=/scratch/work/moustaa1/julia_depot
export JULIA_NUM_THREADS=1      # GPU jobs: single Julia thread is sufficient
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/${OUTDIR}"

cd "${SCRIPT_DIR}"

echo "[$(date)] Starting APSOS_bands_gpu on $(hostname)"
echo "[info] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[info] Lx=$Lx Ly=$Ly  t2=$t2  M=$M  phi=$phi"
echo "[info] Ncheb=$Ncheb  maxdim=$maxdim  num_k=$num_k  Nomega=$Nomega  cutoff=$cutoff  gpu_type=$gpu_type"

julia --project=@. APSOS_bands_gpu.jl \
    $Lx $Ly $t $t2 $M $phi \
    $Ncheb $maxdim \
    $Nomega $Emin $Emax \
    $num_k $cutoff \
    "$OUTDIR" $gpu_type

echo "[$(date)] Done."
