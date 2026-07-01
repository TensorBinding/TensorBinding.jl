#!/bin/bash
#SBATCH -J APSOS-chern-gpu
#SBATCH -o logs/chern_gpu_%j.out
#SBATCH -e logs/chern_gpu_%j.err
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-18g-ia,gpu-h200-141g-ellis,gpu-h200-35g-ia-ellis,gpu-h200-35g-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00

# GPU Chern marker: projector (McWeeny) and C1–C4 assembly on GPU.
# Single job — no array needed; all positions evaluated in the closure loop.

set -euo pipefail

# ── parameters ────────────────────────────────────────────────────────────────
Lx=14
Ly=14
t=1.0
t2=0.2
M=0.1
phi=1.5707963267948966   # π/2
maxdim=200
l_param=$Lx              # locality scale
Lambda=10
num_x=70
num_y=70
cutoff=1e-8
OUTDIR="outputs_chern_gpu"

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

echo "[$(date)] Starting APSOS_chern_gpu on $(hostname)"
echo "[info] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[info] Lx=$Lx Ly=$Ly  t2=$t2  M=$M  phi=$phi"
echo "[info] maxdim=$maxdim  l=$l_param  Lambda=$Lambda  grid=${num_x}x${num_y}  cutoff=$cutoff"

julia --project=@. APSOS_chern_gpu.jl \
    $Lx $Ly $t $t2 $M $phi \
    $maxdim $l_param $Lambda \
    $num_x $num_y $cutoff \
    "$OUTDIR"

echo "[$(date)] Done."
