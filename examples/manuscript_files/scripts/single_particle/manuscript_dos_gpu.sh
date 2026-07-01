#!/bin/bash
#SBATCH -J APSOS-dos-gpu
#SBATCH -o logs/dos_gpu_%j.out
#SBATCH -e logs/dos_gpu_%j.err
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-18g-ia,gpu-h200-141g-ellis,gpu-h200-35g-ia-ellis,gpu-h200-35g-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00

# GPU stochastic DOS: each MPS Chebyshev recursion runs on GPU.
# All N_sample samples run sequentially in a single job — no array needed.
# Increase N_sample for better statistics; GPU throughput makes this practical.

set -euo pipefail

# ── parameters ───────────────────────────────────────────────────────────────
Lx=14
Ly=14
t=1.0
t2=0.2
M=0.1
phi=1.5707963267948966   # π/2
Ncheb=200
maxdim=200
Nomega=300
Emin=-4.0
Emax=4.0
N_sample=500   # samples per job; GPU can handle many in reasonable wall-time
seed=42
cutoff=1e-4    # GPU cutoff floor (ComplexF32 needs >= ~1e-4 to avoid NaN)
OUTDIR="outputs_dos_gpu"

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

echo "[$(date)] Starting APSOS_dos_gpu on $(hostname)"
echo "[info] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[info] Lx=$Lx Ly=$Ly  t2=$t2  M=$M  phi=$phi"
echo "[info] Ncheb=$Ncheb  maxdim=$maxdim  N_sample=$N_sample  Nomega=$Nomega  cutoff=$cutoff"

julia --project=@. APSOS_dos_gpu.jl \
    $Lx $Ly $t $t2 $M $phi \
    $Ncheb $maxdim \
    $Nomega $Emin $Emax \
    $N_sample $seed $cutoff \
    "$OUTDIR"

echo "[$(date)] Done."
