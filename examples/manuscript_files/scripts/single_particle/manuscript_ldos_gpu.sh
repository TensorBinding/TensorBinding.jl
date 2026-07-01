#!/bin/bash
#SBATCH -J APSOS-ldos-gpu
#SBATCH -o logs/ldos_gpu_%j.out
#SBATCH -e logs/ldos_gpu_%j.err
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-18g-ia,gpu-h200-141g-ellis,gpu-h200-35g-ia-ellis,gpu-h200-35g-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00

# Single GPU job: the Chebyshev recurrence runs once for all positions.
# No array needed — all num_x × num_y positions are sampled in one pass
# (each is just a scalar extraction from the diagonal MPS).

set -euo pipefail

# ── parameters ───────────────────────────────────────────────────────────────
Lx=12
Ly=12
t=1.0
t2=0.2
M=0.1
phi=1.5707963267948966   # π/2
Ncheb=150
maxdim=200
Nomega=300
Emin=-4.0
Emax=4.0
num_x=64    # x pixels (reduce=block: MUST be a power of two)
num_y=64    # y pixels (reduce=block: MUST be a power of two)
box_half=0  # point mode only: 0 = no averaging; 2 → 5×5 box
cutoff=1e-7
reduce=block         # block → integrate each block (gap-free, catches thin edge states); point → sample at cells
sublattice=average   # average → clean large-scale map (1 value/unit cell); resolve → A/B columns; auto
gpu_type=ComplexF64  # ComplexF64 is safer at tight cutoff; ComplexF32 is faster if stable
OUTDIR="outputs_ldos_gpu"

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

echo "[$(date)] Starting APSOS_ldos_gpu on $(hostname)"
echo "[info] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[info] Lx=$Lx Ly=$Ly  t2=$t2  M=$M  phi=$phi"
echo "[info] Ncheb=$Ncheb  maxdim=$maxdim  num_x=$num_x  num_y=$num_y  box_half=$box_half  Nomega=$Nomega  cutoff=$cutoff  reduce=$reduce  sublattice=$sublattice  gpu_type=$gpu_type"

julia --project=@. APSOS_ldos_gpu.jl \
    $Lx $Ly $t $t2 $M $phi \
    $Ncheb $maxdim \
    $Nomega $Emin $Emax \
    $num_x $num_y $box_half $cutoff \
    $reduce $sublattice "$OUTDIR" $gpu_type

echo "[$(date)] Done."
