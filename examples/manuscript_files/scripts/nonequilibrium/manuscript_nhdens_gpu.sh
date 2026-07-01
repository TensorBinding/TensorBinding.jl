#!/bin/bash
#SBATCH -J APSOS-nhdens-gpu
#SBATCH -o logs/nhdens_gpu_%j.out
#SBATCH -e logs/nhdens_gpu_%j.err
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-18g-ia,gpu-h200-141g-ellis,gpu-h200-35g-ia-ellis,gpu-h200-35g-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00

# GPU non-Hermitian density-matrix evolution for panel (b).
# This is a single-GPU job: time evolution is sequential in t, while spatial
# density sampling is extracted from each GPU-resident density snapshot.

set -euo pipefail

# -- Hamiltonian parameters: match APSOS_nhdos_gpu.sh -------------------------
L=20
t=1.0
aah_V=0.5
aah_phi=0.0
aah_b=1.618033988749895
gamma0=1.0
loss_b=1.5           # commensurate loss period in sites = N/4
loss_phase=0.0
loss_harmonics=15
scale=4.5

# -- time evolution -----------------------------------------------------------
nsteps=120
dt=0.125
sample_every=1

# -- MPO/GPU controls ---------------------------------------------------------
maxdim=100
cutoff=1e-4
purif_maxiter=30
purif_tol=1e-6

# -- output sampling ----------------------------------------------------------
# reduce=block averages over contiguous intervals on GPU and is the fast
# production mode. Use reduce=point for representative-point sampling.
reduce=block
num_x=128
num_avg=1
OUTDIR="outputs_nhdens_gpu"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# -- environment --------------------------------------------------------------
module load triton/2024.1-gcc
module load cuda/12.2.1
module load julia/1.10.3

export JULIA_DEPOT_PATH=/scratch/work/moustaa1/julia_depot
export JULIA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/${OUTDIR}"

cd "${SCRIPT_DIR}"

echo "[$(date)] Starting APSOS_nhdens_gpu on $(hostname)"
echo "[info] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[info] AAH open chain: L=$L  t=$t  V=$aah_V  phi=$aah_phi  b=$aah_b"
echo "[info] loss: gamma0=$gamma0  loss_b=$loss_b  period_sites=loss_b*N/6  loss_phase=$loss_phase  harmonics=$loss_harmonics  scale=$scale"
echo "[info] nsteps=$nsteps  dt=$dt  sample_every=$sample_every"
echo "[info] maxdim=$maxdim  cutoff=$cutoff  purif_maxiter=$purif_maxiter  purif_tol=$purif_tol"
echo "[info] reduce=$reduce  num_x=$num_x  num_avg=$num_avg  output dir=$OUTDIR"

julia --project=@. APSOS_nhdens_gpu.jl \
    $L $t $gamma0 $loss_b $loss_phase \
    $scale \
    $nsteps $dt $sample_every \
    $maxdim $cutoff \
    $purif_maxiter $purif_tol \
    $num_x $num_avg $reduce \
    "$OUTDIR" \
    $aah_V $aah_phi $aah_b $loss_harmonics

echo "[$(date)] Done."
