#!/bin/bash
#SBATCH -J APSOS-nhstate-gpu
#SBATCH -o logs/nhstate_gpu_%j.out
#SBATCH -e logs/nhstate_gpu_%j.err
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-18g-ia,gpu-h200-141g-ellis,gpu-h200-35g-ia-ellis,gpu-h200-35g-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00

# GPU TDVP propagation of a comb initial state for panel (d).
# This is a single-GPU job: the no-loss and lossy trajectories are sequential
# in time, while all state evolution and amplitude sampling stay GPU-resident.

set -euo pipefail

# -- Hamiltonian parameters: match APSOS_nhdos_gpu.sh / APSOS_nhdens_gpu.sh ---
L=20
t=1.0
aah_V=0.5
aah_phi=0.0
aah_b=1.618033988749895
gamma0=1.0
loss_b=1.5           # commensurate loss period in sites = N/4
loss_phase=0.0       # site-coordinate phase shift inside gamma(n)
loss_harmonics=15    # odd harmonics in the cusped loss profile

# -- TDVP time evolution: panel (d) defaults ---------------------------------
nsteps=600
dt=0.025
sample_every=1

# -- MPS/GPU controls ---------------------------------------------------------
maxdim=100
cutoff=1e-6          # raise to 1e-5/1e-4 if ComplexF32 TDVP becomes unstable
normalize_each_step=false

# -- output sampling ----------------------------------------------------------
# The state trajectory is sampled on a literal central window of sample_window
# sites, so use point sampling with num_x=sample_window.
reduce=point
num_avg=1
component=real       # panel (d): real(<x|psi(t)>)
pointavg=complex     # :complex (coherent avg then component), :abs (mean|<x|psi>|), :abs2 (mean|<x|psi>|^2)

# Initial paired-comb state has support on (|n> + |n+1>)/sqrt(2)
# for n = 0, init_stride, 2*init_stride, ...
# The sampled window is centered in the chain.
init_stride=64
sample_window=256
num_x=$sample_window

OUTDIR="outputs_nhstate_gpu"

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

echo "[$(date)] Starting APSOS_nhstate_gpu on $(hostname)"
echo "[info] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[info] AAH open chain: L=$L  t=$t  V=$aah_V  phi=$aah_phi  b=$aah_b"
echo "[info] loss: gamma0=$gamma0  loss_b=$loss_b  period_sites=loss_b*N/6  loss_phase=$loss_phase  harmonics=$loss_harmonics"
echo "[info] nsteps=$nsteps  dt=$dt  sample_every=$sample_every"
echo "[info] maxdim=$maxdim  cutoff=$cutoff  normalize_each_step=$normalize_each_step"
echo "[info] reduce=$reduce  num_x=$num_x  num_avg=$num_avg  component=$component  pointavg=$pointavg"
echo "[info] init_stride=$init_stride  sample_window=$sample_window  output dir=$OUTDIR"

julia --project=@. APSOS_nhstate_gpu.jl \
    $L $t $gamma0 $loss_b $loss_phase \
    $nsteps $dt $sample_every \
    $maxdim $cutoff \
    $num_x $num_avg $reduce $component $pointavg $normalize_each_step \
    $init_stride $sample_window \
    "$OUTDIR" \
    $aah_V $aah_phi $aah_b $loss_harmonics

echo "[$(date)] Done."
