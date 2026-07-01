#!/bin/bash
#SBATCH -J APSOS-nhdos-gpu
#SBATCH -o logs/nhdos_gpu_%A_%a.out
#SBATCH -e logs/nhdos_gpu_%A_%a.err
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-18g-ia,gpu-h200-141g-ellis,gpu-h200-35g-ia-ellis,gpu-h200-35g-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --array=1-16

# GPU deterministic diagonal-trace non-Hermitian DOS / spectral-weight grid.
# The chain and loss MPO are built on CPU; each hermitized complex-z problem is
# then evaluated with a GPU-resident online MPO-MPO recurrence plus diagonal
# extraction. The complex-energy grid is split over a 16-task Slurm array; each
# task writes one long-form tile CSV.

set -euo pipefail

# -- parameters ---------------------------------------------------------------
L=20
t=1.0
aah_V=0.5
aah_phi=0.0
aah_b=1.618033988749895
gamma0=1.0
loss_b=1.5           # commensurate loss period in sites = N/4
loss_phase=0.0       # site-coordinate phase shift inside gamma(n)
loss_harmonics=15    # odd harmonics in the cusped loss profile
scale=auto           # use the NH universal bound from TensorBinding.nh_kpm_scale
nh_scale_pad=1.25
nh_bound_check_stride=2
Ncheb=50             # NH recurrence order; must be even
Nh=$(( Ncheb / 2 ))  # get_nh_dos_* convention: recurrence order is 2*Nh
maxdim=100
nx=30
ny=30
xmin=-2.25
xmax=2.25
ymin=-2.25
ymax=0.25
legacy_n_random=0    # positional compatibility slot; ignored by diagonal trace
legacy_seed=42       # positional compatibility slot; ignored by diagonal trace
cutoff=1e-8          # ComplexF64 diagonal-trace path is stable at tighter cutoff
OUTDIR_ROOT="outputs_nhdos_gpu"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

(( Ncheb % 2 == 0 )) || { echo "[error] Ncheb=$Ncheb must be even"; exit 1; }

# -- split z-grid over the Slurm array ---------------------------------------
NCHUNKS=16
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NCHUNKS="${SLURM_ARRAY_TASK_COUNT:-$NCHUNKS}"
    ARRAY_MIN="${SLURM_ARRAY_TASK_MIN:-1}"
    CHUNK_ID=$(( SLURM_ARRAY_TASK_ID - ARRAY_MIN ))  # Julia expects 0-based tile_id
    ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}"
else
    CHUNK_ID=0
    NCHUNKS=1
    ARRAY_JOB_ID="manual"
fi

OUTDIR="${OUTDIR_ROOT}/${ARRAY_JOB_ID}"

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

echo "[$(date)] Starting APSOS_nhdos_gpu on $(hostname)"
echo "[info] Slurm array job=${ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-manual} chunk_id=$CHUNK_ID nchunks=$NCHUNKS"
echo "[info] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[info] AAH open chain: L=$L  t=$t  V=$aah_V  phi=$aah_phi  b=$aah_b"
echo "[info] loss: gamma0=$gamma0  loss_b=$loss_b  period_sites=loss_b*N/6  loss_phase=$loss_phase  harmonics=$loss_harmonics"
echo "[info] scale=$scale  nh_scale_pad=$nh_scale_pad  bound_check_stride=$nh_bound_check_stride  Ncheb=$Ncheb  Nh=$Nh  maxdim=$maxdim  cutoff=$cutoff"
echo "[info] method=diagtrace  grid Re=[$xmin,$xmax] nx=$nx  Im=[$ymin,$ymax] ny=$ny"
echo "[info] legacy stochastic args ignored: n_random=$legacy_n_random seed=$legacy_seed"
echo "[info] output dir: $OUTDIR"

julia --project=@. APSOS_nhdos_gpu.jl \
    $L $t $gamma0 $loss_b $loss_phase \
    $scale $Nh $maxdim \
    $nx $ny \
    $xmin $xmax $ymin $ymax \
    $legacy_n_random $legacy_seed $cutoff \
    "$OUTDIR" \
    $CHUNK_ID $NCHUNKS \
    $aah_V $aah_phi $aah_b $loss_harmonics $nh_scale_pad $nh_bound_check_stride

echo "[$(date)] Done."
