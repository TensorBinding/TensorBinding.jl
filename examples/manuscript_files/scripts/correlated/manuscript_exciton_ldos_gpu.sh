#!/bin/bash
#SBATCH -J APSOS-exciton-ldos-gpu
#SBATCH -o logs/exciton_ldos_gpu_%j.out
#SBATCH -e logs/exciton_ldos_gpu_%j.err
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-18g-ia,gpu-h200-141g-ellis,gpu-h200-35g-ia-ellis,gpu-h200-35g-ia
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00

# GPU spatial exciton LDOS (HODC kernel): one MPS Chebyshev recursion per bound
# exciton position X (electron=hole=X) on the 2L-site electron-hole space, run on
# GPU. All num_x positions run sequentially in a single job — no array needed.

set -euo pipefail

# ── parameters ───────────────────────────────────────────────────────────────
L=10
t=-1.0
U=6.0
V0=1.5
scale=10.0
Ncheb=200
maxdim=200
Nomega=400
Emin=-10.0
Emax=10.0
num_x=64       # coarse exciton positions (electron=hole=X) across [1, 2^L]
num_avg=1      # subpositions averaged per coarse X
x_start=1
x_end=$((1 << L))
eta=0.0        # 0 -> 1/(Ncheb+1)
eta_shift=0.0  # eta=0 -> 1/(Ncheb+1-eta_shift); larger shift = broader HODC
m_order=6      # HODC contour order
cutoff=1e-4    # GPU cutoff floor (ComplexF32 needs >= ~1e-4 to avoid NaN)
domainwall=false   # ramp the large-scale wavevector across x=N/2 (true/false)
OUTDIR="outputs_exciton_ldos_gpu"

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

echo "[$(date)] Starting APSOS_exciton_ldos_gpu on $(hostname)"
echo "[info] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[info] L=$L  t=$t  U=$U  V0=$V0  scale=$scale  domainwall=$domainwall"
echo "[info] Ncheb=$Ncheb  maxdim=$maxdim  num_x=$num_x  num_avg=$num_avg  Nomega=$Nomega"
echo "[info] x_range=[$x_start, $x_end]"
echo "[info] kernel=hodc  eta=$eta  eta_shift=$eta_shift  m_order=$m_order  cutoff=$cutoff"

julia --project=@. APSOS_exciton_ldos_gpu.jl \
    $L $t $U $V0 $scale \
    $Ncheb $maxdim \
    $Nomega $Emin $Emax \
    $num_x \
    $eta $m_order $cutoff $domainwall \
    "$OUTDIR" \
    $num_avg $x_start $x_end \
    $eta_shift

echo "[$(date)] Done."
