#!/bin/bash
#
# Submit all (config × seq_len) prefill benchmarks as one array job.
# 5 configs × 5 sequence lengths = 25 tasks (indices 0–24).
#
# Matches module/conda style from install_flash_attn.slurm (Bridges2 GPU-shared).
#
# IMPORTANT — SLURM opens #SBATCH --output before this script runs. The directory
# .../LT2/logs MUST exist at submit time or you get
# JobLaunchFailure (e.g. exit 53). The repo keeps logs/.gitkeep for that; if you
# removed it: mkdir -p .../logs before sbatch.
#
# Do not run the Python benchmark on Bridges2 *login* nodes: PyTorch wheels can
# hit "Illegal instruction" on older CPUs; use this batch script or srun on a GPU node.
#
# Usage:
#   cd <REPO_ROOT>
#   sbatch apps/LT2/slurm/benchmark_prefill_array.slurm

#SBATCH --job-name=bench-prefill-lwa
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --time=120
#SBATCH --mem=128G
#SBATCH --array=0-24

#SBATCH --output=<REPO_ROOT>/apps/LT2/logs/bench_prefill_%A_%a.log
#SBATCH --error=<REPO_ROOT>/apps/LT2/logs/bench_prefill_%A_%a.log
#SBATCH --open-mode=append

set -euo pipefail

module load gcc/13.3.1-p20240614
module load cuda/12.6.1

source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate lindow

export MAX_JOBS=12
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

LINGUA_ROOT=<REPO_ROOT>
cd "${LINGUA_ROOT}"

CONFIGS=(
  "${LINGUA_ROOT}/apps/LT2/configs/600M/looped_600M_pure_full.yaml"
  "${LINGUA_ROOT}/apps/LT2/configs/600M/looped_600M_pure_window.yaml"
  "${LINGUA_ROOT}/apps/LT2/configs/600M/looped_600M_pure_gdn.yaml"
  "${LINGUA_ROOT}/apps/LT2/configs/600M/looped_600M_gdn_4to1.yaml"
  "${LINGUA_ROOT}/apps/LT2/configs/600M/looped_600M_4to1.yaml"
)

SEQ_LENS=(512 1024 2048 4096 16384)

NSEQ=${#SEQ_LENS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}
C=$((TASK_ID / NSEQ))
S=$((TASK_ID % NSEQ))
CONFIG="${CONFIGS[$C]}"
SEQ_LEN="${SEQ_LENS[$S]}"

RESULTS_DIR="${LINGUA_ROOT}/apps/LT2/logs"
mkdir -p "${RESULTS_DIR}"
JSONL="${RESULTS_DIR}/prefill_results_${SLURM_ARRAY_JOB_ID}.jsonl"

echo "=================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Task: array_job=${SLURM_ARRAY_JOB_ID} task_id=${SLURM_ARRAY_TASK_ID}"
echo "Config: ${CONFIG}"
echo "seq_len: ${SEQ_LEN}"
echo "CUDA_HOME: ${CUDA_HOME:-}"
echo "=================================="

# Extra warmup when training YAML uses compile (large graphs)
WARMUP=8
ITERS=25

python -u -m apps.LT2.benchmark_prefill \
  --config "${CONFIG}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size 1 \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --json-out "${JSONL}"

echo "Appended to ${JSONL}"
echo "Finished at: $(date)"
