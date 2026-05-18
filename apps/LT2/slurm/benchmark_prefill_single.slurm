#!/bin/bash
#
# Run one (config, seq_len) pair. Set CONFIG and SEQ_LEN before sbatch, e.g.:
#
#   export CONFIG=<REPO_ROOT>/apps/LT2/configs/600M/looped_600M_pure_full.yaml
#   export SEQ_LEN=4096
#   sbatch --export=ALL,CONFIG,SEQ_LEN apps/LT2/slurm/benchmark_prefill_single.slurm
#
# Or pass on the submit line:
#   CONFIG=... SEQ_LEN=1024 sbatch --export=ALL,CONFIG,SEQ_LEN apps/LT2/slurm/benchmark_prefill_single.slurm
#
# logs/ must exist before sbatch (see logs/.gitkeep in repo).

#SBATCH --job-name=bench-prefill-1
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --time=120
#SBATCH --mem=128G

#SBATCH --output=<REPO_ROOT>/apps/LT2/logs/bench_prefill_%j.log
#SBATCH --error=<REPO_ROOT>/apps/LT2/logs/bench_prefill_%j.log
#SBATCH --open-mode=append

set -euo pipefail

: "${CONFIG:?Set CONFIG to a training YAML path}"
: "${SEQ_LEN:?Set SEQ_LEN (e.g. 512, 1024, 2048, 4096, 16384)}"

module load gcc/13.3.1-p20240614
module load cuda/12.6.1

source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate lindow

export MAX_JOBS=12
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

LINGUA_ROOT=<REPO_ROOT>
cd "${LINGUA_ROOT}"

RESULTS_DIR="${LINGUA_ROOT}/apps/LT2/logs"
mkdir -p "${RESULTS_DIR}"
JSONL="${RESULTS_DIR}/prefill_results_${SLURM_JOB_ID}.jsonl"

echo "CONFIG=${CONFIG} SEQ_LEN=${SEQ_LEN}"
python -u -m apps.LT2.benchmark_prefill \
  --config "${CONFIG}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size 1 \
  --warmup 8 \
  --iters 25 \
  --json-out "${JSONL}"
