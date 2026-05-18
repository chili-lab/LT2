#!/bin/bash
# Submit evaluation jobs for all checkpoints except 25400

CHECKPOINTS_DIR="<REPO_ROOT>/apps/LT2/results/looped_600M_full_512_256_128_weighted_loss/checkpoints"
EXCLUDE_CKPT="0000025400"
WORK_DIR="<REPO_ROOT>"
CONDA_ENV="lindow"

# Create logs directory if it doesn't exist
mkdir -p "$WORK_DIR/logs"

echo "Found checkpoints to evaluate (excluding $EXCLUDE_CKPT):"

# Loop through all checkpoint directories
for ckpt_dir in "$CHECKPOINTS_DIR"/*; do
    if [ -d "$ckpt_dir" ]; then
        ckpt_name=$(basename "$ckpt_dir")

        # Skip the excluded checkpoint
        if [ "$ckpt_name" == "$EXCLUDE_CKPT" ]; then
            echo "  - Skipping $ckpt_name"
            continue
        fi

        echo "  - $ckpt_name"

        # Create temporary slurm script
        cat > "/tmp/submit_eval_${ckpt_name}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=eval_${ckpt_name}
#SBATCH --output=${WORK_DIR}/logs/eval_${ckpt_name}_%j.out
#SBATCH --error=${WORK_DIR}/logs/eval_${ckpt_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

# Activate conda environment
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Change to working directory
cd ${WORK_DIR}

# Create temporary config file for this checkpoint
TEMP_CONFIG=\$(mktemp /tmp/eval_config_${ckpt_name}_XXXXXX.yaml)
cat > \$TEMP_CONFIG << 'EOL'
name: "eval_LT2"
dump_dir: ${CHECKPOINTS_DIR}/${ckpt_name}/evals
ckpt_dir: ${CHECKPOINTS_DIR}/${ckpt_name}

generator:
  max_tokens: 16384
  dtype: bf16

harness:
  tasks:
    - hellaswag
    - task: boolq
      dataset_kwargs:
        trust_remote_code: true
    - piqa
    - task: sciq
      dataset_kwargs:
        trust_remote_code: true
    - winogrande
    - openbookqa
    - arc_easy
    - arc_challenge
    - commonsense_qa
    - copa
  log_samples: true
  verbosity: "INFO"

validation: null
EOL

# Run evaluation
python -m lingua.stool script=apps.LT2.eval config=\$TEMP_CONFIG nodes=1 ngpu=2 ncpu=4

# Clean up temporary config
rm -f \$TEMP_CONFIG
EOF

        # Submit the job
        sbatch "/tmp/submit_eval_${ckpt_name}.sh"

        # Clean up temporary slurm script
        rm -f "/tmp/submit_eval_${ckpt_name}.sh"
    fi
done

echo ""
echo "All jobs submitted!"
echo "Check status with: squeue -u \$USER"
