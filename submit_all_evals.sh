#!/bin/bash
# Submit evaluation jobs for all checkpoints in both directories

WORK_DIR="<REPO_ROOT>"
CONDA_ENV="lindow"

# Define checkpoint directories to process
CKPT_DIRS=(
    "<REPO_ROOT>/apps/LT2/results/looped_600M_full_512_256_128_weighted_loss/checkpoints"
    "<REPO_ROOT>/apps/LT2/results/looped_600M_128_256_512_full_weighted_loss/checkpoints"
)

# Specific checkpoints to exclude (full paths)
EXCLUDE_PATHS=(
    "<REPO_ROOT>/apps/LT2/results/looped_600M_full_512_256_128_weighted_loss/checkpoints/0000025400"
)

# Create logs directory if it doesn't exist
mkdir -p "$WORK_DIR/logs"

echo "==================================="
echo "Submitting evaluation jobs"
echo "==================================="

total_submitted=0

# Loop through each checkpoint directory
for CHECKPOINTS_DIR in "${CKPT_DIRS[@]}"; do
    echo ""
    echo "Processing: $CHECKPOINTS_DIR"
    echo "-----------------------------------"

    if [ ! -d "$CHECKPOINTS_DIR" ]; then
        echo "  WARNING: Directory does not exist, skipping"
        continue
    fi

    # Loop through all checkpoint directories
    for ckpt_dir in "$CHECKPOINTS_DIR"/*; do
        if [ -d "$ckpt_dir" ]; then
            ckpt_name=$(basename "$ckpt_dir")

            # Check if this specific checkpoint path should be excluded
            skip=false
            for exclude_path in "${EXCLUDE_PATHS[@]}"; do
                if [ "$ckpt_dir" == "$exclude_path" ]; then
                    echo "  - Skipping $ckpt_name (excluded)"
                    skip=true
                    break
                fi
            done

            if [ "$skip" = true ]; then
                continue
            fi

            echo "  - Submitting $ckpt_name"

            # Create temporary config file for this checkpoint
            TEMP_CONFIG=$(mktemp /tmp/eval_config_${ckpt_name}_XXXXXX.yaml)
            cat > "$TEMP_CONFIG" << EOL
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

            # Submit job directly via lingua.stool (it handles sbatch internally)
            cd ${WORK_DIR}
            job_output=$(python -m lingua.stool script=apps.LT2.eval config="$TEMP_CONFIG" nodes=1 ngpu=2 ncpu=4 2>&1)
            if [ $? -eq 0 ]; then
                # Extract job ID from lingua.stool output (typically "Submitted batch job XXXXX")
                job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+' || echo "$job_output" | grep -oP 'job \K\d+')
                if [ -n "$job_id" ]; then
                    echo "    → Job ID: $job_id"
                else
                    echo "    → Submitted (check with squeue)"
                fi
                ((total_submitted++))
            else
                echo "    → ERROR: $job_output"
            fi

            # Clean up temporary config
            rm -f "$TEMP_CONFIG"
        fi
    done
done

echo ""
echo "==================================="
echo "Summary: $total_submitted jobs submitted!"
echo "==================================="
echo ""
echo "Useful commands:"
echo "  Check status: squeue -u \$USER"
echo "  View logs:    ls -lh $WORK_DIR/logs/"
echo "  Cancel all:   scancel -u \$USER"
echo ""
