#!/usr/bin/env python3
"""
Script to submit slurm jobs for evaluating all checkpoints except 25400
"""
import os
import subprocess
from pathlib import Path

# Configuration
CHECKPOINTS_DIR = "<REPO_ROOT>/apps/LT2/results/looped_600M_full_512_256_128_weighted_loss/checkpoints"
EXCLUDE_CKPT = "0000025400"
CONDA_ENV = "lindow"
WORK_DIR = "<REPO_ROOT>"

def get_checkpoint_dirs():
    """Get all checkpoint directories except the excluded one"""
    ckpt_path = Path(CHECKPOINTS_DIR)
    ckpts = [d.name for d in ckpt_path.iterdir() if d.is_dir() and d.name != EXCLUDE_CKPT]
    return sorted(ckpts)

def create_slurm_script(ckpt_name):
    """Create a slurm script for a specific checkpoint"""
    script_content = f"""#!/bin/bash
#SBATCH --job-name=eval_{ckpt_name}
#SBATCH --output=logs/eval_{ckpt_name}_%j.out
#SBATCH --error=logs/eval_{ckpt_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {CONDA_ENV}

# Change to working directory
cd {WORK_DIR}

# Create temporary config file for this checkpoint
TEMP_CONFIG=$(mktemp /tmp/eval_config_{ckpt_name}_XXXXXX.yaml)
cat > $TEMP_CONFIG << 'EOL'
name: "eval_LT2"
dump_dir: {CHECKPOINTS_DIR}/{ckpt_name}/evals
ckpt_dir: {CHECKPOINTS_DIR}/{ckpt_name}

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
python -m lingua.stool script=apps.LT2.eval config=$TEMP_CONFIG nodes=1 ngpu=2 ncpu=4

# Clean up temporary config
rm -f $TEMP_CONFIG
"""
    return script_content

def submit_jobs():
    """Submit slurm jobs for all checkpoints"""
    # Create logs directory if it doesn't exist
    logs_dir = Path(WORK_DIR) / "logs"
    logs_dir.mkdir(exist_ok=True)

    checkpoints = get_checkpoint_dirs()
    print(f"Found {len(checkpoints)} checkpoints to evaluate (excluding {EXCLUDE_CKPT}):")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")

    print("\nSubmitting jobs...")
    submitted_jobs = []

    for ckpt in checkpoints:
        # Create temporary slurm script
        script_content = create_slurm_script(ckpt)
        script_path = f"/tmp/submit_eval_{ckpt}.sh"

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Submit the job
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True,
                text=True,
                check=True
            )
            job_id = result.stdout.strip().split()[-1]
            submitted_jobs.append((ckpt, job_id))
            print(f"  ✓ Submitted {ckpt}: Job ID {job_id}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to submit {ckpt}: {e.stderr}")
        finally:
            # Clean up temporary script
            if os.path.exists(script_path):
                os.remove(script_path)

    print(f"\n{len(submitted_jobs)} jobs submitted successfully!")

    # Print summary
    if submitted_jobs:
        print("\nJob Summary:")
        for ckpt, job_id in submitted_jobs:
            print(f"  {ckpt}: {job_id}")

        print("\nTo check job status:")
        print("  squeue -u $USER")
        print("\nTo cancel all jobs:")
        job_ids = " ".join([jid for _, jid in submitted_jobs])
        print(f"  scancel {job_ids}")

if __name__ == "__main__":
    submit_jobs()
