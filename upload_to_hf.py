"""
Upload LT2 checkpoints to HuggingFace Hub.

For each checkpoint step directory:
  1. Consolidate the DCP-sharded checkpoint into a single consolidated.pth
     using lingua.checkpoint.consolidate_checkpoints (skipped if already done).
  2. Upload consolidated.pth + params.json to a sub-directory on the Hub repo,
     tagged by token count.

Step → token mapping  (25400 steps == 10B tokens):
    step  25400  →  10B
    step  50800  →  20B
    step  76200  →  30B
    ...

Usage:
    python upload_to_hf.py \
        --repo_id  YOUR_USERNAME/looped_window_600M \
        --ckpt_root <REPO_ROOT>/apps/LT2/results/looped_600M_full_512_256_128_weighted_loss \
        [--token HF_TOKEN]      # defaults to HF_TOKEN env var
        [--private]             # create repo as private (default: public)
        [--dry_run]             # print what would be done, upload nothing

Dependencies:
    pip install huggingface_hub torch
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# Re-use the existing consolidation function exactly as written in lingua
from lingua.checkpoint import consolidate_checkpoints

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CKPT_ROOT_DEFAULT = (
    "<REPO_ROOT>/apps/LT2/results"
    "/looped_600M_full_512_256_128_weighted_loss"
)

# 25400 steps == 10 B tokens
TOKENS_AT_25400   = 10_000_000_000
STEPS_AT_10B      = 25400
TOKENS_PER_STEP   = TOKENS_AT_25400 / STEPS_AT_10B   # ~393700.787…

CONSOLIDATE_NAME  = "consolidated.pth"
CONFIG_NAME       = "params.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def step_to_tokens_label(step: int) -> str:
    """Return a human-readable token-count label, e.g. '10B', '20B'."""
    billions = (step * TOKENS_PER_STEP) / 1e9
    if abs(billions - round(billions)) < 0.05:
        return f"{round(billions)}B"
    return f"{billions:.1f}B"


def discover_checkpoints(ckpt_root: Path) -> list[tuple[int, Path]]:
    """
    Walk the checkpoints/ sub-directory and return sorted (step, step_dir) pairs.
    Falls back to ckpt_root itself if checkpoints/ doesn't exist.
    """
    candidates = ckpt_root / "checkpoints"
    if not candidates.is_dir():
        candidates = ckpt_root

    step_dirs = [
        (int(p.name), p)
        for p in candidates.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    step_dirs.sort()
    return step_dirs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Consolidate & upload lingua checkpoints to HuggingFace"
    )
    parser.add_argument(
        "--repo_id", required=True,
        help="HuggingFace repo id, e.g. username/looped_window_600M"
    )
    parser.add_argument(
        "--ckpt_root", default=CKPT_ROOT_DEFAULT,
        help="Root dir that contains the checkpoints/ folder"
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace API token (falls back to HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private", action="store_true", default=False,
        help="Create the HF repo as private"
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Print plan only; do not consolidate or upload"
    )
    args = parser.parse_args()

    ckpt_root = Path(args.ckpt_root)
    if not ckpt_root.is_dir():
        sys.exit(f"ckpt_root does not exist: {ckpt_root}")

    # --- discover ---
    checkpoints = discover_checkpoints(ckpt_root)
    if not checkpoints:
        sys.exit("No checkpoint step directories found.")

    # Build plan: (step, step_dir, token_label, hf_prefix)
    plan = []
    print("=" * 70)
    print(f"{'Step':>12}  {'Tokens':>8}  HF path")
    print("-" * 70)
    for step, step_dir in checkpoints:
        tok_label  = step_to_tokens_label(step)
        hf_prefix  = f"checkpoints/{step}_{tok_label}"
        plan.append((step, step_dir, tok_label, hf_prefix))
        print(f"{step:>12}  {tok_label:>8}  {hf_prefix}")
    print("=" * 70)

    if args.dry_run:
        print("\n[dry_run] Nothing uploaded. Remove --dry_run to proceed.")
        return

    # --- token (only needed for actual upload) ---
    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("No HuggingFace token provided. Pass --token or set HF_TOKEN env var.")

    # --- create repo (no-op if already exists) ---
    api = HfApi(token=hf_token)
    print(f"\nCreating repo {args.repo_id} (private={args.private}) …")
    create_repo(
        repo_id=args.repo_id,
        token=hf_token,
        private=args.private,
        exist_ok=True,
        repo_type="model",
    )
    print("Repo ready.")

    # --- upload base_config.yaml once at repo root ---
    base_config = ckpt_root / "base_config.yaml"
    if base_config.exists():
        print("\nUploading base_config.yaml …")
        api.upload_file(
            path_or_fileobj=str(base_config),
            path_in_repo="base_config.yaml",
            repo_id=args.repo_id,
            token=hf_token,
            repo_type="model",
        )

    # --- consolidate + upload each checkpoint ---
    for step, step_dir, tok_label, hf_prefix in plan:
        print(f"\n--- step {step} ({tok_label} tokens) ---")

        # 1. Consolidate (uses lingua.checkpoint.consolidate_checkpoints exactly)
        #    Returns the consolidated/ directory Path; idempotent if already done.
        consolidated_dir = consolidate_checkpoints(str(step_dir))
        consolidated_pth = consolidated_dir / CONSOLIDATE_NAME

        if not consolidated_pth.exists():
            print(f"  [ERROR] consolidation did not produce {consolidated_pth}, skipping.")
            continue

        # 2. Upload consolidated.pth
        print(f"  [upload] {CONSOLIDATE_NAME} → {hf_prefix}/")
        api.upload_file(
            path_or_fileobj=str(consolidated_pth),
            path_in_repo=f"{hf_prefix}/{CONSOLIDATE_NAME}",
            repo_id=args.repo_id,
            token=hf_token,
            repo_type="model",
        )

        # 3. Upload params.json
        params_json = consolidated_dir / CONFIG_NAME
        if params_json.exists():
            print(f"  [upload] {CONFIG_NAME} → {hf_prefix}/")
            api.upload_file(
                path_or_fileobj=str(params_json),
                path_in_repo=f"{hf_prefix}/{CONFIG_NAME}",
                repo_id=args.repo_id,
                token=hf_token,
                repo_type="model",
            )

        print(f"  [done] step {step} uploaded.")

    print("\n" + "=" * 70)
    print("All checkpoints uploaded successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
