# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Prefill throughput benchmark for LoopedWindowTransformer using dummy token IDs.

Runs a single-GPU forward (no FSDP), optional torch.compile matching train configs,
and reports tokens/sec and peak GPU memory.

Attention: by default this **overrides** ``model.attn_impl`` in the YAML to ``fmha``
(xFormers memory-efficient attention) so prefill numbers are comparable across configs
that train with ``flash_attn3`` vs ``fmha``. Use ``--attn-impl`` to pick another backend
or match training exactly.

Example:
  cd /path/to/lingua
  python -m apps.LT2.benchmark_prefill \\
    --config apps/LT2/configs/looped_600M_pure_full.yaml \\
    --seq-len 4096 --batch-size 1

On clusters (e.g. Bridges2), avoid the login node: PyTorch wheels may use CPU
instructions the login CPU lacks, causing "Illegal instruction (core dumped)".
Use an interactive GPU allocation or sbatch (see slurm/benchmark_prefill_*.slurm).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import OmegaConf

from lingua.args import dataclass_from_dict
from lingua.distributed import DistributedArgs
from apps.LT2.transformer import (
    LoopedWindowTransformer,
    LoopedWindowTransformerArgs,
)

logger = logging.getLogger(__name__)


DEFAULT_VOCAB_IF_UNSET = 128256


def _load_args(
    config_path: Path,
    seq_len: int,
    vocab_override: Optional[int],
) -> Tuple[LoopedWindowTransformerArgs, DistributedArgs]:
    cfg = OmegaConf.load(config_path)
    raw: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    model_cfg = dict(raw["model"])
    dist_cfg = dict(raw.get("distributed", {}))
    model_args = dataclass_from_dict(LoopedWindowTransformerArgs, model_cfg)
    dist_args = dataclass_from_dict(DistributedArgs, dist_cfg)

    vocab = vocab_override if vocab_override is not None else model_args.vocab_size
    if vocab < 0:
        vocab = DEFAULT_VOCAB_IF_UNSET
    model_args.vocab_size = vocab
    model_args.max_seqlen = max(seq_len, int(model_args.max_seqlen))
    return model_args, dist_args


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def run_benchmark(
    config_path: Path,
    seq_len: int,
    batch_size: int,
    warmup: int,
    iters: int,
    vocab_override: Optional[int],
    use_compile: Optional[bool],
    device: torch.device,
    attn_impl: str,
) -> Dict[str, Any]:
    model_args, dist_args = _load_args(config_path, seq_len, vocab_override)
    attn_impl_from_config = model_args.attn_impl
    model_args.attn_impl = attn_impl

    compile_on = dist_args.compile if use_compile is None else use_compile
    dtype = _dtype(dist_args.model_dtype)

    torch.backends.cuda.matmul.allow_tf32 = dist_args.matmul_allow_tf32
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(model_args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_args.seed)

    model = LoopedWindowTransformer(model_args)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    if compile_on:
        torch._dynamo.config.cache_size_limit = dist_args.compile_cache_size_limit
        if hasattr(model, "compile"):
            model.compile()
        else:
            logger.warning("Model has no .compile(); skipping torch.compile")

    tokens = torch.randint(
        0,
        model_args.vocab_size,
        (batch_size, seq_len),
        device=device,
    )

    def one_step() -> None:
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                model(tokens)

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = batch_size * seq_len * iters
    tok_per_s = total_tokens / elapsed if elapsed > 0 else float("nan")
    peak_gb = None
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    return {
        "config_path": str(config_path),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "compile": compile_on,
        "model_dtype": dist_args.model_dtype,
        "vocab_size": model_args.vocab_size,
        "max_seqlen": model_args.max_seqlen,
        "loop_count": model_args.loop_count,
        "n_layers": model_args.n_layers,
        "layer_pattern": model_args.layer_pattern,
        "attention_pattern": model_args.attention_pattern,
        "attn_impl": attn_impl,
        "attn_impl_in_config": attn_impl_from_config,
        "elapsed_s": elapsed,
        "tokens_per_s": tok_per_s,
        "peak_mem_gb": peak_gb,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="LoopedWindowTransformer prefill throughput (dummy tokens)")
    p.add_argument("--config", type=Path, required=True, help="Training YAML (model + distributed sections)")
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5, help="Warmup forwards (use more if compile=True)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help=f"Override vocab; if YAML has -1, default fallback is {DEFAULT_VOCAB_IF_UNSET}",
    )
    p.add_argument(
        "--force-compile",
        action="store_true",
        help="Enable torch.compile regardless of YAML distributed.compile",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile even if YAML enables it",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu (cpu is for smoke tests only)")
    p.add_argument(
        "--attn-impl",
        type=str,
        default="fmha",
        help="Attention backend for full-attention TransformerBlock layers (default: fmha / xFormers). "
        "Examples: fmha, flash_attn3, sdpa, flex_attention. Overrides model.attn_impl from the YAML.",
    )
    p.add_argument("--json-out", type=Path, default=None, help="Append one JSON line to this file")
    args = p.parse_args()
    if args.force_compile and args.no_compile:
        p.error("Use only one of --force-compile and --no-compile")
    if args.force_compile:
        use_compile: Optional[bool] = True
    elif args.no_compile:
        use_compile = False
    else:
        use_compile = None

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available")
        sys.exit(1)

    stats = run_benchmark(
        config_path=args.config,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        vocab_override=args.vocab_size,
        use_compile=use_compile,
        device=device,
        attn_impl=args.attn_impl,
    )

    line = json.dumps(stats)
    print(line)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
