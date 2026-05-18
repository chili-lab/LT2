# Looped Window Attention Transformer (LT2)

A transformer variant combining **looped layers** (parameter-efficient depth) and **hybrid attention** (local + global mixing).

## Core Concepts

### Layer Looping

Reuse the same layers multiple times to decouple depth from parameter count:

```yaml
model:
  n_layers: 16      # Physical layers (parameters)
  loop_count: 3     # Effective depth: 16 × 3 = 48
```


**Attention implementation:**
```yaml
attn_impl: "fmha"  # or "flex_attention" — required for sliding window
                   # "sdpa" ignores window settings (falls back to full)
default_sliding_window: 2048
```

### Stochastic Attention Patterns

Two dynamic patterns that resample on every forward pass (useful as regularization):

```yaml
attention_pattern: "random:4:1"         # Fully random distribution per step
attention_pattern: "bookend_random:4:1" # First/last layer of each loop always full; rest random
```

- Deterministic: `seed = base_seed + forward_step`, reproducible across runs and GPUs
- `@torch.compiler.disable` on pattern generation — required for `torch.compile` compatibility
- Zero extra memory overhead

## Architecture & Implementation

**Forward pass** (`transformer.py`):
```python
for loop_idx in range(self.loop_count):
    for layer_idx, layer in enumerate(self.layers):
        sliding_window = self.attention_windows[layer_idx]
        mask = create_causal_mask(seqlen, attn_impl, sliding_window)
        h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
```

**FLOP accounting** (`train.py`) includes looping:
```python
get_num_flop_per_token(..., n_layers=args.model.n_layers * args.model.loop_count, ...)
```

**Memory complexity:**
- Looping adds no extra activation memory (same parameters reused)
- Sliding window attention: O(n × window) vs full O(n²)
- 4:1 pattern with 16 layers → ~80% attention memory savings

## File Structure

```
LT2/
├── transformer.py            # Core model (LoopedWindowTransformer)
├── train.py                  # Training script
├── generate.py               # Inference / generation
├── eval.py                   # LM harness evaluation
├── test_model.py             # Unit tests
└── configs/
    ├── debug.yaml
    ├── looped_1B_4to1.yaml
    ├── looped_1B_alternating.yaml
    ├── looped_600M_random_4to1.yaml
    ├── looped_600M_bookend_random_4to1.yaml
    └── eval.yaml
```

## Quickstart

**Environment:**
```bash
conda activate /scratch/conda-envs/lindow
cd <REPO_ROOT>
```

**Verify installation (30 sec):**
```bash
PYTHONPATH=<REPO_ROOT>:$PYTHONPATH python apps/LT2/test_model.py
# Expected: ALL TESTS PASSED
```

## Training

```bash
# Debug (1 GPU, ~100 steps)
python apps/LT2/train.py config=apps/LT2/configs/debug.yaml

# 1B model (8 GPUs)
torchrun --nproc_per_node=8 apps/LT2/train.py config=apps/LT2/configs/looped_1B_4to1.yaml

# Override parameters
torchrun --nproc_per_node=8 apps/LT2/train.py \
  config=apps/LT2/configs/looped_1B_4to1.yaml \
  model.loop_count=4 \
  model.attention_pattern="1:1" \
  model.default_sliding_window=1024
```

## Generation & Evaluation

```bash
# Generate
python apps/LT2/generate.py ckpt=/path/to/checkpoint max_gen_len=256 temperature=0.7

# Evaluate
torchrun --nproc_per_node=8 apps/LT2/eval.py \
  config=apps/LT2/configs/eval.yaml \
  ckpt_dir=/path/to/checkpoint
```

## Example Configs

**Debug** (quick test):
```yaml
model:
  dim: 512
  n_layers: 4
  loop_count: 2
  attention_pattern: "3:1"
  attn_impl: "fmha"
  default_sliding_window: 512
steps: 100
```

**1B, 4:1** (recommended):
```yaml
model:
  dim: 2048
  n_layers: 16
  loop_count: 3          # 48 effective layers
  attention_pattern: "4:1"
  attn_impl: "fmha"
  default_sliding_window: 2048
steps: 255_000
seq_len: 4096
```

**600M, stochastic** (regularization):
```yaml
model:
  n_layers: 25
  loop_count: 4
  attention_pattern: "random:4:1"   # or "bookend_random:4:1"
  default_sliding_window: 128
```

## Troubleshooting

| Problem | Solutions |
|---------|-----------|
| OOM | Reduce `batch_size` / `seq_len`; use `attention_pattern="7:1"`; reduce `default_sliding_window` |
| Slow training | Reduce `loop_count`; enable `distributed.compile=true`; add GPUs |
| Loss not decreasing | Try `optim.lr=1e-4`; increase `optim.warmup`; reduce `loop_count` |
| NaN gradients | Lower LR; use `distributed.model_dtype=bf16`; check data |
| `torch.compile` error with stochastic patterns | Ensure `@torch.compiler.disable` is on `generate_random_attention_pattern` |

## Architecture Comparison

| Config | Params | Effective Depth | Attention | Notes |
|--------|--------|-----------------|-----------|-------|
| Vanilla 1B | 1B | 25 | Full O(n²) | Baseline |
| Looped 4:1 | 1B | 48 | Mixed | Recommended |
| Looped 1:1 | 1B | 40 | Mixed | More global |
| Stochastic random | ~600M | 100 | Mixed + random | Regularization |
