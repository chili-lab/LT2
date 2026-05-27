<div align="center">

# LT2:Linear-Time Looped Transformers

A family of **looped Transformers** with **subquadratic token mixers** —
linear, sparse, and hybrid attention, unified within a single architecture.

<!-- <p align="center">
 <img src="lingua_overview.svg" width="78%"/>
</p> -->

</div>

> Official codebase accompanying the paper **"LT2: Linear-Time Looped Transformers."**
> Built upon the [Meta Lingua](https://github.com/facebookresearch/lingua) pre-training framework.

---

## 1. Architecture

LT2 replaces the multi-head attention sub-layer of a standard Looped Transformer with a
**subquadratic token mixer**, so that each shared block becomes

$$F_\ell(h) \;=\; h' + \mathrm{FFN}_\ell(h'), \qquad h' \;=\; h + \mathrm{Mixer}_\ell(h)$$

where `Mixer` may be any linear-attention, sparse-attention, or hybrid primitive. Looping
reuses the same parameters `T` times in succession, so a block of `n_layers` attains an
effective depth of `n_layers × T`.

- **LT2-linear** — DPLR linear-attention mixers (GDN, KDA, RWKV7). Loop iterations turn rank-1 state updates into rank-`T` updates.
- **LT2-sparse** — sliding-window, NSA, or DSA attention. A per-loop window of size `w` becomes an effective receptive field of `T·w`.
- **LT2-hybrid (Full+GDN)** — interleaves a small fraction of full attention with GDN; surpasses the standard looped transformer at ~2.7× decode speedup, establishing a new Pareto frontier.

See the paper for the full theoretical analysis and experimental results.

---

## 2. Repository Layout

```
LT2/
 ├─ lingua/             Core training library (forked from Meta Lingua)
 │   ├─ transformer.py     Reference Transformer block
 │   ├─ data.py            Pre-training dataloader
 │   ├─ distributed.py     FSDP / TP / compile wrappers
 │   ├─ checkpoint.py      Distributed checkpointing
 │   ├─ optim.py           Optimizer + LR scheduler
 │   └─ stool.py           SLURM launcher
 ├─ apps/LT2/           LT2 application code
 │   ├─ transformer.py     LT2 model (linear / sparse / hybrid mixers)
 │   ├─ train.py           Training entry point
 │   ├─ eval.py            LM-Evaluation-Harness wrapper
 │   ├─ generate.py        Inference / generation
 │   ├─ benchmark_prefill.py
 │   ├─ configs/
 │   │   ├─ 600M/          0.6B-parameter pre-training recipes
 │   │   └─ 1B/            1.3B-parameter pre-training recipes
 │   ├─ kernel/            Custom Triton / CUDA kernels
 │   ├─ scripts/           Helper scripts
 │   └─ slurm/             Example SLURM job files
 ├─ setup/              Environment + data preparation
 ├─ tokenizer/          Tokenizer files (downloaded)
 └─ requirements.txt
```

---

## 3. Quick Start

The following commands launch a SLURM job that creates a Conda environment for the codebase.
Environment creation takes around five minutes (excluding downloads).

```bash
git clone <THIS_REPO_URL>
cd LT2

bash setup/create_env.sh
# or, if you have access to a SLURM cluster
sbatch setup/create_env.sh
```

Once that is done, activate the environment:

```bash
conda activate lingua_<date>
```

Use the provided script to download and prepare data from HuggingFace
(`fineweb_edu`, `fineweb_edu_10bt`, or `dclm_baseline_1.0`). The command below downloads
`fineweb_edu` and prepares it for training in `./data`, specifying the memory `terashuf`
(the shuffling tool) is allowed to use. By default `nchunks=32`; if you train on fewer than
32 GPUs, set `nchunks` to 1 or to the number of GPUs you have
([details](https://github.com/facebookresearch/lingua/issues/55#issuecomment-2483643076)).

```bash
python setup/download_prepare_hf_data.py fineweb_edu <MEMORY> \
    --data_dir ./data --seed 42 --nchunks <NCHUNKS>
```

Download the tokenizer (Llama 3):

```bash
python setup/download_tokenizer.py llama3 <SAVE_PATH> --api_key <HUGGINGFACE_TOKEN>
```

Now launch a quick debug job to verify the setup. The provided configurations are
templates — edit `dump_dir`, `data.root_dir`, `data.tokenizer.path`, etc., for your environment.

```bash
# stool = SLURM tool
python -m lingua.stool script=apps.LT2.train \
    config=apps/LT2/configs/600M/debug.yaml \
    nodes=1 partition=<partition>

# Or launch locally with torchrun
torchrun --nproc-per-node 8 -m apps.LT2.train \
    config=apps/LT2/configs/600M/debug.yaml

# Or on a single GPU
python -m apps.LT2.train config=apps/LT2/configs/600M/debug.yaml
```

If a `stool` job crashes, it may be relaunched directly:

```bash
sbatch path/to/dump_dir/submit.slurm
```

---

## 4. LT2 Configuration

### Model Fields

The LT2 model is configured through the `model:` section of the YAML config. Key fields:

| Field | Description |
|---|---|
| `n_layers` | Number of *physical* (parameter-sharing) layers in the looped block. |
| `loop_count` | Number of loop iterations `T`. Effective depth is `n_layers × loop_count`. |
| `mixer` | Token-mixer family: `full`, `window`, `gdn`, `kda`, `mamba2`, `hgrn2`, `deltanet`, `retnet`, `nsa`, `dsa`. |
| `attention_pattern` | Depth-level hybrid layout (e.g. `"4:1"` for 4 mixer + 1 full attention). |
| `attn_impl` | `"fmha"`, `"flex_attention"`, or `"sdpa"`. Sliding-window requires `fmha` or `flex_attention`. |
| `default_sliding_window` | Window size for sparse / sliding-window mixers. |
| `use_residual` | Learned zero-init per-iteration residual gate across loop iterations. |
| `gated_attn` | SDPA output gate that suppresses the attention-sink sawtooth (§ 3.4 of the paper). |

Example fragment:

```yaml
model:
  dim: 2048
  n_layers: 20
  loop_count: 4              # T = 4, effective depth = 80
  mixer: "gdn"
  attention_pattern: "4:1"   # 4 GDN layers : 1 full-attention layer
  attn_impl: "fmha"
  default_sliding_window: 2048
  use_residual: true
  gated_attn: true
```

### Recipes Included

`apps/LT2/configs/` provides reference pre-training recipes for the experiments in the paper:

| Scale | Config | Description |
|:---:|---|---|
| 0.6B | `600M/debug.yaml` | Fast smoke test (single GPU). |
| 0.6B | `600M/looped_pure_full_600M.yaml` | Looped Transformer baseline (full attention). |
| 0.6B | `600M/looped_pure_{gdn,kda,mamba2,deltanet,retnet,hgrn2}_600M.yaml` | LT2-linear single-mixer ablations. |
| 0.6B | `600M/looped_pure_{window,nsa,dsa}_600M.yaml` | LT2-sparse single-mixer ablations. |
| 0.6B | `600M/looped_hybrid_gdn_4to1_600M.yaml` | **LT2-hybrid (Full+GDN)**, 4:1 depth interleave. |
| 0.6B | `600M/looped_hybrid_bookend_600M.yaml` | LT2-hybrid (Full+GDN), bookend pattern. |
| 0.6B | `600M/looped_hybrid_128_256_512_full_600M.yaml` | Loop-level hybrid (fine → coarse). |
| 1.3B | `1B/looped_pure_{full,gdn,kda,...}_1B.yaml` | 1.3B single-mixer recipes. |
| 1.3B | `1B/looped_hybrid_gdn_4to1_1B.yaml` | **LT2-hybrid (Full+GDN)** at 1.3B — flagship recipe. |

All recipes train on **FineWeb-Edu** at sequence length 4096 for ~100B tokens (255k steps),
using `T = 4` loops by default.

### Training

```bash
# Single-node debug
torchrun --nproc-per-node 8 -m apps.LT2.train \
    config=apps/LT2/configs/600M/debug.yaml

# Multi-node via stool
python -m lingua.stool script=apps.LT2.train \
    config=apps/LT2/configs/1B/looped_hybrid_gdn_4to1_1B.yaml \
    nodes=8 partition=<your_partition>

# Override fields on the command line
torchrun --nproc-per-node 8 -m apps.LT2.train \
    config=apps/LT2/configs/600M/looped_hybrid_gdn_4to1_600M.yaml \
    model.loop_count=4 \
    model.attention_pattern="4:1" \
    model.default_sliding_window=2048
```

### Generation & Evaluation

```bash
# Free-form generation from a checkpoint
python -m apps.LT2.generate ckpt=/path/to/checkpoint \
    max_gen_len=256 temperature=0.7

# Zero-shot evaluation on LM-Evaluation-Harness suites
torchrun --nproc-per-node 8 -m apps.LT2.eval \
    config=apps/LT2/configs/600M/eval.yaml \
    ckpt_dir=/path/to/checkpoint
```

### Long-Context Efficiency Benchmark

`benchmark_prefill.py` measures prefill / decode throughput across sequence lengths:

```bash
python -m apps.LT2.benchmark_prefill \
    --config apps/LT2/configs/1B/looped_hybrid_gdn_4to1_1B.yaml \
    --seq-len 8192
```

A reference SLURM array script is provided in `apps/LT2/slurm/benchmark_prefill_array.slurm`.

---

## 5. Configuration System

All scripts use [OmegaConf](https://omegaconf.readthedocs.io/) and accept dot-list overrides:

```bash
python -m apps.LT2.train config=apps/LT2/configs/600M/debug.yaml \
    model.dim=1024 \
    optim.lr=2e-4 \
    name=my_run
```

Resolution order: *dataclass defaults → values from `config=...` YAML → command-line overrides.*

A typical `TrainArgs` YAML:

```yaml
dump_dir: /path/to/dumpdir
name: "lt2_hybrid_gdn_4to1_1B"
steps: 255000
seed: 777

optim:
  lr: 3e-4
  warmup: 5000
  lr_min_ratio: 1e-6
  clip: 1.0

distributed:
  fsdp_type: full_shard
  compile: true

model:
  dim: 2048
  n_layers: 20
  loop_count: 4
  mixer: "gdn"
  attention_pattern: "4:1"

data:
  root_dir: ./data/fineweb_edu
  batch_size: 4
  seq_len: 4096
  tokenizer:
    name: tiktoken
    path: ./tokenizer/original/tokenizer.model
```

---

## 6. Dump Directory

```
example_dump_dir/
 ├─ checkpoints/
 │   └─ 0000007000/        DCP-format checkpoint + train state
 ├─ code/                  Snapshot of code at launch time
 ├─ logs/                  Per-GPU stdout/stderr
 ├─ profiling/             Memory + CPU/CUDA traces
 ├─ base_config.yaml
 ├─ metrics.jsonl
 └─ submit.slurm
```

Checkpoints are stored in `.distcp` format and may be converted to standard PyTorch checkpoints
via `torch.distributed.checkpoint.format_utils.dcp_to_torch_save`.

---

## 7. Citation

If this codebase is useful in your work, the LT2 paper reference is at:

```bibtex
@misc{deng2026lt2lineartimeloopedtransformers,
      title={LT2: Linear-Time Looped Transformers}, 
      author={Chunyuan Deng and Yizhe Zhang and Rui-Jie Zhu and Yuanyuan Xu and Jiarui Liu and T. S. Eugene Ng and Hanjie Chen},
      year={2026},
      eprint={2605.20670},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.20670}, 
}
}
```
