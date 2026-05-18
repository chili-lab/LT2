# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

import torch.nn.functional as F

from xformers.ops import fmha, AttentionBias
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    FeedForward,
    FlashAttnMask,
    RMSNorm,
    TiedLinear,
    cross_entropy,
    TransformerBlock,
    Attention,
    apply_rotary_emb,
    repeat_kv,
    _get_fa3_flash_attn_func,
    _sdpa_flash_attn3_fallback,
    flex_attention_comp,
)

@torch.compiler.disable
def create_causal_mask(seqlen, attn_impl, sliding_window):
    """Create attention mask based on implementation and window size."""
    if sliding_window is not None and attn_impl == "fmha":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "fmha":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "flash_attn3":
        # window_size=(left, right): left=sliding_window-1 matches fmha's window_left=sliding_window-1
        # (N-1 tokens left of current + current itself = N tokens total)
        return FlashAttnMask(
            causal=True,
            window_size=(sliding_window - 1, 0) if sliding_window is not None else None,
        )
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@torch.compiler.disable
def generate_random_attention_pattern(
    pattern_type: str,
    window_count: int,
    full_count: int,
    n_layers: int,
    loop_count: int,
    default_sliding_window: Optional[int],
    seed: int,
) -> List[List[Optional[int]]]:
    """
    Generate a random attention pattern for one forward pass.

    Note: Decorated with @torch.compiler.disable to prevent torch.compile from tracing
    through Python random operations with dynamic seeds.

    Args:
        pattern_type: Either "random" or "bookend_random"
        window_count: Number of sliding window layers in ratio
        full_count: Number of full attention layers in ratio
        n_layers: Number of unique layers
        loop_count: Number of loop iterations
        default_sliding_window: Window size for SWA layers
        seed: Random seed for this forward pass

    Returns:
        List of lists (per iteration) with sliding window sizes
    """
    import random

    total_ratio = window_count + full_count
    total_slots = n_layers * loop_count
    target_num_full = int(total_slots * full_count / total_ratio)

    rng = random.Random(seed)

    if pattern_type == "random":
        # Fully random distribution across all slots
        num_window = total_slots - target_num_full
        all_slots = [None] * target_num_full + [default_sliding_window] * num_window
        rng.shuffle(all_slots)

        # Reshape into per-iteration lists
        result = []
        for loop_idx in range(loop_count):
            start_idx = loop_idx * n_layers
            end_idx = start_idx + n_layers
            result.append(all_slots[start_idx:end_idx])

        return result

    elif pattern_type == "bookend_random":
        # First and last layer of each loop are always full
        bookend_full_count = loop_count * 2

        if n_layers < 2:
            raise ValueError(
                f"Bookend random pattern requires at least 2 layers per loop, got {n_layers}"
            )

        if bookend_full_count > target_num_full:
            raise ValueError(
                f"Bookend random pattern needs {bookend_full_count} full attention layers, "
                f"but ratio allows only {target_num_full} total. Increase full attention ratio."
            )

        # Remaining full attention layers to randomly distribute
        remaining_full = target_num_full - bookend_full_count
        non_bookend_per_loop = n_layers - 2
        total_non_bookend_slots = non_bookend_per_loop * loop_count

        # Create and shuffle non-bookend positions
        non_bookend_slots = [None] * remaining_full + [default_sliding_window] * (total_non_bookend_slots - remaining_full)
        rng.shuffle(non_bookend_slots)

        # Build result with bookends
        result = []
        non_bookend_idx = 0

        for loop_idx in range(loop_count):
            iteration_pattern = []
            for layer_idx in range(n_layers):
                if layer_idx == 0 or layer_idx == n_layers - 1:
                    iteration_pattern.append(None)  # Bookend: full attention
                else:
                    iteration_pattern.append(non_bookend_slots[non_bookend_idx])
                    non_bookend_idx += 1
            result.append(iteration_pattern)

        return result

    else:
        raise ValueError(f"Unknown random pattern type: {pattern_type}")


def parse_attention_pattern(pattern: str, n_layers: int, default_sliding_window: Optional[int] = 2048, loop_count: int = 1, seed: int = 42) -> Union[List[Optional[int]], List[List[Optional[int]]], dict]:
    """
    Parse attention pattern specification into a list of sliding window sizes per layer.
    For dynamic random patterns, returns metadata dict instead.

    Args:
        pattern: Can be one of:
                 - Ratio string like "4:1" (4 sliding window layers per 1 full attention, interleaved)
                 - Bookend pattern like "bookend:2" or "sandwich:3" (full attn at start/end, window in middle)
                   The number specifies how many full attention layers at each end
                 - Random ratio like "random:4:1" (4:1 ratio randomly distributed, sampled each forward pass)
                   This creates a fully random pattern where no layer knows if it will get full or SWA
                 - Bookend random like "bookend_random:4:1" (first/last of each loop are full, rest random)
                   This creates a semi-stable pattern with guaranteed full attention at boundaries
                 - Comma-separated list like "2048,2048,None,2048,None" where None means full attention
                 - Per-iteration pattern like "full->128->128->128" (use "->" to separate iterations)
                   Each iteration can use: "full" (full attention), "SWA=N" (sliding window with size N),
                   or a layer pattern like "4:1" or "bookend:2"
        n_layers: Total number of unique layers (before looping)
        default_sliding_window: Default window size to use for sliding window layers
        loop_count: Number of loop iterations (used for per-iteration patterns)
        seed: Random seed for random patterns (only used for validation, actual seed comes from step)

    Returns:
        - If static pattern: List of sliding window sizes (None means full attention)
        - If per-iteration static pattern: List of lists
        - If dynamic random pattern: Dict with metadata for runtime generation

    Examples:
        "4:1" with 20 layers -> [2048]*4 + [None] repeated (interleaved pattern)
        "bookend:2" with 20 layers -> [None, None, 2048, ..., 2048, None, None] (2 full at each end)
        "sandwich:1" with 8 layers -> [None, 2048, 2048, 2048, 2048, 2048, 2048, None]
        "random:4:1" -> metadata dict for dynamic generation each forward pass
        "bookend_random:4:1" -> metadata dict for dynamic generation with bookends
        "2048,None,2048,None" -> [2048, None, 2048, None] (explicit values)
        "full->SWA=128->SWA=128->SWA=128" with loop_count=4 ->
            [[None]*n_layers, [128]*n_layers, [128]*n_layers, [128]*n_layers]
    """
    # Check if it's a dynamic random pattern (returns metadata for runtime generation)
    if pattern.startswith("random:") or pattern.startswith("bookend_random:"):
        pattern_type = "random" if pattern.startswith("random:") else "bookend_random"
        prefix_len = 7 if pattern_type == "random" else 15

        # Extract the ratio
        ratio_part = pattern[prefix_len:]
        parts = ratio_part.split(":")
        if len(parts) != 2:
            raise ValueError(f"{pattern_type.capitalize()} pattern must be '{pattern_type}:window:full', got {pattern}")

        window_count = int(parts[0])
        full_count = int(parts[1])

        if window_count + full_count == 0:
            raise ValueError(f"{pattern_type.capitalize()} pattern must have at least one layer in ratio")

        if window_count > 0 and default_sliding_window is None:
            raise ValueError(
                f"default_sliding_window must be provided for {pattern_type} pattern (got {pattern})"
            )

        # Validate the pattern is feasible
        if pattern_type == "bookend_random":
            total_slots = n_layers * loop_count
            target_num_full = int(total_slots * full_count / (window_count + full_count))
            bookend_full_count = loop_count * 2

            if n_layers < 2:
                raise ValueError(f"Bookend random pattern requires at least 2 layers, got {n_layers}")

            if bookend_full_count > target_num_full:
                raise ValueError(
                    f"Bookend random pattern needs {bookend_full_count} full layers for bookends, "
                    f"but {window_count}:{full_count} ratio allows only {target_num_full} total. "
                    f"Increase full attention ratio or reduce loop_count."
                )

        # Return metadata for dynamic generation
        return {
            "type": "dynamic_random",
            "pattern_type": pattern_type,
            "window_count": window_count,
            "full_count": full_count,
            "default_sliding_window": default_sliding_window,
        }

    # Check if it's a per-iteration pattern (contains "->")
    if "->" in pattern:
        iteration_patterns = [p.strip() for p in pattern.split("->")]
        
        if len(iteration_patterns) != loop_count:
            raise ValueError(
                f"Per-iteration pattern has {len(iteration_patterns)} iterations "
                f"but loop_count is {loop_count}. They must match."
            )
        
        result = []
        for iter_pattern in iteration_patterns:
            # Parse each iteration's pattern
            if iter_pattern.lower() == "full":
                # Full attention for all layers in this iteration
                result.append([None] * n_layers)
            elif iter_pattern.lower().startswith("swa="):
                # Sliding window attention with specified size for all layers
                window_size = int(iter_pattern.split("=")[1])
                result.append([window_size] * n_layers)
            else:
                # Recursively parse as a regular pattern (ratio, bookend, or explicit)
                iter_result = parse_attention_pattern(
                    iter_pattern, n_layers, default_sliding_window, loop_count=1
                )
                result.append(iter_result)
        
        return result
    
    # Check if it's a bookend/sandwich pattern
    if pattern.startswith("bookend:") or pattern.startswith("sandwich:"):
        prefix, num_str = pattern.split(":")
        num_full_each_end = int(num_str)
        
        if num_full_each_end < 0:
            raise ValueError(f"Number of full attention layers must be non-negative, got {num_full_each_end}")
        
        if 2 * num_full_each_end > n_layers:
            raise ValueError(
                f"Cannot have {num_full_each_end} full attention layers at each end "
                f"with only {n_layers} total layers"
            )
        
        # Check if default_sliding_window is needed (if there are middle layers)
        if 2 * num_full_each_end < n_layers and default_sliding_window is None:
            raise ValueError(
                f"default_sliding_window must be provided for bookend/sandwich pattern "
                f"with middle layers (pattern: {pattern}, n_layers: {n_layers})"
            )
        
        # Create pattern: full at start, window in middle, full at end
        result = []
        for i in range(n_layers):
            if i < num_full_each_end or i >= n_layers - num_full_each_end:
                result.append(None)  # Full attention
            else:
                result.append(default_sliding_window)  # Sliding window
        
        return result
    
    # Check if it's a ratio pattern (e.g., "4:1")
    if ":" in pattern:
        parts = pattern.split(":")
        if len(parts) != 2:
            raise ValueError(f"Ratio pattern must be 'window:full', got {pattern}")
        
        window_count = int(parts[0])
        full_count = int(parts[1])
        pattern_length = window_count + full_count
        
        if pattern_length == 0:
            raise ValueError("Pattern must have at least one layer")
        
        # Check if default_sliding_window is needed
        if window_count > 0 and default_sliding_window is None:
            raise ValueError(
                f"default_sliding_window must be provided when pattern has sliding window layers "
                f"(pattern: {pattern})"
            )
        
        # Create the repeating pattern using default_sliding_window
        base_pattern = [default_sliding_window] * window_count + [None] * full_count
        
        # Repeat to fill n_layers
        result = []
        for i in range(n_layers):
            result.append(base_pattern[i % pattern_length])
        
        return result
    
    # Otherwise, treat as comma-separated list
    parts = [p.strip() for p in pattern.split(",")]
    result = []
    for part in parts:
        if part.lower() == "none" or part.lower() == "null":
            result.append(None)
        else:
            result.append(int(part))
    
    if len(result) != n_layers:
        raise ValueError(
            f"Explicit pattern has {len(result)} entries but n_layers is {n_layers}. "
            "They must match or use ratio pattern like '4:1'"
        )
    
    return result


def get_block_size(pattern: str, n_layers: int) -> int:
    """
    Determine the block size from the attention pattern.
    For ratio patterns like "4:1", block size is 5 (4+1).
    For bookend/sandwich patterns, block size is n_layers (treat as single block).
    For explicit patterns, block size is the pattern length.
    """
    if pattern.startswith("bookend:") or pattern.startswith("sandwich:"):
        # Bookend pattern treats all layers as one block
        return n_layers
    elif ":" in pattern:
        parts = pattern.split(":")
        window_count = int(parts[0])
        full_count = int(parts[1])
        return window_count + full_count
    else:
        # For explicit patterns, block size is the pattern length
        parts = [p.strip() for p in pattern.split(",")]
        return len(parts)


_LINEAR_ATTN_TYPES = frozenset(
    {"gdn", "retnet", "deltanet", "kda", "hgrn2", "mla", "mamba2", "nsa", "dsa", "cot_gdp"}
)

# Linear-attention types that need CoT (chain-of-thought) state threaded through
# the outer loop_count loop: at outer step k we pass current_cot_step=k and the
# per-layer V cache from step k-1.
_COT_ATTN_TYPES = frozenset({"cot_gdp"})


def parse_layer_pattern(pattern: str, n_layers: int) -> List[str]:
    """
    Parse a layer pattern into a list of fla/mamba mixer types or "full" per layer.

    Pattern formats:
      - "full"          : all layers are standard full/SWA TransformerBlock (default, backward-compatible)
      - "gdn"           : all layers are GatedDeltaNet (fla)
      - "retnet"        : all layers are MultiScaleRetention / RetNet (fla)
      - "deltanet"      : all layers are DeltaNet (fla)
      - "kda"           : all layers are KimiDeltaAttention (fla)
      - "hgrn2"         : all layers are HGRN2Attention (fla)
      - "mla"           : all layers are MultiheadLatentAttention (fla; needs flash-attn)
      - "mamba2"        : all layers are Mamba2 (fla.layers.mamba2; causal_conv1d optional for fast path)
      - "nsa"           : all layers are NativeSparseAttention (fla.layers.nsa; Triton NSA kernels)
      - "interleaved:N:M:TYPE1:TYPE2" : cycle of N TYPE1 layers then M TYPE2 layers
        TYPE1/TYPE2 ∈ {gdn, retnet, deltanet, kda, hgrn2, mla, mamba2, nsa, full}.
        Example: "interleaved:4:1:gdn:full" → 4 GDN then 1 full, repeated.
    """
    pattern = pattern.strip().lower()

    if pattern == "full":
        return ["full"] * n_layers
    if pattern == "gdn":
        return ["gdn"] * n_layers
    if pattern == "retnet":
        return ["retnet"] * n_layers
    if pattern == "mamba2":
        return ["mamba2"] * n_layers
    if pattern == "deltanet":
        return ["deltanet"] * n_layers
    if pattern == "kda":
        return ["kda"] * n_layers
    if pattern == "hgrn2":
        return ["hgrn2"] * n_layers
    if pattern == "mla":
        return ["mla"] * n_layers
    if pattern == "nsa":
        return ["nsa"] * n_layers
    if pattern == "dsa":
        return ["dsa"] * n_layers
    if pattern == "cot_gdp":
        return ["cot_gdp"] * n_layers

    if not pattern.startswith("interleaved:"):
        raise ValueError(
            f"Unsupported layer_pattern: {pattern!r}. "
            "Expected 'full', 'gdn', 'retnet', 'deltanet', 'kda', 'hgrn2', 'mla', 'mamba2', 'nsa', "
            "or 'interleaved:N:M:TYPE1:TYPE2'."
        )

    parts = pattern.split(":")
    if len(parts) != 5:
        raise ValueError(
            f"Interleaved pattern must be 'interleaved:N:M:TYPE1:TYPE2', got {pattern!r}"
        )

    n_type1, n_type2 = int(parts[1]), int(parts[2])
    type1, type2 = parts[3], parts[4]

    if n_type1 <= 0:
        raise ValueError("N must be > 0 for interleaved pattern")
    if n_type2 < 0:
        raise ValueError("M must be >= 0 for interleaved pattern")
    allowed = {"gdn", "retnet", "deltanet", "kda", "hgrn2", "mla", "mamba2", "nsa", "dsa", "cot_gdp", "full"}
    if type1 not in allowed or type2 not in allowed:
        raise ValueError(f"layer types must be in {allowed}, got {type1!r}, {type2!r}")
    if type1 == type2 and n_type2 > 0:
        raise ValueError("TYPE1 and TYPE2 must be different when M > 0")

    if n_type2 == 0:
        return [type1] * n_layers

    cycle = n_type1 + n_type2
    result: List[str] = []
    for i in range(n_layers):
        result.append(type1 if (i % cycle) < n_type1 else type2)
    return result


class GatedAttention(Attention):
    """Attention with an elementwise head-specific sigmoid gate after SDPA (G1 from the paper).

    Following "Gated Attention for Large Language Models" (Qiu et al., NeurIPS 2025):
    - Gate position G1: applied to the concatenated SDPA output *before* the output projection wo.
    - Gate scores are query-dependent (computed from the pre-norm input x), head-specific, and elementwise.
    - Gate formula: gate = sigmoid(x @ W_gate.T), shape (B, S, n_heads * head_dim)
    - Output: sdpa_out = sdpa_out * gate, then sdpa_out @ wo

    This is the most effective variant from the paper (+0.2 PPL reduction, +2 MMLU points vs baseline).
    It also eliminates attention sinks and massive activations, and improves training stability.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__(dim=dim, head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, rope_theta=rope_theta)
        # Gate weight: maps pre-norm input (dim) -> gating scores (n_heads * head_dim)
        # Head-specific elementwise gate: each head dimension has its own gate score
        self.w_gate = nn.Linear(dim, n_heads * head_dim, bias=False)
        # Initialize to zero so gate starts as sigmoid(0) = 0.5 — a neutral multiplicative identity
        nn.init.zeros_(self.w_gate.weight)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx=None,
        mask=None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        output_shape = xq.shape
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()
        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask_t = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=is_causal, attn_mask=mask_t)
            output = output.transpose(1, 2).contiguous()
        elif attn_impl == "flash_attn3":
            fa3 = _get_fa3_flash_attn_func()
            if fa3 is not None and xq.is_cuda:
                if mask is None:
                    output, _ = fa3(xq, xk, xv, causal=False)
                elif isinstance(mask, FlashAttnMask):
                    if not mask.causal:
                        raise NotImplementedError("FlashAttnMask with causal=False is not supported for flash_attn3")
                    if mask.window_size is None:
                        output, _ = fa3(xq, xk, xv, causal=True)
                    else:
                        wl, wr = mask.window_size
                        output, _ = fa3(xq, xk, xv, causal=True, window_size=(int(wl), int(wr)))
                else:
                    raise NotImplementedError(f"flash_attn3 expects mask None or FlashAttnMask, got {type(mask)}")
            else:
                output = _sdpa_flash_attn3_fallback(xq, xk, xv, mask)
        else:
            raise NotImplementedError(f"Attention implementation {attn_impl} not supported")

        # G1 gate: elementwise head-specific sigmoid gate applied to SDPA output before wo
        # output: (B, S, n_heads, head_dim) -> flatten to (B, S, n_heads * head_dim)
        # gate:   sigmoid(x @ w_gate.T) -> (B, S, n_heads * head_dim)
        output_flat = output.reshape(output_shape)  # (B, S, n_heads * head_dim)
        gate = torch.sigmoid(self.w_gate(x))         # (B, S, n_heads * head_dim)
        output_flat = output_flat * gate

        return self.wo(output_flat)

    def reset_parameters(self, init_std=None, factor=1.0):
        super().reset_parameters(init_std, factor)
        # Keep gate weight at zero init so training starts from neutral sigmoid(0)=0.5
        nn.init.zeros_(self.w_gate.weight)


class GatedTransformerBlock(TransformerBlock):
    """TransformerBlock with G1 gated attention (SDPA output sigmoid gate).

    Drop-in replacement for TransformerBlock. Swaps the standard Attention
    for GatedAttention; everything else (FFN, norms, residuals) is identical.
    """

    def __init__(self, args: "BaseTransformerArgs"):
        # Call nn.Module.__init__ directly to skip TransformerBlock's __init__
        # which would create a plain Attention — we want GatedAttention instead.
        nn.Module.__init__(self)

        assert (args.head_dim is not None) or (args.n_heads is not None), \
            "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert self.n_heads % self.n_kv_heads == 0
        assert args.dim % self.n_heads == 0

        self.attention = GatedAttention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)


class LinearAttentionBlock(nn.Module):
    """Transformer block using Gated Delta Net (GDN) for the attention path.

    Drop-in replacement for TransformerBlock in the looped-window architecture.
    The sliding-window mask from attention_pattern is ignored for GDN layers
    (GDN uses its own linear recurrence); all other arguments are accepted for
    API compatibility but have no effect.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        allow_neg_eigval: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        try:
            from fla.layers import GatedDeltaNet
        except ImportError as exc:
            raise ImportError(
                "flash-linear-attention (fla) is not installed. "
                "Please install it with: pip install flash-linear-attention"
            ) from exc

        target_key_dim = int(0.75 * dim)
        if target_key_dim % n_heads != 0:
            raise ValueError(
                f"GatedDeltaNet: 0.75 * dim ({target_key_dim}) must be divisible by "
                f"n_heads ({n_heads}). Adjust n_heads."
            )

        self.attention = GatedDeltaNet(
            hidden_size=dim,
            num_heads=n_heads,
            head_dim=target_key_dim // n_heads,
            allow_neg_eigval=allow_neg_eigval,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor = None,   # unused, kept for API compat
        tok_idx: torch.Tensor = None,    # unused, kept for API compat
        mask=None,                       # unused, kept for API compat
        attn_impl: str = "flash_attn3",  # unused, kept for API compat
    ) -> torch.Tensor:
        x_normed = self.attention_norm(x)
        attn_out = self.attention(
            x_normed,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        import math

        init_std = init_std or (self.dim ** (-0.5))
        attn = self.attention

        for proj_name in ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "a_proj", "b_proj"]:
            if hasattr(attn, proj_name):
                proj = getattr(attn, proj_name)
                if isinstance(proj, nn.Linear):
                    nn.init.trunc_normal_(
                        proj.weight, mean=0.0, std=init_std,
                        a=-3 * init_std, b=3 * init_std,
                    )
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

        if hasattr(attn, "A_log") and hasattr(attn, "dt_bias"):
            with torch.no_grad():
                A = torch.empty_like(attn.A_log).uniform_(0, 16)
                attn.A_log.copy_(A.log())
                dt = torch.empty_like(attn.dt_bias).uniform_(0, 1)
                dt = torch.exp(
                    dt * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                attn.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        for attr in ["o_norm", "rotary", "g_norm_swish_gate", "g_norm"]:
            obj = getattr(attn, attr, None)
            if obj is not None and hasattr(obj, "reset_parameters"):
                obj.reset_parameters()
                break  # only reset first matching norm

        for conv_name in ["q_conv1d", "k_conv1d", "v_conv1d"]:
            conv = getattr(attn, conv_name, None)
            if conv is not None and hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)


class RetNetLinearAttentionBlock(nn.Module):
    """Transformer block using RetNet (MultiScaleRetention from fla) for the attention path.

    Same outer contract as LinearAttentionBlock (GDN): ignores sliding-window masks from
    attention_pattern; RoPE and retention are handled inside fla.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        retention_mode: str = "chunk",
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        try:
            from fla.layers import MultiScaleRetention
        except ImportError as exc:
            raise ImportError(
                "flash-linear-attention (fla) is not installed. "
                "Please install it with: pip install flash-linear-attention"
            ) from exc

        # Default expand_k=1, expand_v=2 in fla: head dims must divide dim and 2*dim by n_heads.
        self.attention = MultiScaleRetention(
            mode=retention_mode,
            hidden_size=dim,
            num_heads=n_heads,
            layer_idx=layer_idx,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor = None,
        tok_idx: torch.Tensor = None,
        mask=None,
        attn_impl: str = "flash_attn3",
    ) -> torch.Tensor:
        x_normed = self.attention_norm(x)
        attn_out = self.attention(
            x_normed,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))
        attn = self.attention

        for proj_name in ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"]:
            if hasattr(attn, proj_name):
                proj = getattr(attn, proj_name)
                if isinstance(proj, nn.Linear):
                    nn.init.trunc_normal_(
                        proj.weight,
                        mean=0.0,
                        std=init_std,
                        a=-3 * init_std,
                        b=3 * init_std,
                    )
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

        for attr in ["rotary", "g_norm_swish_gate", "g_norm", "o_norm"]:
            obj = getattr(attn, attr, None)
            if obj is not None and hasattr(obj, "reset_parameters"):
                obj.reset_parameters()

        for conv_name in ["q_conv1d", "k_conv1d", "v_conv1d"]:
            conv = getattr(attn, conv_name, None)
            if conv is not None and hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)


class DeltaNetLinearAttentionBlock(nn.Module):
    """Transformer block using DeltaNet (``fla.layers.DeltaNet``) for the attention path.

    Same outer contract as LinearAttentionBlock / RetNet: ignores sliding-window masks;
    uses fla's chunk / fused_recurrent delta rule (default ``mode="chunk"``).
    With default ``expand_k=expand_v=1``, requires ``dim % n_heads == 0``.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        deltanet_mode: str = "chunk",
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        try:
            from fla.layers import DeltaNet
        except ImportError as exc:
            raise ImportError(
                "flash-linear-attention (fla) is not installed. "
                "Please install it with: pip install flash-linear-attention"
            ) from exc

        if dim % n_heads != 0:
            raise ValueError(
                f"DeltaNet (expand_k=expand_v=1): dim ({dim}) must be divisible by n_heads ({n_heads})."
            )

        self.attention = DeltaNet(
            mode=deltanet_mode,
            hidden_size=dim,
            num_heads=n_heads,
            expand_k=1.0,
            expand_v=1.0,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor = None,
        tok_idx: torch.Tensor = None,
        mask=None,
        attn_impl: str = "flash_attn3",
    ) -> torch.Tensor:
        x_normed = self.attention_norm(x)
        attn_out = self.attention(
            x_normed,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))
        attn = self.attention

        for proj_name in ["q_proj", "k_proj", "v_proj", "b_proj", "g_proj", "o_proj"]:
            if hasattr(attn, proj_name):
                proj = getattr(attn, proj_name)
                if isinstance(proj, nn.Linear):
                    nn.init.trunc_normal_(
                        proj.weight,
                        mean=0.0,
                        std=init_std,
                        a=-3 * init_std,
                        b=3 * init_std,
                    )
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

        if hasattr(attn, "o_norm") and hasattr(attn.o_norm, "reset_parameters"):
            attn.o_norm.reset_parameters()

        for conv_name in ["q_conv1d", "k_conv1d", "v_conv1d"]:
            conv = getattr(attn, conv_name, None)
            if conv is not None and hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)


class KDALinearAttentionBlock(nn.Module):
    """Transformer block using Kimi Delta Attention (``fla.layers.KimiDeltaAttention``).

    Same outer contract as other fla linear blocks. Uses ``head_dim = dim // n_heads``.
    ``expand_v`` controls the value head expansion ratio; set to 1.5 to match GDN's
    value_dim (1536 for dim=1024, n_heads=16) for a fair parameter comparison.
    Training uses ``mode="chunk"`` (required by fla KDA during training).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        kda_mode: str = "chunk",
        expand_v: float = 1.0,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        try:
            from fla.layers import KimiDeltaAttention
        except ImportError as exc:
            raise ImportError(
                "flash-linear-attention (fla) is not installed. "
                "Please install it with: pip install flash-linear-attention"
            ) from exc

        if dim % n_heads != 0:
            raise ValueError(
                f"KDA: dim ({dim}) must be divisible by n_heads ({n_heads}) for head_dim = dim // n_heads."
            )
        head_dim = dim // n_heads

        self.attention = KimiDeltaAttention(
            hidden_size=dim,
            head_dim=head_dim,
            num_heads=n_heads,
            expand_v=expand_v,
            mode=kda_mode,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor = None,
        tok_idx: torch.Tensor = None,
        mask=None,
        attn_impl: str = "flash_attn3",
    ) -> torch.Tensor:
        x_normed = self.attention_norm(x)
        attn_out = self.attention(
            x_normed,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        import math

        init_std = init_std or (self.dim ** (-0.5))
        attn = self.attention

        def _init_linear(m: nn.Linear) -> None:
            nn.init.trunc_normal_(
                m.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for proj_name in ["q_proj", "k_proj", "v_proj", "b_proj", "o_proj"]:
            proj = getattr(attn, proj_name, None)
            if isinstance(proj, nn.Linear):
                _init_linear(proj)

        for seq_name in ("f_proj", "g_proj"):
            seq = getattr(attn, seq_name, None)
            if isinstance(seq, nn.Sequential):
                for m in seq:
                    if isinstance(m, nn.Linear):
                        _init_linear(m)

        if hasattr(attn, "A_log") and hasattr(attn, "dt_bias"):
            with torch.no_grad():
                A = torch.empty_like(attn.A_log).uniform_(1, 16)
                attn.A_log.copy_(A.log())
                dt = torch.empty_like(attn.dt_bias, dtype=torch.float32).uniform_(0, 1)
                dt = torch.exp(
                    dt * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                attn.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        if hasattr(attn, "o_norm") and hasattr(attn.o_norm, "reset_parameters"):
            attn.o_norm.reset_parameters()

        for conv_name in ["q_conv1d", "k_conv1d", "v_conv1d"]:
            conv = getattr(attn, conv_name, None)
            if conv is not None and hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)


class HGRN2LinearAttentionBlock(nn.Module):
    """Transformer block using ``fla.layers.HGRN2Attention`` (HGRN2 gated linear RNN).

    Ignores looped-window masks. Requires ``dim % n_heads == 0`` (fla uses
    ``expand_ratio = dim // n_heads``).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        hgrn2_mode: str = "chunk",
        hgrn2_use_short_conv: bool = False,
        hgrn2_conv_size: int = 4,
        hgrn2_conv_bias: bool = False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        try:
            from fla.layers import HGRN2Attention
        except ImportError as exc:
            raise ImportError(
                "flash-linear-attention (fla) is not installed. "
                "Please install it with: pip install flash-linear-attention"
            ) from exc

        if dim % n_heads != 0:
            raise ValueError(
                f"HGRN2: dim ({dim}) must be divisible by n_heads ({n_heads})."
            )

        self.attention = HGRN2Attention(
            mode=hgrn2_mode,
            hidden_size=dim,
            num_heads=n_heads,
            expand_ratio=None,
            use_short_conv=hgrn2_use_short_conv,
            conv_size=hgrn2_conv_size,
            conv_bias=hgrn2_conv_bias,
            elementwise_affine=True,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor = None,
        tok_idx: torch.Tensor = None,
        mask=None,
        attn_impl: str = "flash_attn3",
    ) -> torch.Tensor:
        x_normed = self.attention_norm(x)
        attn_out = self.attention(
            x_normed,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))
        attn = self.attention

        for proj_name in ["q_proj", "f_proj", "i_proj", "o_proj"]:
            proj = getattr(attn, proj_name, None)
            if isinstance(proj, nn.Linear):
                nn.init.trunc_normal_(
                    proj.weight,
                    mean=0.0,
                    std=init_std,
                    a=-3 * init_std,
                    b=3 * init_std,
                )
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

        g_norm = getattr(attn, "g_norm", None)
        if g_norm is not None and hasattr(g_norm, "reset_parameters"):
            g_norm.reset_parameters()

        for conv_name in ["q_conv1d", "f_conv1d", "i_conv1d"]:
            conv = getattr(attn, conv_name, None)
            if conv is not None and hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)


class MLALinearAttentionBlock(nn.Module):
    """Transformer block using ``fla.layers.MultiheadLatentAttention`` (DeepSeek MLA).

    Uses internal RoPE and FlashAttention; ignores ``freq_cis`` / sliding-window masks from
    the looped trainer. Requires ``pip install flash-attn`` (fla raises if ``flash_attn`` is missing).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        rope_theta: float,
        max_seqlen: int,
        mla_q_lora_rank: Optional[int] = None,
        mla_kv_lora_rank: int = 256,
        mla_qk_rope_head_dim: int = 64,
        mla_qk_nope_head_dim: int = 64,
        mla_v_head_dim: int = 64,
        mla_window_size: Optional[int] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        try:
            from fla.layers import MultiheadLatentAttention
        except ImportError as exc:
            raise ImportError(
                "flash-linear-attention (fla) is not installed. "
                "Please install it with: pip install flash-linear-attention"
            ) from exc

        self.attention = MultiheadLatentAttention(
            hidden_size=dim,
            num_heads=n_heads,
            q_lora_rank=mla_q_lora_rank,
            qk_rope_head_dim=mla_qk_rope_head_dim,
            kv_lora_rank=mla_kv_lora_rank,
            v_head_dim=mla_v_head_dim,
            qk_nope_head_dim=mla_qk_nope_head_dim,
            qk_head_dim=None,
            window_size=mla_window_size,
            rope_theta=rope_theta,
            max_position_embeddings=max_seqlen,
            layer_idx=layer_idx,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor = None,
        tok_idx: torch.Tensor = None,
        mask=None,
        attn_impl: str = "flash_attn3",
    ) -> torch.Tensor:
        x_normed = self.attention_norm(x)
        attn_out = self.attention(
            x_normed,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))
        attn = self.attention

        def _init_linear(m: nn.Linear) -> None:
            nn.init.trunc_normal_(
                m.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        def _init_seq_or_linear(obj) -> None:
            if isinstance(obj, nn.Linear):
                _init_linear(obj)
            elif isinstance(obj, nn.Sequential):
                for m in obj:
                    if isinstance(m, nn.Linear):
                        _init_linear(m)
                    elif hasattr(m, "reset_parameters"):
                        m.reset_parameters()

        _init_seq_or_linear(attn.q_proj)
        k_rope = getattr(attn, "k_rope", None)
        if isinstance(k_rope, nn.Linear):
            _init_linear(k_rope)
        _init_seq_or_linear(attn.kv_proj)
        o_proj = getattr(attn, "o_proj", None)
        if isinstance(o_proj, nn.Linear):
            _init_linear(o_proj)

        rot = getattr(attn, "rotary", None)
        if rot is not None and hasattr(rot, "reset_parameters"):
            rot.reset_parameters()

        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)


class Mamba2LinearAttentionBlock(nn.Module):
    """Transformer block using ``fla.layers.mamba2.Mamba2``.

    Same outer contract as LinearAttentionBlock / RetNetLinearAttentionBlock: ignores
    sliding-window masks from attention_pattern.

    ``fla.layers.Mamba2`` uses ``hidden_size``, ``num_heads``, and ``head_dim``
    (= expand * dim // n_heads). ``n_heads`` in the YAML must divide ``expand * dim`` evenly.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        expand: int = 2,
        d_state: int = 128,
        d_conv: int = 4,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.expand = expand

        try:
            from fla.layers.mamba2 import Mamba2 as FLAMamba2
        except ImportError as exc:
            raise ImportError(
                "fla is not installed. Install with: pip install flash-linear-attention"
            ) from exc

        d_inner = expand * dim
        if d_inner % n_heads != 0:
            raise ValueError(
                f"Mamba2 (fla): expand * dim ({d_inner}) must be divisible by n_heads ({n_heads}) "
                "so head_dim = (expand * dim) // n_heads is integral."
            )
        head_dim = d_inner // n_heads

        self.attention = FLAMamba2(
            hidden_size=dim,
            num_heads=n_heads,
            head_dim=head_dim,
            expand=expand,
            state_size=d_state,
            conv_kernel=d_conv,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor = None,
        tok_idx: torch.Tensor = None,
        mask=None,
        attn_impl: str = "flash_attn3",
    ) -> torch.Tensor:
        x_normed = self.attention_norm(x)
        # fla Mamba2: (B, L, D) -> ((B, L, D), None, cache)
        attn_out = self.attention(x_normed)[0]
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        import math

        init_std = init_std or (self.dim ** (-0.5))
        attn = self.attention

        for proj_name in ["in_proj", "out_proj"]:
            proj = getattr(attn, proj_name, None)
            if proj is not None and isinstance(proj, nn.Linear):
                nn.init.trunc_normal_(
                    proj.weight,
                    mean=0.0,
                    std=init_std,
                    a=-3 * init_std,
                    b=3 * init_std,
                )
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

        conv = getattr(attn, "conv1d", None)
        if conv is not None and hasattr(conv, "reset_parameters"):
            conv.reset_parameters()

        if hasattr(attn, "A_log") and hasattr(attn, "dt_bias"):
            with torch.no_grad():
                A = torch.empty_like(attn.A_log).uniform_(0, 16)
                attn.A_log.copy_(A.log())
                dt = torch.empty_like(attn.dt_bias).uniform_(0, 1)
                dt = torch.exp(
                    dt * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                attn.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        if hasattr(attn, "D"):
            nn.init.ones_(attn.D)

        norm = getattr(attn, "norm", None)
        if norm is not None and hasattr(norm, "reset_parameters"):
            norm.reset_parameters()

        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)


class CoTGatedDeltaProductAttention(nn.Module):
    """Gated DeltaProduct realised as a CoT (chain-of-thought) loop.

    Repurposes ``num_householder`` as the *number of outer passes* over the model
    stack rather than a per-token width expansion (compare ``fla.layers.GatedDeltaProduct``,
    which folds N householders into a single ``chunk_gated_delta_product`` call).

    Design (V shared across cot steps, K and beta duplicated per step):
      * ``q_proj``, ``v_proj``, ``o_proj`` exist once.
      * ``k_projs``/``b_projs`` are ``ModuleList``s of length ``num_householder``;
        ``k_conv1ds`` mirrors this. At cot step k (``current_cot_step == k``),
        K and beta are produced by indices ``0..k-1`` and the model has previously
        cached V for steps ``1..k-1`` -- those V's are appended to the current V
        list so the operator sees an increasingly deep V history.
      * Q is computed once per step. Sequences are interleaved along the time axis
        (Q is zero-padded for the leading k-1 sub-steps and lives only on the kth);
        the forget gate ``g`` is non-zero only on the first sub-step.
      * The kernel is :func:`chunk_gated_delta_rule` (NOT ``chunk_gated_delta_product``);
        the product is realised by interleaving and then subsampling
        ``o[:, k-1 :: k, :]``.

    Forward state contract:
      * ``current_cot_step`` (1-indexed) and ``cached_v`` (list of [B,T,value_dim] from
        prior steps in this forward; ``None`` on the first pass) are threaded in by
        :class:`LoopedWindowTransformer`.
      * Returns ``(out, vs)`` where ``vs`` is the V list (newest first) to be cached
        for the next cot step.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_householder: int,
        head_dim: int,
        expand_v: float = 1.5,
        conv_size: int = 4,
        norm_eps: float = 1e-5,
        use_forget_gate: bool = True,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
    ):
        super().__init__()
        import math
        try:
            from fla.modules import FusedRMSNormSwishGate, RMSNorm as FlaRMSNorm, ShortConvolution
        except ImportError as exc:
            raise ImportError(
                "flash-linear-attention (fla) is required for CoTGatedDeltaProduct."
            ) from exc

        if num_householder < 1:
            raise ValueError(f"num_householder must be >= 1, got {num_householder}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_householder = num_householder
        self.head_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * head_dim
        self.value_dim = num_heads * self.head_v_dim
        self.use_forget_gate = use_forget_gate
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.k_projs = nn.ModuleList(
            [nn.Linear(hidden_size, self.key_dim, bias=False) for _ in range(num_householder)]
        )
        self.b_projs = nn.ModuleList(
            [nn.Linear(hidden_size, num_heads, bias=False) for _ in range(num_householder)]
        )

        self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation="silu")
        self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, activation="silu")
        self.k_conv1ds = nn.ModuleList(
            [ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation="silu")
             for _ in range(num_householder)]
        )

        self.v_norm = FlaRMSNorm(self.value_dim, eps=norm_eps)

        if use_forget_gate:
            self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
            A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True
            dt = torch.exp(
                torch.rand(num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
            self.dt_bias._no_weight_decay = True

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = FlaRMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    @staticmethod
    def _interleave(seqs: List[torch.Tensor]) -> torch.Tensor:
        """Interleave a list of [B, T, D] tensors along the time axis: out[:, i::n, :] == seqs[i]."""
        if len(seqs) == 1:
            return seqs[0]
        B, T = seqs[0].shape[:2]
        rest = seqs[0].shape[2:]
        # Stack along a new axis 2: [B, T, n, *rest] -> [B, T*n, *rest]
        return torch.stack(seqs, dim=2).view(B, T * len(seqs), *rest)

    def forward(
        self,
        hidden_states: torch.Tensor,
        current_cot_step: int = 1,
        cached_v: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from einops import rearrange

        n = current_cot_step
        if n < 1 or n > self.num_householder:
            raise ValueError(
                f"current_cot_step={n} out of range [1, num_householder={self.num_householder}]"
            )

        # New V for this pass (will be cached for next pass).
        v_new, _ = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False)
        vs: List[torch.Tensor] = [v_new]
        if cached_v is not None:
            vs.extend(cached_v)

        ks: List[torch.Tensor] = []
        betas: List[torch.Tensor] = []
        for i in range(n):
            k_i, _ = self.k_conv1ds[i](x=self.k_projs[i](hidden_states), cache=None, output_final_state=False)
            ks.append(k_i)
            b_i = self.b_projs[i](hidden_states).sigmoid()
            if self.allow_neg_eigval:
                b_i = b_i * 2.0
            betas.append(b_i)

        q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False)
        q = self._interleave([torch.zeros_like(q)] * (n - 1) + [q])
        k = self._interleave(ks)
        v = self._interleave(vs)
        beta = self._interleave(betas)

        v = self.v_norm(v)

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_heads)

        if self.use_forget_gate:
            g = -self.A_log.float().exp() * F.softplus(
                self.a_proj(hidden_states).float() + self.dt_bias
            )
            # g non-zero only on the first sub-step of each interleaved group.
            g = self._interleave([g] + [torch.zeros_like(g)] * (n - 1))
        else:
            g = torch.zeros(
                hidden_states.shape[0], hidden_states.shape[1] * n, self.num_heads,
                dtype=torch.float32, device=hidden_states.device,
            )

        o, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        # Subsample: keep only the last sub-step output for each original token.
        o = o[:, n - 1 :: n, :]

        if self.use_gate:
            g_out = rearrange(self.g_proj(hidden_states), "b t (h d) -> b t h d", h=self.num_heads)
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o, vs


class CoTGatedDeltaProductBlock(nn.Module):
    """LT2 transformer block wrapping :class:`CoTGatedDeltaProductAttention`.

    Plays the same role as :class:`LinearAttentionBlock` but threads CoT state
    (``current_cot_step``, per-layer V cache) through its forward signature.
    The ``num_householder`` parameter must equal the outer ``loop_count`` of the
    :class:`LoopedWindowTransformer` so step k of the outer loop matches K/beta
    projection index k.

    If ``head_dim_override`` is given, that head dim is used directly (matching
    the reference CoTFormer's explicit head_dim=256 setup). Otherwise the LT2
    GDN convention is used: head_dim = 0.75 * dim / n_heads.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        num_householder: int,
        head_dim_override: Optional[int] = None,
        expand_v: float = 1.0,
        conv_size: int = 4,
        use_forget_gate: bool = True,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        num_total_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.num_householder = num_householder
        # Stored for GPT-2 style residual scaling in init_weights (1/sqrt(2*L) on o_proj/w2).
        self.num_total_layers = num_total_layers

        if head_dim_override is not None:
            head_dim = head_dim_override
        else:
            target_key_dim = int(0.75 * dim)
            if target_key_dim % n_heads != 0:
                raise ValueError(
                    f"CoTGatedDeltaProduct: 0.75 * dim ({target_key_dim}) must be divisible by "
                    f"n_heads ({n_heads}). Adjust n_heads or set cot_gdp_head_dim."
                )
            head_dim = target_key_dim // n_heads

        self.attention = CoTGatedDeltaProductAttention(
            hidden_size=dim,
            num_heads=n_heads,
            num_householder=num_householder,
            head_dim=head_dim,
            expand_v=expand_v,
            conv_size=conv_size,
            norm_eps=norm_eps,
            use_forget_gate=use_forget_gate,
            use_gate=use_gate,
            allow_neg_eigval=allow_neg_eigval,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        current_cot_step: int = 1,
        cached_v: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_out, vs = self.attention(
            self.attention_norm(x),
            current_cot_step=current_cot_step,
            cached_v=cached_v,
        )
        h = x + attn_out
        h = h + self.feed_forward(self.ffn_norm(h))
        return h, vs

    def init_weights(self, init_std=None, factor=1.0):
        """Initialize weights mirroring the reference CoTFormer.

        Reference scheme
        (fla/models/gated_deltaproduct_cotformer/modeling_...py):
          * Every ``nn.Linear`` inside the block: ``xavier_uniform_(gain=2**-2.5)``.
          * After the per-module init, the residual-stream output projections
            (``o_proj`` and FFN ``down_proj``/``w2``) are divided by
            ``sqrt(2 * num_hidden_layers)`` — GPT-2 style residual scaling so the
            activations don't grow with depth.
          * Forget-gate ``A_log`` / ``dt_bias`` use the Mamba2 initialization.

        The ``factor`` argument is ignored here -- this block uses its own scaling
        and does not participate in LT2's ``init_std_factor`` mechanism.
        """
        import math

        attn = self.attention
        xavier_gain = 2 ** -2.5

        def _xavier(m: nn.Linear) -> None:
            nn.init.xavier_uniform_(m.weight, gain=xavier_gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for proj in [attn.q_proj, attn.v_proj, attn.o_proj]:
            _xavier(proj)
        for mlist in [attn.k_projs, attn.b_projs]:
            for proj in mlist:
                _xavier(proj)
        if attn.use_gate:
            _xavier(attn.g_proj)

        if attn.use_forget_gate:
            _xavier(attn.a_proj)
            with torch.no_grad():
                A = torch.empty_like(attn.A_log).uniform_(0, 16)
                attn.A_log.copy_(A.log())
                dt = torch.empty_like(attn.dt_bias).uniform_(0, 1)
                dt = torch.exp(
                    dt * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                attn.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        for conv in [attn.q_conv1d, attn.v_conv1d, *attn.k_conv1ds]:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        for norm_attr in ["o_norm", "v_norm"]:
            obj = getattr(attn, norm_attr, None)
            if obj is not None and hasattr(obj, "reset_parameters"):
                obj.reset_parameters()

        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        # FeedForward.w1/w3 get re-initialised here; w2 is overwritten below.
        self.feed_forward.reset_parameters(init_std, factor)
        for w in [self.feed_forward.w1, self.feed_forward.w3]:
            _xavier(w)
        _xavier(self.feed_forward.w2)

        # GPT-2 style residual scaling for the two output projections.
        scale = 1.0 / math.sqrt(2 * max(1, self.num_total_layers))
        with torch.no_grad():
            attn.o_proj.weight.mul_(scale)
            self.feed_forward.w2.weight.mul_(scale)


def _patch_fla_nsa_flash_attn_to_flash_attn_3() -> None:
    """Point ``fla.ops.nsa.parallel`` at Flash Attention 3 when FA2 is missing or forced.

    fla imports ``from flash_attn import flash_attn_func``; on some systems the FA2 wheel fails
    (e.g. GLIBC too old) while ``flash_attn_interface`` (FA3) still loads. FA3 exposes the same
    ``flash_attn_func`` / ``flash_attn_varlen_func`` kwargs fla uses (``causal``, ``window_size``).

    Set ``LINGUA_FLA_NSA_USE_FLASH_ATTN3=1`` to always use FA3 for the NSA sliding-window path.
    """
    import fla.ops.nsa.parallel as parallel_mod

    force_fa3 = os.environ.get("LINGUA_FLA_NSA_USE_FLASH_ATTN3", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if parallel_mod.flash_attn_func is not None and not force_fa3:
        return
    try:
        import flash_attn_interface as fa3_iface
    except ImportError as exc:
        raise ImportError(
            "NSA needs flash_attn_func for sliding-window (window_size > 0). "
            "flash_attn (FA2) is not available and `import flash_attn_interface` (FA3) failed. "
            "Install flash_attn_3 / FA3 wheels compatible with your CUDA driver, or set "
            "nsa_window_size: 0 to skip the SWA branch."
        ) from exc
    parallel_mod.flash_attn_func = fa3_iface.flash_attn_func
    parallel_mod.flash_attn_varlen_func = fa3_iface.flash_attn_varlen_func


class NSALinearAttentionBlock(nn.Module):
    """Transformer block using ``fla.layers.NativeSparseAttention`` (NSA).

    Ignores looped-window ``freq_cis`` / masks; RoPE and sparse patterns are internal to fla.
    Requires a recent ``flash-linear-attention`` build with NSA Triton ops (CUDA).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        rope_theta: float,
        max_seqlen: int,
        nsa_block_size: int,
        nsa_block_counts: int,
        nsa_window_size: Optional[int],
        nsa_qkv_bias: bool,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        try:
            from fla.layers import NativeSparseAttention
        except ImportError as exc:
            raise ImportError(
                "flash-linear-attention (fla) is not installed. "
                "Please install it with: pip install flash-linear-attention"
            ) from exc

        effective_ws = 512 if nsa_window_size is None else int(nsa_window_size)
        if effective_ws > 0:
            _patch_fla_nsa_flash_attn_to_flash_attn_3()

        assert n_heads % n_kv_heads == 0
        heads_per_kv = n_heads // n_kv_heads
        if heads_per_kv % 16 != 0:
            raise ValueError(
                f"NSA (fla): n_heads ({n_heads}) // n_kv_heads ({n_kv_heads}) = {heads_per_kv} must be "
                "a multiple of 16 (fla.ops.nsa.parallel_nsa group-size constraint). "
                "Example: n_heads=16 with n_kv_heads=1, or n_heads=32 with n_kv_heads=2."
            )

        self.attention = NativeSparseAttention(
            hidden_size=dim,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            qkv_bias=nsa_qkv_bias,
            block_size=nsa_block_size,
            block_counts=nsa_block_counts,
            window_size=nsa_window_size,
            rope_theta=rope_theta,
            max_position_embeddings=max_seqlen,
            layer_idx=layer_idx,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    # fla NSA uses custom autograd + Triton; torch.compile/Dynamo can break
    # (e.g. ParallelNSAFunction.apply becomes None). Run this block eagerly.
    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor = None,
        tok_idx: torch.Tensor = None,
        mask=None,
        attn_impl: str = "flash_attn3",
    ) -> torch.Tensor:
        x_normed = self.attention_norm(x)
        attn_out = self.attention(
            x_normed,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))
        attn = self.attention

        for proj_name in ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"]:
            proj = getattr(attn, proj_name, None)
            if proj is not None and isinstance(proj, nn.Linear):
                nn.init.trunc_normal_(
                    proj.weight,
                    mean=0.0,
                    std=init_std,
                    a=-3 * init_std,
                    b=3 * init_std,
                )
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

        rot = getattr(attn, "rotary", None)
        if rot is not None and hasattr(rot, "reset_parameters"):
            rot.reset_parameters()

        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)


class _SparseMLA(torch.autograd.Function):
    """Wraps the TileLang sparse MLA fwd+bwd kernels as a differentiable op.

    Inputs that need gradients: q, kv.
    indices is int32 and non-differentiable (top-k selection is treated as a
    straight-through / stop-gradient operation, same as the original DSA paper).
    """

    @staticmethod
    def forward(ctx, q, kv, indices, sm_scale, d_v, block_i):
        from .sparse_mla_fwd import sparse_mla_fwd_interface
        out, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale, d_v=d_v, block_I=block_i)
        ctx.save_for_backward(q, kv, out, indices, lse)
        ctx.sm_scale = sm_scale
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        from .sparse_mla_bwd import sparse_mla_bwd
        q, kv, out, indices, lse = ctx.saved_tensors
        dout = dout.contiguous()
        dq, dkv = sparse_mla_bwd(q, kv, out, dout, indices, lse, sm_scale=ctx.sm_scale)
        return dq, dkv, None, None, None, None  # no grad for indices/sm_scale/d_v/block_i


class DSABlock(nn.Module):
    """Transformer block using DeepSeek DSA (Dynamic Sparse Attention) TileLang kernels.

    Follows the three-stage DSA pipeline from deepseek_v32:
      1. Scoring attention: cheap compressed Q_s × KV_s matmul → per-token scores [B*S, S]
      2. Top-k selection: TileLang radix-sort kernel → int32 indices [B, S, kv_group, topk]
      3. Sparse MLA: TileLang fwd+bwd kernels over selected positions

    Gradients flow through q_proj/kv_proj/o_proj/score projections via _SparseMLA.
    Indices are treated as stop-gradient (straight-through), same as the original paper.

    Kernel constraints (sparse_mla_fwd.py / sparse_mla_bwd.py):
      - dsa_dim + dsa_tail_dim == 576  (hardcoded in interface)
      - dsa_dim, dsa_tail_dim must each be a power of 2  (512, 64)
      - dsa_topk % dsa_block_i == 0
      - all kernel inputs must be bfloat16 and contiguous
      - n_kv_heads == 1  (kv_group=1)
    """

    _KERNEL_DIM_PLUS_TAIL: int = 576

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        dsa_dim: int,
        dsa_tail_dim: int,
        dsa_topk: int,
        dsa_block_i: int,
        dsa_score_dim: int,
        dsa_sm_scale: Optional[float] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.model_dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dsa_dim = dsa_dim
        self.dsa_tail_dim = dsa_tail_dim
        self.dsa_topk = dsa_topk
        self.dsa_block_i = dsa_block_i
        self.dsa_score_dim = dsa_score_dim
        self.dsa_sm_scale = dsa_sm_scale

        assert dsa_dim + dsa_tail_dim == self._KERNEL_DIM_PLUS_TAIL, (
            f"dsa_dim ({dsa_dim}) + dsa_tail_dim ({dsa_tail_dim}) must equal "
            f"{self._KERNEL_DIM_PLUS_TAIL} (kernel constraint)"
        )
        assert dsa_topk % dsa_block_i == 0, (
            f"dsa_topk ({dsa_topk}) must be divisible by dsa_block_i ({dsa_block_i})"
        )

        proj_dim = dsa_dim + dsa_tail_dim  # 576

        # Main attention projections
        self.q_proj = nn.Linear(dim, n_heads * proj_dim, bias=False)
        self.kv_proj = nn.Linear(dim, n_kv_heads * proj_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * dsa_dim, dim, bias=False)

        # Scoring projections: cheap low-rank Q_s×K_s to produce per-token similarity scores.
        # score_dim << proj_dim so this is much cheaper than the main attention.
        self.q_score_proj = nn.Linear(dim, n_heads * dsa_score_dim, bias=False)
        self.k_score_proj = nn.Linear(dim, n_kv_heads * dsa_score_dim, bias=False)

        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    @torch.no_grad()
    def _sparse_indices(self, x_normed: torch.Tensor) -> torch.Tensor:
        """Compute causal top-k sparse indices via scoring attention + TileLang top-k.

        1. Project to low-dim score queries/keys.
        2. Compute causal Q_s × K_s^T scores → [B, S, S] (upper triangle masked).
        3. Run TileLang radix-sort top-k per query row.

        Returns int32 tensor [B, S, n_kv_heads, topk].
        Treated as stop-gradient: no grad flows through index selection.
        """
        from .topk_selector import tl_topk
        B, S, _ = x_normed.shape

        # [B, S, n_heads, score_dim] and [B, S, n_kv_heads, score_dim]
        q_s = self.q_score_proj(x_normed).view(B, S, self.n_heads, self.dsa_score_dim)
        k_s = self.k_score_proj(x_normed).view(B, S, self.n_kv_heads, self.dsa_score_dim)

        # GQA: repeat k_s to match n_heads, then mean-pool across heads → [B, S, S]
        k_s_rep = k_s.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)  # [B, S, n_heads, score_dim]
        # scores[b, q, k] = sum_h dot(q_s[b,q,h], k_s[b,k,h]) / n_heads
        scores = torch.einsum("bqhd,bkhd->bqk", q_s.float(), k_s_rep.float()) / self.n_heads  # [B, S, S]

        # Causal mask: zero out future positions (tl_topk picks highest, future → -inf)
        causal_mask = torch.ones(S, S, device=x_normed.device, dtype=torch.bool).triu(diagonal=1)
        scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

        # tl_topk expects [B*S, S] with per-row start/end bounds
        scores_flat = scores.view(B * S, S).contiguous().float()
        # Each query at position s can attend to [0, s]; encode as start=0, end=s+1
        row_idx = torch.arange(S, device=x_normed.device, dtype=torch.int32)
        starts = torch.zeros(B * S, device=x_normed.device, dtype=torch.int32)
        ends = (row_idx + 1).repeat(B)  # end = position + 1 (exclusive)

        topk_per_row = min(self.dsa_topk, S)  # can't select more than S positions
        indices_flat = tl_topk(scores_flat, starts, ends, topk_per_row)  # [B*S, topk]

        # Reshape to [B, S, n_kv_heads=1, topk] and pad if needed
        indices = indices_flat.view(B, S, topk_per_row)
        if topk_per_row < self.dsa_topk:
            pad = indices[:, :, :1].expand(B, S, self.dsa_topk - topk_per_row)
            indices = torch.cat([indices, pad], dim=-1)

        return indices.unsqueeze(2).to(torch.int32).contiguous()  # [B, S, 1, topk]

    @torch.compiler.disable  # TileLang kernels use custom CUDA; disable dynamo
    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor = None,
        tok_idx: torch.Tensor = None,
        mask=None,
        attn_impl: str = "flash_attn3",
    ) -> torch.Tensor:
        B, S, _ = x.shape
        proj_dim = self.dsa_dim + self.dsa_tail_dim

        x_normed = self.attention_norm(x).to(torch.bfloat16)

        # Stage 1+2: scoring attention → top-k indices (no grad)
        indices = self._sparse_indices(x_normed)  # [B, S, 1, topk] int32

        # Stage 3: sparse MLA with fwd+bwd TileLang kernels
        q = self.q_proj(x_normed).view(B, S, self.n_heads, proj_dim).contiguous()
        kv = self.kv_proj(x_normed).view(B, S, self.n_kv_heads, proj_dim).contiguous()

        out, _lse = _SparseMLA.apply(q, kv, indices, self.dsa_sm_scale, self.dsa_dim, self.dsa_block_i)
        # out: [B, S, n_heads, dsa_dim]

        h = x + self.o_proj(out.view(B, S, self.n_heads * self.dsa_dim).to(x.dtype))
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        init_std = init_std or (self.model_dim ** (-0.5))
        for proj in (self.q_proj, self.kv_proj, self.o_proj, self.q_score_proj, self.k_score_proj):
            nn.init.trunc_normal_(proj.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)


@dataclass
class LoopedWindowTransformerArgs(BaseTransformerArgs):
    """Arguments for looped window attention transformer."""

    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False

    # Attention implementation
    # Options: "fmha" (Flash Multi-Head Attention), "sdpa" (Scaled Dot Product Attention), "flex_attention"
    # Note: Only "fmha" and "flex_attention" support sliding window attention
    # Options: "flash_attn3" (Flash Attention 3), "fmha" (xformers), "sdpa", "flex_attention"
    # Note: "flash_attn3" and "fmha" and "flex_attention" support sliding window attention
    attn_impl: str = "flash_attn3"  # Default to Flash Attention 3 for causal and window attn

    # Looping configuration
    loop_count: int = 1  # How many times to loop through the layers
    use_residual: bool = True  # Whether to use learned residual connection between iterations
    use_block_residual: bool = True  # Whether to use learned residual connection across blocks

    # Hybrid attention configuration
    # Can be:
    # - A ratio like "4:1" (4 sliding window : 1 full attention, interleaved)
    # - A bookend pattern like "bookend:2" or "sandwich:2" (full attn at start/end, window in middle)
    # - Explicit list like "2048,2048,None,2048" (None = full attention)
    # - Per-iteration pattern like "full->SWA=128->SWA=128->SWA=128" (different pattern per iteration)
    #   Use "->" to separate iterations. Each can be: "full", "SWA=N", or any regular pattern
    attention_pattern: str = "1:0"  # Default: all sliding window

    # Default sliding window size (used when pattern specifies sliding window)
    # Can be None if pattern doesn't use sliding windows (e.g., "0:1" for pure full attention)
    default_sliding_window: Optional[int] = 2048

    # Number of ensemble passes for inference with random attention patterns
    inference_ensemble_k: int = 5

    # Legacy compatibility: weighted sum over iterations (for old checkpoints)
    legacy_iteration_weights: bool = False

    # Interleaved linear-attn / full layer pattern (orthogonal to attention_pattern).
    # Controls which layers use fla / mamba sequence mixers vs standard TransformerBlock.
    # - "full"      : all layers are standard TransformerBlock (default, backward-compatible)
    # - "gdn"       : all layers are GatedDeltaNet
    # - "retnet"    : all layers are MultiScaleRetention (RetNet)
    # - "deltanet"    : all layers are DeltaNet (fla)
    # - "kda"         : all layers are KimiDeltaAttention (fla)
    # - "hgrn2"       : all layers are HGRN2Attention (fla)
    # - "mla"         : all layers are MultiheadLatentAttention (fla; flash-attn)
    # - "mamba2"      : all layers are Mamba2 (fla.layers.mamba2)
    # - "nsa"         : all layers are NativeSparseAttention (fla; NSA Triton)
    # - "dsa"         : all layers are DSA sparse MLA (TileLang kernel; see DSABlock)
    # - "interleaved:N:M:TYPE1:TYPE2" : cycle of N TYPE1 then M TYPE2 layers
    #   TYPE1/TYPE2 ∈ {gdn, retnet, deltanet, kda, hgrn2, mla, mamba2, nsa, dsa, full}.
    #   Example: "interleaved:4:1:dsa:full"
    # Linear-attention layers ignore the window mask from attention_pattern.
    # Full layers use flash_attn3 (or the configured attn_impl) with the window mask.
    layer_pattern: str = "full"

    # KDA-specific: value head expansion ratio.  expand_v=1.5 gives value_dim=1536 for
    # dim=1024/n_heads=16, matching GDN's value_dim for a fair parameter comparison.
    kda_expand_v: float = 1.0

    # GDN (layer_pattern includes "gdn") — fla.layers.GatedDeltaNet
    gdn_allow_neg_eigval: bool = False

    # CoT-GDP (layer_pattern "cot_gdp") — CoTFormer-style Gated DeltaProduct realised
    # as ``loop_count`` outer passes (V shared, K/beta duplicated, see
    # CoTGatedDeltaProductAttention). ``num_householder`` equals ``loop_count``;
    # the per-layer K/beta ModuleLists are sized to ``loop_count``.
    #
    # Geometry defaults mirror fla's ``gated_deltaproduct_cot_1.3B_new.json``:
    # head_dim=256, num_heads=8, expand_v=1 -> key_dim = value_dim = hidden_size.
    # Set to None to fall back to LT2's 0.75*dim GDN-style key dim with the
    # transformer-wide ``n_heads``.
    cot_gdp_head_dim: Optional[int] = 256
    cot_gdp_num_heads: Optional[int] = 8
    cot_gdp_expand_v: float = 1.0
    cot_gdp_allow_neg_eigval: bool = True
    cot_gdp_use_forget_gate: bool = True
    cot_gdp_use_gate: bool = True
    cot_gdp_conv_size: int = 4

    # HGRN2 (layer_pattern "hgrn2") — fla.layers.HGRN2Attention
    hgrn2_mode: str = "chunk"
    hgrn2_use_short_conv: bool = False
    hgrn2_conv_size: int = 4
    hgrn2_conv_bias: bool = False

    # MLA (layer_pattern "mla") — fla.layers.MultiheadLatentAttention (needs flash-attn)
    mla_q_lora_rank: Optional[int] = None
    mla_kv_lora_rank: int = 256
    mla_qk_rope_head_dim: int = 64
    mla_qk_nope_head_dim: int = 64
    mla_v_head_dim: int = 64
    mla_window_size: Optional[int] = None

    # Mamba2 (layer_pattern "mamba2") — fla.layers.mamba2.Mamba2 (no mamba-ssm needed; causal-conv1d optional for fast path)
    # headdim = (mamba2_expand * dim) // n_heads must be integral
    mamba2_expand: int = 2
    mamba2_d_state: int = 64
    mamba2_d_conv: int = 4

    # NSA (layer_pattern "nsa") — fla.layers.NativeSparseAttention
    nsa_block_size: int = 64
    nsa_block_counts: int = 16
    nsa_window_size: Optional[int] = 512
    nsa_qkv_bias: bool = False

    # DSA (layer_pattern "dsa") — DeepSeek sparse MLA via TileLang kernel
    # Kernel constraint: dsa_dim + dsa_tail_dim == 576 (hardcoded in sparse_mla_fwd.py)
    # dsa_dim and dsa_tail_dim must each be a power of 2; dsa_topk % dsa_block_i == 0.
    dsa_dim: int = 512           # Main K/V head dim (d_v in the kernel)
    dsa_tail_dim: int = 64       # Tail dim; dsa_dim + dsa_tail_dim must == 576
    dsa_topk: int = 2048         # Sparse positions per query per kv-group
    dsa_block_i: int = 64        # Kernel tile size (must divide dsa_topk)
    dsa_score_dim: int = 64      # Per-head dim for the cheap scoring Q_s×K_s attention
    dsa_sm_scale: Optional[float] = None  # Softmax scale; None → 1/sqrt(576) * log2(e)

    # Gated attention: elementwise head-specific sigmoid gate after SDPA output (G1 in paper).
    # See "Gated Attention for Large Language Models" (Qiu et al., NeurIPS 2025).
    # Only applies to "full" layer_type layers (standard TransformerBlock variants).
    use_attn_gate: bool = False


class LoopedWindowTransformer(BaseTransformer):
    """
    A transformer with four key features:
    1. Looping: Reuse the same layers ``loop_count`` times; the loss is the
       cross-entropy of the final hidden state after all loops complete (no
       per-iteration deep supervision).
    2. Hybrid Attention: Mix of sliding window and full attention (controlled by attention_pattern)
    3. Per-Iteration Attention: Different attention patterns for each loop iteration
    4. Optional Residual Connections: Learned residual connections between iterations (controlled by use_residual)
    5. Block Residuals: Learned residual connections across attention blocks (controlled by use_block_residual)

    The attention_pattern allows flexible specification:
    
    Per-Layer Patterns (same pattern for all iterations):
    - "4:1" means 4 sliding window layers followed by 1 full attention layer, repeated (interleaved, block size = 5)
    - "1:1" alternates between sliding window and full attention (block size = 2)
    - "8:0" means all layers use sliding window attention (block size = 8)
    - "bookend:2" or "sandwich:2" means 2 full attention layers at start and end, sliding window in middle
      (useful for global context capture at input/output with efficient processing in between)
    
    Per-Iteration Patterns (different pattern for each iteration):
    - "full->SWA=128->SWA=128->SWA=128" means:
      * Iteration 1: All layers use full attention
      * Iterations 2-4: All layers use sliding window attention with window size 128
    - "full->4:1->bookend:2" means:
      * Iteration 1: All layers use full attention
      * Iteration 2: 4:1 interleaved pattern
      * Iteration 3: Bookend pattern with 2 full attention layers at each end
    - Use "->" to separate iteration patterns. Number of patterns must match loop_count.
    - Each iteration can specify: "full" (full attention), "SWA=N" (sliding window with size N),
      or any regular pattern like "4:1", "bookend:2", or explicit comma-separated values.

    Key insight: Attention weights (Wq, Wk, Wv, Wo) are shared across iterations, but attention masks
    can differ, enabling flexible computation patterns without additional parameters.

    The loop_count allows the same layers to be applied multiple times, which can:
    - Reduce parameter count while maintaining depth
    - Create recurrent-like processing
    - Enable progressive refinement of representations

    When use_residual=True, a learned gated residual connection (initialized to 0) is added between
    iterations. Each dimension has its own gate, allowing fine-grained control over which features
    can bypass the transformer layers.
    
    When use_block_residual=True, a learned residual connection (initialized to 0) is added across
    each attention block (e.g., after each "4:1" block), allowing information to bypass blocks when beneficial.
    """
    
    def __init__(self, args: LoopedWindowTransformerArgs):
        # Pass n_layers=0 to avoid building TransformerBlock layers in the base class.
        # We build the interleaved GDN/full layers ourselves below.
        original_n_layers = args.n_layers
        args.n_layers = 0
        super().__init__(args)
        args.n_layers = original_n_layers

        self.weight_tying = args.weight_tying
        self.loop_count = args.loop_count
        self._loop_prefix_end = args.n_layers
        self.attn_impl = args.attn_impl
        self.args = args  # Store args for dynamic pattern generation
        self.inference_ensemble_k = args.inference_ensemble_k

        # Parse layer pattern: each layer is a fla/mamba mixer type or "full"
        self.layer_types = parse_layer_pattern(args.layer_pattern, args.n_layers)

        n_heads = args.n_heads or (args.dim // (args.head_dim or 64))
        head_dim = args.head_dim or args.dim // n_heads
        n_kv_heads = args.n_kv_heads or n_heads
        self.layers = nn.ModuleList()
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "gdn":
                self.layers.append(
                    LinearAttentionBlock(
                        dim=args.dim,
                        n_heads=n_heads,
                        ffn_dim_multiplier=args.ffn_dim_multiplier,
                        multiple_of=args.multiple_of,
                        norm_eps=args.norm_eps,
                        allow_neg_eigval=args.gdn_allow_neg_eigval,
                    )
                )
            elif layer_type == "cot_gdp":
                self.layers.append(
                    CoTGatedDeltaProductBlock(
                        dim=args.dim,
                        n_heads=args.cot_gdp_num_heads if args.cot_gdp_num_heads is not None else n_heads,
                        head_dim_override=args.cot_gdp_head_dim,
                        ffn_dim_multiplier=args.ffn_dim_multiplier,
                        multiple_of=args.multiple_of,
                        norm_eps=args.norm_eps,
                        num_householder=args.loop_count,
                        expand_v=args.cot_gdp_expand_v,
                        conv_size=args.cot_gdp_conv_size,
                        use_forget_gate=args.cot_gdp_use_forget_gate,
                        use_gate=args.cot_gdp_use_gate,
                        allow_neg_eigval=args.cot_gdp_allow_neg_eigval,
                        num_total_layers=args.n_layers,
                    )
                )
            elif layer_type == "retnet":
                self.layers.append(
                    RetNetLinearAttentionBlock(
                        dim=args.dim,
                        n_heads=n_heads,
                        ffn_dim_multiplier=args.ffn_dim_multiplier,
                        multiple_of=args.multiple_of,
                        norm_eps=args.norm_eps,
                        layer_idx=layer_idx,
                    )
                )
            elif layer_type == "deltanet":
                self.layers.append(
                    DeltaNetLinearAttentionBlock(
                        dim=args.dim,
                        n_heads=n_heads,
                        ffn_dim_multiplier=args.ffn_dim_multiplier,
                        multiple_of=args.multiple_of,
                        norm_eps=args.norm_eps,
                        layer_idx=layer_idx,
                    )
                )
            elif layer_type == "kda":
                self.layers.append(
                    KDALinearAttentionBlock(
                        dim=args.dim,
                        n_heads=n_heads,
                        ffn_dim_multiplier=args.ffn_dim_multiplier,
                        multiple_of=args.multiple_of,
                        norm_eps=args.norm_eps,
                        expand_v=args.kda_expand_v,
                        layer_idx=layer_idx,
                    )
                )
            elif layer_type == "hgrn2":
                self.layers.append(
                    HGRN2LinearAttentionBlock(
                        dim=args.dim,
                        n_heads=n_heads,
                        ffn_dim_multiplier=args.ffn_dim_multiplier,
                        multiple_of=args.multiple_of,
                        norm_eps=args.norm_eps,
                        hgrn2_mode=args.hgrn2_mode,
                        hgrn2_use_short_conv=args.hgrn2_use_short_conv,
                        hgrn2_conv_size=args.hgrn2_conv_size,
                        hgrn2_conv_bias=args.hgrn2_conv_bias,
                        layer_idx=layer_idx,
                    )
                )
            elif layer_type == "mla":
                self.layers.append(
                    MLALinearAttentionBlock(
                        dim=args.dim,
                        n_heads=n_heads,
                        ffn_dim_multiplier=args.ffn_dim_multiplier,
                        multiple_of=args.multiple_of,
                        norm_eps=args.norm_eps,
                        rope_theta=args.rope_theta,
                        max_seqlen=args.max_seqlen,
                        mla_q_lora_rank=args.mla_q_lora_rank,
                        mla_kv_lora_rank=args.mla_kv_lora_rank,
                        mla_qk_rope_head_dim=args.mla_qk_rope_head_dim,
                        mla_qk_nope_head_dim=args.mla_qk_nope_head_dim,
                        mla_v_head_dim=args.mla_v_head_dim,
                        mla_window_size=args.mla_window_size,
                        layer_idx=layer_idx,
                    )
                )
            elif layer_type == "mamba2":
                self.layers.append(
                    Mamba2LinearAttentionBlock(
                        dim=args.dim,
                        n_heads=n_heads,
                        ffn_dim_multiplier=args.ffn_dim_multiplier,
                        multiple_of=args.multiple_of,
                        norm_eps=args.norm_eps,
                        expand=args.mamba2_expand,
                        d_state=args.mamba2_d_state,
                        d_conv=args.mamba2_d_conv,
                        layer_idx=layer_idx,
                    )
                )
            elif layer_type == "nsa":
                self.layers.append(
                    NSALinearAttentionBlock(
                        dim=args.dim,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        ffn_dim_multiplier=args.ffn_dim_multiplier,
                        multiple_of=args.multiple_of,
                        norm_eps=args.norm_eps,
                        rope_theta=args.rope_theta,
                        max_seqlen=args.max_seqlen,
                        nsa_block_size=args.nsa_block_size,
                        nsa_block_counts=args.nsa_block_counts,
                        nsa_window_size=args.nsa_window_size,
                        nsa_qkv_bias=args.nsa_qkv_bias,
                        layer_idx=layer_idx,
                    )
                )
            elif layer_type == "dsa":
                self.layers.append(
                    DSABlock(
                        dim=args.dim,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        ffn_dim_multiplier=args.ffn_dim_multiplier,
                        multiple_of=args.multiple_of,
                        norm_eps=args.norm_eps,
                        dsa_dim=args.dsa_dim,
                        dsa_tail_dim=args.dsa_tail_dim,
                        dsa_topk=args.dsa_topk,
                        dsa_block_i=args.dsa_block_i,
                        dsa_score_dim=args.dsa_score_dim,
                        dsa_sm_scale=args.dsa_sm_scale,
                        layer_idx=layer_idx,
                    )
                )
            else:
                if args.use_attn_gate:
                    self.layers.append(GatedTransformerBlock(args))
                else:
                    self.layers.append(TransformerBlock(args))

        # Parse attention pattern
        attention_result = parse_attention_pattern(
            args.attention_pattern,
            args.n_layers,
            args.default_sliding_window,
            args.loop_count,
            args.seed
        )

        # Check if we have dynamic random patterns, per-iteration patterns, or a single pattern
        if isinstance(attention_result, dict) and attention_result.get("type") == "dynamic_random":
            # Dynamic random pattern: metadata for runtime generation
            self.attention_windows = None
            self.per_iteration_attention = True
            self.dynamic_random_pattern = attention_result
            # For block residuals, use layer count as block size (treat as one block)
            first_pattern = f"bookend:{args.n_layers}"  # Dummy pattern for block size calculation
            self.forward_step = 0  # Counter for deterministic randomness
        elif isinstance(attention_result[0], list):
            # Per-iteration patterns: List[List[Optional[int]]]
            self.attention_windows = attention_result
            self.per_iteration_attention = True
            self.dynamic_random_pattern = None
            # For block residuals, use the first iteration's pattern to compute block size
            first_pattern = self._reconstruct_pattern_string(attention_result[0], args.n_layers)
        else:
            # Single pattern for all iterations: List[Optional[int]]
            self.attention_windows = attention_result
            self.per_iteration_attention = False
            self.dynamic_random_pattern = None
            first_pattern = args.attention_pattern

        # Compute block boundaries for block-level residuals
        self.block_size = get_block_size(first_pattern, args.n_layers)
        self.n_blocks = (args.n_layers + self.block_size - 1) // self.block_size  # Ceiling division

        assert args.vocab_size > 0

        # Learned gated residual connection between iterations (init = 0) - only if enabled
        if args.use_residual:
            if args.legacy_iteration_weights:
                # Legacy format: scalar per iteration [loop_count]
                self.residual_weight = nn.Parameter(torch.zeros(self.loop_count))
            else:
                # New format: per-dimension gating [loop_count, dim]
                self.residual_weight = nn.Parameter(torch.zeros(self.loop_count, args.dim))
        else:
            self.residual_weight = None

        # Learned residual connection across blocks (init = 0) - only if enabled
        if args.use_block_residual:
            self.block_residual_weight = nn.Parameter(torch.zeros(self.n_blocks))
        else:
            self.block_residual_weight = None

        # Legacy: weighted sum over iterations (for old checkpoints)
        if args.legacy_iteration_weights:
            self.iteration_weights = nn.Parameter(torch.ones(self.loop_count) / self.loop_count)
        else:
            self.iteration_weights = None

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(
                args.dim,
                args.vocab_size,
                bias=False,
            )
    
    def _reconstruct_pattern_string(self, pattern_list: List[Optional[int]], n_layers: int) -> str:
        """Helper to reconstruct a pattern string from a list for block size computation."""
        # Simple heuristic: if all same, return that pattern; otherwise treat as explicit
        if all(w == pattern_list[0] for w in pattern_list):
            if pattern_list[0] is None:
                return "0:1"  # All full attention
            else:
                return "1:0"  # All sliding window
        else:
            # Return explicit pattern
            return ",".join("None" if w is None else str(w) for w in pattern_list)



    def _generate_dynamic_pattern(self):
        """Generate a random attention pattern and increment the step counter."""
        pattern = generate_random_attention_pattern(
            pattern_type=self.dynamic_random_pattern["pattern_type"],
            window_count=self.dynamic_random_pattern["window_count"],
            full_count=self.dynamic_random_pattern["full_count"],
            n_layers=self.args.n_layers,
            loop_count=self.loop_count,
            default_sliding_window=self.dynamic_random_pattern["default_sliding_window"],
            seed=self.args.seed + self.forward_step,
        )
        self.forward_step += 1
        return pattern

    def _iteration_windows_for(self, loop_idx: int, dynamic_pattern):
        if dynamic_pattern is not None:
            return dynamic_pattern[loop_idx]
        if self.per_iteration_attention:
            return self.attention_windows[loop_idx]
        return self.attention_windows

    def _forward_one_layer(
        self,
        layer_idx: int,
        h: torch.Tensor,
        iteration_windows,
        freq_cis,
        tok_idx,
        mask,
        attn_impl: str,
        seqlen: int,
        kv_loop_idx: int,
        cot_cache: Optional[List[Optional[List[torch.Tensor]]]] = None,
    ):
        layer = self.layers[layer_idx]
        layer_type = self.layer_types[layer_idx]
        if layer_type in _COT_ATTN_TYPES:
            # CoT layers receive the current outer-loop index (1-indexed) as their
            # cot step and read/write a per-layer V cache.
            cached_v = cot_cache[layer_idx] if cot_cache is not None else None
            h, vs = layer(h, current_cot_step=kv_loop_idx + 1, cached_v=cached_v)
            if cot_cache is not None:
                cot_cache[layer_idx] = vs
            return h
        if layer_type in _LINEAR_ATTN_TYPES:
            return layer(h)
        sliding_window = iteration_windows[layer_idx]
        layer_mask = (
            create_causal_mask(seqlen, attn_impl, sliding_window)
            if mask is None
            else mask
        )
        return layer(
            h,
            freq_cis,
            tok_idx=tok_idx,
            mask=layer_mask,
            attn_impl=attn_impl,
        )

    def _run_layer_blocks(
        self,
        h: torch.Tensor,
        iteration_windows,
        freq_cis,
        tok_idx,
        mask,
        attn_impl: str,
        seqlen: int,
        kv_loop_idx: int,
        layer_end_exclusive: int,
        cot_cache: Optional[List[Optional[List[torch.Tensor]]]] = None,
    ):
        """Apply block-structured stack for layers in ``[0, layer_end_exclusive)``."""
        for block_idx in range(self.n_blocks):
            h_block_input = h
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, len(self.layers))
            loop_end = min(end_idx, layer_end_exclusive)
            if start_idx >= loop_end:
                continue
            for layer_idx in range(start_idx, loop_end):
                h = self._forward_one_layer(
                    layer_idx,
                    h,
                    iteration_windows,
                    freq_cis,
                    tok_idx,
                    mask,
                    attn_impl,
                    seqlen,
                    kv_loop_idx,
                    cot_cache=cot_cache,
                )
            if self.block_residual_weight is not None:
                h = h + self.block_residual_weight[block_idx] * h_block_input
        return h

    def _run_loops(self, h, freq_cis, tok_idx, mask, attn_impl, seqlen, target, dynamic_pattern):
        """Run all loop iterations. Returns averaged loss (training) or final logits (inference)."""

        # CoT state: per-layer V cache, accumulated across outer loop iterations.
        # Only allocated when at least one layer is CoT-style (e.g. ``cot_gdp``).
        cot_cache: Optional[List[Optional[List[torch.Tensor]]]] = (
            [None] * len(self.layers)
            if any(lt in _COT_ATTN_TYPES for lt in self.layer_types)
            else None
        )

        # Legacy mode: weighted sum of hidden representations across iterations
        if self.iteration_weights is not None:
            h_accum = torch.zeros_like(h)
            for loop_idx in range(self.loop_count):
                h_input = h
                iteration_windows = self._iteration_windows_for(loop_idx, dynamic_pattern)
                h = self._run_layer_blocks(
                    h,
                    iteration_windows,
                    freq_cis,
                    tok_idx,
                    mask,
                    attn_impl,
                    seqlen,
                    loop_idx,
                    self._loop_prefix_end,
                    cot_cache=cot_cache,
                )
                if self.residual_weight is not None:
                    h = h + self.residual_weight[loop_idx] * h_input

                h_accum = h_accum + self.iteration_weights[loop_idx] * h

            logits = self.output(self.norm(h_accum))
            return cross_entropy(logits, target) if target is not None else logits

        # Single final-CE mode: run all layers ``loop_count`` times, compute cross
        # entropy once on the final hidden state. Matches the reference CoTFormer
        # training signal (no per-iteration deep supervision).
        for loop_idx in range(self.loop_count):
            h_input = h
            iteration_windows = self._iteration_windows_for(loop_idx, dynamic_pattern)
            h = self._run_layer_blocks(
                h,
                iteration_windows,
                freq_cis,
                tok_idx,
                mask,
                attn_impl,
                seqlen,
                loop_idx,
                self._loop_prefix_end,
                cot_cache=cot_cache,
            )
            if self.residual_weight is not None:
                h = h + self.residual_weight[loop_idx] * h_input

        logits = self.output(self.norm(h))
        return cross_entropy(logits, target) if target is not None else logits

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "flash_attn3",
    ):
        if attn_impl is None:
            attn_impl = self.attn_impl

        bsz, seqlen = token_values.shape
        h = self.tok_embeddings(token_values)
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        dynamic_pattern = self._generate_dynamic_pattern() if self.dynamic_random_pattern is not None else None

        # Training: single pass with uniform loss
        if target is not None:
            return self._run_loops(h, freq_cis, tok_idx, mask, attn_impl, seqlen, target, dynamic_pattern)

        # Inference: ensemble over K random patterns for dynamic random attention
        logits = self._run_loops(h, freq_cis, tok_idx, mask, attn_impl, seqlen, None, dynamic_pattern)
        if self.dynamic_random_pattern is not None and self.inference_ensemble_k > 1:
            for _ in range(self.inference_ensemble_k - 1):
                logits = logits + self._run_loops(
                    h, freq_cis, tok_idx, mask, attn_impl, seqlen, None, self._generate_dynamic_pattern()
                )
            logits = logits / self.inference_ensemble_k
        return logits
    
    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))

        # Re-init loop-specific learnable scalars after meta->to_empty()
        if self.residual_weight is not None:
            nn.init.zeros_(self.residual_weight)
        if self.block_residual_weight is not None:
            nn.init.zeros_(self.block_residual_weight)

        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


# Optional policy for activation checkpointing
def get_no_recompute_ops():
    return None


# Optional FSDP grouping plan for large models
def build_fsdp_grouping_plan(model_args: LoopedWindowTransformerArgs):
    group_plan: Tuple[int, bool] = []
    
    # Grouping embeddings and output separately
    group_plan.append(("tok_embeddings", False))
    
    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", False))
    
    group_plan.append(("output", True))
    
    return group_plan


# Tensor parallelism implementation
def tp_parallelize(model, tp_mesh, model_args: LoopedWindowTransformerArgs, distributed_args):
    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0
    
    # Embedding layer tp
    main_plan = {}
    main_plan["tok_embeddings"] = ColwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    )
    main_plan["norm"] = SequenceParallel()
    main_plan["output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )
    
    parallelize_module(
        model,
        tp_mesh,
        main_plan,
    )
    
    # Attention layers tp
    for layer_idx, layer in enumerate(model.layers):
        layer_type = model.layer_types[layer_idx]
        layer_plan = {}

        if layer_type in _LINEAR_ATTN_TYPES:
            # Linear-attention layers: only parallelize the FFN (attention not sharded here)
            layer_plan["attention_norm"] = SequenceParallel()
            layer_plan["ffn_norm"] = SequenceParallel()
            layer_plan["feed_forward.w1"] = ColwiseParallel()
            layer_plan["feed_forward.w3"] = ColwiseParallel()
            layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))
        else:
            layer_plan["attention"] = PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            )
            layer_plan["attention_norm"] = SequenceParallel()
            layer_plan["attention.wq"] = ColwiseParallel()
            layer_plan["attention.wk"] = ColwiseParallel()
            layer_plan["attention.wv"] = ColwiseParallel()
            layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

            # Feedforward layers tp
            layer_plan["feed_forward"] = PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            )
            layer_plan["ffn_norm"] = SequenceParallel()
            layer_plan["feed_forward.w1"] = ColwiseParallel()
            layer_plan["feed_forward.w3"] = ColwiseParallel()
            layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(layer, tp_mesh, layer_plan)

        if layer_type not in _LINEAR_ATTN_TYPES:
            # Adjusting the number of heads and kv heads according to the tp size
            attn_layer = layer.attention
            attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
            attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
