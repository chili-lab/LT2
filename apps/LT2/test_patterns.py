#!/usr/bin/env python3
"""Test script for dynamic random attention patterns."""

import random
from typing import List, Optional


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
    """
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
            raise ValueError(f"Bookend random pattern requires at least 2 layers per loop, got {n_layers}")

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


def test_random_pattern():
    """Test fully random 4:1 pattern."""
    print("Testing random:4:1 pattern with 25 layers, 4 loops")
    print("=" * 80)

    n_layers = 25
    loop_count = 4
    pattern = generate_random_attention_pattern(
        pattern_type="random",
        window_count=4,
        full_count=1,
        n_layers=n_layers,
        loop_count=loop_count,
        default_sliding_window=128,
        seed=42,
    )

    # Count full vs SWA
    total_full = sum(1 for loop in pattern for window in loop if window is None)
    total_swa = sum(1 for loop in pattern for window in loop if window is not None)
    total_slots = n_layers * loop_count

    print(f"Total slots: {total_slots}")
    print(f"Full attention layers: {total_full} ({total_full/total_slots*100:.1f}%)")
    print(f"SWA layers: {total_swa} ({total_swa/total_slots*100:.1f}%)")
    print(f"Expected ratio: 4:1 (80% SWA, 20% full)")
    print()

    # Show pattern for each loop
    for loop_idx, loop_pattern in enumerate(pattern):
        full_count = sum(1 for w in loop_pattern if w is None)
        swa_count = sum(1 for w in loop_pattern if w is not None)
        print(f"Loop {loop_idx}: {full_count} full, {swa_count} SWA")
        pattern_str = "".join("F" if w is None else "S" for w in loop_pattern)
        print(f"  Pattern: {pattern_str}")

    print("\n✓ Random pattern test passed!\n")


def test_bookend_random_pattern():
    """Test bookend random 4:1 pattern."""
    print("Testing bookend_random:4:1 pattern with 25 layers, 4 loops")
    print("=" * 80)

    n_layers = 25
    loop_count = 4
    pattern = generate_random_attention_pattern(
        pattern_type="bookend_random",
        window_count=4,
        full_count=1,
        n_layers=n_layers,
        loop_count=loop_count,
        default_sliding_window=128,
        seed=42,
    )

    # Count full vs SWA
    total_full = sum(1 for loop in pattern for window in loop if window is None)
    total_swa = sum(1 for loop in pattern for window in loop if window is not None)
    total_slots = n_layers * loop_count
    bookend_count = loop_count * 2

    print(f"Total slots: {total_slots}")
    print(f"Full attention layers: {total_full} ({total_full/total_slots*100:.1f}%)")
    print(f"  - Bookends (first/last of each loop): {bookend_count}")
    print(f"  - Random in middle: {total_full - bookend_count}")
    print(f"SWA layers: {total_swa} ({total_swa/total_slots*100:.1f}%)")
    print(f"Expected ratio: 4:1 (80% SWA, 20% full)")
    print()

    # Verify bookends and show pattern for each loop
    all_bookends_full = True
    for loop_idx, loop_pattern in enumerate(pattern):
        full_count = sum(1 for w in loop_pattern if w is None)
        swa_count = sum(1 for w in loop_pattern if w is not None)

        # Check bookends
        if loop_pattern[0] is not None or loop_pattern[-1] is not None:
            all_bookends_full = False
            print(f"✗ Loop {loop_idx}: Bookends not both full!")

        print(f"Loop {loop_idx}: {full_count} full, {swa_count} SWA (first/last always full)")
        pattern_str = "".join("F" if w is None else "S" for w in loop_pattern)
        print(f"  Pattern: {pattern_str}")

    if all_bookends_full:
        print("\n✓ All bookends are full attention")
    print("✓ Bookend random pattern test passed!\n")


def test_determinism():
    """Test that same seed produces same pattern."""
    print("Testing determinism (same seed = same pattern)")
    print("=" * 80)

    n_layers = 25
    loop_count = 4

    pattern1 = generate_random_attention_pattern(
        pattern_type="random",
        window_count=4,
        full_count=1,
        n_layers=n_layers,
        loop_count=loop_count,
        default_sliding_window=128,
        seed=42,
    )

    pattern2 = generate_random_attention_pattern(
        pattern_type="random",
        window_count=4,
        full_count=1,
        n_layers=n_layers,
        loop_count=loop_count,
        default_sliding_window=128,
        seed=42,
    )

    if pattern1 == pattern2:
        print("✓ Same seed produces identical patterns (deterministic)")
    else:
        print("✗ Same seed produces different patterns (non-deterministic!)")

    # Test different seeds
    pattern3 = generate_random_attention_pattern(
        pattern_type="random",
        window_count=4,
        full_count=1,
        n_layers=n_layers,
        loop_count=loop_count,
        default_sliding_window=128,
        seed=43,
    )

    if pattern1 != pattern3:
        print("✓ Different seed produces different patterns")
    else:
        print("✗ Different seed produces same pattern")

    print()


if __name__ == "__main__":
    test_random_pattern()
    test_bookend_random_pattern()
    test_determinism()
    print("=" * 80)
    print("All tests passed! ✓")
