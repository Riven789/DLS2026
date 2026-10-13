"""autodiff_memory.py -- illustrate activation-memory vs. gradient checkpointing (Module Q30).
Pure stdlib. Simulates a feed-forward net's backward pass and counts how many activation tensors
must be kept alive WITHOUT checkpointing vs. WITH sqrt-spacing checkpointing (the Chen et al. 2016 idea).
NOTE: this is a MEMORY-ACCOUNTING simulation, not a real autograd run.
"""
import math

def activations_saved_naive(n_layers):
    """Reverse-mode AD without checkpointing: every layer's activation is kept for backward."""
    return n_layers

def activations_saved_checkpointed(n_layers):
    """sqrt-checkpointing: store ~sqrt(n) checkpoints; recompute the rest segment-by-segment.
    Peak = checkpoints kept + the longest segment recomputed at once."""
    seg = max(1, round(math.sqrt(n_layers)))
    n_checkpoints = math.ceil(n_layers / seg)
    return n_checkpoints + seg        # stored checkpoints + one segment live during recompute

def recompute_cost(n_layers):
    """Extra forward work from checkpointing: ~one extra forward over the recomputed segments."""
    return n_layers     # roughly one extra full forward pass worth of layer recomputes

if __name__ == "__main__":
    print(f"{'layers':>8} {'naive_act':>10} {'ckpt_act':>9} {'mem_ratio':>10} {'extra_fwd':>10}")
    for n in (16, 64, 256, 1024, 4096):
        naive = activations_saved_naive(n)
        ckpt = activations_saved_checkpointed(n)
        print(f"{n:>8} {naive:>10} {ckpt:>9} {naive/ckpt:>9.1f}x {recompute_cost(n):>10}")
    print("\nLesson: checkpointing cuts peak activation memory from O(n) toward O(sqrt(n)),")
    print("at the cost of ~one extra forward pass -- the classic compute-for-memory trade.")
