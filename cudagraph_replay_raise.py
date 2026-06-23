"""
The case you asked for:
    warmup OK, capture OK, but REPLAY raises — precisely because the CPU-side
    if/else guard does NOT run during replay.

Mechanism (read this first)
---------------------------
A CUDA graph has NO branches at replay. It just re-launches the kernels recorded
at capture. So "replay goes to else" never literally happens. What actually bites
you is this chain:

  1. In EAGER, the forward protects a kernel with a guard:
         if indices_in_range:  gather
         else:                 clamp/rebuild, then gather
     The guard needs a CPU<->GPU sync (.item() / bool reduction) to read the data.
  2. To make the forward CAPTURABLE you must DELETE that guard (a sync is illegal
     inside a graph). The graph therefore records ONLY the bare gather — no
     protection.
  3. warmup + capture run with valid data -> fine.
  4. At REPLAY the static buffer now holds out-of-range data. The recorded gather
     blindly reads out of bounds -> CUDA illegal memory access. The if/else that
     would have saved you in eager simply isn't in the graph.

Run on a machine with an NVIDIA GPU:
    pip install loguru
    CUDA_LAUNCH_BLOCKING=1 python cudagraph_replay_raise.py
(The script also sets CUDA_LAUNCH_BLOCKING itself so the fault localizes to the
 exact replay call.)
"""

import os

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # must be set before CUDA init

import sys

import torch
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | <level>{message}</level>",
)


def rule(title: str) -> None:
    logger.info("─" * 72)
    logger.info(title)
    logger.info("─" * 72)


assert torch.cuda.is_available(), "this repro needs an NVIDIA GPU"
dev = "cuda"

N = 4                 # number of rows we gather (graph shape, fixed)
POOL = 1000           # source pool size; valid indices are 0..POOL-1
BAD_INDEX = 10 ** 7   # wildly out of range -> guaranteed illegal access

pool = torch.arange(POOL, device=dev, dtype=torch.float32)
idx = torch.arange(N, device=dev, dtype=torch.long)   # static index buffer (0..3)
out = torch.empty(N, device=dev)


# =============================================================================
# EAGER — the GUARDED forward. Handles bad indices fine, never crashes.
# =============================================================================
def eager_forward() -> None:
    # This guard reads the data on the CPU -> it is a SYNC. Perfectly fine in
    # eager. (This is the `if/else` people write to stay safe.)
    in_range = bool((idx < pool.numel()).all().item())   # <-- CPU sync
    if in_range:
        logger.debug("  if-branch  | indices in range -> direct gather")
        torch.index_select(pool, 0, idx, out=out)
    else:
        logger.warning("  else-branch| out-of-range indices -> clamp first (eager only)")
        safe = idx.clamp(max=pool.numel() - 1)
        torch.index_select(pool, 0, safe, out=out)


rule("EAGER — guarded forward survives bad indices")
idx.copy_(torch.arange(N, device=dev))
eager_forward()
logger.success(f"eager, good idx: out={out.tolist()}  (no error)")

idx.fill_(BAD_INDEX)
eager_forward()
logger.success(f"eager, BAD idx={BAD_INDEX}: else-branch clamped, out={out.tolist()}  (no error)")


# =============================================================================
# GRAPH — to be capturable, the guard is DROPPED. Only the bare gather is recorded.
# =============================================================================
def graph_forward() -> None:
    # No guard here: the `.item()` sync above cannot be captured, so the
    # capturable version is just the raw gather. Nothing protects it.
    torch.index_select(pool, 0, idx, out=out)


rule("GRAPH — warmup OK, capture OK")
idx.copy_(torch.arange(N, device=dev))   # valid indices for warmup + capture

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        graph_forward()
torch.cuda.current_stream().wait_stream(s)
logger.success("warmup: OK")

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    graph_forward()
logger.success("capture: OK  (bare gather recorded, no bounds guard inside)")

rule("GRAPH — replay #1 with VALID data")
idx.copy_(torch.arange(N, device=dev))
g.replay()
torch.cuda.synchronize()
logger.success(f"replay(good idx): out={out.tolist()}  (no error)")

rule("GRAPH — replay #2 with BAD data -> RAISES (the guard isn't in the graph)")
idx.fill_(BAD_INDEX)
logger.info(f"wrote idx={BAD_INDEX} into the static buffer, then replay() ...")
try:
    g.replay()
    torch.cuda.synchronize()
    logger.error("unexpected: replay did NOT raise")
except RuntimeError as e:
    logger.error(f"replay RAISED: {str(e).strip().splitlines()[0]}")
    logger.info(
        "=> warmup & capture were fine; replay crashed because the recorded gather "
        "ran on out-of-range data with NO if/else guard. In eager the else-branch "
        "clamped it; that guard could not be captured (it needs a CPU sync), so the "
        "graph has no protection. THIS is what 'replay 不走 CPU 的 if/else 导致报错' means."
    )
    logger.warning("CUDA context is now corrupted; nothing else can run. Exiting.")
    sys.exit(1)
