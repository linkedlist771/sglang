"""
Minimal reproducer: a Python `if/else` and CUDA graph capture/replay.

The point being demonstrated
----------------------------
Take a forward with a degenerate-case branch:

    if num_tokens > 0:
        <normal, graph-safe path>
    else:
        <special handling for the empty / per-rank-0-token case>

* EAGER mode: BOTH branches are reachable and BOTH run fine. The else-branch can
  freely do data-dependent / synchronizing work (.item(), .cpu(), boolean-mask
  indexing, a non-contiguous .view(), a resize, ...). Eager never complains.

* CUDA GRAPH mode:
    - capture traces the Python ONCE; only the branch taken at capture time is
      recorded. `if num_tokens > 0` is baked -> the else-branch becomes
      unreachable via replay (Part B).
    - if you DO force the else-branch into the captured region, the very ops that
      were harmless in eager now RAISE, because data-dependent / synchronizing
      ops are illegal inside a graph (Part C). This is the "走 else 就会 raise,
      但 eager 不会出问题" case.
    - the correct fix: express the degenerate case as DATA in a static buffer
      (a per-rank token count) and mask on it -- one graph handles count=0..N
      with no branch at all (Part D).

Run on a machine with an NVIDIA GPU:
    pip install loguru          # if not already installed
    python cudagraph_branch_repro.py
"""

import sys

import torch
from loguru import logger

# ---- loguru: compact, colorized, level-tagged output --------------------------
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format=(
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level:<8}</level> | "
        "<level>{message}</level>"
    ),
)


def rule(title: str) -> None:
    logger.info("─" * 72)
    logger.info(title)
    logger.info("─" * 72)


assert torch.cuda.is_available(), "this repro needs an NVIDIA GPU"
dev = "cuda"

N = 4       # captured number of tokens; graph shapes are FIXED at this value
FEAT = 6    # feature width


# =============================================================================
# The forward under study. `num_tokens` is a plain Python int from the shape.
# =============================================================================
def forward(x: torch.Tensor, out: torch.Tensor) -> None:
    num_tokens = x.shape[0]
    if num_tokens > 0:
        logger.debug(f"  if-branch   | num_tokens={num_tokens} | graph-safe: out = x*2")
        out.copy_(x * 2.0)
    else:
        # "Special handling" for the degenerate / 0-token case. The op below is a
        # CPU<->GPU SYNC (.item()) standing in for ANY data-dependent work an
        # else-branch typically wants: counting valid rows, dynamic resize, a
        # non-contiguous .view(), boolean-mask indexing, etc. All fine in eager,
        # all ILLEGAL to capture into a CUDA graph.
        logger.warning(
            f"  else-branch | num_tokens={num_tokens} | data-dependent .item() sync"
        )
        k = int((x.abs().sum(dim=1) > 0).sum().item())   # <-- the offending sync
        out.zero_()
        if k:
            out[:k].copy_(x[:k] + 100.0)


# =============================================================================
# PART A — EAGER: both branches run, both are fine
# =============================================================================
rule("PART A | EAGER mode — both branches work")

x_full = torch.ones(N, FEAT, device=dev)
o_full = torch.empty(N, FEAT, device=dev)
forward(x_full, o_full)
logger.success(f"eager, num_tokens={N}: out[0]={o_full[0].tolist()}  (no error)")

x_empty = torch.ones(0, FEAT, device=dev)   # 0-token rank
o_empty = torch.empty(0, FEAT, device=dev)
forward(x_empty, o_empty)
logger.success("eager, num_tokens=0: else-branch ran, out.shape="
               f"{tuple(o_empty.shape)}  (no error)")


# =============================================================================
# PART B — GRAPH: the if/else is baked at capture; replay never re-runs Python
# =============================================================================
rule("PART B | CUDA GRAPH — branch is frozen at capture")

static_in = torch.ones(N, FEAT, device=dev)
static_out = torch.empty(N, FEAT, device=dev)

# warmup on a side stream (required before capture)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        forward(static_in, static_out)
torch.cuda.current_stream().wait_stream(s)

logger.info("capturing with num_tokens=N -> the if-branch is recorded ...")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    forward(static_in, static_out)
logger.success("capture done")

logger.info("replaying twice with DIFFERENT data; watch for any branch logs ...")
for val in (1.0, 5.0):
    static_in.fill_(val)
    g.replay()
    torch.cuda.synchronize()
    logger.success(
        f"replay(input={val}): out[0]={static_out[0].tolist()}  "
        f"(always if-branch x*2; no 'if/else' log appeared -> Python did not run)"
    )


# =============================================================================
# PART C — GRAPH: forcing the else-branch INTO the graph -> it RAISES
# =============================================================================
rule("PART C | CUDA GRAPH — capturing the else-branch raises")

empty_in = torch.ones(0, FEAT, device=dev)     # capture with num_tokens=0
empty_out = torch.empty(0, FEAT, device=dev)

# warmup (eager) — note this same else-branch is totally fine here
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        forward(empty_in, empty_out)           # eager: OK
torch.cuda.current_stream().wait_stream(s)
logger.success("eager warmup of the else-branch: OK (no error)")

logger.info("now capturing the SAME else-branch into a graph ...")
g_bad = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g_bad):
        forward(empty_in, empty_out)           # else-branch -> .item() sync
    logger.error("unexpected: capture did NOT raise")
except RuntimeError as e:
    first_line = str(e).strip().splitlines()[0]
    logger.error(f"capture RAISED (as expected): {first_line}")
    logger.info(
        "=> the data-dependent op that was harmless in eager is illegal in a graph. "
        "This is exactly why an `if num_tokens == 0` special path must not be the "
        "thing a graph depends on."
    )


# =============================================================================
# PART D — the CORRECT pattern: express '0 tokens' as DATA, no branch
# =============================================================================
rule("PART D | correct fix — one graph, count buffer drives the degenerate case")

count = torch.zeros(1, dtype=torch.int32, device=dev)   # live valid-row count
row_idx = torch.arange(N, device=dev).unsqueeze(1)      # [N,1] static helper
masked_in = torch.ones(N, FEAT, device=dev)
masked_out = torch.empty(N, FEAT, device=dev)


def forward_masked() -> None:
    valid = (row_idx < count).to(masked_in.dtype)       # data-driven [N,1] mask
    masked_out.copy_(masked_in * 2.0 * valid)           # padding rows -> 0


s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        forward_masked()
torch.cuda.current_stream().wait_stream(s)

count.fill_(N)
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    forward_masked()
logger.success("captured a single graph (no if/else inside)")

for c in (N, 0, 2):
    count.fill_(c)
    g2.replay()
    torch.cuda.synchronize()
    kept = int((masked_out.sum(dim=1) != 0).sum().item())
    logger.success(f"replay(count={c}): rows kept={kept}, out.sum={masked_out.sum().item():.1f}")

logger.info(
    "count=0 reproduces the '0 tokens on this rank' behaviour correctly, through "
    "ONE captured graph, with NO `if num_tokens == 0`. Drive the degenerate case "
    "with a count buffer + mask, never a Python branch the graph cannot see."
)
