"""
Standalone multi-process (TP) CUDA-graph capture/replay repro for a custom decode op.

WHY THIS EXISTS
---------------
A single-graph unit test (capture one bs, replay the same inputs) almost always
PASSES even when the op is broken inside SGLang, because it does not reproduce the
three conditions that actually break replay in the framework:

  (1) SHARED POOL: SGLang captures dozens of batch sizes into ONE shared
      graph memory pool (`global_graph_memory_pool`, cuda_graph_runner.py:1190).
      Scratch allocated *inside* your op during capture of bs=8 can be
      overwritten by the capture of bs=16/24/... -> replay reads garbage -> IMA.

  (2) PADDING: a real batch is padded UP to the nearest captured bs. The padded
      rows [raw_bs:bs] carry seq_lens=fill_value and out_cache_loc=0
      (populate_from_forward_batch, cuda_graph_runner.py:296-298). If your kernel
      dereferences those padded rows, it can walk off the KV cache.

  (3) REPLAY != CAPTURE INPUTS: capture uses benign dummy values (all in-bounds);
      replay feeds real seq_lens / indices that point far into the KV cache.

This harness reproduces all three under real NCCL TP. Plug your op into `user_op`.

RUN
---
    python test/srt/cuda_graph_replay_repro.py --tp 2

    # localize the faulting kernel + address (recommended):
    compute-sanitizer --tool memcheck --destroy-on-device-error kernel \
        python test/srt/cuda_graph_replay_repro.py --tp 2

    # A/B: prove it is graph-specific (this path never uses a graph):
    python test/srt/cuda_graph_replay_repro.py --tp 2 --eager

    # isolate suspect (2)/(3): single captured bs, replay at the exact bs:
    python test/srt/cuda_graph_replay_repro.py --tp 2 --single-bs 8 --no-pad
"""

import argparse
import bisect
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# --------------------------------------------------------------------------- #
# Mirror SGLang's default decode capture sizes (server_args._generate_cuda_graph_batch_sizes)
# --------------------------------------------------------------------------- #
def make_capture_bs(max_bs: int):
    bs = [1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 512, 16))
    bs = sorted({b for b in bs if b <= max_bs} | {max_bs})
    return bs


# =========================================================================== #
#                         >>> PLUG YOUR OP HERE <<<                           #
# =========================================================================== #
#
# `model_step` is what gets captured and replayed. It must:
#   - read ONLY from the pre-allocated static buffers (sliced to the current bs)
#   - write its result into the pre-allocated `buf.hidden_out` (no fresh output
#     tensor each call, or its address won't survive replay)
#   - allocate NO scratch with torch.empty/zeros/cat/arange on the forward path
#     (pre-allocate everything once in `build_static_buffers` instead)
#
# The placeholder below does a representative decode pattern:
#   gather KV by out_cache_loc -> matmul -> TP all-reduce -> write hidden_out.
# Replace the marked block with your real custom op.
# --------------------------------------------------------------------------- #
def model_step(buf, bs: int, num_tokens: int, tp_group, weight):
    out_cache_loc = buf.out_cache_loc[:num_tokens]          # [num_tokens] int
    seq_lens = buf.seq_lens[:bs]                            # [bs] int
    hidden_out = buf.hidden_out[:num_tokens]               # [num_tokens, H]

    # ---- BEGIN replace with your custom op -------------------------------- #
    # Representative: index the KV cache by out_cache_loc (this is exactly the
    # kind of indexing that walks off the end on padded rows / stale indices).
    gathered = buf.kv_cache.index_select(0, out_cache_loc.long())  # [num_tokens, H]
    x = gathered @ weight                                          # [num_tokens, H]
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.all_reduce(x, group=tp_group)                         # TP reduce
    hidden_out.copy_(x)
    # ---- END replace ------------------------------------------------------ #
    return hidden_out


# --------------------------------------------------------------------------- #
# Static buffers (mirror DecodeInputBuffers): pre-allocate ONCE, reuse forever.
# --------------------------------------------------------------------------- #
class Buffers:
    pass


def build_static_buffers(device, max_bs, hidden, kv_len, seq_len_fill_value):
    b = Buffers()
    b.max_num_token = max_bs  # decode: num_tokens_per_bs == 1
    b.seq_len_fill_value = seq_len_fill_value
    b.kv_capacity = kv_len

    b.input_ids = torch.zeros((max_bs,), dtype=torch.int64, device=device)
    b.positions = torch.zeros((max_bs,), dtype=torch.int64, device=device)
    b.req_pool_indices = torch.zeros((max_bs,), dtype=torch.int64, device=device)
    b.seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int32, device=device)
    b.out_cache_loc = torch.zeros((max_bs,), dtype=torch.int64, device=device)
    b.num_token_non_padded = torch.zeros((1,), dtype=torch.int32, device=device)

    # Fake KV cache + persistent output buffer. A correct op stays within
    # [0, kv_capacity); a buggy one (reading padded/stale indices) walks past it.
    b.kv_cache = torch.randn((kv_len, hidden), dtype=torch.float32, device=device)
    b.hidden_out = torch.zeros((max_bs, hidden), dtype=torch.float32, device=device)
    return b


def fill_capture_dummies(buf, bs, num_tokens):
    """Capture-time inputs: benign, all in-bounds (mirrors capture_one_batch_size)."""
    buf.seq_lens.fill_(buf.seq_len_fill_value)
    buf.req_pool_indices.zero_()
    buf.out_cache_loc.zero_()                 # index 0 -> always valid
    buf.input_ids[:num_tokens].zero_()
    buf.positions[:num_tokens].zero_()
    buf.num_token_non_padded[...] = num_tokens


def populate_for_replay(buf, raw_bs, captured_bs, kv_capacity, rng):
    """
    Mirror populate_from_forward_batch (cuda_graph_runner.py:282).
    Real data into [:raw_bs]; padded rows [raw_bs:captured_bs] keep fill/zero.
    """
    num_tokens = captured_bs
    raw_num_token = raw_bs
    if captured_bs != raw_bs:
        buf.seq_lens.fill_(buf.seq_len_fill_value)   # padded rows -> fill_value
        buf.out_cache_loc.zero_()                    # padded rows -> 0 (sentinel)

    # Real per-request state: indices spread across the KV cache (this is what
    # capture's all-zero dummy never exercised).
    real_seq = torch.randint(1, kv_capacity, (raw_bs,), device=buf.seq_lens.device, dtype=torch.int32, generator=rng)
    real_loc = torch.randint(0, kv_capacity, (raw_bs,), device=buf.out_cache_loc.device, dtype=torch.int64, generator=rng)
    buf.seq_lens[:raw_bs].copy_(real_seq)
    buf.out_cache_loc[:raw_num_token].copy_(real_loc)
    buf.req_pool_indices[:raw_bs].copy_(
        torch.arange(raw_bs, device=buf.req_pool_indices.device, dtype=torch.int64)
    )
    buf.num_token_non_padded[...] = raw_num_token
    return num_tokens


def worker(rank, world_size, args):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    tp_group = dist.group.WORLD

    def log(msg):
        print(f"[rank{rank}] {msg}", flush=True)

    torch.manual_seed(1234 + rank)
    max_bs = args.single_bs or args.max_bs
    buf = build_static_buffers(device, max_bs, args.hidden, args.kv_len, args.seq_len_fill)
    weight = torch.randn((args.hidden, args.hidden), dtype=torch.float32, device=device)

    capture_bs = [args.single_bs] if args.single_bs else make_capture_bs(max_bs)
    log(f"capture_bs = {capture_bs}")

    if args.eager:
        # A/B path: no graph at all. If this passes but graph mode fails,
        # the bug is graph-specific (suspect 1/2/3), not your kernel's math.
        rng = torch.Generator(device=device); rng.manual_seed(7 + rank)
        for step in range(args.steps):
            raw_bs = capture_bs[step % len(capture_bs)]
            populate_for_replay(buf, raw_bs, raw_bs, args.kv_len, rng)
            model_step(buf, raw_bs, raw_bs, tp_group, weight)
        torch.cuda.synchronize()
        log("EAGER ok")
        dist.destroy_process_group()
        return

    # ----- Warmup (SGLang warms up on a side stream before capture) -------- #
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fill_capture_dummies(buf, max_bs, max_bs)
            model_step(buf, max_bs, max_bs, tp_group, weight)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    log("warmup ok")

    # ----- Capture every bs into ONE shared pool (suspect #1) -------------- #
    shared_pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for bs in reversed(capture_bs):           # SGLang captures largest-first
        num_tokens = bs
        fill_capture_dummies(buf, bs, num_tokens)
        # re-warm this exact shape on the side stream so capture is clean
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                model_step(buf, bs, num_tokens, tp_group, weight)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=shared_pool):
            model_step(buf, bs, num_tokens, tp_group, weight)
        graphs[bs] = g
    torch.cuda.synchronize()
    log(f"captured {len(graphs)} graphs into shared pool")

    # ----- Replay with REAL, padded inputs (suspects #2/#3) ---------------- #
    rng = torch.Generator(device=device); rng.manual_seed(7 + rank)
    for step in range(args.steps):
        raw_bs = max(1, int(torch.randint(1, max_bs + 1, (1,), generator=rng).item()))
        # all TP ranks must replay the SAME captured graph -> agree on bs
        agree = torch.tensor([raw_bs], device=device)
        dist.all_reduce(agree, op=dist.ReduceOp.MAX, group=tp_group)
        raw_bs = int(agree.item())

        idx = bisect.bisect_left(capture_bs, raw_bs)
        captured_bs = capture_bs[min(idx, len(capture_bs) - 1)]
        if args.no_pad:
            captured_bs, raw_bs = raw_bs, raw_bs  # exact match, no padding

        num_tokens = populate_for_replay(buf, raw_bs, captured_bs, args.kv_len, rng)
        graphs[captured_bs].replay()
        torch.cuda.synchronize()              # surface IMA at the replay site
        out = buf.hidden_out[:raw_bs]
        assert torch.isfinite(out).all(), f"non-finite output at step {step}, raw_bs={raw_bs}, captured_bs={captured_bs}"
        if step % 16 == 0:
            log(f"replay step {step}: raw_bs={raw_bs} -> graph bs={captured_bs} ok")

    log("ALL REPLAYS OK")
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=2, help="number of GPUs / processes")
    p.add_argument("--max-bs", type=int, default=64)
    p.add_argument("--single-bs", type=int, default=0, help="capture only this one bs")
    p.add_argument("--no-pad", action="store_true", help="replay at exact captured bs")
    p.add_argument("--eager", action="store_true", help="A/B: skip cuda graph entirely")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--kv-len", type=int, default=4096, help="fake KV cache rows")
    p.add_argument("--seq-len-fill", type=int, default=1)
    p.add_argument("--steps", type=int, default=128)
    args = p.parse_args()

    assert torch.cuda.device_count() >= args.tp, "not enough GPUs for requested --tp"
    mp.spawn(worker, args=(args.tp, args), nprocs=args.tp, join=True)


if __name__ == "__main__":
    main()
