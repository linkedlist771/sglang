"""
Microbenchmark for SGLang manager IPC payload serialization.

This intentionally avoids starting an SGLang server. It compares the default
msgpack transport against the whole-protocol pickle fallback used by
SGLANG_USE_PICKLE_IPC.
"""

from __future__ import annotations

import argparse
import pickle
import time
import uuid
from array import array
from collections import OrderedDict
from typing import Any, Callable
from unittest.mock import patch

import zmq

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchStrOutput,
    BatchTokenIDOutput,
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
    sock_recv,
    sock_send,
    wrap_as_pickle,
)
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402


def _time_us(fn: Callable[[], Any], iters: int) -> float:
    for _ in range(min(20, iters)):
        fn()

    start = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    return (time.perf_counter_ns() - start) / iters / 1_000.0


def _encode(payload: Any, mode: str) -> bytes:
    if mode == "msgpack":
        return msgpack_encode(payload)
    if mode == "pickle":
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    raise ValueError(f"Unknown mode: {mode}")


def _decode(data: bytes, mode: str) -> Any:
    if mode == "msgpack":
        return msgpack_decode(data)
    if mode == "pickle":
        return pickle.loads(data)
    raise ValueError(f"Unknown mode: {mode}")


def _bench_zmq_roundtrip(payload: Any, mode: str, iters: int) -> float:
    ctx = zmq.Context.instance()
    addr = f"inproc://bench-io-struct-ipc-{uuid.uuid4()}"
    receiver = ctx.socket(zmq.PAIR)
    sender = ctx.socket(zmq.PAIR)
    receiver.linger = 0
    sender.linger = 0
    receiver.bind(addr)
    sender.connect(addr)

    def once():
        sock_send(sender, payload)
        return sock_recv(receiver)

    try:
        with patch("sglang.srt.managers.io_struct._USE_PICKLE_IPC", mode == "pickle"):
            return _time_us(once, iters)
    finally:
        sender.close(0)
        receiver.close(0)


def _api_time_stats() -> APIServerReqTimeStats:
    stats = APIServerReqTimeStats()
    stats.enable_metrics = True
    stats.created_time = time.time()
    stats.api_server_dispatch_time = time.perf_counter()
    return stats


def _scheduler_time_stats() -> SchedulerReqTimeStats:
    stats = SchedulerReqTimeStats()
    stats.enable_metrics = True
    now = time.perf_counter()
    stats.wait_queue_entry_time = now
    stats.forward_entry_time = now + 0.001
    stats.prefill_finished_time = now + 0.002
    return stats


def _make_tokenized_req(
    idx: int, input_len: int, output_len: int, include_time_stats: bool
) -> TokenizedGenerateReqInput:
    return TokenizedGenerateReqInput(
        rid=f"rid-{idx}",
        input_text=None,
        input_ids=array("l", range(input_len)),
        mm_inputs=None,
        sampling_params=SamplingParams(
            max_new_tokens=output_len,
            temperature=0.0,
            stop_token_ids=[1, 2],
        ),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
        time_stats=wrap_as_pickle(_api_time_stats()) if include_time_stats else None,
    )


def _output_arrays(batch_size: int, tokens_per_output: int) -> list[array]:
    return [array("l", range(tokens_per_output)) for _ in range(batch_size)]


def _make_batch_token_id_output(
    batch_size: int, tokens_per_output: int, include_time_stats: bool
) -> BatchTokenIDOutput:
    return BatchTokenIDOutput(
        rids=[f"rid-{i}" for i in range(batch_size)],
        http_worker_ipcs=[None] * batch_size,
        finished_reasons=[None] * batch_size,
        decoded_texts=[""] * batch_size,
        decode_ids=_output_arrays(batch_size, tokens_per_output),
        read_offsets=[0] * batch_size,
        output_ids=_output_arrays(batch_size, tokens_per_output),
        skip_special_tokens=[True] * batch_size,
        spaces_between_special_tokens=[True] * batch_size,
        no_stop_trim=[False] * batch_size,
        prompt_tokens=[1024] * batch_size,
        reasoning_tokens=[0] * batch_size,
        completion_tokens=[tokens_per_output] * batch_size,
        cached_tokens=[0] * batch_size,
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        output_token_logprobs_val=None,
        output_token_logprobs_idx=None,
        input_top_logprobs_val=None,
        input_top_logprobs_idx=None,
        output_top_logprobs_val=None,
        output_top_logprobs_idx=None,
        input_token_ids_logprobs_val=None,
        input_token_ids_logprobs_idx=None,
        output_token_ids_logprobs_val=None,
        output_token_ids_logprobs_idx=None,
        output_token_entropy_val=None,
        output_hidden_states=None,
        routed_experts=None,
        indexer_topk=None,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
        retraction_counts=[0] * batch_size,
        cached_tokens_details=None,
        dp_ranks=[0] * batch_size,
        time_stats=(
            wrap_as_pickle(
                [_scheduler_time_stats() for _ in range(batch_size)]
            )
            if include_time_stats
            else None
        ),
        spec_verify_ct=[],
        spec_num_correct_drafts=[],
        spec_correct_drafts_histogram=[],
    )


def _make_batch_str_output(
    batch_size: int, tokens_per_output: int, include_time_stats: bool
) -> BatchStrOutput:
    return BatchStrOutput(
        rids=[f"rid-{i}" for i in range(batch_size)],
        http_worker_ipcs=[None] * batch_size,
        finished_reasons=[None] * batch_size,
        output_strs=["x"] * batch_size,
        output_ids=_output_arrays(batch_size, tokens_per_output),
        prompt_tokens=[1024] * batch_size,
        completion_tokens=[tokens_per_output] * batch_size,
        reasoning_tokens=[0] * batch_size,
        cached_tokens=[0] * batch_size,
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        output_token_logprobs_val=None,
        output_token_logprobs_idx=None,
        input_top_logprobs_val=None,
        input_top_logprobs_idx=None,
        output_top_logprobs_val=None,
        output_top_logprobs_idx=None,
        input_token_ids_logprobs_val=None,
        input_token_ids_logprobs_idx=None,
        output_token_ids_logprobs_val=None,
        output_token_ids_logprobs_idx=None,
        output_token_entropy_val=None,
        output_hidden_states=None,
        routed_experts=None,
        indexer_topk=None,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
        retraction_counts=[0] * batch_size,
        cached_tokens_details=None,
        dp_ranks=[0] * batch_size,
        time_stats=(
            wrap_as_pickle(
                [_scheduler_time_stats() for _ in range(batch_size)]
            )
            if include_time_stats
            else None
        ),
        spec_verify_ct=[],
        spec_num_correct_drafts=[],
        spec_correct_drafts_histogram=[],
    )


def make_payloads(args: argparse.Namespace) -> OrderedDict[str, Any]:
    tokenized_reqs = [
        _make_tokenized_req(
            i, args.input_len, args.output_len, args.include_time_stats
        )
        for i in range(args.batch_size)
    ]
    return OrderedDict(
        [
            ("tokenized_generate", tokenized_reqs[0]),
            (
                "batch_tokenized_generate",
                BatchTokenizedGenerateReqInput(batch=tokenized_reqs),
            ),
            (
                "batch_token_id_output",
                _make_batch_token_id_output(
                    args.batch_size,
                    args.tokens_per_output,
                    args.include_time_stats,
                ),
            ),
            (
                "batch_str_output",
                _make_batch_str_output(
                    args.batch_size,
                    args.tokens_per_output,
                    args.include_time_stats,
                ),
            ),
        ]
    )


def run(args: argparse.Namespace) -> None:
    payloads = make_payloads(args)
    print(
        "mode,payload,encoded_bytes,encode_us,decode_us,zmq_roundtrip_us",
        flush=True,
    )
    for name, payload in payloads.items():
        for mode in ("msgpack", "pickle"):
            data = _encode(payload, mode)
            encode_us = _time_us(lambda: _encode(payload, mode), args.iters)
            decode_us = _time_us(lambda: _decode(data, mode), args.iters)
            zmq_us = _bench_zmq_roundtrip(payload, mode, args.iters)
            print(
                f"{mode},{name},{len(data)},"
                f"{encode_us:.2f},{decode_us:.2f},{zmq_us:.2f}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--tokens-per-output", type=int, default=1)
    parser.add_argument("--include-time-stats", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
