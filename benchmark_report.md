# Benchmark Report: msgpack migration concurrency sweep

## Summary

The throughput drop on `dev/msgpack-migration` is visible in the measured concurrency sweep at higher concurrency. At concurrency 1 and 8, throughput is effectively unchanged. From concurrency 128 upward, the branch is slower on output token throughput, with the largest measured drop at concurrency 1024.

Follow-up with an opt-in whole-protocol pickle IPC fallback (`SGLANG_USE_PICKLE_IPC=1`) on the c1024 case recovered the throughput gap in one measured run: `14681.70 tok/s`, compared with the earlier c1024 confirmation of `14082.96 tok/s` on main and `12594.39 tok/s` on this branch using msgpack. The fallback is not enabled by default.

Primary result, using mean TTFT and output token throughput:

| Concurrency | Prompts | Main TTFT ms | This TTFT ms | TTFT delta % | Main out tok/s | This out tok/s | Throughput delta % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 14.12 | 13.98 | -0.99 | 563.07 | 563.18 | 0.02 |
| 8 | 512 | 36.33 | 35.94 | -1.06 | 4128.86 | 4129.32 | 0.01 |
| 128 | 640 | 348.58 | 472.79 | 35.63 | 16444.02 | 16090.56 | -2.15 |
| 256 | 1280 | 793.13 | 1052.22 | 32.67 | 16168.76 | 15775.08 | -2.43 |
| 512 | 2560 | 1729.37 | 1907.95 | 10.33 | 15441.56 | 14794.28 | -4.19 |
| 1024 | 5120 | 17611.73 | 15707.03 | -10.81 | 14913.57 | 13339.73 | -10.55 |

Positive TTFT delta is worse. Positive throughput delta is better.

## Setup

Artifact directory:

```text
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410
```

Repository:

```text
/data/users/lmzheng/gitrepos/sglang-3
```

Compared revisions:

| Label | Ref | SHA | Loaded package |
|---|---|---|---|
| main | `main` | `4e1d25117bf3134a3a43d55f461ea8708853f0b3` | `0.5.13.post2.dev578+g4e1d25117` |
| this | `dev/msgpack-migration` | `24841a30f25f821419dbeeac9c618276d7556295` | `0.5.13.post2.dev706+g24841a30f` |

Model and server:

```text
model=Qwen/Qwen3-1.7B
gpu=0
host=127.0.0.1
port=30000
```

Workload from `case.md`:

```text
concurrency: 1, 8, 128, 256, 512, 1024
num_prompts: max(5 * concurrency, 512)
random_input_len: 1024
random_output_len: 1024
```

Important fixed-length detail:

```text
--random-range-ratio 1
```

This is required for fixed 1024-token random output lengths in `bench_serving.py`. Without it, the random workload uses a length range and the effective output length is not fixed at 1024.

Methodology:

- One discard benchmark run, then one measured benchmark run, for each branch and concurrency.
- Execution order was high to low concurrency: `1024, 512, 256, 128, 8, 1`.
- Results are reported in the `case.md` order: `1, 8, 128, 256, 512, 1024`.
- The client/load generator stayed on the main worktree for both branches, so only the server code changed.
- Each branch used a separate server worktree and a separate `uv` environment.
- The server was launched once per branch and reused across that branch's concurrency sweep.

## Worktree and environment isolation

The run used detached worktrees rather than switching the active repo checkout in place:

```bash
ART=/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410

git worktree add --detach "$ART/worktrees/main" 4e1d25117bf3134a3a43d55f461ea8708853f0b3
git worktree add --detach "$ART/worktrees/this" 24841a30f25f821419dbeeac9c618276d7556295
```

Runtime identity was verified after server startup:

```text
main server cwd:  /data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/worktrees/main
main import:      /data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/worktrees/main/python/sglang/__init__.py
main version:     0.5.13.post2.dev578+g4e1d25117

this server cwd:  /data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/worktrees/this
this import:      /data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/worktrees/this/python/sglang/__init__.py
this version:     0.5.13.post2.dev706+g24841a30f
```

The server environments were separate:

```text
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/uv-envs/server-main
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/uv-envs/server-this
```

The client environment was fixed to main:

```text
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/uv-envs/client-main
```

## Commands

### Server command

The server command template was:

```bash
cd "$SERVER_WORKTREE"

export CUDA_VISIBLE_DEVICES=0
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export LD_PRELOAD=/usr/lib64/libnuma.so.1
export CUDA_HOME=/usr/local/cuda-13.0
export UV_PROJECT_ENVIRONMENT="$ART/uv-envs/server-$branch"

with-proxy uv run --project "$SERVER_WORKTREE/python" \
  sglang serve \
  --model Qwen/Qwen3-1.7B \
  --enable-metrics \
  --enable-request-time-stats-logging \
  --host 127.0.0.1 \
  --port 30000
```

For `main`, `SERVER_WORKTREE` was:

```text
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/worktrees/main
```

For `this`, `SERVER_WORKTREE` was:

```text
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/worktrees/this
```

### Benchmark command

For each concurrency `c`, prompt count was:

```bash
n=$((5 * c))
if [ "$n" -lt 512 ]; then
  n=512
fi
```

Each concurrency ran once as a discard:

```bash
--output-file /dev/null
```

Then once as the measured run:

```bash
--output-file "$ART/results/${branch}_c${c}_measured.jsonl"
```

The benchmark command template was:

```bash
cd "$ART/worktrees/main"

export UV_PROJECT_ENVIRONMENT="$ART/uv-envs/client-main"

with-proxy uv run --project "$ART/worktrees/main/python" \
  python "$ART/worktrees/main/python/sglang/bench_serving.py" \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model Qwen/Qwen3-1.7B \
  --dataset-name random \
  --request-rate inf \
  --random-range-ratio 1 \
  --temperature 0 \
  --flush-cache \
  --warmup-requests 16 \
  --output-details \
  --disable-tqdm \
  --seed 1 \
  --output-file "$output_file" \
  --num-prompts "$n" \
  --max-concurrency "$c" \
  --random-input-len 1024 \
  --random-output-len 1024
```

## Validation

All 12 measured JSONL files passed validation:

```text
type,branch,concurrency,status
bench,main,1024,ok
bench,main,512,ok
bench,main,256,ok
bench,main,128,ok
bench,main,8,ok
bench,main,1,ok
bench,this,1024,ok
bench,this,512,ok
bench,this,256,ok
bench,this,128,ok
bench,this,8,ok
bench,this,1,ok
```

Validation checks:

- Exactly one JSONL row per measured run.
- `completed == max(5 * concurrency, 512)`.
- No non-empty request errors.
- Every measured output length is exactly 1024.
- `random_input_len == 1024`.
- `random_output_len == 1024`.
- `random_range_ratio == 1.0`.
- `total_output_tokens == num_prompts * 1024`.
- Port `30000` was clear after shutdown.

## Detailed results

### Mean TTFT and throughput

| Concurrency | Prompts | Main TTFT ms | This TTFT ms | TTFT delta ms | TTFT delta % | Main out tok/s | This out tok/s | Throughput delta tok/s | Throughput delta % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 14.12 | 13.98 | -0.14 | -0.99 | 563.07 | 563.18 | 0.12 | 0.02 |
| 8 | 512 | 36.33 | 35.94 | -0.39 | -1.06 | 4128.86 | 4129.32 | 0.45 | 0.01 |
| 128 | 640 | 348.58 | 472.79 | 124.21 | 35.63 | 16444.02 | 16090.56 | -353.46 | -2.15 |
| 256 | 1280 | 793.13 | 1052.22 | 259.09 | 32.67 | 16168.76 | 15775.08 | -393.68 | -2.43 |
| 512 | 2560 | 1729.37 | 1907.95 | 178.57 | 10.33 | 15441.56 | 14794.28 | -647.28 | -4.19 |
| 1024 | 5120 | 17611.73 | 15707.03 | -1904.70 | -10.81 | 14913.57 | 13339.73 | -1573.84 | -10.55 |

### TTFT percentiles

| Concurrency | Main median TTFT ms | This median TTFT ms | Main p95 TTFT ms | This p95 TTFT ms |
|---:|---:|---:|---:|---:|
| 1 | 14.05 | 13.93 | 14.77 | 14.63 |
| 8 | 37.98 | 38.04 | 44.08 | 42.59 |
| 128 | 319.29 | 347.69 | 481.40 | 1073.80 |
| 256 | 677.46 | 1215.79 | 1279.81 | 1446.21 |
| 512 | 1718.28 | 1908.40 | 2475.45 | 2644.37 |
| 1024 | 10021.04 | 10049.23 | 55103.05 | 60945.09 |

### Run durations

| Concurrency | Main duration s | This duration s |
|---:|---:|---:|
| 1 | 931.13 | 930.94 |
| 8 | 126.98 | 126.97 |
| 128 | 39.85 | 40.73 |
| 256 | 81.06 | 83.09 |
| 512 | 169.77 | 177.19 |
| 1024 | 351.55 | 393.03 |

## Interpretation

The measured throughput regression is concentrated at higher concurrency:

- Concurrency 1 and 8 are effectively unchanged.
- Concurrency 128 and 256 show a small but consistent output throughput drop, about 2.1% to 2.4%.
- Concurrency 512 shows a larger output throughput drop, about 4.2%.
- Concurrency 1024 shows the largest output throughput drop, about 10.6%.

Mean TTFT is mixed because the high-concurrency TTFT distribution is broad. At concurrency 1024, this branch has lower mean TTFT but worse p95 TTFT and lower throughput. For the CPU/IPC question, the throughput table is the cleaner signal: at high concurrency the branch processes fewer output tokens per second under the same workload and client.

## Artifacts

Summary and raw metrics:

```text
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/concurrency_summary.md
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/concurrency_summary.csv
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/concurrency_raw_metrics.csv
```

Full logs:

```text
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/logs
```

Measured JSONL outputs:

```text
/data/users/lmzheng/benchmarks/ttft-msgpack-concurrency-sweep-fixedlen-20260622-142410/results
```
