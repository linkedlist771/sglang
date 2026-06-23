# 把 mctlassEx 融合 MoE 接进 SGLang DeepseekV2(支持 CUDA Graph)

> 目标:把你那套基于对称内存的融合算子(`mctlassEx.SymmBuffer` + `MegaMaskedGroupedGEMM`)
> 接进 `models/deepseek_v2.py` 的 MoE 路径,作为一个新的 a2a 后端 `mctlass`,并且与
> CUDA Graph 兼容。
>
> 参照对象:SGLang 已有的 `megamoe` 后端(`deep_gemm` 版),代码在
> `python/sglang/srt/layers/moe/mega_moe.py`。你要做的就是一个**同构副本**,把
> `deep_gemm` 换成 `mctlassEx`。本文给出所有 hook 点、CUDA Graph 不变量、最小骨架。

---

## 0. 核心结论(先看这个)

1. **没有独立的 Dispatcher 对象**。融合 MoE 的"分发"被拆成两段:
   - `pre_dispatch`:**纯本地** kernel,把本 rank 的 token 量化打包进 `SymmBuffer`,并把
     padding 槽位填无效值。**零通信**。
   - 融合大 kernel(`MegaMaskedGroupedGEMM`):在 kernel **内部**用对称内存做 all-to-all +
     G1 + swiglu + G2 + combine。

2. **CUDA Graph 兼容不靠 graph-runner adapter**。`megamoe` 在 `cuda_graph_runner.py` 里
   没有注册任何回调(对比 `deepep` 有 `DeepEPCudaGraphRunnerAdapter`)。它靠 3 条不变量
   做到 graph-safe(见 §2),你照搬即可。

3. **接线只动 5 个文件**(见 §4),核心逻辑全在你新建的 `mctlass_moe.py` 里。

---

## 1. SGLang 里融合 MoE 现有的 5 个 hook 点

下面是 `megamoe` 接进 DeepseekV2 的全部接触点。你的 `mctlass` 后端要在**同样的位置**
插入对应逻辑。

| # | 位置 | 作用 |
|---|---|---|
| 1 | `layers/moe/utils.py:23` `MoeA2ABackend` 枚举 + `is_megamoe()` | 注册后端名 |
| 2 | `server_args.py:212,621` 的 `"megamoe"` 字面量 + §3197 的自动配置 | CLI/env 选择后端 |
| 3 | `layers/quantization/fp8.py:1196` `process_weights_after_loading` | 加载后做一次权重重排 |
| 4 | `models/deepseek_v2.py:705` `DeepseekV2MoE.forward` 开头 | forward 分流到融合路径 |
| 5 | `layers/moe/mega_moe.py` 整个文件 | buffer 单例 + pre_dispatch + 融合调用 |

### 1.4 forward 分流(deepseek_v2.py)

`DeepseekV2MoE.forward` **第一件事**就是检查融合路径,命中就直接 return:

```python
def forward(self, hidden_states, forward_batch=None, ..., input_ids_global=None):
    from sglang.srt.layers.moe.mega_moe import forward_mega_moe, should_use_mega_moe
    if should_use_mega_moe(self, hidden_states):
        return forward_mega_moe(self, hidden_states, forward_batch,
                                input_ids_global=input_ids_global)
    # ... 否则走 deepep / normal
```

注意 `megamoe` **不在** `self._enable_a2a_moe` 里(那个只含 deepep/mooncake/nixl/mori/
ascend_fuseep/flashinfer)。融合路径完全靠这个**早返回短路**处理,这是最干净的插入点。
你的 `mctlass` 后端照此加一个 `should_use_mctlass_moe` / `forward_mctlass_moe` 即可。

---

## 2. CUDA Graph 兼容:为什么不用 adapter,以及 3 条不变量

CUDA Graph 的硬约束:**capture 期间禁止 `cudaMalloc`、禁止 host 侧动态分支、禁止 CPU 同步
(如 `dist.barrier()`),且每次 replay 的 tensor 地址/形状必须不变**。融合 MoE 靠下面 3 条
满足它,你必须逐条对齐。

### 不变量 ① 对称缓冲区一次性分配 + 进程级缓存

`mega_moe.py:38,61-94` 用一个 module 级 dict 缓存 buffer:

```python
_MEGA_MOE_SYMM_BUFFER: dict = {}
def _get_mega_moe_symm_buffer(group, num_experts, num_max_tokens_per_rank, ...):
    key = (id(group), num_max_tokens_per_rank, num_experts, num_topk, hidden, intermediate)
    buf = _MEGA_MOE_SYMM_BUFFER.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe(...)   # 唯一一次分配 + 建链
        _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf
```

**机制**:第一次 forward 发生在 capture 之前的 warmup(eager 模式,graph 外),此时触发
`SymmBuffer(...)` 的 cudaMalloc + 对称内存建链。等到真正 capture 时,`buf` 已缓存,直接复用,
**capture 内零分配**。这跟 DeepEP 的 `DeepEPBuffer._buffer` 单例是同一个套路。

> ⚠️ 你测试脚本里在 `create_inputs()` 里**每次都** `SymmBuffer(...)` 一个新 buffer、结尾
> `buffer.destroy()` —— 接进 sglang 后**必须改成 dict 缓存、整个进程只建一次**,绝不能 per-forward
> 分配或 destroy。

### 不变量 ② 缓冲区尺寸由常量 `padded_max` 决定,与真实 token 数解耦

`buf.x` 形状是 `[padded_max, hidden]`,真实只用 `[0, num_tokens)`,其余 padding。
`padded_max = SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`(默认 1024,
`environ.py:602`),并 `assert num_tokens <= padded_max`(`mega_moe.py:193`)。

**机制**:输出 tensor 的地址和形状静态固定 → graph 可捕获。capture 时 `num_tokens` 等于
CUDA Graph 的 padded batch size(`cuda_graph_max_bs * draft_tokens`),所以你必须保证
`padded_max >= cuda_graph_max_bs * (speculative_num_draft_tokens or 1)`。

### 不变量 ③ pre_dispatch 自己处理 padding(替代 DeepEP 的 clean_buffer)

`mega_moe_pre_dispatch.cuh:105-116` 的 pad 路径:把 `[num_tokens, padded_max)` 的
`topk_idx` 填 `-1`、`topk_weights` 填 `0`。

**机制**:buffer 是复用的,上一轮残留必须清掉,否则融合 kernel 的 masked 计数会读到脏数据。
DeepEP 用单独的 `clean_low_latency_buffer`;融合版把"清零"折进了 pre_dispatch,**不需要 graph
外的额外步骤,也不需要 graph-runner adapter**。

> ⚠️ 你测试脚本里只 `buffer.x[:num_tokens].copy_(...)`,没填 padding——因为每次都是新 buffer。
> 接进 sglang 复用 buffer 后,**pre_dispatch 必须填 padding**。

### 不变量 ④(你的 kernel 特有的坑)去掉 host 侧 barrier

你测试里 `MegaMaskedGroupedGEMM` 调用传了 `barrier=torch.distributed.barrier()`。
**`dist.barrier()` 是 CPU 同步的集合通信,绝对不能进 CUDA Graph capture**。

- DeepEP 的 `_deepep_precompile_tp_barrier()` 只在 `SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE`
  阶段跑,capture/replay 时是 no-op。
- 你的 kernel 若内部需要跨 rank 同步,必须用**对称内存上的 device-side signal**(GPU 端
  spin-wait),而不是 host barrier。把 `barrier=` 参数在 sglang 路径里去掉/换成 device signal。

### 为什么 megamoe 不需要 `should_use_*` 在 capture 时区分?

`mega_moe.py:102-103`:capture 模式直接 `return True`,强制走融合路径。这保证 capture 和
replay 选的是**同一条分支**(graph 录的是融合 kernel,replay 也必须是融合 kernel)。
非 capture 时才按 `num_tokens <= cap` 动态判断。你照搬这个逻辑即可。

---

## 3. 最小实现骨架:`layers/moe/mctlass_moe.py`

把 `mega_moe.py` 复制一份,把 `deep_gemm` 换成 `mctlassEx`、把 FP8 换成你的 INT8+append_scale。
下面是剥到最小的骨架(省略号处填你 kernel 的细节):

```python
# python/sglang/srt/layers/moe/mctlass_moe.py
"""mctlassEx 融合 MoE 路径,接 DeepseekV2 的 megamoe 风格 a2a 后端。"""
from __future__ import annotations
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional
import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


# ---- 不变量①:进程级单例缓存,graph 外只建一次 ----
_MCTLASS_SYMM_BUFFER: dict = {}


def _get_mctlass_symm_buffer(group, num_experts, num_max_tokens_per_rank,
                             num_topk, hidden, intermediate, hidden_align,
                             intermediate_align):
    from mctlassEx import SymmBuffer

    key = (id(group), num_max_tokens_per_rank, num_experts, num_topk,
           hidden, intermediate)
    buf = _MCTLASS_SYMM_BUFFER.get(key)
    if buf is None:
        buf = SymmBuffer(
            group, num_experts, num_max_tokens_per_rank, num_topk,
            hidden, intermediate, hidden_align, intermediate_align,
        )
        _MCTLASS_SYMM_BUFFER[key] = buf
    return buf


def should_use_mctlass_moe(moe: "DeepseekV2MoE", hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_mctlass():            # §4.1 新增的枚举
        return False
    if not getattr(moe.experts, "_mctlass_weights_built", False):
        return False
    # 不变量④:capture 时无条件走融合路径,保证 capture/replay 分支一致
    if get_is_capture_mode():
        return True
    global_num_tokens = get_dp_global_num_tokens()
    max_tokens = max(global_num_tokens) if global_num_tokens else hidden_states.shape[0]
    cap = envs.SGLANG_MCTLASS_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    return max_tokens <= cap


def forward_mctlass_moe(moe, hidden_states, forward_batch=None,
                        input_ids_global=None) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    # 共享专家:融合路径里单独算(不 fuse 进 MoE kernel),可选 alt_stream overlap
    shared_output = moe._forward_shared_experts(hidden_states)
    y = _run_mctlass_routed(moe, hidden_states, forward_batch,
                            input_ids_global, num_tokens)
    if shared_output is not None:
        y.add_(shared_output)
    return y


def _run_mctlass_routed(moe, hidden_states, forward_batch, input_ids_global, num_tokens):
    from sglang.srt.distributed.parallel_state import get_moe_ep_group
    from mctlassEx import MegaMaskedGroupedGEMM

    H = moe.config.hidden_size
    I = moe.config.moe_intermediate_size
    E = moe.experts.num_experts
    K = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    cap = envs.SGLANG_MCTLASS_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    assert num_tokens <= cap, f"num_tokens={num_tokens} > cap={cap}"

    # ---- 路由(topk) ----
    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states, router_logits,
            num_token_non_padded=(forward_batch.num_token_non_padded
                                  if forward_batch is not None else None),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id),
            **topk_kwargs,
        )
        topk_ids = topk_output.topk_ids.to(torch.int64)
        topk_weights = topk_output.topk_weights.to(torch.float32)
    else:
        topk_ids = hidden_states.new_empty((0, K), dtype=torch.int64)
        topk_weights = hidden_states.new_empty((0, K), dtype=torch.float32)

    ep_group = get_moe_ep_group().device_group
    k_align = 512
    hidden_align = ...           # = align(hidden + 4, k_align),按你的 append_scale 规则
    inter_align = (I + 4 + k_align - 1) // k_align * k_align
    buf = _get_mctlass_symm_buffer(ep_group, E, cap, K, H, I, hidden_align, inter_align)

    # ---- 不变量②③:pre_dispatch = 本地 quant+pack 进 buf,并填 padding ----
    # 必须是 graph 可捕获的 kernel(纯 GPU 算子),不能有 .cpu()/.item()/dist.barrier()
    mctlass_pre_dispatch(
        hidden_states, topk_ids, topk_weights,
        buf.x, buf.topk_idx, buf.topk_weights,
        num_tokens=num_tokens,        # 内部把 [num_tokens, cap) 填 idx=-1, w=0
        hidden_align=hidden_align,
    )

    # ---- 融合大 kernel:dispatch + G1 + swiglu + G2 + combine ----
    y = torch.empty((max(num_tokens, 1), H), dtype=torch.bfloat16,
                    device=hidden_states.device)
    swiglu_limit = getattr(moe.config, "swiglu_limit", 0.0) or 0.0
    gemm = MegaMaskedGroupedGEMM()
    gemm(
        y,
        moe.experts.mctlass_l1_weights,     # (E_loc, I*2, H) int8 + scale
        moe.experts.mctlass_l2_weights,     # (E_loc, H, I)   int8 + scale
        buf,
        A_packed=True,
        scale_b1=moe.experts.mctlass_b1_scales,
        scale_b2=moe.experts.mctlass_b2_scales,
        activation_clamp=float(swiglu_limit),
        # 不变量④:绝不传 host barrier!跨 rank 同步用 buf 上的 device signal
    )
    y = y[:num_tokens]
    if not moe.experts.should_fuse_routed_scaling_factor_in_topk:
        y.mul_(moe.routed_scaling_factor)
    return y


def mctlass_pre_dispatch(
    hidden_states: torch.Tensor,   # [num_tokens, hidden] bf16
    topk_ids: torch.Tensor,        # [num_tokens, top_k] int64
    topk_weights: torch.Tensor,    # [num_tokens, top_k] float32
    buf_x: torch.Tensor,           # [padded_max, hidden_align] int8  (对称内存槽位)
    buf_topk_idx: torch.Tensor,    # [padded_max, top_k] int64
    buf_topk_weights: torch.Tensor,  # [padded_max, top_k] float32
    num_tokens: int,
    hidden_align: int,
) -> None:
    """本地 quant + pack + 写对称内存 + 填 padding。**CUDA Graph 安全**。

    等价于测试脚本里的:
        x_q, x_scale = quantize_per_channel_symmetric(x)
        x_packed = append_scale_batch_cuda(x_q, x_scale)
        buffer.x[:N].copy_(x_packed); buffer.topk_idx[:N].copy_(...); ...
    但去掉了 append_scale_batch_cuda 里的 `.cpu().numpy().tobytes()`(host 同步,
    graph capture 会直接报错),改成纯 GPU 的字节重解释 `scale.view(torch.int8)`。

    graph 安全要点:
      - 无 .cpu()/.item()/.numpy()/.tolist()/dist.barrier();
      - num_tokens 在某个被 capture 的 graph 里是常量(= 该 graph 的 padded bs),
        所以下面所有静态切片 [:num_tokens] / [num_tokens:] 录的都是定长 GPU op;
      - 先填 padding 再写有效区,保证复用 buffer 时无脏残留(不变量③)。
    """
    padded_max = buf_x.shape[0]

    # 不变量③:先把 [num_tokens, padded_max) 的路由槽位填无效值(idx=-1 => 不路由到任何专家)。
    # buf_x 的尾部不必清:topk_idx=-1 已让这些 padding token 不参与任何专家的 masked 计数。
    if num_tokens < padded_max:
        buf_topk_idx[num_tokens:].fill_(-1)
        buf_topk_weights[num_tokens:].fill_(0.0)

    if num_tokens == 0:
        # 0-token rank 角落场景:本 rank 无本地 token,但仍托管专家、会收别的 rank 的 token。
        # 只需保证整张 topk_idx 已是 -1(上面 fill 覆盖了 [0, padded_max))。
        return

    # ---- 本地 int8 对称量化:每 token 一个标量 scale(沿 hidden 维求 absmax)----
    amax = hidden_states.abs().amax(dim=-1, keepdim=True)            # [N, 1]
    scale = (amax.to(torch.float32) / 127.0).clamp_min_(1e-10)      # [N, 1] f32
    x_q = (
        torch.round(hidden_states / scale).clamp_(-128, 127).to(torch.int8)
    )                                                               # [N, H] int8

    # ---- 把 4 字节 float32 scale 追加到每行末尾(纯 GPU 字节重解释)----
    # [N,1] f32 --view--> [N,4] int8(小端 4 字节);无 host 拷贝,可进 graph。
    scale_bytes = scale.contiguous().view(torch.int8)               # [N, 4]
    packed = torch.cat([x_q, scale_bytes], dim=-1)                  # [N, H+4]

    # ---- pad 到 hidden_align(= align(hidden+4, 512))----
    pad = hidden_align - packed.shape[-1]
    if pad > 0:
        packed = F.pad(packed, (0, pad))                            # [N, hidden_align]

    # ---- 写进对称内存的本地槽位 [0, num_tokens) ----
    buf_x[:num_tokens].copy_(packed)
    buf_topk_idx[:num_tokens].copy_(topk_ids)
    buf_topk_weights[:num_tokens].copy_(topk_weights)


def build_mctlass_moe_experts_weights(experts) -> None:
    """加载后调用一次:把 w13/w2 量化重排成 mctlass 需要的 int8+per-channel scale 布局。"""
    if getattr(experts, "_mctlass_weights_built", False):
        return
    # 用你的 quantize_per_channel_symmetric 逻辑(从测试脚本搬过来),离线做一次:
    #   experts.mctlass_l1_weights, experts.mctlass_b1_scales = quant(experts.w13_weight.data)
    #   experts.mctlass_l2_weights, experts.mctlass_b2_scales = quant(experts.w2_weight.data)
    ...
    experts._mctlass_weights_built = True
```

> `mctlass_pre_dispatch` 上面给的是 PyTorch 算子拼出来的**可直接跑**版本(quant + 字节 pack +
> pad + scatter),已满足 graph 安全。性能不够再融成一个 kernel —— 但语义和 graph 约束保持不变:
> **全程在 GPU 上,无 host 同步,且填 padding。**
>
> ⚠️ 不要照搬测试脚本的 `append_scale_batch_cuda`:它内部 `scale.cpu().numpy().tobytes()` 是
> host 同步,CUDA Graph capture 会报 "operation would make the legacy stream depend on a
> capturing blocking stream"。上面用 `scale.contiguous().view(torch.int8)` 在 GPU 上做等价的
> 4 字节小端重解释,无 host 拷贝。

---

## 4. 接线 checklist(5 处改动)

### 4.1 注册枚举 — `layers/moe/utils.py`
```python
class MoeA2ABackend(Enum):
    ...
    MCTLASS = "mctlass"          # 新增
    def is_mctlass(self):        # 新增
        return self == MoeA2ABackend.MCTLASS
```

### 4.2 CLI/env — `server_args.py`
- 把 `"mctlass"` 加进 `moe_a2a_backend` 的 `Literal[...]`(§613)和帮助文本里的列表(§212)。
- (可选)仿照 §3197 的 `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` 加一个开关自动设
  `moe_a2a_backend = "mctlass"`。
- 在 `environ.py` 加 `SGLANG_MCTLASS_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(1024)`。

### 4.3 权重重排 — `layers/quantization/fp8.py:1196` 附近
仿照 megamoe 的分支:
```python
if get_moe_a2a_backend().is_mctlass():
    from sglang.srt.layers.moe.mctlass_moe import build_mctlass_moe_experts_weights
    build_mctlass_moe_experts_weights(layer)
    return
```
> 注意:这个在 `process_weights_after_loading` 里,发生在 server 启动早期、**warmup 之前**,
> 满足"权重在 graph 外准备好"。

### 4.4 forward 分流 — `models/deepseek_v2.py:705`
在 `DeepseekV2MoE.forward` 顶部、`should_use_mega_moe` 检查旁边加:
```python
from sglang.srt.layers.moe.mctlass_moe import (
    forward_mctlass_moe, should_use_mctlass_moe,
)
if should_use_mctlass_moe(self, hidden_states):
    return forward_mctlass_moe(self, hidden_states, forward_batch,
                               input_ids_global=input_ids_global)
```

### 4.5 不用动 `cuda_graph_runner.py`
**这是关键**:`megamoe` 在 graph runner 里没有任何注册,你的 `mctlass` 同样不需要。
只要 §2 的 3 条不变量满足,buffer 在 warmup 自然分配、capture 内只复用,就 graph-safe。
(对比 `deepep` 之所以需要 `DeepEPCudaGraphRunnerAdapter`,是因为它的 buffer 在 normal /
low-latency 两种布局间切换、需要 graph 外 `clean_buffer`;融合版没有这个切换,所以不需要。)

---

## 5. 从测试脚本到 sglang 的 4 个必改点

你那个 `mctlassEx` 测试脚本能直接跑,但接进 sglang + CUDA Graph 前,这几处必须改:

| 测试脚本里的写法 | 接进 sglang 必须改成 |
|---|---|
| `create_inputs()` 里每次 `SymmBuffer(...)` | dict 缓存,进程级只建一次(不变量①) |
| 结尾 `buffer.destroy()` | **删掉**,buffer 生命周期 = 进程 |
| `buffer.x[:num_tokens].copy_(...)` 不填 padding | pre_dispatch 必须填 `[num_tokens, cap)` 为 idx=-1(不变量③) |
| `MegaMaskedGroupedGEMM(..., barrier=torch.distributed.barrier())` | 去掉 host barrier,换 device-side signal(不变量④) |

---

## 6. 分阶段验证(bring-up)

1. **离线对拍(不开 graph)**:`--moe-a2a-backend mctlass --disable-cuda-graph`,跟 deepep
   baseline 比 logits。先确认数值对(用你测试里的 `debug_allclose`,rtol=1e-2)。
2. **buffer 单例验证**:加日志确认 `_MCTLASS_SYMM_BUFFER` 在整个 server 生命周期只 miss 一次。
3. **开 graph**:去掉 `--disable-cuda-graph`,设 `cuda_graph_max_bs <= cap`。capture 阶段
   若报 "operation not permitted during capture",90% 是 §2④ 的 host barrier 或某处
   `.item()/.cpu()` 漏了。
4. **padding 正确性**:故意让某 rank `num_tokens=0`(你测试的 `--zero-token-rank`),确认
   融合 kernel 不读脏 padding。
5. **spec decode**:若开投机采样,`cap >= cuda_graph_max_bs * speculative_num_draft_tokens`。

---

## 附:数据流总览

```
DeepseekV2MoE.forward
  └─ should_use_mctlass_moe()  ── capture 时恒 True;否则 num_tokens<=cap
      └─ forward_mctlass_moe()
          ├─ _forward_shared_experts()         # 共享专家,单独算
          └─ _run_mctlass_routed()
              ├─ gate + topk                    # 路由
              ├─ _get_mctlass_symm_buffer()     # 不变量①:单例,warmup 分配
              ├─ mctlass_pre_dispatch()         # 不变量②③:本地 quant+pack+pad 进 buf
              └─ MegaMaskedGroupedGEMM(buf)     # 不变量④:kernel 内 all-to-all,无 host sync
```

对称内存 buffer 全程复用、地址形状不变、无 host 同步、padding 自洽 → 整条路径可被
CUDA Graph capture/replay。
