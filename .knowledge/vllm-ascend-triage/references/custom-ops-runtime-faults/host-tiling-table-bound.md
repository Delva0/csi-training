# `<name> is out of range: N`: N is the op's own table constant, not an engine knob

Scope: a pre-launch `TORCH_CHECK` raised from a `csrc/<op>/` **host adapter** into a
plain Python traceback at request time — recorded shape `RuntimeError: batchIdx is out
of range: 1024` from `torch.ops._C_ascend.batch_matmul_transpose` inside
`vllm_ascend/attention/sfa_v1.py::_v_up_proj` (#8392, DeepSeek-V3.2 w4a8, DP2×TP8 + EP,
`deepseek_mtp` n=3, `FULL_DECODE_ONLY`, `--max-num-batched-tokens 4096`). The op ran no
kernel; nothing was launched.

Not this page: a device dump or `fault kernel_name=` (`index.md` device-fault rules), an
`ascendc/…h:137 Index N out of range[0 M)` assert inside a **kernel**
(`gather-index-out-of-range.md`), a `<tensor> shape is invalid` out of
`<op>_tiling.cpp` (`tiling-attribute-rejection.md`), or a Python `IndexError`.

## 1. The number in the message is the op's constant — reconcile it with the reporter's knob first

The reporter's own analysis usually names an engine parameter. In #8392 it named the
cudagraph bound: "`max_num_seqs × (1 + num_speculative_tokens) = 16 × 4 = 64`, times DP=2
gives 128". The message printed **1024**. That one comparison refutes the whole stated
mechanism before any source read, and it kills all three "expected behaviour" proposals
derived from it (request-count batch dim, capture-size accounting, "support cudagraph for
the drafter"). Do this arithmetic as the first move whenever an op refusal prints a round
number: the printed value is a constant in `csrc/`, and the reporter's parameter is a
hypothesis to test against it.

## 2. Read the host adapter: which tensor dim indexes the table, and how deep it is

```bash
rg -n '<the message stem, e.g. batchIdx>|out of range' csrc/
sed -n '1,130p' csrc/batch_matmul_transpose/op_host/batch_matmul_transpose.h
```

Verified at **both** vllm-ascend v0.13.0 (`6281c120`, the #8392 reporter's release line)
and v0.18.0rc1 (`99e1ea0`):

| Anchor | Fact |
|---|---|
| `op_host/batch_matmul_transpose.h:29` (both pins) | `constexpr uint32_t MAX_CAPTURE_NUM = 1024;` — a fixed table depth; the name mentions capture but nothing derives it from the engine's capture sizes |
| `:80-83` (both pins) | `OpShape{.batchSize = tensorAShape[1], .m = tensorAShape[0], …}`, then `int32_t batchIdx = opShape.m - 1` (`:103` at v0.13.0, `:104` at v0.18.0rc1) |
| `:107-112` / `:107-113` | `global_tiling_data[tilingSize * batchIdx]` written by `aclrtMemcpy(… ACL_MEMCPY_HOST_TO_DEVICE)`, else `TORCH_CHECK(false, "batchIdx is out of range: ", batchIdx)` |

So the index is the **first dim of the tensor the caller passed**, and the caller
(`sfa_v1._v_up_proj`) passes `x.view(-1, local_num_heads, kv_lora_rank)` — the first dim
being the batch's **token count**. `batchIdx 1024` therefore means 1025 tokens in one
forward: a prefill chunk or mixed batch larger than the table, not a spec-decode bucket.

## 3. Find the caller's guard — and check the reporter's release, in both directions

```bash
rg -n 'BMM_TRANS_MAX_SUPPORTED_TOKENS' vllm_ascend/       # constant + the guard
sed -n '<guard lines>p' vllm_ascend/attention/sfa_v1.py   # the custom op vs torch.bmm
```

At v0.13.0 the constant is `sfa_v1.py:47`, the guard `:478`; at v0.18.0rc1 they are `:65`
and `:810`. The guarded branch calls the custom op, the `else` runs
`torch.bmm(x.transpose(0, 1), self.W_UV)` — the fallback **is** the repair, and it is
present in *both* releases.

Two consequences, and the recorded run got this wrong once:

- "The pinned tree has the guard, so the reporter's release lacks it — tell them to
  upgrade" is only valid if you can grep their release. In #8392 the snapshot pinned
  v0.18.0rc1 (five releases newer) while the reporter ran v0.13.0rc1 — and the
  v0.13.0 release, which the #8587 snapshot in the same corpus holds, carries the same
  constant and the same guard. The claim was therefore unsupported, and the answer's
  rank 1 rested on it. Check the nearest release you can actually grep, say which one
  that is, and when nothing in the corpus holds the reporter's build, call the version
  attribution unverified instead of asserting it.
- With the guard present the refusal means the guard's scalar and the op's index
  disagreed: `num_input_tokens` comes from `x.shape[0]` while the op's `m` is
  `numel / (local_num_heads × kv_lora_rank)`, so any layout where those differ (heads
  gathered or sliced across TP/CP, a 4-D input, `dsa_cp` o_proj variants) passes the
  guard and still overflows. That is one print, not an upgrade: ask for
  `x.shape` + `num_input_tokens` logged at the call site, or the `VLLM_LOGGING_LEVEL=DEBUG`
  rerun that prints them.

## 4. The graph-mode theory dies at the `TORCH_CHECK`

The check runs on the **host before launch**, so it fires identically in eager, in
`PIECEWISE` and inside a captured graph, and a prefill chunk is outside the decode graph
anyway. Any ask of the form "make the drafter use cudagraph", "raise
`cudagraph_capture_sizes`" or a `cudagraph_mode` A/B cannot move this failure — say so
instead of running the graph ladder. Conversely, a `107030` / `507011` / MTE record in the
same paste belongs to a *different* lane (`../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`).

## 5. Explain "it worked on the previous model" by the backend, not by `max_model_len`

The sibling attention impl uses a **different operator with no such table**:
`mla_v1._v_up_proj` calls `torch_npu.npu_transpose_batchmatmul` (`mla_v1.py:840-848` at
v0.18.0rc1). So a V3.1 (MLA) deployment cannot hit this while a V3.2 (DSA/`sfa_v1`)
deployment can; #8392's reporter credited `max_model_len=16384` instead. Two `rg`s settle
it — which impl class the model selects, and whether that impl's `_v_up_proj` reaches a
`torch.ops._C_ascend.*` op at all.

## 6. Ladder, with the threshold arithmetic that makes the workaround cheap

1. Pick the threshold = the constant. `--max-num-batched-tokens 1024` (equal, not 512) keeps
   every chunk legal; the reporter's estimate of 512 came from the wrong bound.
2. Add the caller guard, copied from the newer tree, when their release really lacks it.
   The `else` branch already exists, so it is the two-line diff.
3. Do **not** enlarge `MAX_CAPTURE_NUM` in `csrc/`: the buffer is a device allocation of
   `tilingSize × N` and the change needs a rebuild for a slower un-tiled regime.
4. Perf claims need this boundary in view: `MLAPO_MAX_SUPPORTED_TOKENS = 1024`
   (`mla_v1.py:52` at v0.13.0) gates the fused MLAPO preprocess at
   `sfa_v1.py:763` (v0.13.0) / `:1079` (v0.18.0rc1), so a launch at
   `--max-num-batched-tokens 4096` **already** runs its big chunks on the native path.
   Benchmarks taken at 4096 and at 1024 measure two different code paths, and
   `_v_up_proj` is reached on both (`sfa_v1.py:895` / `:1226`), so no branch shields it.
   When a reporter rejects the workaround because it "severely degrades prefill", that
   coincidence is the reply.
