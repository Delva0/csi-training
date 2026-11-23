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
(`sfa_v1._v_up_proj`) passes `x.view(-1, self.num_heads, kv_lora_rank)` — the first dim
being the batch's **token count** (`num_heads` and `local_num_heads` coincide wherever the
custom op is reachable, §3). `batchIdx 1024` therefore means 1025 tokens in one forward —
not a spec-decode bucket, and (on DP>1) not this rank's batch either: §2a.

## 2a. The rows are `num_input_tokens` — on DP>1 that is the DP-wide padded width

The count that reaches the op is not the rank's scheduled tokens. `sfa_v1.mla_forward`
sizes its buffers from `attn_metadata.num_input_tokens` (`torch.empty((num_input_tokens,
self.q_lora_rank))` at `sfa_v1.py:703-704`, `q_pe` at `:698`, the `mla_preprocess` call
below them) and hands `npu_sparse_flash_attention`'s TND output straight to `_v_up_proj`
(`:848`), so `m` is the **model input's** token dim. At the reported pins that dim has
already been replaced by the maximum over DP ranks:

| Anchor (v0.13.0rc1 `1b5d5ab`) | Fact |
|---|---|
| `worker/model_runner_v1.py:475-484` | `max_tokens_across_dp = torch.max(num_tokens_across_dp).item()` in `_sync_metadata_across_dp` |
| `:644-649` | `num_input_tokens = maybe_padded_num_tokens` — the DP-wide max, **unconditionally**, in eager and in both graph modes |
| `:870` (`:771-772` for positions) | the model input is `input_ids.gpu[:num_input_tokens]` and the tail positions are zero-filled from the local total: the extra rows are **padding**, not a request |
| `:1055` / `sfa_v1.py:179` | `num_actual_tokens = slot_mapping_size = total_num_scheduled_tokens` is **local**, and `has_prefill = any(query_lens_cpu > self.decode_threshold)` is computed over the local per-request lengths |

That closes the arithmetic the recorded #8390 run left as a hypothesis. Taking the custom
branch means `has_prefill` was False, so *every* local request carried ≤
`1 + num_speculative_tokens` tokens and the local batch was ≤ `max_num_seqs × (1 + n)` —
64 at that launch (`--max-num-seqs 16`, `n=3`) — while `m` was 1025. The rows the op saw
therefore came from a **sibling DP rank's prefill chunk** (≤ `--max-num-batched-tokens`,
4096 there) padded onto this rank; the reporter's "total token count of the batch" is right
about the dim and wrong about the owner, and its "only in `piecewise`" control is confounded
with FlashComm1 (last bullet of §3). Nothing here is `piecewise`-specific: the FULL
uniform-decode block rewrites `query_start_loc` before the metadata is built
(`:1032-1050`), so `has_prefill` reads False on a `FULL_DECODE_ONLY` launch too, and a DP>1
box then shows the same op refusal on either mode.

One print at the call site decides it: `x.shape[0]` (and `attn_metadata.num_input_tokens`)
against `attn_metadata.num_actual_tokens`. Greater means padding — the DP-wide max when
DP>1 (compare it with that step's sibling-rank `num_scheduled_tokens`), otherwise the local
cudagraph bucket from `pad_for_cudagraph` (`model_runner_v1.py:614-615`) — and a bucket is a
round capture size, so an odd width like `1025` is the DP max and not a bucket, which is what
retires the recorded run's "piecewise bucket" variant. Equality would mean
the rows *are* the local batch, which this branch bounds at `max_num_seqs × (1 + n)` (64 at
#8390), so an equal-width print beside `batchIdx` ≥ the constant means the launch itself
differs (a larger `max_num_seqs`, or the `else` branch running with a wide layout) — not a
table that needs to grow.

Two consequences for the ladder. Rung 1 of §6 works *because* of this: capping
`--max-num-batched-tokens` at the constant caps each rank's padded contribution, hence the
DP max, while a per-rank `--max-num-seqs` cut does not touch it. And a
`--data-parallel-size 1` launch cannot produce this signature through the custom branch at
all, unless `max_num_seqs × (1 + n)` itself exceeds the constant — which makes DP size a
cheap attribution test instead of another source read. The same DP-wide max is the number
in `../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md` §10's host `RuntimeError`
(`63` vs `6151`), so one sibling rank's prefill chunk explains that lane and this one.

## 3. Find the caller's guard — and check the reporter's release, in both directions

```bash
rg -n 'BMM_TRANS_MAX_SUPPORTED_TOKENS' vllm_ascend/       # constant + the guard
sed -n '<guard lines>p' vllm_ascend/attention/sfa_v1.py   # the custom op vs torch.bmm
```

At v0.13.0 the constant is `sfa_v1.py:47`, the guard `:478`; at v0.18.0rc1 they are `:65`
and `:810`. The guarded branch calls the custom op, the `else` runs
`torch.bmm(x.transpose(0, 1), self.W_UV)` — the fallback **is** the repair, and it is
present in *both* releases.

Consequences — the recorded run got the first one wrong, and the `rc` builds change who
reaches the op at all:

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
  `numel / (heads × kv_lora_rank)`, so any layout where those differ (heads
  gathered or sliced across TP/CP, a 4-D input, `dsa_cp` o_proj variants) passes the
  guard and still overflows. That is one print, not an upgrade: ask for
  `x.shape` + `num_input_tokens` logged at the call site, or the `VLLM_LOGGING_LEVEL=DEBUG`
  rerun that prints them.
- **The `rc` builds do not carry that constant at all, and their proxy guard is the
  defect.** Verified in vllm-ascend v0.13.0rc1 (`1b5d5ab`, the reporter's own release in
  the #8390 case view): `rg -n 'BMM_TRANS_MAX_SUPPORTED_TOKENS|MLAPO_MAX_SUPPORTED_TOKENS' .`
  returns **nothing**, and `_v_up_proj` (`sfa_v1.py:479-492`) gates the custom op on
  `… and not self.enable_sfa_cp and not has_prefill`, above its own comment:

  ```text
  # TODO(zzzzwwjj): We should not judge by whether `has_prefill` or not.
  # The true criteria for judgment is tensorA's shape[0] <= 1024 (num_tokens <= 1024).
  # This is a bug in the previous code.
  ```

  `has_prefill = any(query_lens_cpu > self.decode_threshold)` (`:179`) with
  `decode_threshold = 1 + num_speculative_tokens` (`:136-140`), so the proxy is a
  per-request query-length test and any >1024-token batch it labels decode reaches the
  op. When the reporter's env block and the case pin are both this rc, the version
  attribution **is** verifiable — quote that TODO and the missing constant, put "move to
  the release line's final (or backport the two-line `else`-branch condition)" at rung 1-2,
  and keep the mechanism open for the batch that was mislabelled. What stays unsupported
  is the converse: a newer pin's constant never proves an older reporter lacks the guard.
- **Do not derive a head/token factor.** In the branch that can crash, `m` *is* the token
  count: the custom branch views with `self.num_heads` (`:487`) while the `else` uses
  `self.local_num_heads` (`:496`), and the two differ only when `enable_sfa_cp` is true
  (`local_num_heads = num_heads × tp_size`, `:378-382`) — which the same `if` excludes. So
  `batchIdx 1024` is 1025 tokens, not `tokens/4` or `tokens/8`; deriving a factor from
  `kv_lora_rank`/`v_head_dim` (the recorded #8390 run spent three blocks there) buys
  nothing the paste cannot already say. *Which* token count it is — the padded
  `num_input_tokens`, not the local schedule — is §2a.
- **`piecewise` in such a launch is forced, and that also fixes who can reach the op.**
  `enable_sfa_cp = enable_sp() and hasattr(hf_config, "index_topk")` (`:148-150`) and an
  assert forbids FlashComm1 with `FULL_DECODE_ONLY` (`:152-156`, *"Please set graph_mode to
  'piecewise' or disable FlashComm1"*) — so a DSA model launched with
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` prints `piecewise` for that reason, not as a free
  choice, and the `not self.enable_sfa_cp` term means the custom op is **unreachable**
  there. A crash inside `batch_matmul_transpose` therefore says SP/FlashComm1 was off at
  run time: ask for the effective `env` rather than reconciling the block that mentions it.

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
   every chunk legal — including the DP-wide padding of §2a, which is why this lever works
   while a `--max-num-seqs` cut does not; the reporter's estimate of 512 came from the wrong
   bound.
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
