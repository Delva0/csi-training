# A bare `setup failed!` is the caller's own pack — start at the deepest `vllm_ascend` frame

Verified at vllm-ascend `v0.7.3rc1` (`f025df05f614`) and vllm `v0.7.3`
(`ed6e9075d31e`), recorded for **#393**: `torch-npu 2.5.1.dev20250308`, an
ms-swift 3.2.1 GRPO run whose rollout engine is the **V0** vLLM engine
(`vllm/engine/llm_engine.py:1391` → `vllm/worker/worker_base.py:420`), Qwen2.5-class
model. It trains ~5 steps (`Train: 0%|… 5/8724`) and then kills the rank.

## The signature

```text
Train: 0%|… 5/8724 [02:25<…]mki_log log dir:/root/atb/log exist
[rank0]:   File ".../vllm_ascend/attention.py", line 869, in forward
[rank0]:     torch_npu._npu_flash_attention_qlens(
[rank0]: RuntimeError: setup failed!
```

No op name, no code, no device dump: the message is the ATB/op-plugin *host-side
operation setup* refusing a variant pack (shape/dtype/attribute). Nothing was
launched, so no card, HCCL, OOM or driver claim applies. The `mki_log … atb` line
immediately before it is ATB's first-use banner (string in no export — label it as
consistent-with, not proof).

## Which of the eight `setup failed!` surfaces this is

| Text | Owner | Page |
|---|---|---|
| **bare `setup failed!`**, no op name, deepest `vllm_ascend` frame is a `torch_npu._npu_*`/ATB attention call | the **call site in that frame**: its own mask/size/attribute arguments | this page |
| `SelfAttentionOperation setup failed!` + `op_plugin_atb` frames at a proposer line, spec decode on | ATB at the drafter (join point; the Python line is a victim) | `../spec-decode-mtp/atb-op-setup-failure-at-proposer.md` |
| `ReshapeAndCacheOperation setup failed!` on a `-310p` image with `--kv-cache-dtype fp8`/`bfloat16` | 310P has no non-fp16 KV path | `../soc-platform-paths/310p-kv-cache-dtype.md` |
| `<Op>Operation setup failed!` beside `507018` / `RunCpuKernel` | the op ran on the CPU/AICPU fallback kernel | `../soc-platform-paths/310p-cpu-fallback-op-fault.md` |
| `SelfAttentionOperation setup failed!` with **no** `--speculative-config` at all, a dense model and one **long** prompt (12 732 tokens on `--max-num-batched-tokens 14336`) | the dense V1 one-shot prefill branch handing ATB a `[prompt, prompt]` mask — branch first, kernel contract not derivable (§ below) | this page, second section |
| `SelfAttentionOperation setup failed!` **plus** an ATB log line `tensor dimNum 0 is invalid … inTensor [N] CheckTensorShape failed` | the caller passed a `None`/empty tensor into the pack — find the never-bound attribute, do not read the op's shape contract | `atb-input-dim0-none.md` (recorded #2779) |
| `PagedAttentionOperation setup failed!` beside `atb::OperationSetup` / `libop_plugin_atb.so` frames, on a V1 serve that started cleanly, at the **second** request (or any request whose prompt shares a cached prefix), the dumped config showing `enforce_eager=True`, `chunked_prefill_enabled=True`, `enable_prefix_caching=True` | the state decided by the dumped `SchedulerOutput` picks the branch: `ChunkedPrefill` ⇒ `_npu_paged_attention_splitfuse`, whose mask is the only argument that changed. Both flags are **V1 defaults**, so no flag was misused and `--enforce-eager` is spent | this page, section `PagedAttentionOperation` |
| `RopeOperation setup failed!` (ATB **Rope**) at the first forward inside a `torch.compile` warm-up, `op_plugin_atb` frames only, **no** Python frame in the paste, on a VL / M-RoPE model (GLM-4.1V) that starts on A2 and dies on an `-a3` image | the rope call site in `vllm_ascend/ops/rotary_embedding.py` `_rope_forward_oot`: three routes, and which one ran is the question. No export carries the ATB Rope source, so the answer is a one-launch route flip, not a source walk | this page, section `#3706` |

All of them mean the same phase. What differs is *who assembled the pack*, and the
paste answers that: read the deepest `vllm_ascend` frame, then read the size
arguments it passes.

A **`CreateOperation failed!`** message is not in this table and is not a pack question at
all: it fires before any tensor is described, so the owner is the ATB build × SoC —
`atb-create-operation-failed.md` (recorded #5170, 310P).

## `PagedAttentionOperation`: the scheduler state picks the branch, not the op name

Recorded #2700: vllm-ascend `v0.10.0rc1` (`4604882a3e2c`) / vllm `v0.10.0`
(`6d8d0a24c02b`), MedGemma-4B (Gemma-3 family) TP1, bf16, `max_model_len=131072`,
`enforce_eager=True`. The reporter's first chat request answered; the **second** — a new
completion whose prompt shares 256 tokens with the first — killed EngineCore on its
first step:

```text
[rank0]:[E902 …] PagedAttentionOperation setup failed!
Exception raised from OperationSetup at … op_plugin_atb.dir/compiler_depend.ts:151
frame #2: atb::OperationSetup(atb::VariantPack, atb::Operation*, atb::Context*) … libop_plugin_atb.so
… RuntimeError: The Inner error is reported as above … the current working operator name is PagedAttentionOperation.
… ERR00100 PTA call acl api failed.
```

Both `enable_chunked_prefill` and `enable_prefix_caching` are **V1 defaults** for a
non-pooling model — `vllm/engine/arg_utils.py:1530-1536`, "V1 always uses chunked
prefills and prefix caching" — so nothing here is a misconfiguration, and the engine's
own `Dumping input data … with config` line prints them as `chunked_prefill_enabled=True,
enable_prefix_caching=True`. Start from that line rather than from the launch shell.

**The dumped `SchedulerOutput` decides the branch** (`model_runner_v1.py:1119-1138`,
where `seq_lens = num_computed_tokens + num_scheduled_tokens`):

| test, in order | #2700's numbers | state |
|---|---|---|
| `np.array_equal(seq_lens, num_scheduled_tokens)` (`:1119-1120`) | `283` vs `27` → no | — |
| `np.all(num_scheduled_tokens == 1)` (`:1122-1123`) | `27` → no | — |
| `np.all(num_valid_tokens == 1)` (`:1129-1133`) | no spec tokens | — |
| `chunked_prefill_enabled` (`:1135-1136`) | `True` | **ChunkedPrefill** |
| otherwise (`:1138`) | — | PrefillCacheHit (Ascend scheduler on **and** chunked prefill off) |

So `num_computed_tokens=256` + `num_scheduled_tokens={req: 27}` + `prompt_token_ids_len=283`
+ `prefix_cache_stats(queries=283, hits=256)` is one subtraction, `283 = 256 + 27`.
`ChunkedPrefill` takes `attention_v1.py`'s `else` branch: `head_size == 192` diverts to
`vanilla_chunked_prefill` (`:391-417`), everything else calls
`_npu_paged_attention_splitfuse` (`:419`) — and the paste's name is a *paged attention*,
which is the same fact a second time.

**Why request 1 was clean: the token budget, not a warm-up.** A 283-token prompt fits one step (`max_num_batched_tokens` defaults to
`DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048`, `vllm/config.py:2391-2402`, floored at `5120`
for a multimodal model, `vllm/utils/__init__.py:86`), so request 1 was
`seq_lens == num_scheduled_tokens` → `PrefillNoCache` → `_npu_flash_attention` (`:343`),
and its generated tokens were `DecodeOnly` → `_npu_paged_attention` (`:376`). The
prefix hit is what creates the third state; `--enforce-eager` is already true in the
dump and removes nothing.

**The pack's variable is the mask**, built for this state only (`_make_attention_mask`,
`model_runner_v1.py:844-859` → `get_splitfuse_attn_mask`, `attention/attention_mask.py:54-96`).
The builder is seeded at `min(max_model_len, PAGED_ATTENTION_MASK_LEN=10000)`
(`:220-222`); `max_seq_len = max(seq_lens) = 283` stays inside it, and the return is
`index_select(mask[:283, :283], 0, position)[:, :283]` — rows are the absolute positions
`256…282`, columns `0…282`. The #393/#2239 "mask narrower than the context" story does
**not** apply here (that branch passes a literal `128`); this shape arithmetic closes.
What ATB then rejects is in no export: `Ascend/op-plugin` in this snapshot carries no
`paged_attention` string at all — one grep, then stop.

### Ladder

1. `--no-enable-prefix-caching`: the second identical request returns to `PrefillNoCache`.
2. `--max-num-batched-tokens 256` (below the prompt's 283): reproduces the branch on a
   **fresh** prompt with no cache involved — step 1 is `PrefillNoCache` (256 = 256),
   step 2 is `ChunkedPrefill`. This is the ablation that separates "prefix cache" from
   "splitfuse", and it is the cheap repro to hand the reporter.
3. Both still die ⇒ the pack itself is the defect at that release: quote the release lane
   (which tag/rc the reporter is on) and ask for the plog at that second plus four numbers —
   `num_scheduled_tokens`, `num_computed_tokens`, the prompt length, the model's `head_dim`.
   Nothing was launched, so `ASCEND_LAUNCH_BLOCKING=1` and `--enforce-eager` explain
   nothing here.

Not the capability lane, and not the ❌-row page: at this pin `Gemma-3 | ✅` and the VL
table's `Gemma3 | ✅` (`docs/source/user_guide/support_matrix/supported_models.md:32`,
`:71`). The only other page naming `PagedAttentionOperation` —
`../feature-enable-recipes/unsupported-model-row.md` — owns the ACL-graph **replay**
form (`super().replay()` frame, a `❌` row); read the row's verdict before choosing
between them.

Run: 17 assistant turns / 31 tool calls, 47.5 K characters of reasoning and **zero**
characters of message text, `output_truncated` — while the attachment's dumped
`SchedulerOutput` (assistant turn 9) already fixed the state and both ablations. Ship the
verdict from that block instead of walking mask shapes.

## #3706: `RopeOperation setup failed!` on an `-a3` image, same model fine on A2

Recorded #3706: GLM-4.1V-9B-Thinking on `vllm-ascend:v0.10.2rc1-a3-openeuler`;
A2 starts the same model. The paste is a screenshot OCR, so the frames are interleaved —
what is certain is the wording and the phase:

```text
(EngineCore_DP0 pid=7998) INFO … Compiling a graph for dynamic shape takes 1.93 s
[rank0]:[E1023 17:02:23…] RopeOperation setup failed!
Exception raised from OperationSetup at build/third_party/op-plugin/op_plugin/CMakeFiles/op_plugin_atb.dir/compiler_depend.ts:151
frame #0: c10::Error … ; frame #2: atb::OperationSetup(…) in libop_plugin_atb.so …
```

`OperationSetup` ⇒ this page's family (the caller's pack), **not**
`CreateOperation failed!` (`atb-create-operation-failed.md`). The phase is decidable from
the line above it: the first forward inside the `torch.compile` warm-up, before any
request. Everything below was re-verified at vllm-ascend v0.11.0rc0 (`00ba0710`) /
vllm v0.11.0rc3 (`8ce5d319`), the #3706 view's pins — one rc newer than the reported image,
so quote the scope.

- **Which rope route vllm-ascend takes** (`vllm_ascend/ops/rotary_embedding.py`
  `_rope_forward_oot`): (a) the custom kernel `torch.ops._C_ascend.rotary_embedding`, gated
  on `query.dtype == float16 and neox_style and head_size % 32 == 0 and
  enable_custom_op()` and `not is_310p()` (`:31-33`, `:50-56`); (b) `torch_npu.
  npu_apply_rotary_pos_emb` when `self.cos`/`self.sin` were precomputed, whose own comment
  says it "requires head_size and rotary_dim equal 128 and neox_style is True" (`:63-72`);
  (c) `torch_npu._npu_rotary_embedding(positions, q_rot, k_rot, self.head_size,
  self.cos_sin_cache, is_neox_style)`, reached for `rotary_dim < head_size` (`:73-90`) and
  for the full-rotary default (`:96-105`). The failing op is one of (b)/(c) — the dtype gate
  in (a) is what keeps a **bf16** serve off the custom kernel and on the `torch_npu` route,
  which makes `--dtype float16` a genuine one-launch **route** flip, not a precision tweak.
- **`npu_rotary_mul` is not on this path**, so do not credit the VL vision tower: it occurs
  only in vllm-ascend's own Qwen-VL copies (`models/qwen2_vl.py:93`, `qwen2_5_vl.py:104`,
  `qwen2_5_vl_without_padding.py:111`), which `models/__init__.py` registers **per arch**
  (`Qwen2VLForConditionalGeneration`, `Qwen2_5_VL…`, `Qwen3VL…`) and GLM never selects; and
  upstream `glm4_1v.py:81` imports `apply_rotary_pos_emb_vision` from `qwen2_vl`, which is
  pure-torch (`apply_rotary_emb_torch`). One grep of the pin's `register_model()` answers
  this in a single call.
- **The ATB API↔op mapping is not locally decidable, and that is quotable.** In the #3706
  view `Ascend/op-plugin` is a 107-file partial export with **zero** `rotary` hits and no
  `atb/` directory, and `rg -ln 'op_plugin_atb|atb::'` over the whole view returns nothing.
  One grep, then stop: name which route is unverified and spend the turn on the flip.
- **Two one-launch ablations, and only these:** `--enforce-eager` (removes `torch.compile`,
  the phase the paste shows — require its own line
  `Compilation disabled, using eager mode by default` in the result) and `--dtype float16`
  (flips the route per the gate above). A2-versus-A3 is a *dispatch/dtype* difference on
  that line, not a driver or privilege question — do not carry `--privileged`, the
  driver-pairing or the `platform_config` lane into it.
- **Coverage, stated before any recipe:** `Glm4vForConditionalGeneration` is not in the
  pin's `register_model()` and has no row in `docs/source/user_guide/support_matrix/
  supported_models.md` (the GLM rows there are GLM-4.5 ✅, GLM-4 ❌#2255, GLM-4-0414
  ❌#2258, GLM-4V ❌#2260), so there is **no validated recipe** for this pair and the op
  fault is the manifestation — later attempts will surface *different* errors.
- **Do not rank "upgrade for the partial-rotary fix"**: the release note
  "Fix accuracy issue on models whose rope dim is not equal to head dim, e.g. GLM4.5"
  (#2601) sits under **v0.10.1rc1**, i.e. the reporter's v0.10.2rc1 image already has it.
  Read the version heading above the bullet before citing a fix as a rung.

#11080's sibling lesson applies: this run shipped no message either (15 turns, 743
characters of text, cut inside a reasoning block), while the phase, the route table and both
ablations were already in hand.

## The recorded mechanism at `f025df05f614` (source-verified sizes)

`vllm_ascend/attention.py:866-876` is reached only when `attn_metadata.block_tables`
is non-empty, `num_decode_tokens == 0` **and** `not chunked_prefill_enabled`
(`:866`). With chunked prefill off, the only code that gives a *prompt* a block
table is a prefix-cache hit — `vllm_ascend/worker/model_runner.py:590-645`
(`prefix_cache_hit = computed_block_nums is not None and len(...) > 0 and
self.sliding_window is None and inter_data.is_prompt`) — and upstream defaults
`enable_prefix_caching` to `bool(envs.VLLM_USE_V1)`, i.e. **False** on the V0 path
(`vllm/engine/arg_utils.py:217-220`). So the first fact to state is a derivation,
not a guess: **the rollout engine set `enable_prefix_caching=True` explicitly and
runs chunked prefill off.**

The pack's variable input is the mask, and its bound is one block:

| Evidence | Claim |
|---|---|
| `attention.py:612-617` | this branch builds `self.compress_mask = …get_attn_mask(128, dtype, device)` — a literal `128` |
| `attention.py:58-88`, `:502-505` | `get_attn_mask(n)` returns `attn_mask_cache[:n, :n]`, so `128` ⇒ a contiguous `[128, 128]` mask; the builder is a per-process singleton seeded from `128` |
| `platform.py:120-123` | `block_size` defaults to `128` and prefix caching *requires* 128 — so `128` is exactly one KV block |
| `model_runner.py:614-645` | what the same branch's metadata can actually carry: **partial** hit → `query_lens = seq_lens − prefix_cache_len`; **full** hit → `query_lens = 1`, `context_lens = seq_lens − 1`. Neither is bounded by one block |
| `attention.py:867-874` | that `[128, 128]` tensor becomes `mask=` while `seq_len=self.query_lens_tensor_cpu` and `context_lens=self.seq_lens_tensor_cpu` |

Why five steps survive: a fresh prompt has no computed block, so `block_tables` is
empty and the step takes the `_npu_flash_attention` + `max_prefill_seq_len` mask
path (`:608-612`, `:850-860`) at any length. The `qlens` branch needs a *hit*, and
it fires as soon as one arrives with a suffix (or a context) larger than a block —
a data-order trigger, not a step count.

**Confidence:** the sizes and the branch membership are source-verified at that
ref; ATB's own validation text is not in the snapshot, so "the pack is rejected
because the mask does not cover those lengths" is the hypothesis the two ablations
below settle.

## `SelfAttentionOperation` with **no** spec decode: the one-shot prefill mask

Derived at vllm-ascend `v0.9.2rc1` for #2239 (Qwen2.5-7B bf16 on a 910B4/32 GB,
`torch-npu 2.5.1.post1.dev20250619`, CANN 8.1.RC1, `--max-model-len 14336`,
`--max-num-batched-tokens 14336`, one 12 732-token prompt, cause left open). The
named op appears with no drafter anywhere, so read the **branch** before the ATB
contract — the same message has three different owners in the table above.

- Backend: V1, non-MLA, non-torchair → `AscendAttentionBackend`
  (`platform.py:228-229`) → `attention/attention_v1.py`.
- Branch: `model_runner_v1.py` selects `PrefillNoCache` when
  `seq_lens == num_scheduled_tokens`, i.e. **the whole prompt fit in one step**
  because `--max-num-batched-tokens` ≥ prompt length. That branch calls
  `torch_npu._npu_flash_attention` (`attention_v1.py:346`) with
  `mask = AttentionMaskBuilder.get_attn_mask(max(seq_lens))` — a dense
  `[12732, 12732]` bf16 tensor. A prompt that does **not** fit one step takes the
  splitfuse branch instead (`_npu_paged_attention_splitfuse`, `:420`).
- `PAGED_ATTENTION_MASK_LEN` is **not** a mask cap: the builder is seeded with
  `min(max_model_len, int(os.getenv("PAGED_ATTENTION_MASK_LEN", 10000)))`
  (`model_runner_v1.py:213-215`) and `get_attn_mask`/`_update_attn_cache` grow the
  cache on demand (`attention/attention_mask.py:49-53`, `:98-103`). Do not report
  "masks above 10000 are unsupported" out of that default.
- The version lane was clean and proves nothing: `requirements.txt:27` pins
  `torch-npu==2.5.1.post1.dev20250619`, exactly the reporter's pta, and that
  release raises CANN to 8.1.RC1 *to enable* chunked prefill/APC — which they had.
- Discriminators, cheapest first: does a **short** prompt serve (branch vs op)?
  then `--max-num-batched-tokens 8192` (forces chunked → splitfuse) and, if it
  still dies, `VLLM_USE_V1=0` (the V0 `attention.py` path uses
  `npu_fused_infer_attention_score`). Unlike #393 above, the batched-token knob is
  the lever here, precisely because it changes the branch.
- What cannot be settled: no exported tree carries ATB's `SelfAttention` contract
  (`Ascend/op-plugin` exports no `atb/` sources; `Ascend/pytorch` and
  `Ascend/torchair` contain no `SelfAttention` string). Ship the branch, the mask
  size and the two ablations; do not derive the kernel's limit.

## Ladder

1. `enable_prefix_caching=False` on the rollout engine (ms-swift exposes the vLLM
   engine kwargs; the exact arg name is not in this snapshot — ask for the launch
   config rather than quoting one). Removes the branch entirely. **Most decisive.**
2. `enable_chunked_prefill=True` instead: the same step takes the splitfuse path
   (`:619-629` builds `chunk_mask` from `max_seq_len`; `:880-905` calls
   `_npu_paged_attention_splitfuse`), so the mask is sized by the schedule.
3. Keep both flags and still crash ⇒ out of this page: `--enforce-eager` then
   `ASCEND_LAUNCH_BLOCKING=1`, and get the plog.
4. If they must keep prefix caching without chunked prefill, the fix is the code at
   `:615-617` (build the mask from the branch's own maxima) — say so as the patch
   proposal, not as the first action.

Not useful: `--max-num-batched-tokens`, `--max-model-len`, capture sizes and
`HCCL_*` (nothing was launched); `--enforce-eager` alone (the ATB call is on the
eager path too).

## What this snapshot cannot settle

- ATB (`cann/ascend-transformer-boost`) and op-plugin sources are unavailable and
  `_npu_flash_attention_qlens` is in neither exported tree (the pytorch export is
  `1133/1134 TRUNCATED`), so the op's mask/shape contract is a hypothesis: one
  grep per tree, then stop — do not hunt the emitter.
- Whether the reporter's engine really enabled prefix caching is derived from the
  branch, not read from their config. Ask for the rollout engine's kwargs.
- Version scope: the V0 `vllm_ascend/attention.py` layout is this pin's; re-grep
  this branch on any other ref before quoting `path:line`.
- Precedent: `scripts/history-titles.sh -i 'setup failed|qlens|atb'` returns 0 hits;
  the prior-month corpus has no equivalent (it is not evidence of absence).

## Deliver

Verdict first (a real vllm-ascend prefill-path crash, host-side op setup, no
hardware involvement; the two-flag combination is what reaches it), then the one
decisive ablation, then the ranked second cause (query/context bookkeeping in
`_add_seq_group`), then what to send back: the rollout engine's full kwargs
(prefix caching / chunked prefill / `max_model_len` / `max_num_batched_tokens`),
the failing iteration's prompt lengths or `max_query_len`, and the plog window.
