# FIA TND head-dim refusal: `queryD(256), keyD(256) and valueD(256) must be same equal 192/128/64`

## Scope

A lane-C paste whose deepest frame is `npu_fused_infer_attention_score` /
`torch_npu`, with

```text
RuntimeError: npu_fused_infer_attention_score_symint … NPU function error:
  call aclnnFusedInferAttentionScoreV3 failed, error code is 561002
[ERROR] … ERR00100 PTA call acl api failed.  E89999: Inner Error!
E89999: [PID: …] <ts> When layout is TND, queryD(256), keyD(256) and valueD(256) must be
  same equal 192/128/64, or queryD and keyD equal 192 and valueD equal 128.
  [FUNC:CheckInputShapeWhenLayoutIsTND][FILE:prompt_flash_attention_tiling.cpp][LINE:3701]
        FusedInferAttentionScore do tiling failed, ret is -1.
        Check NnopbaseExecutorDoTiling(executor) failed   (… TilingAndUpdateBinInfo / MatchCache / RunForWorkspace)
```

and a traceback through `patch_attention_layer.py:67 → attention/attention_v1.py:593 →
:691 → :668 → :539 _forward_v1_style`. Recorded #4135: vllm-ascend **v0.11.0rc1**
(`c5fe179`) + vllm v0.11.0 (`b8b302c`), image `vllm-ascend:v0.11.0rc1`, Qwen3-Next-80B-A3B-Instruct
TP4 `--enforce-eager`, serve up, request dies.

The tell is that the sentence **prints the caller's own dim and the set the tiling
accepts**. That makes the refused property the model's `head_dim`, and it is neither of
the two siblings this wording is routinely confused with:

| Sentence | Refused property | Page |
|---|---|---|
| `When layout is TND, not support Page Attention` (`RunBigKernelTilingWithParams`) — no dims, verb `not support` | a layout × cache-kind combination of the installed toolkit | `tiling-capability-refusal-cann-release.md` |
| `queryT(16) must be equal to the last element of actualSequenceLengthQ(4)` (`CheckFASeqlenDataInTND`) | padding bookkeeping: two *sequence-length* counts disagree | `../spec-decode-mtp/fia-tnd-seqlen-mismatch.md` |
| `queryD(256), keyD(256) and valueD(256) must be same equal 192/128/64` (`CheckInputShapeWhenLayoutIsTND`) | the **head dim** of q/k/v against `{64,128,192}` (or the MLA pair 192/128) | this page |

So `must be equal` alone does not mean bookkeeping: read *what* is compared. Here the
compared names are `*D`, both sides are constants of the model and the tiling, and no
amount of sequence padding moves them. `561002` / `ERR00100` / `E89999` are shared by all
three and name no lane.

## Step 1 — the two numbers, one from the paste and one from the checkpoint

The paste already carries the caller's number (`queryD(256)`). The other side is the
model's `head_dim`: `config.json`'s `head_dim`, not `hidden_size / num_heads` — at vllm
v0.11.0 Qwen3-Next defaults `head_dim=256` with `num_attention_heads=16`,
`num_key_value_heads=2` (`vllm/transformers_utils/configs/qwen3_next.py:196-211`) and the
attention layer takes it verbatim (`vllm/model_executor/models/qwen3_next.py:640-643`).
A 256-dim full-attention layer therefore *cannot* use this tiling; the DeepSeek MLA shape
192/128 is the second clause of the same message for exactly that reason. Do not derive
the dim from a served `--max-model-len` or from hidden size.

## Step 2 — which attention state reached `_forward_v1_style`, from the paste

`_forward_v1_style` is the **last** branch of the dispatch (`attention_v1.py:651-670`):
`PrefillNoCache` → `_forward_prefill_no_cache` → unpaged `torch_npu._npu_flash_attention`
(`:325,:350`), `DecodeOnly`/`PrefillCacheHit` → their own calls, and everything else —
`ChunkedPrefill` — reaches the paged-TND FIA call at `:539-553`. The state is chosen by
`_build_attn_state` (`worker/model_runner_v1.py:1576-1600`): `PrefillNoCache` **only**
when `seq_lens == num_scheduled_tokens` for every request in the batch, i.e. each prompt
is scheduled whole. Any batch that continues a partially computed prefill (a chunked
long prompt, a mixed batch, a resumed request) falls to `ChunkedPrefill`.

That is why the paste's own disaster dump is the confirmation, and the cheapest read in
the session — `Dumping scheduler output for model execution: SchedulerOutput(...)` prints
all three fields on one line. #4135's:

```text
scheduled_new_reqs=[NewRequestData(…, prompt_token_ids_len=17, …), NewRequestData(…, prompt_token_ids_len=25, …)],
scheduled_cached_reqs=CachedRequestData(req_ids=['chatcmpl-45a8…'], resumed_from_preemption=[false],
                                        num_computed_tokens=[350]),
num_scheduled_tokens={chatcmpl-724d…: 17, chatcmpl-8795…: 25, chatcmpl-45a8…: 1}, total_num_scheduled_tokens=43
```

— one request with `num_computed_tokens=350` still taking 1 token beside two fresh
prompts: `seq_lens != num_scheduled_tokens`, hence `ChunkedPrefill`, hence this wall. The
chunk budget is `DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` (`vllm/utils/__init__.py:88`), and
the serve log's own `Chunked prefill is enabled with max_num_batched_tokens=2048` states
it; a prompt that fits one chunk is scheduled whole and never sees the op. #4135's
reporter later confirmed the control on the same image and model — short `curl` chats
answer (`Who are you?`, 12 prompt tokens) while the request that killed the engine was
the one whose batch carried a continuation.

## Step 3 — the tree's only head-size escape is 192

`_forward_v1_style` special-cases **one** size: `if self.head_size == 192:` →
`vanilla_chunked_prefill` (`attention_v1.py:503-518`, comment: the paged/splitfuse path
"maybe crash at such scenario", a TODO to delete once the kernel supports 192). Every
other head size, 256 included, goes straight to the paged-TND `npu_fused_infer_attention_score`
(`:519-553`: `block_table` + `input_layout="TND"` + `sparse_mode=3`). So there is no
in-tree fallback to find, and `--enforce-eager` cannot help — it is already on in this
report and does not change `_build_attn_state`; nothing launched, so
`ASCEND_LAUNCH_BLOCKING=1` is inert too.

## Step 4 — the toolkit is fixed by the image tag, so check that before "upgrade CANN"

The `not support Page Attention` page's rung 1 is a CANN comparison, and #4373 was
reporter-confirmed solved by moving to CANN 8.3. **For a run inside the release image that
rung is inert**, and one read says so: `rg -n '^FROM' Dockerfile*` at the pin.
vllm-ascend v0.11.0rc1 builds `FROM quay.io/ascend/cann:8.3.rc1-910b-ubuntu22.04-py3.11`
(`Dockerfile:18`, `Dockerfile.a3:18`, `Dockerfile.openEuler:18`), i.e. exactly the
`CANN >= 8.3.RC1` floor of `docs/source/installation.md:15` — the image cannot be below
it, the reporter's `docker run` mounts only the driver (`/usr/local/Ascend/driver/lib64`,
`version.info`), and `ascend-toolkit/latest/version.cfg` would only restate the base tag.
A case whose fix *was* a toolkit upgrade (#4373, reporter-confirmed at CANN 8.3) therefore
cannot have been on that image — its toolkit sat below the floor — which is why the same
advice moves that case and not this one. Ask for `version.cfg` only when the paste shows a
pip/conda install, an older image tag, or a host-mounted toolkit.

Do not read the `block_size` warnings as the cause either: `platform.py:264-280` prints
`If chunked prefill or prefix caching is enabled, block size must be set to 128.` and
`When running qwen3-next model, block_size needs to be restored to its original value.`
for **every** Qwen3-Next serve with chunked prefill on; they are a shape hint (the model
is special-cased here), not this refusal.

## Ladder

1. **`--max-num-batched-tokens` ≥ the longest prompt** (8192/16384, or whatever the
   reporter's real prompt needs), all else identical: the prompt is then scheduled whole,
   `PrefillNoCache` runs the unpaged `_npu_flash_attention` and the paged-TND call is
   never made. Label it *unblock*, and label the unpaged path's own D=256 acceptance a
   **hypothesis** — the op is a different kernel (`torch_npu._npu_flash_attention`), not
   the one that refused, and nothing in this snapshot proves its dim set. One run
   settles it.
2. Keep single requests (or at least never let a batch mix a continuation with a fresh
   prompt) if the budget cannot cover the prompts: the state gate is per batch, not per
   request.
3. If a 256-dim model must serve concurrently at that pin, it is a **toolkit/coverage
   gap**, not a serve-line defect: hand over the framework-free repro (FIA with
   `input_layout="TND"`, `block_table`, `sparse_mode=3`, `head_dim=256`) and say the
   refusal cannot be repaired from the launch line.
4. Version lane: the fix, if any, is later than this pin. #4135's thread points at a
   dev-branch PR (#4202, maintainer-suggested, **unverified** here — no PR data in this
   snapshot); do not name a released tag as fixed.

## Rules

- Classify on what the sentence compares: `*D(query/key/value)` ⇒ head dim (this page);
  `queryT` vs `actualSequenceLengthQ` ⇒ padding (`../spec-decode-mtp/fia-tnd-seqlen-mismatch.md`);
  no dims and the verb `not support` ⇒ toolkit release
  (`tiling-capability-refusal-cann-release.md`).
- Reachability is the discriminator, not the launch pair: `_forward_v1_style` is what the
  plugin means to do for chunked prefill, so the question is which batch state got there.
  Read the dumped `SchedulerOutput` before any source walk; it is one line of the paste.
- Never open the FIA/GE tiling source: `prompt_flash_attention_tiling.cpp` is in
  `cann/ops-transformer`, not in any export. One `rg` per tree, then the state gate.
- Coverage, not cause: at v0.11.0rc1 `rg -i 'qwen3_next|qwen3-next' tests/` returns
  nothing, so a green CI is no counter-evidence for this model.
- Delivery cost recorded: #4135 had this whole chain verified by its 39th message (image
  base CANN, `head_dim=256`, the `ChunkedPrefill` gate, `DEFAULT_MAX_NUM_BATCHED_TOKENS=2048`)
  and shipped **zero** message text — 20 assistant turns / 30 tool calls, 46.7 K
  characters of reasoning against 86 characters (the opening line only), `output_truncated`
  at the 16 384-token cap while checking whether the unpaged path and the e2e tree would
  support the sidestep. The sidestep is handed over as a hypothesis with its one run; it is
  never worth the answer.
