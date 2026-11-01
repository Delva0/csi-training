# Sampling path, top-k/top-p kernels, and output-quality reports

Verified in the read-only clones on 2026-11-01 at the tags named per statement. Use for three
report classes that look alike but need different first answers: a **performance/latency**
report on `ApplyTopKTopPCustom`, an **output-quality** report (repetition, loops, garbled text),
and **wrong content under MTP/speculative decoding** at a release-candidate tag, where the answer is
usually the rc→final correctness delta rather than the sampler.

## Which sampler code is even in play — check the tag first

Sampling overrides are not present in every release, so establish the surface before reading
kernels:

| tag | sampling path |
| --- | --- |
| `v0.7.1rc1` | none in the plugin (no `vllm_ascend/sample/`, no `patch_*sampler.py`): V0/V1 sampling is upstream `_multinomial`, which already uses the exponential race (`vllm v0.7.1:vllm/model_executor/layers/sampler.py:585-604`) |
| `v0.7.3rc2`–`v0.7.3.post1` | `vllm_ascend/patch/patch_rejection_sampler.py:46` (`q[start:end].exponential_(1.0, generator=...)`, spec-decode path) and at `v0.7.3` also `vllm_ascend/sample/sampler_v1.py` → `sample/ops/ascend_topk_topp_sampler.py`, which just delegates to upstream `random_sample` |
| `v0.8.5rc1` | none in the plugin: no `vllm_ascend/sample/` and no `patch_*sampler.py` in the tree; sampling is upstream vLLM (matrix pairs it with vLLM `v0.8.5.post1`) |
| `v0.9.0rc2`–`v0.9.1rc3` | `vllm_ascend/patch/worker/patch_common/patch_sampler.py` + `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` in `vllm_ascend/envs.py` |
| `v0.9.2rc1` | the env var is gone (removed by the "Clean up v0.9.1 code" commit `830332ebf`, PR #1672); a report quoting `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` is 0.9.x-era and inert here |
| `v0.18.0` … `v0.22.1rc1` | `vllm_ascend/sample/sampler.py`: `apply_top_k_top_p = _apply_top_k_top_p_ascendc` when `get_ascend_device_type() in [A2, A3]`, else `_apply_top_k_top_p_pytorch`; the AscendC side calls `torch.ops._C_ascend.npu_apply_top_k_top_p`. No env switch |
| `v0.23.0`, `v0.24.0rc1`, `v0.25.1rc1`+ | `_apply_top_k_top_p_torch_npu` (`torch_npu.npu_top_k_top_p`); the AscendC kernel is deleted from `csrc/` (#12232 merged 2026-07-23, cherry-picks #12822 → `v0.23.0`/`v0.23.0.post1`, #12823 → `v0.24.0rc1`) |
| `main` (2026-11-01) | back to `apply_top_k_top_p = _apply_top_k_top_p_pytorch`; comment at `vllm_ascend/sample/sampler.py:115-119` says `npu_top_k_top_p` "degrades severely (5-28ms) when k is large or the batch mixes k values, while sort+mask is consistently ~1ms" |

Consequence: the maintainer comment in #10821 ("replaced by `torch_npu.npu_top_k_top_p`, better
performance") describes the 0.23–0.25 line, and later code on `main` contradicts the performance
half of it. Never forward that claim as the fix for a 0.18.x report without checking the tag's
sampler and measuring. On `main` the sort-based implementation has moved to
`vllm_ascend/worker/v2/sample/apply_top_k_top_p.py` (the MRv1 `vllm_ascend.sample` package is
scheduled for deletion), and the issue closed with the reporter noting that CANN reimplemented the
operator — so the durable fix for this class may arrive through CANN/torch_npu rather than the
plugin.

## `ApplyTopKTopPCustom` slow path (v0.18.0 layout)

Files at `v0.18.0`: `csrc/apply_top_k_top_p_custom/op_host/apply_top_k_top_p_custom_tiling.cpp`
and `csrc/apply_top_k_top_p_custom/op_kernel/apply_top_k_top_p_custom.h`. The tree is later moved
and then removed; at `v0.23.0`+ it no longer exists, so grep the reported tag.

- Tiling: `onlyTopK_ = (kDimNum != 0 && pDimNum == 0) ? ONLY_TOP_K_KEY : 0`,
  `onlyTopP_ = (pDimNum != 0 && kDimNum == 0) ? ONLY_TOP_P_KEY : 0`, summed in `SetTilingKey()`;
  `ONLY_TOP_K_KEY = 1`, `ONLY_TOP_P_KEY = 2`, `K_VALUE_MAX = 1024` and `BATCH_MODE = 1` are the
  constants at `op_host/apply_top_k_top_p_custom_tiling.cpp:44-48`. **Both k and p present ⇒
  tilingKey 0 ⇒ `Process()`**, even when the top-k bound is a no-op. The kernel dispatch is
  `op_kernel/apply_top_k_top_p_custom.cpp:26/:31/:36` — key 1 runs `ProcessTopK()` on the same
  class, while key 2 constructs a **different** class, `ApplyTopPCustom` in
  `op_kernel/apply_top_p_custom.h:35` (there is no `ProcessTopP()` method on
  `ApplyTopKTopPCustom`), so a "route the no-op top-k case to tilingKey 2" proposal is a call-site
  change, not a one-line redirect.
- Locators inside `op_kernel/apply_top_k_top_p_custom.h` at `v0.18.0` (declarations `:42-62`):
  `Process()` `:243`, `InitProcess()` `:321`, `ProcessKLtKMax()` `:356`, `ScatterCumtomImpl()`
  `:395` (the forward `continue` scan plus `DataCopyPad`), `GetFirstKLoop()` `:412`,
  `ProcessRemain()` `:463`, `ScatterFromFirstKLoop()` `:482`, `ProcessTopK()` `:521`,
  `ScatterCumtomImplTopK()` `:635` (backward walk with `break`). Cite these instead of
  re-deriving them by `sed` windows.
- `Process()` has a per-row shortcut that a `top_k >= vocab` request cannot reach:
  `InitProcess()` sets `kthValue = mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue]` and the loop
  dispatches `ProcessKLtKMax` only when the loaded chunk's last element is `< kthValue`. With
  `kValue == vocabSize` the index is 0, i.e. the smallest sorted value, so every row takes
  `ProcessRemain` → `ScatterCumtomImpl`.
- `ScatterCumtomImpl` scans forward (`continue`, never `break`) and issues one
  `DataCopyPad(... {1, sizeof(outputT), ...})` — a 4-byte write to a random HBM address — between
  `SToMTE3Sync()`/`MTE3ToSSync()` per retained token. `ScatterCumtomImplTopK` (top-k-only path,
  ~line 635) walks backwards and `break`s, which is why an effective `top_k` stays in the
  millisecond range.
- The Python fallback at the same tag already treats `k == vocab` as a no-op
  (`_apply_top_k_top_p_pytorch`: `no_top_k_mask = (k == logits.shape[1])` → cutoff `-inf`); the
  AscendC tiling has no equivalent branch. That asymmetry, not "the kernel always sorts", is the
  defect to propose.
- `K_VALUE_MAX = 1024` is in `op_host/apply_top_k_top_p_custom_tiling.cpp:45`, not in the kernel
  header; the report says the k ≤ 1024 limit was removed by #6098 (2026-01) — not re-verified here.

### Workarounds for that class

1. Give every request an effective `top_k < vocab`. Reporter-measured (batch 384, vocab 151936):
   `top_k=50` ≈ 4.7 ms vs `top_k=vocab` + `top_p=0.9` ≈ 224 ms per call; one such request in a
   batch drags the whole batch. Reported numbers, not re-run here; it changes the candidate space,
   so it needs an A/B quality check.
2. In-process, bind the sort-based implementation instead:
   `AscendTopKTopPSampler.__init__` does `self.apply_top_k_top_p = apply_top_k_top_p` from the
   module global, so patching `vllm_ascend.sample.sampler.apply_top_k_top_p` (or the device-type
   dispatch) before the sampler is built removes the custom op. Candidate — not executed here;
   measure per-call latency and logprobs compatibility.
3. Upgrade to a tag carrying #12232 (`v0.23.0`, `v0.24.0rc1`, `v0.25.1rc1`), with the caveat above.

## Output-quality reports: repetition, loops, garbled text

When the only discriminator is "garbled with graph capture, clean without it" (and especially with
DP + a sparse/DSA attention model), go to the dummy-run slot-invalidation class in
[glm-moe-accuracy.md](glm-moe-accuracy.md) first: that is a KV-correctness defect and no sampling
parameter will explain it.

- **A reply made of one repeated punctuation or marker token from the very first token, on
  several releases at once, is checkpoint integrity, not the sampler.** The recorded #8180 report
  ("Qwen3-32B on 4x910B4, replies always exclamation marks, on 0.13.0, 0.15.0rc1, 0.10.0rc1 and
  0.9.2rc1") was resolved by the reporter himself: the model files were corrupted. A symptom that
  is identical across four releases, one of them a working baseline on the same hardware, is not a
  release-specific plugin defect; and a garbage token from position 1 is not a
  `temperature`/`top_p` effect. Ask for per-file sizes/hashes against the published checkpoint list
  (a truncated download reproduces this exactly), the safetensors load log and the first response's
  `token_ids`/`finish_reason`, and re-download before reading any source — the maintainer's own
  first reply in that thread asked for HDK, model weights and the serve command, in that order.
  Keep the report's *other* symptom separate: the same thread also described the first request
  after ~5 idle minutes failing with `curl: Recv failure: connection reset` while the second
  succeeded, a front-end/idle-timeout question and no evidence for the corruption. Garbage that
  differs *by DP rank* rather than appearing on every rank is a different class — see
  [moe-dp-ep-accuracy.md](moe-dp-ep-accuracy.md).
- **A reply that starts correctly and then collapses into one repeated punctuation token, only at
  raised concurrency, on a single release line, is a runtime defect — not checkpoint integrity and
  not a sampling parameter.** Recorded shape (#9666: `v0.17.0rc1`, Qwen3.5-9B, A2, serve line
  `--data-parallel-size 1 --tensor-parallel-size 1` while `npu-smi` lists four `VLLMWorker_TP`
  processes, so the effective sharding is not settled by the paste; `--max-num-seqs 8`,
  `cudagraph_mode: FULL_DECODE_ONLY`, `--async-scheduling`, no `--speculative-config`, a repro that
  posts 100 requests over 30 threads): the body begins `{\n"intentName` and then repeats `!` to
  `max_tokens` (`finish_reason: length`), occasionally, under both
  `temperature 0.01 / top_p 0.01` and `0.5 / 0.7`. The reporter's own follow-up states `v0.18.0` no
  longer reproduces it, so the finding is the [`v0.17.0rc1`, `v0.18.0`] boundary plus "upgrade" as
  the workaround, labelled reporter-confirmed. That boundary spans both trees (the plugin tag pairs
  with upstream `v0.18.0`), so do not attribute it to plugin code alone. In-range commits are candidates only — sampler-kernel
  mask fix #8531 (`696dcc926`), PIECEWISE replay sync #8469 (`785026432`, which cannot explain a
  `FULL_DECODE_ONLY` run because FULL already synced) and short-prompt attention state #8088
  (`054fde7b7`) — and a 276-commit range grep is not evidence of a root cause. Ask for the same
  prompt at `--max-num-seqs 1` and with the graph off: reproducing single-stream or eager moves it
  to the decoding/greedy classes below.

1. **Read the request's sampling parameters before any source.** The `SamplingParams(...)` line or
   chat-completions body in the report is the primary evidence. A Qwen3 thinking model served with
   `temperature=0.0`, `top_k=-1`, `top_p=1.0`, `repetition_penalty=1.0` (the #957 shape) is on
   greedy decoding, which the Qwen3 model documentation warns against. The 0.6/0.95/40 triple does
   sit in the tag's `docs/source/tutorials/multi_npu.md`, but that page is `# Multi-NPU (QwQ 32B)` —
   every command serves `Qwen/QwQ-32B` (`:45`, `:87`, `SamplingParams` at `:86`). `v0.8.5rc1` ships
   **no** Qwen3 tutorial, so cite those values as QwQ-32B's documented sampling, not as a Qwen3
   recipe.
2. **Read a release-note line's `## vX.Y.Z` section before citing it as "documented at this tag".**
   The "Improved and reduced the garbled code in model output. But if you still hit the issue, try to
   change the generation config value, such as `temperature`" line is `release_notes.md:117` at
   `v0.8.5rc1` — inside the `## v0.7.3rc1` section (`:91`, next header `## v0.7.1rc1` at `:119`).
   The tag's own `## v0.8.5rc1` section (`:3-25`) carries Highlights/Core/Other and **no** Known
   Issues at all, so at this tag the hint is an older release's generic sampling advice, not a
   0.8.5 statement. Label it with the section it sits in; never with the published `latest` site.
3. **Thread evidence is a hypothesis until re-run.** #957's only substantive reply (not a
   maintainer fix) advises not using temperature 0 for Qwen3; the issue then closed stale with no
   root cause. State it as "reporter/user claim + model guidance", and the confirming check is the
   same prompt at the model's documented sampling.
4. **Eliminate the plugin's sampling path by tree inspection.** At `v0.8.5rc1` the plugin ships no
   sampler override, so a repetition report there cannot be a plugin-sampler defect; the plugin's
   output-affecting overrides are `vllm_ascend/ops/{layernorm,activation,rotary_embedding,
   common_fused_moe}.py` plus attention. That single `git ls-tree` check replaces a speculative
   kernel hunt. The tag also has no `vllm_ascend/models/qwen3_moe.py`: the plugin's
   `CustomQwen3MoeForCausalLM` (plus its `packed_modules_mapping` tweak for W8A8) lands in
   `17f05b108` (#925), first tag `v0.9.0rc1`, so a Qwen3-235B-A22B report on `v0.8.5rc1` runs the
   upstream generic MoE class and has no plugin MoE model to blame. Even where a later tag adds
   `CustomQwen3MoeForCausalLM`, that class is a thin subclass and its expert blocks are
   upstream's, whose patched `forward_oot` is the subject of
   [moe-dp-ep-accuracy.md](moe-dp-ep-accuracy.md). For a Qwen3-VL/Qwen3.5-VL-derived
   checkpoint the attention override is `vllm_ascend/patch/worker/patch_qwen3vl.py`: at `v0.19.1rc1` it rebinds
   `Qwen3Attention.forward` / `Qwen3MoeAttention.forward` (lines 70-71) to a fused
   `torch.ops.vllm.triton_split_qkv_rmsnorm_mrope` path driven by `mrope_section` /
   `mrope_interleaved` / `rotary_dim`, and TP-chunks
   `Qwen3VLForConditionalGeneration._get_deepstack_input_embeds` (lines 72-73); the
   `pos_embed_interpolate` half is gated on `not vllm_version_is("0.19.1")`. A derived model whose
   attention class overrides `forward` itself, or does not inherit those two upstream classes,
   silently runs the unfused path — so before attributing a vLLM-vs-HF mismatch to the checkpoint
   adaptation, check that the patch's assignment targets are the classes the derived model actually
   instantiates. Candidate check; no report in the recorded batch confirmed it either way.
5. Ask for/collect `finish_reason` (length vs stop), `enable_thinking`, and whether the loop starts
   after a fixed token count — a repetition that continues to `max_tokens` *and* reproduces
   single-stream at `temperature 0` or on every release is decoding behaviour, not decode corruption
   of the KV cache. Running to `max_tokens` alone does not settle it: the concurrency-only,
   one-release case in the bullet above also finished on `length`. A "garbled only with thinking
   disabled" title does not establish that `enable_thinking` is the variable: the recorded
   Qwen3.5-122B-A10B retest (`#10390`, `v0.18.0`, TP8 + `--enable-expert-parallel`,
   `--async-scheduling`, `qwen3_5_mtp` x3, `FULL_DECODE_ONLY`,
   `--default-chat-template-kwargs '{"enable_thinking":false}'`) that claimed the bug
   "persisted" on `v0.20.2rc1` ran `temperature: 0.0` and emitted
   `NH2, NH2, NH2, ...` — the greedy-loop class above. Read the retest's request body before the
   flag. The one tutorial advisory against `enable_thinking: false` is GLM-5.3(-Flash)'s
   (`1a4d4e6bb`, #16589, first tag `v0.28.0.rc1`; at `v0.30.0` the surviving file is
   `docs/source/tutorials/models/GLM5.3-Flash.md:187`, "Do not set `enable_thinking: false` / `thinking:
   false` ... otherwise the output quality may degrade", plus FAQ `:1901-1903`) and it is about a model
   that always reasons. No tutorial at `v0.18.0`, `v0.19.1rc1` or `v0.20.2rc1` mentions
   `enable_thinking` at all (`git grep enable_thinking <tag> -- docs/source/tutorials` is empty), so
   do not forward that mechanism for a Qwen3.5 report.
6. **Check the paste before the hypothesis.** When the reporter's own flag names and JSON keys are
   mangled by look-alike characters (`"type"` pasted as `"tуре"`, `"image_url"` as `"iлаgе uг1"`,
   `"assistant"` as `"assis Lant"`), the text went through a transliteration or OCR hop, so the
   *character set* of the quoted model output (mixed Cyrillic/Devanagari/CJK, `㏑` where `\n` should
   be) is not evidence of a broken tokenizer, wrong-vocab sampling or a corrupted detokenizer. Judge
   the model from a structured response body, usage/`finish_reason` counters and a retest with
   explicit parameters; character-level "garbling" analysis of a mangled paste is a detour.

### Repetition with `--async-scheduling` + MTP before `v0.19.1rc1`: the missing `num_computed_tokens` correction

Report shape (`vllm-project/vllm-ascend#9639`): `v0.18.0`, GLM-4.7-w8a8-with-float-MTP, 1P1D PD
with `MooncakeConnectorV1` (`use_ascend_direct: true`) inside a `MultiConnector` plus
`AscendStoreConnector`, `--async-scheduling`, `num_speculative_tokens: 3`,
`cudagraph_mode: FULL_DECODE_ONLY`, `--max-model-len 131072`; the reporter's own follow-up is
"short inputs are fine, once the context is long it very likely repeats". The thread carries one
automated reply that calls long-context repetition "a known PD + MTP pattern" and offers "disable
MTP or shorten the context" — stated in the thread, no mechanism, so it cannot be the finding.
Both launch scripts `export VLLM_VERSION=0.14.1`, which contradicts `collect_env.py`'s vLLM 0.18.0 /
vllm-ascend 0.18.0; that export is stale launcher noise, so pin from `collect_env.py` and check
what it would have flipped in
[env-vars-and-additional-config.md](env-vars-and-additional-config.md).

Verified in the plugin tree (`git grep -n async_scheduling <tag> -- vllm_ascend/platform.py`):

- **`v0.18.0` has no async/spec-decode gate at all** — the grep is empty, so `--async-scheduling`
  with an MTP draft runs as configured.
- `811271d12` (#7787) later adds one whose own comment names the mechanism: "Ascend does not
  implement the GPU-side num_computed_tokens correction
  (`update_num_computed_tokens_for_batch_change`) required for async spec decode, which causes
  accuracy divergence".
- `34f2ed0e0` (#8461, "fix spec decode and async bug") deletes that gate and instead records an
  async NPU→CPU `seq_lens` event in `worker/model_runner_v1.py`, gated on the attention backend
  (`AscendAttentionBackend`/`AscendMLABackend` only — SFA/GDN backends are skipped), citing
  upstream vLLM #38556.
- **Both commits first appear in `v0.19.1rc1`** (`git tag --contains`), so no release tag ever
  shipped the "disable async scheduling" behaviour: pre-`v0.19.1rc1` tags run the combination that
  the plugin itself describes as divergence-prone.

Triage: make that the first candidate for a long-context repetition/divergence report on
`--async-scheduling` + MTP, with the same prompt at `--no-async-scheduling` (or without the
speculative config) as the separating check, and `v0.19.1rc1` as the fix boundary — label the causal
link a candidate, since #9639's reporter never ran the A/B.

Neighbours to keep apart:

- `5dba75bad` (#8540, first tag `v0.19.1rc1`, "[P/D] In scenarios where TP is not equal, the KV cache
  at the MTP layer is not handled") is the Mooncake connector dropping the MTP layer's KV when P and
  D shard TP differently. #9639's `kv_connector_extra_config` declares `tp_size: 8`/`dp_size: 2`
  for both the `prefill` and the `decode` side, so that precondition is absent — ask before
  invoking it.
- A block-aligned first anomaly (`(prompt+prefix) % block_size == block_size - 1`) under a sliding
  window is the KV-lifetime free in [pd-disaggregation.md](pd-disaggregation.md), not this
  async+MTP divergence; without a phase from the thread, leave repetition a candidate here rather
  than borrowing either mechanism.
- The tag's own notes do **not** document this report: `v0.18.0:docs/source/user_guide/release_notes.md`
  `### Known Issue` (`:112-123`) names GLM5/DeepSeek-V3.2 PD empty-output-or-garbled (#8853, `:121`)
  and GLM 5/5.1 PD `TP16 DP2` GPQA (#8844, `:122`), not GLM-4.7. The same tag's
  `docs/source/tutorials/models/GLM4.x.md` does carry a multi-node GLM-4.x recipe for 2 × A2
  (`:176`, quoting the older "not recommended to deploy multi-node on Atlas 800 A2 (64G × 8)"),
  so multi-node GLM-4.x is documented-and-caveated rather than unsupported.

## Wrong or mixed-language content under MTP/spec decode: check the rc→final delta

A report that pastes no request parameters at all ("the curl answer is not right, the content mixes
Chinese and English", Qwen3.5-MoE, PD separated) is not the sampling class above — there is nothing to
read, so go to the thread and then to what changed between the reported rc and its final release.

**First rule out the decode-side KV-lifetime race, which wears the same symptom.** If the report is a
PD deployment with `--async-scheduling` and a sliding window, and the first anomalous token sits at a
fixed block boundary (`(prompt+prefix) % block_size == block_size - 1`), it is the premature
sliding-window free, not the sampler — the mechanism, the fix tags and the sha-vs-tag trap are in
[pd-disaggregation.md](pd-disaggregation.md) (`#13518`/`42a70b7ba` in `v0.23.0`; upstream `#47728`).
Everything below assumes no such alignment.

`v0.18.0rc1` → `v0.18.0` (108 commits) carries two decode-path correctness fixes, both verified by
`git grep` at the tags:

| commit / PR | what it changes | rc1 | v0.18.0 |
| --- | --- | --- | --- |
| `9a0b786f2` "Fix race in non-blocking num_accepted_tokens" (#8764, cherry-pick on `releases/v0.18.0`) | adds `num_accepted_tokens_event.synchronize()` before the accepted-token counts are copied into the runner's buffer | absent | `vllm_ascend/worker/model_runner_v1.py:2044-2045` |
| `696dcc926` "fix kernels in sample when mask is not static or draft_token_id is invalid" (#8531) | replaces `tl.load(ptr + offsets - 1, mask)` with `tl.load(ptr + tl.maximum(offsets - 1, 0), mask)` in `vllm_ascend/ops/triton/reject_sample.py` and `ops/triton/spec_decode/utils.py` | `reject_sample.py:164`, `:390` read `offsets - 1` directly | `:165`, `:393` use `tl.maximum(...)` |

Mechanism (from the code, not re-run): the accepted-token counts come from a non-blocking
device→pinned copy and async scheduling deliberately skips the sync, so the CPU-side numpy view can be
read stale in `_build_attention_metadata`; and the Triton kernels evaluate the `-1` pointer before
applying a non-static mask, which the compiler is free to order as a real negative-offset access.
Either way the wrong draft-token accounting lands in decode, which is how "correct first answer, then
garbage/mixed language" arises.

How the recorded thread reaches this point: the report that pasted nothing pointed at a sibling
issue, whose reporter had cherry-picked #8531/#7904/#8764 into `0.18.0rc1` and reported the hang
fixed while an accuracy regression remained in a further issue. So for this class: follow the thread
chain, name the rc→final set, say which commit is absent at the reported tag, and label the rest
(candidate fix vs co-variate).

Boundary: which of the two fixes explains a given report needs a runtime check (acceptance metrics,
per-request behaviour with spec decode off); a maintainer naming a main-only PR alongside a
cherry-pick describes an upgrade path, not the reported build — see the ancestry rule in
`SKILL.md` step 6.

### When the reported rc is the newest tag of its line

A v0.20.2rc1 report has no final to compare against: `git tag --list 'v0.20.2*'` returns only
`v0.20.2rc1` (2026-11-01), and the line keeps receiving cherry-picks on
`origin/releases/v0.20.2rc`. That branch range *is* the rc→final delta —
`git log --oneline v0.20.2rc1..origin/releases/v0.20.2rc` is 25 commits — but read all the titles:
a `--grep` filter for the symptom's subsystem returned only `7cb5d6180` and `42984bfe9` and missed
`91e6b539d`, a second MTP-placeholder fix under a title with no decode keyword. A thread with zero
`comments` (vllm-ascend#10405: closed, no maintainer reply) names no fix at all, so the branch list is
the only usable answer.

### The MTP `-1` placeholder decode defect (PD decode node only)

Verified at `v0.20.2rc1`, fixed after the tag, not reproduced:

- Producer: `vllm_ascend/core/recompute_scheduler.py:152` sets
  `request.spec_token_ids = [PLACEHOLDER_TOKEN_ID] * self.num_spec_tokens` only when
  `is_mtp_kv_consumer` and `max_model_len >= num_tokens + num_spec_tokens`. `PLACEHOLDER_TOKEN_ID` is
  `-1` (upstream `v0.20.2`: `vllm/v1/sample/rejection_sampler.py:30 PLACEHOLDER_TOKEN_ID: tl.constexpr = -1`),
  so the defect needs a **PD decode (KV-consumer)
  node running MTP on the recompute scheduler** (`--additional-config
  '{"recompute_scheduler_enable":true,...}'`, which the PD tutorials set). A single-node MTP report —
  no `kv_transfer_config` — never creates the placeholder, so this is not its mechanism.
- Consumer at the tag: `vllm_ascend/sample/rejection_sampler.py:874`, `:882`, `:899` and `:1167`,
  `:1175`, `:1192` index `draft_probs`/`target_probs` with `draft_tokens.flatten()` directly, so `-1`
  reads the last vocabulary row (Python negative indexing) instead of being rejected.
- Fixes, both on `origin/releases/v0.20.2rc` only (`git tag --contains` empty): `7cb5d6180` (#10259,
  backport of #10043) masks placeholders to 0 before indexing and rejects placeholder drafts
  (`placeholder_mask = draft_tokens == PLACEHOLDER_TOKEN_ID`, branch tip `:868`/`:1166`); `91e6b539d`
  (#10308) also sanitizes the forward buffer, `_sanitize_placeholder_input_ids_for_forward` →
  `input_ids.masked_fill_(input_ids == PLACEHOLDER_TOKEN_ID, 0)` at `model_runner_v1.py:1311-1317`,
  because embedding kernels reject `-1`.
- Same producer, different consumer on the 0.13 line: the grammar path (`grammar_bitmask`) asserts on
  those `-1` spec tokens and kills the EngineCore when structured output is on; upstream stops
  advancing at `-1` from `v0.14.0rc1`
  ([structured-output-guidance.md](structured-output-guidance.md)).
- Symptom fit: sampling from the wrong probability row accepts a token the target model did not
  choose, which reads as fluent-but-wrong or mixed-language output on the D node's first decode
  steps. Read from the diffs only — the confirming check is per-request behaviour on the D node with
  MTP off, and the workaround is the branch cherry-pick, since no released tag contains either fix.

## Release-to-release output regression with `temperature > 0`: the torch_npu `exponential_` build (0.7.x)

Same model, same flags, two vllm-ascend builds where the older one reads coherently and the newer
one emits token soup (random URLs, half-words, switching languages — the recorded run produced
strings like "faucwu.com" and "SetProperty(...)") under `temperature=1, top_p=1`: that is a
sampler-RNG defect, not the model forward and not the KV cache. The cheapest differential to ask
for is greedy vs `temperature > 0` on the same prompt set, because greedy decoding exercises no
RNG at all.

- Mechanism: both upstream samplers sample through the exponential race.
  `vllm v0.7.1:vllm/model_executor/layers/sampler.py:585-604` and
  `vllm v0.7.3:...:517-536` are the same `_multinomial` body (`q = torch.empty_like(probs)`,
  `q.exponential_()` or `q[...].exponential_(generator=...)`, then `probs.div_(q).argmax(dim=1)`),
  and `random_sample` uses the same trick. Any error in the `exponential_` kernel therefore biases
  every sampled token. **The vLLM code did not change between those two versions**, so the
  differentiator is the torch-npu build, not the plugin's sampling surface.
- Verified boundary: `vllm-ascend#221` (Qwen2.5-7B-Instruct, 0.7.1 vs 0.7.3). The reporter found it
  first ("怀疑torch npu的exponential_算子精度有问题。采样阶段规避了torch_npu的exponential_算子后精度正常");
  maintainer ganyi1996ppo confirmed the same op and reported it to torch_npu; Yikun closed it as
  fixed by "[CI/Build] Bump torch_npu to dev20250307.3" — PR #265 (main, merged 2025-03-07) and #266
  (v0.7.3-dev), whose body reads "Update torch-npu version to fix torch npu exponential_ accuracy.
  With this update, the percision issue when setting `temperature > 0` is fixed." The diff moves
  `torch_npu-2.5.1.dev20250226` → `2.5.1.dev20250307` in `pta_install.sh`, the CI workflow and the
  docs (`torch-npu >= 2.5.1.dev20250307`).
- Statement to make: the defect is in the **`dev20250226`-era nightly** and fixed from
  `2.5.1.dev20250307`; a report is this class only if the installed `torch_npu.__version__` predates
  the fix. The pin is not in `requirements.txt` for most 0.7.x tags — read `pta_install.sh` at the
  tag: `v0.7.1rc1` installs `dev20250218`, `v0.7.3rc1` `dev20250308`, `v0.7.3rc2` `dev20250320`,
  and `v0.7.3` final drops the script for `torch-npu==2.5.1` in `requirements.txt` (a release wheel
  whose kernel was not inspected here — collect `torch_npu.__version__` in-container rather than
  inferring from the tag).
- Unconfirmed alternative, mention only as an experiment: the maintainer's first suggestion in the
  same thread was to replace `npu_add_rms_norm` with `npu_rms_norm` + `torch.add`
  (`vllm_ascend/ops/layernorm.py:32`). Nothing in the thread ties it to the resolution; the
  discriminating test is greedy vs `temperature > 0`, or swapping the torch-npu build.
- Do not confuse this with the `random_sample()` storage-lifetime race documented below: same call
  site and same operator family, but a different era (v0.20+) and a different symptom (masked/OOV
  ids winning `argmax`), with a `record_stream` fix.

## `<pad>` / id 0 / garbage ids only when `temperature > 0`: the side-stream `q` in `random_sample()`

A sampled run that intermittently emits token id 0, an out-of-vocab id, or short runs of repeated
garbage while **greedy decoding is byte-identical over tens of thousands of tokens** isolates the
sampler, not the model forward and not the KV cache. That differential (greedy clean, sampled dirty,
rate rising with concurrency) is the reporter's in #18754 and is the cheapest thing to ask for.

Boundary: greedy id-0 fill does occur in a different class — a **first call only**, all-100-token
`token_ids: [0, 0, ...]` at `temperature=0` on `enable_sleep_mode=True` (recorded on 0.18.0/0.23.0/
0.26.0rc1, clean on the second call). The greedy/sampled differential only isolates this sampler when
the corruption recurs across calls; a first-call-only run belongs to
[first-request-only-output.md](first-request-only-output.md), and its `q.record_stream` lifetime is
not the mechanism there.

Verified at `v0.20.2rc1`, `vllm_ascend/sample/sampler.py`:

- `random_sample()` (`:18-41`) allocates `q = torch.empty_like(probs)` inside
  `npu_stream_switch(global_stream())`, fills it with `q.exponential_()` (and per-request
  `q[i].exponential_(generator=...)` for seeded requests), leaves the switch with only
  `torch.npu.current_stream().wait_stream(global_stream())` (`:40`), then consumes it in
  `probs.div_(q).argmax(dim=-1)` (`:41`). Stream-order only: no event, and no `record_stream`
  declaring the buffer's lifetime on the consumer stream.
- `enable_async_exponential` (`--additional-config`, default `False`, and forced off under
  `VLLM_BATCH_INVARIANT` — `ascend_config.py:230-232`) replaces that path: `do_async_exponential()`
  (`:79-92`) pre-generates `q` on the side stream before the forward, records
  `self.async_exponential_event`, and `forward_native` does `self.async_event.synchronize()` before
  `probs.div_(self.q)` (`:160-163`); the pre-generation is invoked from
  `worker/model_runner_v1.py:1970-1974`. Introduced by `5d02eed16` (#4501), first tag `v0.13.0rc1`.
- The maintainer fix for the default path is a lifetime declaration, not an event:
  `fc0ce85b5` ("Fix inf in Qwen30b RL sampling by recording stream for q", #13394, 2026-08-06) adds
  `q.record_stream(torch.npu.current_stream())`. Its PR body's mechanism is the allocator reusing
  `q`'s storage while the consumer still reads it, made likelier because Ascend `exponential_()` is
  a multi-kernel sequence — and its symptom is a masked/OOV id winning `argmax` with `-inf` logprob,
  which is exactly the "garbage id" family, not a numeric drift.

### Release boundary — grep the line per tag, not `tag --contains`

`git tag --contains fc0ce85b5` answers `v0.27.1rc1`, but the line is already in earlier trees via
cherry-picks, so a `tag --contains` answer overstates the gap:

```sh
for t in v0.25.1rc1 v0.26.0rc1 v0.27.1rc1; do printf "%s: " $t; \
  git show $t:vllm_ascend/sample/sampler.py | grep -c record_stream; done
```

`v0.20.2rc1`, `v0.23.0` and `v0.25.1rc1` have none; `v0.26.0rc1` and later have
`q.record_stream(...)` (backport `255954399`, #14687 on the v0.26.0rc branch), the original
`fc0ce85b5` first lands in `v0.27.1rc1`, and `2c905d735` (#19461) carries it on `origin/releases/v0.23.0`
without any tag containing that commit. So the statement to make is "absent through `v0.25.1rc1`,
present from `v0.26.0rc1`", which is not what the ancestry command says.

### What stays a candidate

The reporter's own attribution — `wait_stream`-only being insufficient under aclgraph replay plus
async scheduling — is a code fact for the `wait_stream` part, but which of the two mechanisms
(missing event vs missing storage lifetime) produces a given corruption is not established here; the
maintainer fix addresses the lifetime, so prefer it. The applicability tests are behavioural, and
they are cheap to name in a reply: `--additional-config '{"enable_async_exponential": true}'` (the
event-synced path, and the isolating test because it changes only the `q` generation;
reporter-measured at parity throughput), `--no-async-scheduling`, and `--enforce-eager` (both also
change scheduling or graph mode, so they confirm the region rather than the mechanism). None of the
three was run here.

## Next checks that would settle the open points

- Per-call latency of `npu_apply_top_k_top_p` vs the sort+mask fallback on the reported build
  (profiler table or PMU MTE3 ratio), at the batch/vocab the reporter used.
- Precision/logprobs parity between the three implementations before recommending a swap.
- For repetition: same prompt and seed at the documented sampling; if it still loops, compare the
  plugin `ops/` numerics (`layernorm`, `rotary_embedding`) against a CPU/torch reference.
