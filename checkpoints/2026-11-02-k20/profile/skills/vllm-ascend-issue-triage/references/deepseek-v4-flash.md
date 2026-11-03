# DeepSeek-V4-Flash: introduction points, documented recipes, MTP/DSA graph capture

Applies to DeepSeek-V4 / V4-Flash / V4-Pro reports (`--tokenizer-mode deepseek_v4`,
`vllm_ascend/models/deepseek_v4.py`, quantized `vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp`) and, with
the package and parser differences in the V4.1 section below, to V4.1-Flash reports.
The model uses Compress-4/Compress-128 sparse attention (DSA) plus MTP speculative decoding, so
many "startup crash / hang / wrong output" reports land in the DSA metadata contract and in the
documented flag cluster, not in a kernel.

A DSpark/V4-Flash **decode freeze** (`Running > 0`, zero generation throughput, `/health` 200) with
speculative decoding on has its own candidate and log hook in
[pd-disaggregation.md](pd-disaggregation.md) (all-`-1` recovered tokens -> empty `parse_output`);
read that before tracing DSA/MoE kernels.

## Which trees contain the model

- Main line: `7bce23cc0 [Feature] Support DeepseekV4 (#9270)` adds
  `vllm_ascend/models/deepseek_v4.py`; first tag containing it is `v0.20.2rc1` (2026-06-03), then
  every later tag (`git tag --contains 7bce23cc0`).
- The v0.13.0 release line got a separate initial-support commit
  `bbe9bc0d9 [v0.13.0][Feature] Add DeepSeek v4 initial support (#8648)`; only `v0.13.0rc3`
  (2026-04-24) contains it. That tree has `models/deepseek_v4.py`, `deepseek_v4_mtp.py`, a
  `deepseek_v4` config patch, and the tutorial at **`docs/source/tutorials/DeepSeek-V4.md`**
  (flat path; the `tutorials/models/DeepSeek-V4-Flash.md` layout only appears later). Its A3
  image tag is `quay.io/ascend/vllm-ascend:v0.13.0rc3-a3` and the doc says the model "is
  integrated in image v0.13.0rc3".
- Consequence for reports: `v0.18.0` (2026-04-30) and `v0.19.1rc1` have **no**
  `vllm_ascend/models/deepseek_v4.py` (`git ls-tree -r --name-only <tag> | grep deepseek_v4` →
  nothing), so a DeepSeek-V4 serve line claiming 0.18.0/0.19.x can only come from a floating or
  preview image. In one report the tag was `quay.io/ascend/vllm-ascend:deepseekv4-a3` with a
  claimed "v0.18.0"; the thread's only remedy comment named the `v0.21.0rc1-a3` image as working,
  and another commenter saw no hang in the pasted log at all. Say which tree you grepped and that
  the exact revision is unidentifiable — do not answer "the model did not exist
  in that release", because a backport line can carry it.
- Version numbers of the two introduction points are not ordered by date: the 0.13.0 branch was
  cut after 0.18/0.19 (see [revision-pinning.md](revision-pinning.md)).

## V4.1-Flash is a separate package, not a V4 config (v0.29-v0.30 line)

- `vllm_ascend/models/deepseek_v41/` holds the model (17 files at `v0.30.0rc1`/`v0.30.0rc2`, 15 at
  `v0.29.0.rc1`, none at `v0.28.0.rc1`); `git ls-tree -r --name-only <tag> | grep -c
  'models/deepseek_v41/'` answers whether a tag carries it. Attention is `attention/dsa_v41.py` plus
  `attention/context_parallel/dsa_v41_cp.py`, and it ships Engram (`models/deepseek_v41/engram/`) and
  DSpark (`dspark.py`) rather than `deepseek_v4_mtp.py`, so a V4 frame or file path quoted from this
  family does not transfer.
- Serve flags differ: `--tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice`, on **model runner V1 only** (the
  documented scripts export `VLLM_USE_V2_MODEL_RUNNER=0`). A V4.1 report reusing `deepseek_v4`
  parsers is on the wrong surface; the parsing class itself is unchanged —
  [serving-toolcall-reasoning-streaming.md](serving-toolcall-reasoning-streaming.md).
- The tutorial predates the package (`7a5b24735`, 2026-09-11) and its support sentence moved
  between copies (`v0.30.0rc2` requires `main` plus a dated upstream vLLM revision; `main` says
  "v0.30.0rc1 and later"). Its A2 tabs also prescribe `--max-model-len 1048576` while §11 says the
  A2 configuration "is not revalidated". Quote the revision you read; the tab matrix, the
  890 bytes/token KV figure, the sibling recipes and the `VLLM_ENGINE_READY_TIMEOUT_S=360000`
  restart illusion are in [model-tutorial-recipes.md](model-tutorial-recipes.md).

## Documented serve lines (v0.21.0rc1 and v0.23.0rc1, `docs/source/tutorials/models/DeepSeek-V4-Flash.md`)

- A2 (64G × 8): TP=8, DP=1, `--max-model-len 133120`, `--max-num-batched-tokens 8192`,
  `--max-num-seqs 32`, `--no-enable-prefix-caching`, `--enable-expert-parallel`,
  `--quantization ascend`, `--additional-config` with `enable_dsa_cp: true` and
  `multistream_overlap_shared_expert: true`; env `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`,
  `HCCL_BUFFSIZE=1024`, `HCCL_OP_EXPANSION_MODE="AIV"`, jemalloc preload.
- A3 (128G × 8 / "Atlas 800 A3"): DP=4, TP=4, `--max-model-len 1048576`,
  `--max-num-batched-tokens 10240`, `--max-num-seqs 64`, `--api-server-count 1`,
  `enable_cpu_binding`, no `enable_dsa_cp`, `multistream_overlap_shared_expert: true`.
- **Both recipes put `"enforce_eager": true` inside `--speculative-config`**
  (`'{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}'`). A report using
  `method: mtp` / `deepseek_mtp` without it is off the documented path and enters the DSv4 MTP
  graph-capture path below. `method: deepseek_mtp` only logs
  "method `deepseek_mtp` is deprecated and replaced with mtp" — cosmetic, not the failure.
- FLASHCOMM1=1 is documented in both tabs. `VLLM_ASCEND_ENABLE_FUSED_MC2=1` was in the
  v0.13.0rc3 A3 doc, is absent from the v0.21/v0.23 tutorial, and the A3 nightly config
  `tests/e2e/nightly/single_node/models/configs/DeepSeek-V4-Flash-W8A8-A3.yaml` (already present
  at `v0.20.2rc1`) pins `VLLM_ASCEND_ENABLE_FUSED_MC2: "0"`. That same config runs **DP=2/TP=8**
  while the tutorial's A3 tab runs DP=4/TP=4 — compare a report against the shape that matches
  it, and treat FUSED_MC2=1 on a DP4/TP4 run as a named delta and candidate only (the older doc
  recommended it).
- One recorded A3 run served traffic and then hung with "requests never return": DP=4/TP=4,
  `--max_model_len 1024000`, `VLLM_ASCEND_ENABLE_FUSED_MC2=1`, `multistream_dsa_preprocess:
  false`, and `"method": "deepseek_mtp"` **without** `enforce_eager`; its metrics kept printing
  `Running: 1 reqs` with non-zero generation throughput while clients got nothing, and the thread
  closed stale with no cause (a later comment named the `v0.21.0rc1-a3` image as working). Against
  the documented A3 tab that command differs in three flags (enforce_eager omitted, FUSED_MC2 on,
  1024000 vs 1048576) — the first hypotheses, all unverified; this symptom is steady-state, not
  the startup assert above, so do not merge the two write-ups.
- CI coverage boundary at `v0.20.2rc1`: the nightly A3 config above runs
  `--speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}'`
  (eager drafter, so it does not exercise the graph-capture path), while the PR-level
  `tests/e2e/multicard/4-cards/test_deepseek_v4.py` serves a `-mtp` checkpoint with **no**
  `--speculative-config` at all. A green CI therefore says nothing about MTP + `FULL_DECODE_ONLY`
  graph capture.
- Opt-in patch delta at `v0.20.2rc1`: the nightly A3 config also exports
  `VLLM_ASCEND_APPLY_DSV4_PATCH: "1"`, which gates
  `vllm_ascend/patch/platform/{patch_speculative_config,patch_kv_cache_coordinator}` imports
  (`vllm_ascend/envs.py:116`, `patch/platform/__init__.py:46`, default **0** — an omitted value
  runs unpatched). `patch_speculative_config` rewrites a `deepseek_v4` draft config to
  architecture `DeepSeekV4MTPModel`. By `v0.21.0rc1` both imports are unconditional and the env
  var is gone. A v0.20.2rc1 DSv4 MTP report that never sets it is on the unpatched path: name the
  delta, but do not present it as the cause of a separately proven failure.
- The same gate exists earlier under a different name: at `v0.13.0rc3`
  `patch/platform/__init__.py` imports `patch_kv_cache_coordinator`, `patch_kv_cache_utils`,
  `patch_model_config`, `patch_speculative_config`, `patch_transformers` and `patch_vllm_config`
  only inside `if envs.USE_MULTI_BLOCK_POOL:` — `USE_MULTI_BLOCK_POOL` is
  `bool(int(os.getenv("USE_MULTI_BLOCK_POOL", '0')))` (`vllm_ascend/envs.py:143-144`, default 0), and
  `patch_speculative_config.hf_config_override` is what rewrites a `deepseek_v4` draft config to
  `DeepSeekV4MTPModel`. Every recipe in that tag's own `docs/source/tutorials/DeepSeek-V4.md`
  exports it before `vllm serve` (lines 142, 181, 345, 416, 487). The tutorial has **no offline
  example** at that tag (its only `offline` mention is the benchmarks command at line 628), so an
  offline script copied out of the serve tab silently drops the variable and runs the
  MTP/DSv4 patch set unimported — a candidate cause for an offline-only failure, not a proven one.
- **"Online serving works, offline `LLM()` fails" is not a patch-application asymmetry.** At the
  paired vLLM `v0.13.0`, `EngineArgs.create_engine_config` calls
  `current_platform.pre_register_and_update()` (`vllm/engine/arg_utils.py:1312`), and
  `NPUPlatform.pre_register_and_update` is what runs `adapt_patch(is_global_patch=True)`
  (`vllm_ascend/platform.py:128-129`; `adapt_patch` at `vllm_ascend/utils.py:372` just imports
  `vllm_ascend.patch.platform`). Offline `LLM(...)` builds its config through the same call, so both
  entry points apply the same patch set; the CLI-only call at `arg_utils.py:2014` passes the argparse
  parser and only appends `"ascend"` to the `--quantization` choices. Check the env gate and the
  config delta (e.g. an offline script that omits `--speculative-config`, `additional_config` or
  `max-model-len`) before hypothesising that patches are applied only under `vllm serve`.

## `AssertionError` in `dsa_v1.py` during MTP graph capture (v0.20.2rc1 and the 0.21 line)

Symptom: startup aborts while capturing the drafter's graph —
`capture_model -> _warmup_and_capture -> _dummy_run -> drafter.dummy_run ->
build_for_graph_capture -> build` and a bare `AssertionError` at
`vllm_ascend/attention/dsa_v1.py:507` (`assert self.prefill_ratio_to_sas_metadata is not None`).

Mechanism, verified at `v0.20.2rc1`:

- `dsa_v1.py:495-509` `build()` reads `kwargs.get("prefill_ratio_to_sas_metadata")` /
  `decode_ratio_to_sas_metadata` and asserts both non-`None`;
  `dsa_v1.py:1326-1337` `build_for_graph_capture()` forwards `**kwargs` to it.
- `spec_decode/llm_base_proposer.py:493-496` calls `build_for_graph_capture(common_attn_metadata,
  state)` with no kwargs → assert. The **eager** build in the same file does pass them:
  `:730-738`, under `if self.use_compress:`, `prefill/decode/common_ratio_to_sas_metadata=dict()`
  plus `block_size`.
- Gate that makes the workaround work:
  `:160 self.use_cuda_graph = self.runner._use_aclgraph() and not
  self.speculative_config.enforce_eager`; `:426-427` forces
  `aclgraph_runtime_mode = CUDAGraphMode.NONE` when `use_cuda_graph` is false, so the
  `CUDAGraphMode.FULL` branch at `:446` (which contains the failing call) never runs.
  `SpeculativeConfig.enforce_eager` is an upstream field at `vllm/config/speculative.py:73` in
  vLLM `v0.20.2` ("Override the default enforce_eager from model_config").

Workaround: add `"enforce_eager": true` to `--speculative-config` (the thread proposes it and
maintainers repeat it; one reporter instead manually applied PR #9957 and reported it fixed).
Both documented recipes carry the flag. Cost: the drafter runs eager — a reporter asked about the
perf hit and the thread has no measurement.

Fix and availability: `1930088f9 [Feature] Support dsv4 mtp graph (#11062)` (2026-07-01) adds
`extra_attn_metadata_args` (the three ratio dicts + `block_size`) to the graph-capture build,
populates `positions_cpu`, and switches to `build_for_drafting` for draft steps after the first.
`git tag --contains 1930088f9` starts at `v0.23.0rc1` (2026-07-19); it is in neither
`v0.20.2rc1` nor `v0.21.0rc1`. So the upgrade answer is "v0.23.0rc1 or later", while the
thread's PR #9957 is a separate, still-open/unmerged proposal at fetch time (last modified
2026-10-09) — never cite it as a released fix.

Boundary: established by tag reads, not by running the fix. On any tag re-check the call site
(`git grep -n 'build_for_graph_capture' <tag> -- 'vllm_ascend/spec_decode/*'`) and whether it
passes kwargs, since the file moves and is renamed between releases.

## `SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered`

- Traceback shape: `vllm_ascend/models/deepseek_v4.py:1351 for name, loaded_weight in weights:`
  ← upstream `model_executor/model_loader/weight_utils.py multi_thread_safetensors_weights_iterator`.
  At `v0.23.0` `deepseek_v4.py:1351` is exactly that `for` statement, so the plugin frame is the
  consumer and the raise happens inside safetensors parsing a checkpoint shard, before any weight
  is applied.
- Read it as a shard-integrity failure: the file's header and length disagree (the declared
  tensor ranges do not cover the file), i.e. an incomplete/truncated download or an overwritten
  file — not a model, quantization or attention bug. The
  `--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}'` flag only
  selects the iterator that surfaces it.
- Next check: compare each shard's byte size with the published ModelScope/HF file list (or
  re-download and verify), then re-serve. The recorded run never performed this check, so the
  conclusion stays a candidate; if shard sizes match, the iterator/reader becomes the suspect.
- Doc trap in the same report: the thread replied that the linked tutorial URL is the
  **main-branch** documentation, "not recommended" on the 0.23.0 image, and that switching to the
  0.23.0 docs produced new errors still under review. Quote the versioned doc (or the in-tree file) and keep the doc
  version as a separate statement from the code finding.

## Stray garbled characters in otherwise correct output on a floating `deepseekv4` image

Report shape (issue `https://github.com/vllm-project/vllm-ascend/issues/9430`, claimed
"框架版本 0.18.0", image `quay.io/ascend/vllm-ascend:deepseekv4`, DeepSeek-V4-Pro w4a8 on a 4-machine
2P2D split with `MooncakeHybridConnector`, 1-token `deepseek_mtp`, `FULL_DECODE_ONLY`,
`--no-enable-prefix-caching`, `recompute_scheduler_enable: true`): the reply is coherent except for
inserted meaningless characters, i.e. local corruption rather than a bad answer. The thread is open
with 7 comments and no fix commit.

- The `0.18.0` label cannot be the served tree: `v0.18.0` and `v0.19.1rc1` contain **no**
  `deepseek_v4` file (`git ls-tree -r --name-only <tag> | grep -c deepseek_v4` → 0) while
  `v0.20.2rc1` has 7 including `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`.
  So this is the floating/preview-image class from "Which trees contain the model" above — say which
  trees you grepped and that the revision is unidentifiable. A commenter also flagged that the image
  is A2 while the pasted serve line is A3-shaped; keep both as separate observations.
- Thread claims, neither verified: an automated reply proposes a decode-side sampling/detokenisation
  out-of-range read, advises checking `max_model_len` against the context the weights really support
  and whether the MTP rejection sampler drops invalid tokens, and suggests disabling MTP; the
  reporter later wrote that the v0.20 release resolved it. No commit, no bisect, no config delta
  attached to that statement.
- Delta against the documented A3 recipe that is worth naming first (both are recorded in the
  sections above): `"enforce_eager": true` missing from `--speculative-config`, which puts the run on
  the DSv4 MTP graph-capture path, and `VLLM_ASCEND_APPLY_DSV4_PATCH=1` unset, which leaves
  `patch_speculative_config` and `patch_kv_cache_coordinator` unimported (`envs.py:116`, default 0).
  The v0.20.2rc1 nightly A3 config sets both and is single-node DP=2/TP=8, so the PD dimension has no
  CI analogue.
- Confirming test for a new report of this shape: the same prompt and topology on a tag that actually
  contains the model, with `--speculative-config` removed, then with the two flags above restored.
  Until that A/B exists, do not attribute stray characters to a kernel, to the sampler, or to the
  Mooncake transfer.
- A **new** report with the same stray-character symptom is separated from the decode-side KV-lifetime
  race by the first-anomaly phase only: an exact `(prompt+prefix) % block_size == block_size - 1`
  boundary under `--async-scheduling` + a sliding window goes to
  [pd-disaggregation.md](pd-disaggregation.md); #9430's own thread never reports a phase, so it stays
  unattributed here rather than inheriting either mechanism.

## Accuracy below official on a reasoning model: the `deepseek_v4` tokenizer-mode default

Report shape: `--tokenizer-mode deepseek_v4` with `--reasoning-parser deepseek_v4`, an eval
harness (AISBench `VLLMCustomAPIChat`) that sets `chat_template_kwargs={"thinking": False}` and
`pred_postprocessor=extract_non_reasoning_content`, and a score well under the model card
(#10413: GPQA 57-63 against 71, vllm-ascend 0.20.2rc1 / vLLM 0.20.2,
`DeepSeek-V4-Flash-w8a8-mtp`, A2 4P1D). Check prompt construction before any kernel, quant or
sampling explanation.

`--tokenizer-mode deepseek_v4` replaces the HF chat template: `vllm/tokenizers/deepseek_v4.py`
wraps `apply_chat_template` and builds the prompt with `encode_messages` from
`vllm/tokenizers/deepseek_v4_encoding.py`. At vLLM `v0.20.2` (`:32-50`) that wrapper does:

- `thinking = kwargs.get("thinking", False)`, `enable_thinking = kwargs.get("enable_thinking",
  False)`, `thinking = thinking or enable_thinking`, `thinking_mode = "thinking" if thinking
  else "chat"` — so **omitting both kwargs runs chat (non-thinking) mode**;
- `reasoning_effort = kwargs.get("reasoning_effort")`, then `if reasoning_effort not in
  ("max", "high"): reasoning_effort = None` — `medium`/`low`/`minimal`/`none` are silently
  dropped rather than rejected;
- `drop_thinking=kwargs.get("drop_thinking", True)` for historical reasoning blocks.

`main` inverts the default and widens the mapping: with neither key present
`thinking_enabled = True`; `reasoning_effort` defaults to `"high"` when thinking is on,
`"none"` forces chat mode, and `low`/`minimal`/`medium` map to `"low"`. The flip is **not
tag-bounded here**: `git log -S` on the changed statements surfaced only
`4d51588e23 [Feat] DeepSeek V4 Rebased (#40860)` as a candidate, and its first tag was not
established. Say "at v0.20.2 the default is chat mode; on `main` it is thinking" and require
the reporter's tag before recommending an upgrade.

**A request-level `reasoning_effort` turns thinking back on, over any serve-line
`--default-chat-template-kwargs '{"thinking": false}'`.** At `v0.20.2rc1` and `v0.21.0rc1` the
plugin ships `vllm_ascend/patch/platform/patch_deepseek_v4_thinking.py`: it widens the request
model's `reasoning_effort` annotation, then wraps `ChatCompletionRequest.build_chat_params` so that
`if self.reasoning_effort is None or "enable_thinking" in user_kwargs: return params`, otherwise
`chat_template_kwargs["enable_thinking"] = self.reasoning_effort != "none"` (`:39-73`; the same
guard is at `v0.20.2rc1`). The tokenizer wrapper then evaluates
`thinking = kwargs.get("thinking", False) or kwargs.get("enable_thinking", False)`, so a client
that sends `reasoning_effort: "max"` runs **thinking** mode even though the serve line defaults
`thinking: false`. From vLLM `v0.22.1` the same branch is upstream at
`vllm/entrypoints/openai/chat_completion/protocol.py:487-488` (the plugin file is gone from
`v0.22.1rc1`), so the rule is not tag-specific. The plugin file is re-added later for the
V4-0731 effort options (`v0.25.1rc1`, `v0.26.0rc1`), so read the tag's own tree instead of
inferring from this paragraph whether the plugin or upstream owns the bridge. Two consequences when
triaging: a serve line whose default says `{"thinking": false}` proves nothing about the request a
client sent — read the client's payload; and `extra_body={"thinking": {"type": "enabled"}}` lands at the *request top
level*, not in `chat_template_kwargs`, so it is not what sets the mode. To force chat mode per
request, send `chat_template_kwargs: {"enable_thinking": false}` — the guard keys on that name, not
on `thinking`. This refutes a "thinking was disabled, so quality dropped" hypothesis instead of
supporting it. At `v0.21.0` the tokenizer's effort mapping also changed from v0.20.2's
`not in ("max","high") -> None` to `none` -> chat mode, `max`/`xhigh` -> `"max"`, anything else ->
`"high"`.

Consequence: an eval score compared against a thinking-mode official number while the harness
disables thinking is a configuration mismatch, not a plugin defect. The workaround is
`chat_template_kwargs={"thinking": true}` (or `enable_thinking: true`) plus a
`reasoning_effort` in {`high`,`max`}, which the reporter confirmed in-thread. The reasoning
parser for this tokenizer mode is `DeepSeekV3ReasoningParser`
(`vllm/reasoning/__init__.py:31-33`), so a `reasoning_content` split problem belongs to
[serving-toolcall-reasoning-streaming.md](serving-toolcall-reasoning-streaming.md) instead. Two decoys in the same report: an odd
`--max-num-batched-tokens 60` on a `kv_consumer` decode node (chunked prefill is a prefill
concern) and `max_out_len=65536` against `--max-model-len 135000` (truncation, not accuracy).

## Triage rules for this topic

These rules are kept with the topic they decide; SKILL.md carries the routing line and only
cross-cutting rules.

- **`SafetensorError: ... incomplete metadata, file not fully covered` is shard integrity, not a
  model bug.** It is raised inside safetensors parsing a checkpoint file while the loader's weight
  iterator runs; the plugin frame above it is only the consumer. Treat it as a truncated or
  overwritten shard (compare its size with the published file list, re-download) and keep that a
  candidate until the sizes are checked; the DeepSeek-V4 instance is recorded above.
