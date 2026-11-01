# Gemma 3 / Gemma 4 on Ascend: config nesting and support boundaries

Applies to reports naming `gemma-3-*` (text or `Gemma3ForConditionalGeneration`) or
`gemma-4-*` (`Gemma4ForConditionalGeneration`) on NPU. Two failure classes dominate, and
neither is diagnosed by reading attention kernels:

- **Gemma3 on 0.9.x**: a multimodal *config* read at start-up (`AttributeError: ...
  num_hidden_layers`), which looks like an ACL-Graph/capture fault because the log line
  above it says "PIECEWISE compilation enabled ... only ACL Graph mode".
- **Gemma4 up to v0.23**: a *support boundary*, not a defect in the reporter's flags.

Both are answered from source plus the issue thread in a few calls; capture sizes,
prefix caching and CANN versions do not change either answer.

## Gemma3: `AttributeError: 'Gemma3Config' object has no attribute 'num_hidden_layers'`

Mechanism (verified in three trees):

- transformers `Gemma3Config` is a wrapper: the text fields live on
  `text_config: Gemma3TextConfig` (`src/transformers/models/gemma3/configuration_gemma3.py`,
  `text_config` at `:202-206`; `num_hidden_layers: 26` belongs to the text config at `:77`).
  The wrapper has no `num_hidden_layers`, so reading it raises.
- vllm-ascend read the wrapper instead of the text config inside `update_aclgraph_sizes`:
  `num_hidden_layers = vllm_config.model_config.hf_config.num_hidden_layers` —
  `vllm_ascend/utils.py:134` at `v0.9.0rc1` and `v0.9.0rc2`, `:307` at `v0.9.2rc1`.
  It is reached from `vllm_ascend/platform.py:189` in the PIECEWISE branch of
  `check_and_update_config` (the "PIECEWISE compilation enabled" log is `:183` there).
- The same statement hit `Qwen2AudioConfig` in #1760/#1780, which is why the fix is generic
  rather than Gemma-specific.
- Upstream already exposes the intended accessor: `ModelConfig.hf_text_config =
  get_hf_text_config(hf_config)` exists at vLLM `v0.9.0` (`vllm/config.py:528`). Two
  details cost rounds if guessed: `hf_text_config` is an instance attribute, so
  `grep 'def hf_text_config'` finds nothing; and the upstream clone's `main` tree has no
  `vllm/config.py` at all (it is the `vllm/config/` package), so grep the tagged file with
  `git show <tag>:vllm/config.py`.

Fix and release boundaries:

- `d1c640841` `[Bugfix] Fix num_hidden_layers when Qwen2-Audio 7B (#1803)` adds
  `get_max_hidden_layers(hf_config)` (recursive scan, `max` over every integer
  `num_hidden_layers`, `ValueError("Not found num_hidden_layers in model config.")` when
  none) and calls it only when `hasattr(hf_config, 'num_hidden_layers')` is false. Its
  message closes `#359` (the gemma-3-27b-it new-model issue), `#1760`, `#1276`, `#1780`.
  First tag `v0.10.0rc1` (`git describe --contains` → `v0.10.0rc1~85`); absent at
  `v0.9.2rc1`.
- The branch later disappears with `5f2ef5a0b` `[BugFix] Remove legacy capture-size pruning
  update_aclgraph_sizes (#9962)`, first tag `v0.21.0rc1`; `def update_aclgraph_sizes` has
  zero hits at `v0.22.1rc1`+. A 0.9.x report is fixed from `v0.10.0rc1`, and the raising
  statement no longer exists from `v0.21.0rc1`.

Workaround and its limit:

- `--enforce-eager` does avoid this exact crash at 0.9.x: `platform.py:169`
  (`if enforce_eager or compilation_config.level == NO_COMPILATION`) short-circuits before
  the `update_aclgraph_sizes(vllm_config)` call, logging "Compilation disabled, using eager
  mode by default". That is a code fact about the start-up traceback only.
- It does not make Gemma3 work on 0.9.x. #1183 (closed, duplicate) carries yiz-liu's "Try
  add --enforce-eager, should be OK then. Your problem is noted and we will work on fixing
  it." and Yikun's "I think gemma3 will not work in graph mode and eager mode"; #359's
  comments report gemma-3-12b-it running in eager with one prompt and failing with two or
  more. Recommend the upgrade and label `--enforce-eager` as traceback avoidance.
- Working pointer, reporter-verified not maintainer-verified: a comment on #6257 reports
  gemma-3-1b-it offline inference working on `v0.15.0rc1` with vLLM 0.16, torch/torch-npu
  2.9.0, Python 3.11, `LLM(model=..., tensor_parallel_size=<all NPUs>,
  trust_remote_code=True, max_model_len=2048)`.

Report shapes seen: #1183 (`0.9.0rc1`, full traceback) and #6257 (`0.9.2rc1`, body is
screenshots only). For the image-only body, do not guess the exception from image
dimensions or the model family; the same raising statement at `v0.9.2rc1` plus the thread's
later working configuration is the finding.

## Gemma4: support boundaries, not attention bugs

- Upstream vLLM: Gemma4 first exists in `v0.19.0` (`8adcf8c40a` / `08ed2b9688`, #38826,
  first tags `v0.19.0` and `v0.19.1rc0`). There is no
  `vllm/model_executor/models/gemma4*.py` at `v0.18.0`. A report labelled
  `v0.18.0rc1` / "vLLM 0.18.0" that serves `Gemma4ForConditionalGeneration` therefore runs
  a newer install than the label (upgraded inside the container); treat the image tag as
  environment context and pin the source by symbol, not by line.
- vllm-ascend: no Gemma4 model or doc file at `v0.18.0rc1` or `v0.23.0rc1`; the only
  gemma-named tree entry at both tags is `tests/e2e/models/configs/gemma-3-4b-it.yaml`.
  Maintainers in #8370 say "It seems that Gemma4 is not supported in v0.18.0rc1 yet" and
  "gemma4 does not supported in this version, and we are working on it".
- Support entry points (`git describe --contains`): `8cc7bcd4c` A5 Gemma4 (#10643) and
  `ae7203aae` large-head attention on A2 (#11091) first in `v0.23.0rc1`; `8cecff754`
  "Support Gemma4 graph execution on A2/A3" (#11536) and `4c7616410` the Gemma4 deployment
  tutorial (#11055) first in `v0.24.0rc1`. So a Gemma4 FULL_DECODE_ONLY/ACL-Graph failure
  at ≤ v0.23 is a support-gap answer with an upgrade to ≥ `v0.24.0rc1`; do not spend the
  pass on capture sizes or kernels. Once the model is supported, a mode-specific crash is the
  decode-pad / FlashComm1 class in
  [full-decode-only-graph-mode.md](full-decode-only-graph-mode.md), not a Gemma boundary.
- The documented recipe (`docs/source/tutorials/models/Gemma4.md`, ≥ `v0.24.0rc1`) is
  `--trust-remote-code --tensor-parallel-size 4 --max-model-len 32768` with either
  `--enforce-eager` or `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`
  ("PIECEWISE is also supported"), and no `pa_shape_list`. Compare the reported serve line
  against it.

## `ValueError: not enough values to unpack (expected 13, got 9)` in `attention_v1.update_graph_params`

Static finding at `v0.18.0rc1` / `v0.18.0` (`vllm_ascend/attention/attention_v1.py`); no
NPU run was recorded.

- Two full-graph capture helpers append tuple *shapes* of different arity into one list:
  `full_graph_fia` (`:533`, appends 13 fields at `:583`) and `full_graph_pa` (`:626`,
  appends 9 fields at `:658-670`). The list is `graph_params.attn_params[num_tokens]`,
  keyed only by token count and shared by every layer; `GraphParams`/`get_graph_params()`
  are one process-global (`compilation/acl_graph.py:250-272`).
- `update_graph_params` (`:388`) selects its unpack from `using_paged_attention(num_tokens,
  vllm_config)` alone — 9 fields at `:410-420`, 13 fields at `:479-493` — while capture
  (`forward_impl`, `:920-928`) takes PA only when `attn_state == DecodeOnly and
  using_paged_attention(...) and self.sliding_window is None`, otherwise FIA. The capture
  decision is per layer; the update decision is per run.
- `using_paged_attention` (`vllm_ascend/attention/utils.py:44`) is False under any
  speculative config, False on A5, False unless `cudagraph_mode == FULL_DECODE_ONLY`, else
  `runtime_shape in get_ascend_config().pa_shape_list`. `pa_shape_list` defaults to `[]`
  (`ascend_config.py:127`; documented as "The custom shape list of page attention ops",
  default `[]`, in `docs/source/user_guide/configuration/additional_config.md`). The
  maintainer's first question in #8370 is whether it was set: "Normally, if this option is
  not specified, paged attention ops should not run."
- Read the error as a disagreement about the PA/FIA choice, not a metadata schema bug: with
  the default empty `pa_shape_list` every layer captures and updates through FIA, so this
  error implies `pa_shape_list` was set (or that build's predicates differ); a 9-tuple in
  the list proves a PA capture happened. Interleaved sliding-window/global models
  (Gemma3/Gemma4) put both arities in one list because sliding layers are excluded from PA
  at capture while the update branch is not per-layer — either message ("expected 9, got
  13" or "expected 13, got 9") is the same defect.
- Update runs on every FULL decode step, not only on a replay: `model_runner_v1.py:1827-1841`
  calls `update_full_graph_params(..., num_tokens_padded, ...)` when
  `cudagraph_runtime_mode == FULL and not capturing`.
- Check order: the serve line's `--additional-config` (`pa_shape_list`), whether the model
  has sliding-window layers, `--enforce-eager` (no capture, so no update), then the release.
  The PA path was removed by `f6b33f49c` `[Refactor][Attention] Remove paged attention
  (#11899)`, first tag `v0.24.0rc1`, so the class cannot occur later.
- Boundary: the #8370 traceback frames (`attention_v1.py:636`,
  `model_runner_v1.py:1962`) match no tag in 0.17–0.20 — at `v0.18.0rc1`/`v0.18.0` `:636`
  sits inside `full_graph_pa` and `model_runner_v1.py:1962` is `valid_modes=` /
  `num_active_loras=`. The direction of that report's mismatch stays unverified, and the
  model-support answer (previous section) comes first.

## Locators that shorten the next pass

- `git log --all --oneline -i --grep='gemma4'` and
  `git log --all --oneline --diff-filter=A -- 'tests/e2e/models/configs/gemma*'` find the
  model entry points; a plain `--grep='gemma'` in vllm-ascend returns mostly Gemma4
  Model-Runner-V2 spec-decode and doc noise.
- `git describe --contains <sha>` gives the first tag directly. Do not order `git tag --contains`
  with `sort -V`: it puts a final release before its own rc tags, which is exactly the distinction a
  support-boundary answer turns on.
- `git ls-tree -r --name-only <tag> | grep -i gemma` settles "does this tag know the model
  at all" in one call, in either repository.

## Triage rules for this topic

These rules are kept with the topic they decide; SKILL.md carries the routing line and only
cross-cutting rules.

- **A `ValueError: not enough values to unpack (expected N, got M)` in
  `attention_v1.update_graph_params` is a PA-versus-FIA arity disagreement.** Capture needs
  `DecodeOnly and using_paged_attention(...) and sliding_window is None`
  (`attention_v1.py:920-928` at `v0.18.0rc1`) while the update branches on
  `using_paged_attention(num_tokens, cfg)` alone (`:397`), and the 9-field PA tuple (`:658-670`)
  and 13-field FIA tuple (`:583`) share one `graph_params.attn_params[num_tokens]` list. That
  predicate needs `FULL_DECODE_ONLY`, no speculative config, non-A5 and
  `num_tokens in pa_shape_list` (default `[]`), so check `--additional-config '{"pa_shape_list": ...}'`
  and interleaved sliding-window layers before reading kernels; PA is removed in `v0.24.0rc1`.
