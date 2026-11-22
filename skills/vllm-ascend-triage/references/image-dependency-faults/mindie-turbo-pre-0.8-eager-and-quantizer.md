# `mindie_turbo` on the pre-graph releases (≤ v0.7.3.post1): eager ops, the quantizer, empty output

## Symptom this page answers

A report on a release whose compatibility matrix **does** name a MindIE Turbo version
(so the pairing is sanctioned), with a *quality* complaint rather than a crash —
"some results have no content", a few samples empty or truncated under concurrency,
accuracy below the plugin-free run. Recorded **#1954**: image
`quay.io/ascend/vllm-ascend:v0.7.3.post1-openeuler`, then a bare
`pip install mindie_turbo`, Qwen3-30B on 2×910B3, 40 concurrent GPQA requests —
"some inference results may lack content … not reproduced after uninstalling
mindie_turbo". The record has no resolution, so the mechanism stays **open**; what the
tree settles is *which lanes are even possible* at that ref, and they are not the ones
`mindie-turbo-graph-break.md` ranks.

## Version gate — read this before that page's §2/§4

- At vllm-ascend v0.7.3.post1 (**`c69ceaca1ecc`**, the pin for #1954) compile is forced
  off wholesale: `vllm_ascend/platform.py:110-114` `check_and_update_config` rewrites any
  level other than `NO_COMPILATION` and logs `Compilation level %s is not supported on
  NPU now, forcing compilation level to NO_COMPILATION`. There is no Dynamo trace and no
  ACL graph of the forward, so a graph break, `fullgraph` and
  `VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE` **cannot** be this report's cause.
- The `PIECEWISE compilation enabled on NPU` / `Compilation disabled, using eager mode`
  log pair and `platform.py:162-182` are 0.9.x anchors. Re-read that function at whatever
  ref the report names before quoting either lane; the flip between the two behaviours is
  inside one method, so one `sed -n` decides it.
- What remains at this ref is eager **operator substitution** and the **quantizer**, i.e.
  numerics. Rank accordingly and drop capture-side checks (#1534's shape) entirely.

## What the plugin can touch at this ref — three sites, nothing else

- **Workers, unconditionally**: `vllm_ascend/worker/worker_v1.py:77-82` (V1) and
  `worker.py:77-82` (V0) call `try_register_lib("mindie_turbo", <info>, <exception_info>)`
  with no flag or env gate.
- **`try_register_lib` itself**: `vllm_ascend/utils.py:23-35` logs `logger.info(lib_info)`
  when the import succeeds and `logger.warning_once(exception_info)` when it raises. So
  this ref's log has **three** states and only the first means the plugin ran:
  `MindIE Turbo is installed. vLLM inference will be accelerated with MindIE Turbo.` ·
  `MindIE Turbo is installed but unable to import mindie_turbo, skip MindIE Turbo
  acceleration.` (installed but broken — the `undefined symbol` form is
  `abi-undefined-symbol.md` / #1144) · nothing at all (`find_spec` returned `None`).
  "No info line" alone therefore does not separate *absent* from *present-and-failing* on
  this ref, unlike the silent `except Exception: pass` of the 0.9.x helper.
- **The quantizer**: `vllm_ascend/quantization/quantizer.py:39-46` —
  `AscendQuantizer.get_quantizer` imports `mindie_turbo.MindIETurboQuantizer` and raises
  `NotImplementedError("There is no available ascend quantizer.")` when the import fails;
  call sites `vllm_ascend/quantization/quant_config.py:153,225,321`. At this ref w8a8
  exists **only** through the plugin (`docs/source/faqs.md:82`: *"If you're using vllm
  0.7.3 version, w8a8 quantization is supporeted with the integration of vllm-ascend and
  mindie-turbo"*), so on a quantized checkpoint every quantized linear/MoE method is
  plugin code — the largest plugin-owned surface on this release.
- **The forward being substituted is the tree's own copy** for this family:
  `vllm_ascend/models/qwen3.py` and `qwen3_moe.py`, registered at
  `vllm_ascend/models/__init__.py:34-39` (`Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`),
  each "Adapted from vllm/model_executor/models/<file>.py". A numeric symptom therefore
  has two candidate owners — the substituted op and the downstream copy
  (`../upstream-downstream-drift/patch-copy-vs-upstream.md`) — and a same-release run
  without the plugin separates them in one launch.

## The pairing contract, and the reporter's install line

- Matrix rows at this pin: `docs/source/developer_guide/versioning_policy.md:70-71` —
  `v0.7.3.post1 | v0.7.3 | 3.9 - 3.11 | 8.1.0 | 2.5.1 / 2.5.1 | 2.0rc1` and the same for
  `v0.7.3`; those are the **only two filled rows** of the MindIE Turbo column.
- The same pin states it twice more: `setup.py:371`
  `extras_require={"mindie_turbo": ["mindie-turbo==2.0rc1"]}` and
  `docs/source/conf.py:76 'pip_mindie_turbo_version': "2.0rc1"` (the value the tutorials
  substitute into `pip install mindie_turbo==…`).
- A reporter who followed the issue's own words — `pip install mindie_turbo`, unpinned —
  almost certainly resolved a newer plugin. Rung 1 is therefore `pip show mindie-turbo`,
  then the *same* run repeated pinned at `2.0rc1`. Whether some newer plugin pairs with
  this release is not answerable from the tree: say so (`mindie-turbo-graph-break.md` §3
  has the same rule) and point at the vendor table.

## Ranking for "some results have no content" under concurrency

1. **The response body, not the engine.** Empty `content` beside a populated
   `reasoning`/`reasoning_content` and `finish_reason: length` is the reasoning-parser
   truncation branch, and an eval harness that reads only `content` scores those samples
   as "no answer" — under 40-way concurrency the longest CoT is exactly what runs out of
   budget. One raw body decides it: `content`, `reasoning`, `finish_reason`,
   `completion_tokens` → `../accuracy-benchmarks/reasoning-output-routing.md` and
   `../accuracy-benchmarks/harness-vs-engine-attribution.md`. Take this gate **before**
   any plugin theory, and note the trap: a faster plugin should *reduce* thinking-budget
   truncation, so the install/uninstall correlation the reporter asserts cuts both ways
   and is a premise, not a proof.
2. **The version contract** above (`pip show mindie-turbo`). Cheap, and the only lane the
   reporter's own environment line already makes suspect.
3. **A quantized checkpoint**: then the plugin owns all quantized numerics (§quantizer).
   Check the checkpoint's `config.json` `quant_method` / a `…-W8A8` directory name and
   ablate with the bf16 checkpoint on the same serve line.
4. **Eager substitution numerics at large dynamic batch.** The pin documents the class:
   *"Some performance tuning features in `mindie_turbo` currently have certain scenario
   limitations, so environment variables are used to control whether they are enabled"*
   (`docs/source/developer_guide/performance/optimization_and_tuning.md:161`), with
   the switches themselves in the vendor MindIE doc only. Check: sweep concurrency
   1 → 8 → 16 → 40 at fixed seed and count empty/short samples; a threshold, not a rate,
   points at the batched path.
5. **The plugin never loaded in that process** — the three log states above, then
   `python -c "import mindie_turbo"` in the launching interpreter.

Do not read "unpinned plugin + quality complaint" as "unsupported combination": the pin
**recommends** turbo for Qwen3 (`docs/source/user_guide/release_notes.md:9`, *"Mindie
Turbo is recomanded to improve the performance of Qwen3"*) and lists the plugin as a
functional feature (`docs/source/user_guide/suppoted_features.md:37`), so the honest
verdict is a sanctioned pair with an open concurrency/reporting mechanism.

## What to ask back

- `pip show mindie-turbo` and which `pip install` line produced it.
- One failing response body whole (`content`, `reasoning`, `finish_reason`,
  `completion_tokens`, `return_token_ids: true`) **and** the same prompt at
  concurrency 1.
- The serve command plus the three plugin log lines (`MindIE Turbo is installed.` /
  `unable to import` / neither).
- The checkpoint: `config.json` `quant_method`, or the weights directory name.
- Whether the harness reads `content` only, `reasoning` too, or strips think tags.

## Scope and limits

- `mindie_turbo`'s own source is in no case export, so which operators it replaces is
  unknown at this ref too; do not enumerate them from symbols the message does not carry.
- The **vllm** tree in these snapshots is not the reporter's: #1954's pair is vllm
  v0.7.3, while the export carries vllm v0.9.1. Any `vllm/reasoning/*`,
  `vllm/entrypoints/**` or `vllm/compilation/**` anchor is then off-ref — name the
  response fields as the reporter prints them and ask for the version, instead of grading
  a parser that may not exist at their ref.
- Nothing here transfers upward: from v0.8.4rc2 the MindIE Turbo column is empty and
  compile is on again, so a 0.9.x report belongs in `mindie-turbo-graph-break.md`.
