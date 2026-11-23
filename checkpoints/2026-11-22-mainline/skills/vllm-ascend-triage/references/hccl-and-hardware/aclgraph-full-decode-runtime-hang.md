# A full/partial ACL graph that captured fine, then hangs the first request

Scope: the paste carries the mode's own success banner and every rank's
`Graph capturing finished in N secs`, and the failure appears only once a request
is served — ending in the bare `TimeoutError: RPC call to execute_model timed out.`
and `EngineDeadError` of `rpc-watchdog-timeouts.md`. Recorded **#4178**:
vllm-ascend `v0.11.0rc1` (`c5fe179`) + vllm `v0.11.0` (`b8b302c`), 8×910B4,
`Qwen3-VL-8B-Instruct` at TP8 with `--max-num-seqs 16 --max-model-len 16384`,
chunked prefill + prefix caching on, `compilation_config={"cudagraph_mode":
"FULL_DECODE_ONLY"}`, one image prompt, `max_tokens=512`; the title says the same
hang reproduces on Qwen2.5-VL-72B / Qwen2.5-72B / Qwen3-VL-8B.

This page is about **order, not mechanism**. Two of the three lanes that this text
usually sends investigators into are already closed by the paste, and the third one
— the mode's own request path at a release candidate — is usually a *version*
answer that no source walk at the pinned ref can confirm.

## What the paste already settles (four lines, one call)

| Line in the paste | What it closes |
|---|---|
| `FULL_DECODE_ONLY compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode` (`vllm_ascend/platform.py:220`) or `PIECEWISE compilation enabled on NPU` | the mode survived the whole gate chain, so the "setting never took effect" lane (`../startup-failures/graph-mode-not-applied.md`) does not apply |
| `Starting to capture ACL graphs for cases: [16, 8, 4, 2, 1], mode: FULL, uniform_decode: True` (`worker/model_runner_v1.py:3449-3452`) **plus** a per-rank `Graph capturing finished in 6 secs, took 0.16 GiB` (`:3567-3569`) | capture completed on every rank: `../startup-failures/acl-graph-capture.md` and its `insufficient available streams` wrapper are out, and the bucket list was already filtered to `≤ max_num_seqs` |
| the red `platform.py:236` block whose text says "capturing too many batch sizes can lead to OOM … **or inference hangs**" | *canned*: printed for **every** full-graph launch (same note at `../deploy-sizing/tp-scaling-oom-at-startup.md:117`). It is the project's own risk statement about an experimental mode, quotable as such, never this incident's cause (SKILL.md rule 11) |
| `ERROR … [dump_input.py:69] Dumping input data for V1 LLM engine` + `[dump_input.py:76] Dumping scheduler output …` then `TimeoutError: RPC call to execute_model timed out.` | the dumped scheduler output **is** the step whose RPC never returned: `execute_model_with_error_logging` dumps exactly that `scheduler_output` before re-raising (`vllm/v1/engine/core.py:254-271`, `vllm/logging_utils/dump_input.py:66-77`) |

Read two fields of that dump before theorising, because they name the shape of the
hung step:

- `scheduled_new_reqs=[NewRequestData(… prompt_token_ids_len=440 …)]` with
  `total_num_scheduled_tokens=440` — the stalled step is the **first (prefill)
  step** of the request, not a decode-graph replay. That reading is checkable and
  worth stating, because it reframes the answer: under `FULL_DECODE_ONLY` only
  `uniform_decode=True` keys are registered for `FULL` with the captured buckets
  (`vllm/v1/cudagraph_dispatcher.py:59-93`), and a 440-token non-uniform step is in
  no `FULL` key set, so `dispatch` returns `NONE` (`:96-125`) — the graph did not
  replay on the step that hung. What is left is the mode's *second-order* effects
  (a compile/`splitting_ops` shape the mode does not prepare, or state the capture
  left behind), which is exactly the reading that is not settled by argument.
- `compilation_config={… "cudagraph_mode":[2,0] …}` — `CUDAGraphMode.FULL_DECODE_ONLY`
  **is** the tuple `(FULL, NONE) = (2, 0)` (`vllm/config/compilation.py:39-43`), so
  the engine core that timed out ran with the mode in effect. `[2,1]` is
  `FULL_AND_PIECEWISE`. This is the config the failing process actually used, not
  the frontend's pre-worker dump.

`bash <skill_root>/scripts/executor-timeout-check.sh <paste|->` now prints this
mode/capture/step summary together with the watchdog verdict.

## The two checks to run before any source read

1. **Which models does the pinned tree validate for that mode?**
   `rg -n 'FULL_DECODE_ONLY' <ascend>/tests/` — the **bare token**, because a compound
   `cudagraph_mode.*FULL_DECODE_ONLY` misses any case whose kwarg wraps: at v0.11.0rc1
   the multicard call closes `"FULL_DECODE_ONLY"}) as runner:` on the line *after*
   `compilation_config={"cudagraph_mode":` (:45-46), so the compound form returns only
   the two singlecard lines and the mode's only unwired case silently disappears.
   Then confirm a hit is wired:
   `rg -n '<test-file-basename>' <ascend>/.github/ --hidden` (`_e2e_test.yaml:77,91` →
   `singlecard/test_aclgraph.py`, `:92` → `test_aclgraph_mem.py`).
   At v0.11.0rc1 the answer was **three lines in two files**:
   `tests/e2e/multicard/test_full_graph_mode.py:46` (`Qwen/Qwen3-30B-A3B` TP2,
   `max_model_len=1024`, `max_tokens=32`) — **no workflow invoked it** — and
   `tests/e2e/singlecard/test_aclgraph.py:156,175`, both inside
   `test_models_with_aclgraph_full_decode_only`
   (`Qwen/Qwen3-0.6B` + `vllm-ascend/DeepSeek-V2-Lite-W8A8`, `max_model_len=1024`),
   which `@pytest.mark.skip("… it fails intermittently and needs investigation")` at
   `:103` disables in the one file CI *does* wire — so the mode itself ran in no CI
   job. The wired ACL-graph runs are PIECEWISE (`:181` →
   `tests/e2e/multicard/test_single_request_aclgraph.py`, DP2, **no**
   `cudagraph_mode` key), plus the wired MTP correctness cases that pass `FULL`, which
   `platform.py:237-242` sends to `NONE` (rule 16). No VL model, no 72B, no 16 K
   context is covered anywhere.
   When the reporter's model sits outside that set, the coverage verdict is the
   first sentence of the answer (same shape as rule 14's "no validated recipe for
   this combination").
2. **Does the reporter have a working control?** In #4178 the default-mode launch is
   the reporter's normal setup and only the mode changes the outcome, so the mode —
   not their command — is the lever, and the cheapest action is the fallback plus a
   bug report. Without a working control, run check 1 first: the rest of their
   config becomes a candidate again.

## Ranked causes, each with its one check

1. **The mode is not validated for that model family at this ref, and the fix landed
   after it** — highest prior for an rc report. Check: the coverage grep above, the
   next rc's release notes / the issue tracker; the recorded case's reporter later
   reported it fixed on `0.11.0rc2` (later public observation, not a reply to the
   investigators). Reporter action: drop `cudagraph_mode` (PIECEWISE default) or
   `--enforce-eager`, and attach the banner + capture lines to their upstream report.
2. **Capture-size / memory pressure inside the mode.** The `platform.py:236` block
   advises exactly this, so it is the project's suggested knob and a hypothesis, not
   a diagnosis. Check: one re-run with
   `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}`
   (single bucket) and, separately, a lower `gpu_memory_utilization`.
3. **A worker stall under the mode** (device fault, collective entered by a
   different rank set, an `update_stream` event never recorded). Check: the per-rank
   worker log and plog for the hang window plus `npu-smi info` taken *during* the
   hang; the recorded paste has no worker text at all, so this lane needs artefacts,
   not more reading. `--enforce-eager` decides whether the mode owns it.
4. **The hung step's own shape** — a VL/multimodal prefill step (the dump carries
   `mm_features`, a `[1, 16, 106]` grid and 440 tokens) under a decode-only graph
   mode. Check: the same prompt in the mode without the image, then with
   `--enforce-eager`; that splits tokenizers/mm path from graph path in two runs.

## Why the mechanism is usually *not* in the snapshot

An rc report whose behaviour disappears in the next rc has its fix commit **after**
the pinned ref, so the tree cannot contain it and no amount of walking the
graph-input-update path will name it. At v0.11.0rc1 the only mode-specific
provisioning in the runner is the MLA `cos`/`sin` buffer allocation
(`worker/model_runner_v1.py:374-377`) and the post-replay
`update_attn_params` / `update_mla_attn_params` on `self.update_stream`
(`:2321-2333`); nothing in that path is VL- or mrope-specific. Quote that as a
*coverage* fact ("no mode-specific handling exists for this family at this ref"),
then deliver. The recorded session treated the missing provision as an invitation
and spent 22 turns and 48 K characters of reasoning on it, emitting **zero** message
text before the cap — see `../evidence-workspace/delivery-discipline.md`.

Ask back for: the full per-rank worker log, the plog for the hang minutes,
`npu-smi info` from during the hang, the same command with `cudagraph_mode` removed,
and the `cudagraph_capture_sizes` if one was set.

## Scope

Pinned at vllm-ascend `c5fe179` (v0.11.0rc1) + vllm `b8b302c` (v0.11.0); the banner
and capture line numbers move between releases, so re-grep
`FULL_DECODE_ONLY compilation enabled on NPU`, `Starting to capture ACL graphs for
cases` and the `CUDAGraphMode` tuple values per version. The classification —
*mode engaged, capture finished, the request path hangs* — holds whenever those
lines are present; when the paste instead stops inside the capture progress bar, go
to `../startup-failures/acl-graph-capture.md`, and when it never prints the banner at
all, `../startup-failures/graph-mode-not-applied.md`. For the same stall on a 310P
card, the documented mask/graph limits are the owner
(`../soc-platform-paths/` pages, recorded #12985) rather than this page.
