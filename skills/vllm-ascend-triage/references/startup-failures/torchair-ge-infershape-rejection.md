# Torchair graph build refused by GE shape inference (`E89999` / `InferShape`)

Symptom family: the engine gets past weight loading and profiling, prints
`Capturing torchair graph, this usually takes 1.0~3.0 mins.`, and the first
compiled dummy run dies with a **host-side graph-build** error. No kernel ran, so
no device is implicated:

```
RuntimeError: E89999: Inner Error!
E89999: [PID: 975] … op[Transpose_5], attr[perm], has wrong value[1], it should be
        less than input shape size[0][FUNC:TransposeCommonInferShape][FILE:transformation_ops.cc][LINE:1039]
        Call InferShapeAndType for node:Transpose_5(Transpose) failed[FUNC:Infer][FILE:infershape_pass.cc][LINE:118]
        process pass InferShapePass on node:Transpose_5 failed, ret:4294967295[FUNC:base_pass.cc][LINE:565]
        [Call][PreRun] Failed, graph_id:0, session_id:2.[FUNC:CompileGraph][FILE:graph_manager.cc][LINE:4654]
        [Compile][Graph]Compile graph failed, error code:1343225857 …[FUNC:ge_api.cc][LINE:1365]
```

Recorded #5484 (DeepSeek-V3.2 `w8a8-QuaRot`, 2×A2 DP2/TP8/EP, `--quantization
ascend`, `torchair_graph_config.enabled` with `graph_batch_sizes:[16]`, v0.11.0
image, reporter's own `0.11.0-dev` rebuild).

## Step 0 — read the frame stack as "graph build", and do not hunt the node name

The Python frames are `worker_v1.py compile_or_warm_up_model` →
`model_runner_v1.py capture_model` → `torchair/torchair_model_runner.py
_capture_model` → `_compile_torchair_graph` → `_dummy_run(is_torchair_compile=True)`
(§`v0.11.0` = `2f1aed98`: `torchair_model_runner.py:286`, `:245`, `:223`), then
`torch_npu/dynamo/torchair/…` → `fx2ge_converter.py` → `_ge_graph.py` → GE
`CompileGraph`. Consequences, in the order they save calls:

- **Not** a card/HCCL/driver/allocator fault. `npu-smi`, `plog`, `ASCEND_LAUNCH_BLOCKING`,
  `asys health`, `--device`/`--privileged` and the DP address/port are all
  irrelevant: nothing was launched. The engine is healthy; its graph is ill-typed.
- **`Transpose_5` is a GE node id, not a Python symbol.** Nothing in any tree
  contains that string: `cann/ge` (owner of `transformation_ops.cc`,
  `infershape_pass.cc`, `ge_api.cc`) is in `SOURCES.json` `unavailable`, and the
  `torch_npu/dynamo/torchair/**` package is not in the pytorch export either
  (`torch_npu/dynamo/` holds only `__init__.py` and `trace_rule.py`). So
  `grep Transpose_5` and "which `.permute` produced it" walks cannot converge —
  recorded #5484 spent ~10 calls grepping `vllm_ascend/torchair/` transpose sites
  and shipped no answer.
- The message does say one thing: a traced tensor is **rank 1** at that node, i.e.
  a `permute`/`transpose(0,1)` met a `[N]`-shaped operand — a per-channel scale
  vector, a squeezed activation, or a shape that symbolic/dynamic inference
  collapsed. Which call site is a **confirmation**, not the answer.

## Step 1 — the two ablations that split the axes (this is the answer's spine)

Take both from the `--additional-config` line; do not add flags beside it:

1. **Graph off**: delete the `"torchair_graph_config"` key (keep
   `ascend_scheduler_config`) and restart. `--enforce-eager` is *not* this
   ablation — `ascend_config.py:285` at v0.11.0 raises `Can't enable graph mode and
   eager mode at the same time. Please set 'enforce_eager=False' …`, so a launch
   carrying both produces a second, unrelated startup failure that reads as "the
   fix did not work". Serving in eager/ACL-graph mode confines the defect to the
   torchair path and is itself a usable deployment.
2. **Validated checkpoint, identical command**: run the recipe's own weights
   (`vllm-ascend/DeepSeek-V3.2-Exp-W8A8`) through the same `vllm serve` line. A
   clean compile makes the checkpoint's quant description the owner; a failure
   makes the stack/code the owner.

`graph_batch_sizes` is **not** the missing-fix lane at this pin: the fix
"deepseek with torchair doesn't work as expect when `graph_batch_sizes` is set"
(#2760) is already in `release_notes.md`'s v0.10.2rc1 section, i.e. older than
v0.11.0. Lower the value only as a cheap third row, not as the theory.

## Step 2 — the checkpoint axis: a rotated/third-party quant format

`--quantization ascend` is driven by the checkpoint's own
`quant_model_description.json`, and the tree at v0.11.0 has **no** rotation
handling at all: `rg -il 'quarot|qua_rot|hadamard' vllm_ascend/` → 0 hits. **That is
a v0.11.0 fact and not a property of the project** — the same grep returns nine
files at v0.18.0rc1, `patch/worker/patch_draft_quarot.py` among them (landed with
the `## v0.16.0rc1` section, `release_notes.md:74,132`), so a later report about a
rotated checkpoint or a dropped spec-decode acceptance rate is a draft-basis
question, never "the engine cannot handle rotations":
`../spec-decode-mtp/quarot-rotation-basis-draft-acceptance.md`. A name like `DeepSeek-V3.2-w8a8-QuaRot` is therefore outside
the validated set that the tutorial tab names, and its description can route
modules to a scheme (`FLOAT`/`W8A8` vs `W8A8_DYNAMIC` rows) whose fused path the
torchair graph never exercised. Two checks, one command each:

- `python - <<'P'` over the checkpoint dir printing, per module family, the
  `quant_model_description.json` scheme values plus `config.json`'s
  `model_type`/`quantization_config` — compare with the validated release's.
- The Step 1.2 swap. Quote the tutorial tab's literal checkpoint id, not "the
  official one".

Keep the mechanism open: the paste cannot show whether the rank-1 tensor comes
from the rotation, from a `FLOAT`-rowed linear, or from a scale vector.

## Step 3 — diff the launch against the tab that validates this shape

For DeepSeek-V3.2 W8A8 on A2, `docs/source/tutorials/multi-node_dsv3.2.md`
(A2 W8A8 tab at v0.11.0, `:298-340`) is the reference: `--data-parallel-size 2
--data-parallel-size-local 1 --tensor-parallel-size 8 --enable-expert-parallel`,
`--max-num-seqs 16`, `--max-model-len 17450`, `--max-num-batched-tokens 17450`,
`--gpu-memory-utilization 0.9`, `PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"`,
and the same `torchair_graph_config` block. Report the delta as a list (recorded
#5484 differed on checkpoint, `max_model_len`/`max_num_batched_tokens` 8096,
`max_num_seqs` 8, util 0.92, and dropped the alloc-conf export) and mark which
deltas can plausibly change **ranks**: the checkpoint can, a smaller token budget
cannot make a rank-1 tensor.

Coverage note worth one line: at v0.11.0 no `tests/e2e/**` case compiles
DeepSeek-V3.2 in torchair mode (`rg -ln -i 'v3\.2|v32' tests/` → only
`tests/ut/torchair/test_utils.py`), so the recipe tab is docs-only and the CI
signal for this cell is absent.

## Step 4 — localise the node only after the axes are split

Ask the reporter for the **GE graph dump** of the failed compile (torchair's own
dump option in the installed `torchair`/`torch_npu` version — read it off
`torchair.get_npu_backend`/`CompilerConfig` in *their* tree, since that package is
not exported here), then read node `Transpose_5`'s input shape and its producer
node. The producer's Python `path:line` in `vllm_ascend/torchair/**` is the patch
site, and it is worth requesting only for a reporter who must stay on torchair;
otherwise the deliverable is the eager/ACL-graph deployment plus the checkpoint
answer.

## Delivery

The G1 message is three sentences: this is a GE shape-inference refusal inside the
torchair-compiled DS3.2 forward (host-side, before any kernel), so not hardware;
the two ablations that split it are *remove the `torchair_graph_config` key* (not
`--enforce-eager`) and *the tutorial's own W8A8 checkpoint with the same command*;
then the recipe delta list and the dump request. The recorded #5484 run had that
verdict by its 35th turn, kept hunting the producing `.permute`, and spent its
remaining output budget on a final 10.9 K-character reasoning block: 1.2 K
characters of message text total, cut at 16 384 tokens with the whole ranked
answer unsent. Version-upgrade rung: the reporter rebuilt `0.11.0-dev` from source
and reported the same error (public follow-up, 2025-12-30), so "your tag is old"
is **spent** here — rank the checkpoint/flag axes, not the tag.
