# The base model is N tok/s and the LoRA serve is a fraction of that

## Symptom and the arithmetic that comes first

Recorded #1812 (v0.9.2rc1, Qwen3-8B, Atlas 300I Duo / 310P, `--dtype float16`,
`--enforce-eager`, multi-LoRA): the base model reaches **9 tok/s**, the same
model with `--enable-lora` and an adapter request reaches **2 tok/s**. No error,
no OOM, single stream, and the reporter already knows that `--enforce-eager` is
mandatory on this SoC.

Convert before theorising: 9 tok/s = 111 ms/step, 2 tok/s = 500 ms/step, so the
adapter adds ≈390 ms **per decode step**. A rank-16 adapter on a dense 8B adds
on the order of 1 % of the model's FLOPs. A 4–5× step time is therefore
dispatch, a backend change or a lost graph — never "LoRA is expensive".

Recorded #1629 (vllm-ascend v0.9.1rc1 `c30ddb83` + vllm v0.9.1 `b6553be1`,
**2×910B3 TP2**, DeepSeek-R1-Distill-Qwen-32B = dense 64-layer Qwen2, CANN
8.1.RC1): the same ratio on a graph-capable SoC — base-only serve **10-12
tok/s**, the same model under `--enable-lora --max-loras 3` and one adapter
**2-3 tok/s** (`Avg generation throughput` 2.4 → 1.6 in the server metrics),
reporter-confirmed. 83-100 ms/step versus 333-500 ms/step, i.e. the adapter
costs ~250-400 ms **per step**. Both halves of that report — the slow serve and
the `VLLM_USE_V1=1` serve that will not start — are one cause: the PyTorch
punica binding (`pytorch-punica-compile-guard.md`).

## Step 1 — which punica backend does this release bind?

`vllm_ascend/lora/punica_npu.py` binds `bgmv_*`/`sgmv_*` in `__init__`:

| ref | binding | consequence |
|---|---|---|
| v0.9.1rc1 `c30ddb83` | unconditional `from vllm.lora.ops.torch_ops import …` at `vllm_ascend/lora/punica_wrapper/punica_npu.py:6` (the module sits under `lora/punica_wrapper/` at this ref; later refs use `vllm_ascend/lora/punica_npu.py`); `platform.py:236` returns `PunicaWrapperNPU`; no AscendC LoRA kernel in the tree | same PyTorch per-apply cost, **and** the math is dynamo-visible, so the `VLLM_USE_V1=1` + `PIECEWISE` path cannot even start (`pytorch-punica-compile-guard.md`) |
| v0.9.2rc1 `b5b7e0e` | `from vllm.lora.ops.torch_ops import …` unconditionally (`punica_npu.py:6`); there is no `vllm_ascend/lora/lora_ops.py` in the tree and no `csrc/kernels/bgmv_*.cpp` | every apply is PyTorch: gather the adapter weight stack, cast, `einsum`, scaled strided write-back (`vllm/lora/ops/torch_ops/lora_ops.py:68-80` at vllm v0.9.2 `a5dd03c`), per slice |
| v0.13.0rc2 `a17059a` | AscendC `vllm_ascend/lora/lora_ops.py` (`csrc/kernels/bgmv_*.cpp` shipped) **unless** `get_ascend_device_type() == AscendDeviceType._310P` **or** `lora_config.max_lora_rank >= 128` (`punica_npu.py:27-39`) | 310P still takes the PyTorch path; so does any A2/A3 serve whose `--max-lora-rank` crosses 128 — a trap, since the obvious "be generous with the rank" advice makes it slower |

`bash <skill_root>/scripts/lora-graph-descriptor-audit.sh <vllm-ascend-root>`
part 1 prints this binding plus its predicate in one call, so the version
question does not need a source walk each time.

## Step 2 — can this SoC amortise the launches at all?

- **300I series is eager-only in the tree's own tutorial**:
  `docs/source/tutorials/single_node_300i.md:194,237` — "For 300I series, only
  eager mode is supported", with `enforce_eager=True` in both the offline and
  the API examples. The generic graph switch agrees: `use_aclgraph =
  compilation_config.level == PIECEWISE and not enforce_eager`
  (`vllm_ascend/worker/model_runner_v1.py:316-318`). **Version bound on that
  premise**: it is the v0.11-and-earlier tutorial; from at least v0.18.0 the 310P
  tutorial (`docs/source/tutorials/hardwares/310p.md:7-11`) documents
  `FULL_DECODE_ONLY` on 300I Duo with TP>1 capture-count limits, so check the
  reporter's release before calling the launch overhead structural
  (`../soc-platform-paths/atlas-300i-310p.md`, v0.18.0 block).
- **310P also loses the torch-npu fused ops**: `vllm_ascend/platform.py:206-208`
  (`if not is_310p(): compilation_config.custom_ops = ["all"]`), which is why the
  300I tutorial hand-lists `["none", "+rms_norm", "+rotary_embedding"]`. The
  package's *own* AscendC kernels are a separate, build-time switch: a `310`
  SOC build must set `COMPILE_CUSTOM_KERNELS=1` (`setup.py:88-94`).

So on 310P there is no graph and no fused custom-op path to amortise a few
thousand tiny launches per step: the overhead is structural, not a flag.

On A2/A3 the platform *can* compile and capture — but that only helps if the
**adapter path is graph-visible**, and that is a property of the punica binding,
not of the SoC. Where the binding is transitive PyTorch there is no graph to be
had at any SoC: #1629's `VLLM_USE_V1=1` serve died in the profile run's first
compiled forward, and the run that served was the eager one, paying the same
per-slice dispatch as 310P. So a "LoRA is slow" report needs the SoC, the graph
mode **and** the binding before any cause is ranked — and "A2 has ACL graph" is
never by itself the reason an adapter would be cheap.

Check the project's support claim while you are here: `LoRA | 🟢 Functional` in
`docs/source/user_guide/support_matrix/supported_features.md:11` is a
*functional* claim with no performance statement, and the LoRA guide's own tip
at v0.9.2rc1 is "if you fail to run … with LoRA, disable graph mode and try
again" (`docs/source/user_guide/feature_guide/lora.md:10-11`) — at v0.13.0rc2
the same page instead says LoRA runs with ACLGraph and enables it in the
example (`:8,16`). Quote the page at the reported ref, not the newer one.

## Step 3 — the one A/B that splits per-step from per-adapter-token

Two different measurements, and only the second decides the mechanism:

- **A base-only serve** (`vllm serve <model>` with no `--enable-lora`) is the
  upper bound and the lane gate: #1629's 10-12 tok/s there versus 2-3 tok/s with
  the adapter is what proves the LoRA path owns the cost. If this number is also
  a few tok/s, leave this page for `../performance-throughput/index.md`.
- **A base-model *request* on the same `--enable-lora` server** is the mechanics
  test: send the identical prompt to the base model name. Base-only decode steps
  still enter the punica wrapper, and at v0.9.2rc1 (likewise v0.9.1rc1: guards
  at `punica_npu.py:34,61,90`, none in `_shrink_decode:44` / `_expand_decode:71`)
  only the prefill variants short-circuit on `no_lora`, so a base request that is
  **also** ~2 tok/s confirms a per-step cost paid by all traffic; a base request
  that is still ~9 tok/s means the cost is per adapter token and the rank /
  `target_modules` / dtype matter more. Label this a hypothesis until that A/B
  runs — newer refs added guards, so re-grep `no_lora` in `punica_npu.py` at the
  reported ref.

## Step 4 — the ladder

0. If `VLLM_USE_V1=1` on this command will not start at all, stop here and read
   `pytorch-punica-compile-guard.md` first: the same binding produces both halves
   and its `--enforce-eager` rung is the one line that gets the serve up.
1. The base-only serve, then the base-model request on the LoRA server (above):
   2 minutes each, decide the lane and then per-step vs per-token, and tell the
   reporter whether their base traffic is affected too.
2. Concurrency sweep (1 / 4 / 8 concurrent): dispatch cost is roughly constant
   per step, so aggregate tok/s should scale even while single-stream does not —
   if it does, the deployment is usable at batch > 1 without any change.
3. `--max-loras 1 --max-lora-rank <the adapter's real rank>`: shrinks the
   gathered adapter tensors. Small gain at v0.9.2rc1; at v0.13.0rc2 keep the
   rank **below 128** or the backend flips back to PyTorch (step 1).
4. **Merge the adapter offline** (`peft` `merge_and_unload`, matching dtype) and
   serve without `--enable-lora`. When the backend is PyTorch and the SoC cannot
   capture, this is the only route to base-model speed; it also gives the honest
   "what does LoRA cost me" comparison. Cost: no dynamic adapter swapping.
5. Only then a feature request (AscendC LoRA ops for that SoC, or a
   graph-capable LoRA path), attached to a profile.

## Traps

- **PR #1591 is not the fix.** It is this release's own "Atlas 300I series
  performance improved" highlight (`docs/source/user_guide/release_notes.md:9`,
  v0.9.2rc1, 2025-07-11) — the reporter already has it, and the release notes
  name no LoRA/310P performance work.
- Do not propose dropping `--enforce-eager` on 300I: it is required, and the
  graph path is what is unavailable.
- `bgmv_*` moved from `vllm/lora/ops/torch_ops.py` to the package
  `vllm/lora/ops/torch_ops/lora_ops.py`; on a partial export a `sed`/`rg` on the
  old path fails and looks like "the op does not exist".
- Model-shape correction belongs in the answer, not in the estimate: count the
  LoRA-target linears at the pinned model implementation (fused
  `qkv_proj`/`gate_up_proj` collapse several targets into one call with several
  slices) before quoting a per-step call count.
