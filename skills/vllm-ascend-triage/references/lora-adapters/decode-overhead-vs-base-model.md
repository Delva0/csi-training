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

## Step 1 — which punica backend does this release bind?

`vllm_ascend/lora/punica_npu.py` binds `bgmv_*`/`sgmv_*` in `__init__`:

| ref | binding | consequence |
|---|---|---|
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
thousand tiny launches per step: the overhead is structural, not a flag. On
A2/A3 with ACL graph the same adapter is far cheaper — which is why a "LoRA is
slow" report must state the SoC and the graph mode before any cause is ranked.
Check the project's support claim while you are here: `LoRA | 🟢 Functional` in
`docs/source/user_guide/support_matrix/supported_features.md:11` is a
*functional* claim with no performance statement, and the LoRA guide's own tip
at v0.9.2rc1 is "if you fail to run … with LoRA, disable graph mode and try
again" (`docs/source/user_guide/feature_guide/lora.md:10-11`) — at v0.13.0rc2
the same page instead says LoRA runs with ACLGraph and enables it in the
example (`:8,16`). Quote the page at the reported ref, not the newer one.

## Step 3 — the one A/B that splits per-step from per-adapter-token

On the **same** `--enable-lora` server, send the identical prompt to the base
model name. Base-only decode steps still enter the punica wrapper, and at
v0.9.2rc1 only the prefill variants short-circuit on `no_lora`
(`punica_npu.py:34,61,90`; `_shrink_decode:44` and `_expand_decode:71` have no
guard), so a base request that is **also** ~2 tok/s confirms a per-step cost
paid by all traffic; a base request that is still ~9 tok/s means the cost is
per adapter token and the rank / `target_modules` / dtype matter more. Label
this a hypothesis until that A/B runs — newer refs added guards, so re-grep
`no_lora` in `punica_npu.py` at the reported ref.

## Step 4 — the ladder

1. The base-model request on the LoRA server (above): 2 minutes, decides
   per-step vs per-token, and tells the reporter whether their base traffic is
   affected too.
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
