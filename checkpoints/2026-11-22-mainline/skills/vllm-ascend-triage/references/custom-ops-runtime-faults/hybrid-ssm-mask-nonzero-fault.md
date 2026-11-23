# A `NonZero` device fault from the hybrid-SSM state mask

## Scope

Lane C: a serve that had been healthy for **hours** dies with a built-in CANN
`NonZero` kernel, on a hybrid linear-attention model (GDN/Mamba — Qwen3-Next,
Qwen3.5). Recorded #7774 at vllm-ascend **v0.17.0rc1** (`e20f0b1a`) + vllm v0.17.0
(`b31e9326`), CANN 8.5.1, torch/torch_npu 2.9.0, Qwen3.5-35B-A3B, TP8, `--dtype
bfloat16`, `--max_model_len 136000`, `--enable-prefix-caching`,
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`,
`--async-scheduling`. The reporter closed the issue on **0.19.1rc1** ("solved by the
latest version"). Inherit `index.md`, `../first-checks/signature-routes/index.md`
and the SKILL.md rules.

**Variant — the paste names no kernel, and the onset is engine start (recorded #4483).**
Qwen3-Next-80B-A3B at vllm-ascend **v0.11.0rc1**, many TP/DP ranks:
`RuntimeError: copy between host and device_opapi, UCE ERROR, error code is 507053`,
raised from `core.py` → `collective_rpc("determine_available_memory")` →
`worker_v1.py profile_run()` → `model_runner_v1.py` `hidden_states[logit_indices]`, so
the engine never finishes profiling (`worker proc VllmWorker-0 died unexpectedly`,
`EngineCore failed to start`) and the title says "multi-request concurrency". Same
owner line, older layout: at that pin the masked clear sits in the **model file**,
`vllm_ascend/models/qwen3_next.py:331` `initial_state[~has_initial_state, ...] = 0`
under `if attn_metadata.num_prefills > 0:` (verified in the v0.11.0rc2 tree
`a2e4c3fe`) — the `patch/worker/patch_qwen3_next.py` copy is the newer pin's layout,
so grep both spellings. It fires at `profile_run` because that dummy run *is* a
prefill. The reporter closed the issue as fixed for `>= v0.13.0rc1` with no mechanism
named, so the owner here is **hypothesis**; what the record settles is the lane —
this paste is a version question, not a card fault.

## What the paste proves

| Fragment | Pins down | Does not |
|---|---|---|
| `Kernel Run failed. opType: 59, NonZero` + `launch failed for NonZero, errno:507053` | the named kernel **is** the faulting one — a CANN built-in, so no `csrc/<op>/` exists to read and "which custom op" is not a check | where in Python it was launched |
| `errorStr: The DDR address of the MTE instruction is out of range`, `error code = 0x800000`, `subErrType:4`, `retCode=0x31, [vector core exception]`, several `core id`/`blk` numbers | an aivec read past the tensor it was given — the input descriptor, not the card | a per-card health problem: `blk:595/596` is which AIV core reported |
| `synchronize stream failed, runtime result = 507053` / `aclrtSynchronizeStream failed` | the *reporting* point: the first host sync after the kernel | a second fault |
| `rtGetMemUceInfo … 507899` + `checkUceErrAndRepair` (printed as `[rank1]:… error code is 507899`) | torch_npu's post-mortem UCE query — the driver's answer to "was this memory poison", not the cause | anything about HBM health until the plog says so |

Nothing here is a capture-phase refusal (`107030`, `CoreDim`) and nothing is an HCCL
collective: the collective lanes in `../hccl-and-hardware/index.md` do not apply
even when `507899` appears, because the same code is what
`rtGetMemUceInfo` returns for any device exception.

## Where a device `NonZero` comes from in this tree

Do **not** grep `nonzero`: at v0.17.0rc1 that returns host-side `np.nonzero`
(`worker/model_runner_v1.py:794`, `vllm/v1/.../cross_attention.py`), EPLB's
`valid.nonzero(as_tuple=True)` and unrelated multimodal/model files. The op is
produced by a **boolean-mask advanced index** (`x[~mask] = 0`, `t[mask].contiguous()`),
which torch_npu lowers to `NonZero` (indices) + an index write. Grep the spelling:

```bash
rg -n '\[~' vllm_ascend/vllm_ascend/            # 3 sites at v0.17.0rc1
rg -n 'has_initial_state' vllm/ vllm_ascend/    # the mask definition
```

The sites that matter at that pin:

- `patch/worker/patch_qwen3_5.py:180` — `initial_state[~has_initial_state, ...] = 0`
  in the Ascend copy of `Qwen3_5GatedDeltaNet._forward_core`, inside
  `if attn_metadata.num_prefills > 0:` (`:178`), right after
  `initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()`.
- `patch/worker/patch_qwen3_next.py:239` — the same line for Qwen3-Next.
- Upstream twins, if the patch is not the active path at your pin:
  `vllm/model_executor/models/qwen3_next.py:776`, and the KDA form
  `layers/kda.py:406` (`zero_idx = non_spec_state_indices_tensor[~has_initial_state]`).
  The mask itself is built in `vllm/v1/attention/backends/gdn_attn.py:295`
  (`has_initial_state = context_lens_tensor > 0`, prefill only, filtered by
  `~spec_sequence_masks` when a drafter is on).
- Other producers to rule out from the serve line, not from source: EPLB
  (`--enable-eplb`, `eplb/core/eplb_worker.py:216,237`) and PCP
  (`worker/pcp_utils.py:297`). A `NonZero` on a serve with none of the three
  configured is a report gap to state, not a card fault.

Two consequences of that placement, and they are why the shape reads as a
long-run crash: the clear runs **once per GDN layer on every prefill step** — any
new request's first chunk, including a partial prefill from a prefix-cache hit —
while the decode graph never runs it, so the failure tracks request arrivals and
not uptime. `#7863` is the same owner with a *different* reporting point (the
data-dependent `Nonzero` refused **at capture**); cite it for symptom and owner,
never for the line.

## The version lane is the deliverable

- At **v0.19.1rc1** the clearing is the Triton `clear_ssm_states`
  (`ops/triton/fla/utils.py:98`) and the masked assign survives only on 310P
  (`_310p/ops/fla/gdn_310.py:195`) — that mapping is carried in
  `../pd-disaggregation/recipe-contract-diff.md` §*fault only at raised scale*.
  #7774's reporter then confirmed the same workload runs on 0.19.1rc1. So on a
  v0.17-era pin the ranked answer is: the hybrid-SSM state clear, no in-snapshot
  fix, upgrade to ≥ v0.19.1rc1 (when only the older pin is exported, the newer
  tree's line is a **check** to name, not a derivation — do not spend turns on it).
- **Two pins, two confirmations, one action.** #7774 → 0.19.1rc1, #4483 → ≥ v0.13.0rc1;
  both reporter-closed as fixed by a later version, neither naming a mechanism. On any pin older
  than the fix, the first action is the upgrade to a release at or past it, and
  card/driver/`npu-smi` rows are the branch taken only if a single-card, single-request
  repro of the same paste faults (the recorded #4483 answer ranked a device UCE second
  and the version lane third — the two closings invert that order).
- State coverage before any mechanism: at v0.17.0rc1 Qwen3.5 has **no row** in
  `docs/source/user_guide/support_matrix/supported_models.md`, while
  `tests/e2e/multicard/4-cards/test_qwen3_5.py` covers 27B and 35B-A3B at **tp4**,
  `max_model_len 4096`, capture sizes `[4,8,12,16]`, plus a FULL_DECODE_ONLY +
  MTP3 case. The reporter's TP8 / 136 K / no-spec / prefix-caching /
  async-scheduling combination is validated by nothing — say it in the first
  paragraph, then answer the version lane.

## Checks, cheapest first

1. **All ranks' kernel lines, not one.** Whether the other ranks print the same
   `NonZero`/`opType: 59` in the same second separates a launch-input problem from
   one card (`../hccl-and-hardware/error-report-timestamps.md`).
2. **The reporter's serve line against the pin's validated one** for that model
   (the e2e file above): the deltas are the discriminators, and each is one flag.
3. **Prefill-step ablations** — `--no-enable-prefix-caching`, a lower
   `--max-num-batched-tokens`, `--max-num-seqs 1` — change *how often* a prefill
   step reaches the masked clear. They are documented here as **regime tests, not
   fixes**: no recorded run reports them as a workaround, and the semantic path
   stays.
4. **The release.** The only confirmed answer in this family, at two pins (#7774 →
   0.19.1rc1, #4483 → ≥ v0.13.0rc1) — take it before asking for per-card health.

## Stop rules

- The kernel lives in CANN (`cann/ops-*`, `cann/opbase` are `unavailable` in
  `SOURCES.json`), so *why* the address arithmetic overran cannot be derived here.
  A recorded session spent 19 turns and 45 K characters of reasoning on that
  derivation (MTE-out-of-range branches, allocator/graph-replay theories, async
  scheduling ordering) with the launching line already found on turn 8 and 39
  characters of message text delivered — `../evidence-workspace/delivery-discipline.md`.
- `ASCEND_LAUNCH_BLOCKING=1` buys nothing here — the paste already names the
  kernel. Use it only when the named op is a *victim* (a generic 507xxx with no op
  name, `../hccl-and-hardware/latched-runtime-error-masking.md`).
- Do not convert it into a hardware case: no `npu-smi`, driver or CANN-pairing
  ranking until the same `NonZero` line (or, on the #4483 paste, the same generic
  `507053` at `profile_run`) appears across ranks *and* a single-request repro on one
  card faults. A UCE code alone is not that evidence — `507053` is
  `ACL_ERROR_RT_DEVICE_MEM_ERROR` and is also what a faulting kernel's reporting sync
  returns.
