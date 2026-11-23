# GLM-5/5.1 single-node: the release's own row outranks the operator name

Scope: a GLM-5/5.1 (`glm_moe_dsa`) serve that starts, serves traffic, then dies on a device
record — `fftsplus aicore error` or `fftsplus aivector error`, `error code = 0x800000`,
`The DDR address of the MTE instruction is out of range`, `Failed to submit kernel task,
retCode=0x7150050`, `runtime result = 507011` — typically on
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` with
`--speculative-config '{"method": "deepseek_mtp", …}'` and/or `--enable-expert-parallel`, on
the 0.19–0.21 line. The first owner is **not** the operator the launcher names: it is the
reported release's own `### Known Issues` row, then the issue that row tracks.

Pin-checked in the vllm-ascend **v0.19.1rc1** tree (`da421afa`); the #8843 contents are the
issue **body** a recorded read returned at that pin (bodies only — comments are outside the
snapshot, so treat any "resolution" as unverified).

## The row

`docs/source/user_guide/release_notes.md`, section `## v0.19.1rc1 - 2026.04.30` (`:3`),
`### Known Issues` (`:199`), `:201`:

> When running GLM-5 / GLM-5.1 models in single-node (non-PD-disaggregated) scenarios,
> incorrect results or runtime errors may occur. See #8843 for details and workarounds.

`:203` in the same block is the pairing the image does not ship: torch-npu
`2.9.0.post1+git4c901a4`, and `2.9.0.post2` once CANN is 9.0.0.

Search by **wording, not line number.** A later tree that keeps the section keeps the row at a
shifted line — at v0.21.0rc1 (`80610e44`) the section starts `:188` and the row sits at `:386`
(torch-npu `:388`). The line also grows newer GLM rows (v0.21.0rc1 `:60` "GLM5/GLM5.1 W4A8
deployments have known issues in some advanced configurations" #9395, `:63` MTP3 + FlashComm
acceptance #9803, `:62` the PD 200K `MoeDistributeDispatchV2` hang #9958) — quote the row whose
own `## v…` heading covers the reporter's build, and do not carry one section's rows onto
another.

## What the tracked issue adds: the operator name is not the owner

#8843's body is a v0.19.1rc1 launch of `GLM-5.1-w8a8`: `dp1`, `tp16`,
`--enable-expert-parallel`, `--max-num-seqs 8 --max-model-len 40960`,
`--quantization ascend`, chunked prefill + prefix caching + `--async-scheduling`,
`FULL_DECODE_ONLY` with `enable_npugraph_ex`, `{"num_speculative_tokens": 3, "method":
"deepseek_mtp"}`, and `HCCL_OP_EXPANSION_MODE="AIV"`, `HCCL_BUFFSIZE=200`,
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, `VLLM_ASCEND_BALANCE_SCHEDULING=1`,
`VLLM_ASCEND_ENABLE_MLAPO=1`; its startup prints `Enabled custom fusions: norm_quant,
act_quant`. It dies with:

- `fftsplus aicore error, core id 12, error code = 0x800000` on `chipId:1` and, in the same
  second, `core id 4` on `chipId:4` — two dies, `cube error info: 0`, `subErrType:4`;
- `The DDR address of the MTE instruction is out of range`, `Failed to submit kernel task,
  retCode=0x7150050`, `runtime result = 507011`, `synchronized memcpy failed … 107030`,
  `Invalid_Argument(EL0003)`;
- and the launcher's "current working operator name is **`rejection_greedy_sample_triton`**".

The #8944 paste on the same release carries the same chain but `fftsplus aivector error`,
`chipId:6`, `core 27`, and names **`aclnnAddRmsNormBias`**. So across runs of this class the
operator, the die and the core all move while the device record and the retCode chain repeat:
read the name as *the first launch that noticed a poisoned stream*, and do not rank
`csrc/<op>` kernel arithmetic from it. Field-by-field reading of that record is
`../custom-ops-runtime-faults/aclnn-error-dump-reading.md`; the op-specific closed questions
are `../custom-ops-runtime-faults/add-rms-norm-bias-op-map.md`.

## The "workarounds" the row points at are not in the issue

#8843 is ~46 KB and everything past the environment block is log (collect_env, model download,
the serve block, the fault). Two windows are enough — `/repos/vllm-project/vllm-ascend/issues/8843`
with `offset`/`max_chars` ≈ 14–16 KB: window 1 holds the serve line, window 3 the fault
block. There is no fix paragraph to find, so do not spend a fourth fetch. The row's
"workarounds" resolve to the same release's tutorial: `docs/source/tutorials/models/GLM5.md:256`
— "For single-node deployment, we recommend using `dp1tp16` and turn off expert parallel in
low-latency scenarios".

**The documented single-node recipes are inside the declared set.** `:212-247` (A2 tail) is
`dp1 / tp8 / --enable-expert-parallel` with `FULL_DECODE_ONLY`, `deepseek_mtp` 3,
`enable_npugraph_ex`, `fuse_muls_add`, `multistream_overlap_shared_expert` and
`HCCL_OP_EXPANSION_MODE="AIV"`; `:137-208` (A3 tail) is the same with `tp16`. So "I followed
the tutorial" is not a defence, and the ablations that tutorial recommends are the first rungs.
An 8×`--device=/dev/davinciN` docker line is eight NPUs (one rank each) — the A2 single-node
shape — not evidence against it.

## Ladder, before any kernel read

1. Get the **serve** line — the report pastes `docker run` and the log, not the command (#8944
   sent neither `vllm serve` nor its flags). The row is scoped to the scenario (model family ×
   single node), so match it first.
2. `--enforce-eager` **plus** `ASCEND_LAUNCH_BLOCKING=1` on the same load. The tracked log
   itself warns "Since the operator is called asynchronously, the stacktrace may be
   inaccurate"; eager also removes the fusion passes and the captured graph, so the operator
   named afterwards belongs to the failing step.
3. `--enable-expert-parallel` off (the tutorial's own single-node advice), optionally `dp1tp16`.
4. MTP off (`--speculative-config` removed), then `cudagraph_mode` default — two separate runs.
5. `pip show torch_npu` against `:203`; CANN 9.0.0 ⇒ `2.9.0.post2`.
6. Only if the operator is still named after 2–4: that op's page and the failing rank's tiling
   I-line (`add-rms-norm-bias-op-map.md`).

`HCCL_OP_EXPANSION_MODE="AIV"` sits in the documented recipe, so the HCCL AIV-collective lane
is a **discriminator, not a cause**: if the plog's `fault kernel_name=` is
`aiv_reduce_scatter_*` / `aiv_all_gather_*` / `aiv_all_to_all_*`
(`../hccl-and-hardware/aiv-collective-kernel-fault.md`), the owner is that template and
unsetting AIV is the ablation, never the fix.

## Limits

- The row is a **scenario-level admission**. It names no kernel and no flag, so it supports
  "this combination may fault at this release" and nothing more — enough to refuse 「坏卡」 and
  "the AddRmsNormBias kernel has an addressing bug", not enough to name the mechanism.
- It does not cover a PD-disaggregated GLM-5.x deployment; the MTP-graph capture variant on this
  same release is `../spec-decode-mtp/draft-graph-capture-device-fault.md` (#9278 / #10700).
- #8843's resolution and any fix PR are outside the snapshot: a row that disappears from a later
  release is not itself a fix (decide that from source or ask, per
  `index.md` rule 6).
