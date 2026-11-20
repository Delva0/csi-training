# Device fault surfacing at the drafter's valid-count D2H copy (#10546)

## Signature

A device dump arrives inside a Python traceback whose deepest `vllm_ascend`
frame is the **valid-count copy to host**:

- `vllm_ascend/worker/model_runner_v1.py:1393 counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)`
  in `_copy_valid_sampled_token_count` (`:1379-1398`), reached from
  `worker.py sample_tokens` → `model_runner_v1.py sample_tokens` →
  `propose_draft_token_ids` (`:1628` guarded by `disable_padded_drafter_batch`
  at `:1627`, or `:1498`, or the `elif` branch at `:2217`);
- batched with `Execution_Error_SDMA(EI0012)` / `rtMemcpyAsyncEx execution
  failed, reason=the model stream execute failed`, an `aivec error … timeout or
  trap error … subErrType:4` block, and an HCCL task line carrying
  `tag:[AllGather_group_name_<N>]`, `size`, `numBlocks`, `rankSize`,
  `isOpbase:[1]`, `flagMem`, `Remote rank: [n]`;
- the dump's own advice block — `1. Network connection exception 2. The peer
  process exits abnormally 3. The input or output memory address is not
  allocated …` — plus `Since the operator is called asynchronously, the
  stacktrace may be inaccurate`.

Recorded #10546: vllm-ascend `v0.20.2rc1` / vllm `v0.20.2`, DeepSeek-V4-flash
1P1D through `load_balance_proxy_server_example.py`, P launched with
`launch_online_dp.py --dp-size 1 --tp-size 8` and D with `--dp-size 2 --tp-size 4
--dp-size-local 2` (the worker name `Worker_DP1_TP1_EP5` also implies EP), 10-way
concurrency, `srclen=24, dstlen=24` (six `int32` rows on that DP rank,
`counts.shape[0] == 6`). The launcher's `--template-path ./run_dp_template_decode.sh`
contents — the serve flags and env the D ranks actually ran with — are **not**
pasted.

## The frame is a sync point — and it is also free evidence

Read the sync-point rule first (`../hccl-and-hardware/collective-error-chain.md`):
the message says the *model stream* failed, so the copy is where an earlier
asynchronous collective gave up, not a slicing bug in `counts_cpu`. That is the
first sentence of the answer (`SKILL.md` rule 1).

Then use the frame to close a gap the reporter usually leaves open. The event,
the side stream and the pinned CPU buffer are created **only** when spec decode
and async scheduling are both on, and the function returns early otherwise:

- vllm `vllm/v1/worker/gpu_model_runner.py:831` `if self.num_spec_tokens:` →
  `:840-846` `if self.use_async_scheduling:` creates
  `valid_sampled_token_count_event`, `…_copy_stream` and `…_cpu`
  (`int32[max_num_reqs]`, pinned);
- `:593-595` `use_async_spec_decode = use_async_scheduling and num_spec_tokens > 0`;
- vllm-ascend's copy is the upstream body with `torch.npu` for `torch.cuda`
  (`gpu_model_runner.py:4472-4492`, copy at `:4486`), so the same guard applies.

So a paste that lands on that `copy_` line **proves** the launch shape:
`--async-scheduling` plus a spec-decode `--speculative-config`, with the padded
drafter batch on — `disable_padded_drafter_batch: bool = False` at vllm
`vllm/config/speculative.py:120`, guard at `model_runner_v1.py:1627`. Say that
instead of asking whether async scheduling was used. What is still unknown is the
flag set the launcher's template rendered (`run_dp_template_decode.sh`: the
shared-expert/multistream knobs, `HCCL_BUFFSIZE`, `VLLM_ASCEND_*`), the twin DP
engine's log and the plog.

## Resolve the collective by rank size, not by group name

`AllGather_group_name_<N>` is a ProcessGroup-level name whose number is an
internal index — it does not name a vllm-ascend group. Compare `rankSize` with
the launch's DP / PCP / TP / EP sizes. Verified sizes at this pin:

- `mc2` and `fc3_quant_x` are both built from the EP-like `group_ranks`
  (`vllm_ascend/distributed/parallel_state.py:85-110`), i.e.
  `DP × PCP × TP` ranks; `fc3_quant_x` exists only when
  `multistream_overlap_gate` is set (`:105-108`), and
  `fc3_all_gather_and_maybe_unpad_impl` (`distributed/utils.py:10-36`) then
  gathers over it and un-pads by hand from `num_tokens_across_dp_cpu`.

So on DP2×TP4 a **two-rank** collective is *not* the shared-expert/fc3 group —
the recorded run was already drafting a mechanism around "shared-expert DP
AllGather" from the rank size alone, and the arithmetic refutes it. The
size-2 candidates are the DP group itself (or TP/EP when those are 2). Rank size
narrows the set; it does not establish the owner (the hccl topic's rule 3
applies to group sizes too).

## Cheapest checks, in order

1. `--enforce-eager` **then** `ASCEND_LAUNCH_BLOCKING=1` re-run, so the
   traceback names the real op instead of the copy (blocking mode is refused
   while any graph mode is on: the gate is in `AscendPlatform.check_and_update_config`,
   `compilation_config.cudagraph_mode != CUDAGraphMode.NONE` **and**
   `ASCEND_LAUNCH_BLOCKING == "1"` → `ValueError: ACL graph is incompatible with
   ASCEND_LAUNCH_BLOCKING=1` (`vllm_ascend/platform.py:457-469` at v0.20.2rc1;
   grep the message, the line moves between releases).
2. One request or `--max-num-seqs 1`: splits batch geometry / rank imbalance
   from hardware.
3. `--speculative-config '{… "disable_padded_drafter_batch": true}'` removes
   this frame (draft-side only, one launch); `--async-scheduling` off removes
   the copy entirely because the event stays `None`; MTP off removes the draft
   path (the spec-decode topic's row 1).
4. The twin DP engine's log and the P-side log for the same second — the dump's
   canned cause 2 (peer exit) is only real if another rank died first — and the
   plog on the `chipId` the dump names. Canned cause 3 (a freed buffer) is worth
   one look at the collective's `flagMem`, not a mechanism on its own.
5. Diff the launch against the project's validated command before proposing any
   source fix; for DeepSeek-V4 that is the nightly yaml below.

## DeepSeek-V4 MTP launch gates at this pin

- `VLLM_ASCEND_APPLY_DSV4_PATCH` (`vllm_ascend/envs.py:116`, default `0`) gates
  `patch/platform/__init__.py:46-48`, which imports `patch_kv_cache_coordinator`
  and `patch_speculative_config`. That second patch's `hf_config_override`
  (`patch_speculative_config.py:15-24`) remaps
  `deepseek_v3 / deepseek_v32 / deepseek_v4 / glm_moe_dsa` →
  `model_type="deepseek_mtp"`, and for `deepseek_v4` also
  `architectures=["DeepSeekV4MTPModel"]`. Without the flag the *draft* config
  keeps the base model type, so an MTP launch is not the validated path even
  when the engine starts. The tree's own tests set it
  (`tests/e2e/multicard/4-cards/test_deepseek_v4.py:31,72`).
- At this ref there is **no** DeepSeek-V4 row in
  `docs/source/user_guide/support_matrix/supported_models.md` and no
  `docs/source/tutorials/models/DeepSeek-V4*.md`; the only in-tree recipe is
  `tests/e2e/nightly/single_node/models/configs/DeepSeek-V4-Flash-W8A8-A3.yaml`,
  scoped to A3 — a reporter on 910B/A2 is a deviation to name, not a verdict.
  It sets `VLLM_ASCEND_APPLY_DSV4_PATCH=1`, `VLLM_ASCEND_ENABLE_FUSED_MC2=0`,
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, `USE_MULTI_BLOCK_POOL=1`,
  `USE_MULTI_GROUPS_KV_CACHE=1`, `HCCL_BUFFSIZE=1024`, runs
  `--speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}'`,
  `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`,
  `--block-size 128`, `--async-scheduling`, and
  `--additional-config '{"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,…}'`.
  Read the file, do not retype the env set from here.

## Related precedent (comparison rows, not causes)

- #9482 (v0.19.0/main): `--async-scheduling` + APC + MTP overflowed
  `num_accepted` / `num_computed_tokens`, and the reporter attributed it to
  `_copy_valid_sampled_token_count` — PR #8766 had moved the `current_stream()`
  read *inside* the `with torch.npu.stream(...)` block. In the pinned tree the
  read is outside (`model_runner_v1.py:1387-1389`), i.e. the form #9482 objects
  to is present; its symptom is bookkeeping overflow, not a device trap, so it
  is a row to compare, not the cause here
  (`async-spec-bookkeeping-desync.md`).
- #8784 (same model family, TP8 + EP, `deepseek_mtp`, `FULL_DECODE_ONLY`,
  `multistream_overlap_shared_expert`, 48 concurrency): `aivec error … The DDR
  address of the MTE instruction is out of range … subErrType:4` under
  `AclrtSynchronizeDeviceWithTimeout 507035` — the same error family at a
  different sync point, which is why the attribution run decides ownership.

## Scope, and what this page does not establish

Verified at vllm-ascend `367b8e6` (`v0.20.2rc1`) + vllm `bc150f5` (`v0.20.2`).
Version-specific, re-grep after an upgrade: the event-creation guard and
`use_async_spec_decode` (upstream), the group arithmetic in
`distributed/parallel_state.py`, and the DSV4 env/patch names.

The recorded #10546 run established the sync-point reading and the config facts
above but **shipped no answer** (14 assistant turns, every message a one-line
status note; `findings.md` written on turn 12, the last turn reasoning-only), and
it never resolved the trapped two-rank collective to a component. So deliver the
frame as a sync point, the proven launch shape, the ranked candidates (peer exit
/ rank-divergent collective / capture-replay overlap of the side-stream copy),
the attribution run first, and name the missing artefacts: the rendered serve
command from `run_dp_template_decode.sh`, the twin DP engine's log, the P-side
log and the plog. Do not hand the reporter one mechanism from this page.
