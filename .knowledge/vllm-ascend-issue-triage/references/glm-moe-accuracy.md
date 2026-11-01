# GLM MoE accuracy and garbled-output reports (GLM-4.5/4.5-Air, GLM-5.x)

Scope: GLM MoE reports where simple prompts answer and complex/long prompts degrade, output turns
into a repeated single token, or wrong content shows up only sometimes. Verified by reading tags in
the clones plus the upstream PRs on 2026-11-01; nothing was executed on an NPU. In the 4.x line two
independent defects with different release boundaries live behind one symptom — separate them before
answering, because the obvious "upgrade one rc" advice fixes one and not the other. In the 5.x line
the class to check first is the dummy/graph-capture `slot_mapping` defect at the end of this file.
A non-GLM MoE accuracy collapse that appears only with `data_parallel_size` / expert parallel
is the separate 0.9-line class in [moe-dp-ep-accuracy.md](moe-dp-ep-accuracy.md); the defects
below are GLM-specific.

## Defect A: the partial-rotary rope rotated the whole head (fixed in `v0.10.1rc1`)

- GLM-4.5 and GLM-4.5-Air use partial rotary. Upstream `vllm` `v0.10.0`
  `model_executor/models/glm4_moe.py` builds `get_rope(self.head_dim, rotary_dim=self.head_dim,
  partial_rotary_factor=<config, default 0.5>, ...)`, and `get_rope`
  (`vllm/model_executor/layers/rotary_embedding.py`) then does
  `if partial_rotary_factor < 1.0: rotary_dim = int(rotary_dim * partial_rotary_factor)`, i.e.
  `rotary_dim = head_size / 2` (64 of 128).
- vllm-ascend `v0.10.0rc1` `vllm_ascend/ops/rotary_embedding.py::rope_forward_oot` reshaped the
  whole `q`/`k` to `(tokens, -1)` and called
  `torch_npu._npu_rotary_embedding(positions, query, key, self.head_size, self.cos_sin_cache,
  neox_style)` — the full head width, with a cos/sin cache sized for half of it. The
  pass-through half of every head is rotated as well, so every attention result is wrong once the
  context is long enough for the rotation to matter.
- Fix `93754d806` / PR #2601 ("Fix long context seq accuracy problem for `GLM4.5`") adds a
  `if self.rotary_dim < self.head_size:` branch that slices `[..., :rotary_dim]` and
  `[..., rotary_dim:]`, rotates only the slice, and `torch.cat`s the halves back.
  `git tag --contains 93754d806` starts at **`v0.10.1rc1`**; the commit is absent from
  `v0.10.0rc1`.
- The PR describes the symptom exactly: "long context seq accuracy problem ... When
  `max_tokens=1000`, there is cyclic output problem like `00 00 00 00 ...`" — degenerate repeated
  tokens rather than mixed-language text. Match that against reports of "simple prompt fine,
  complex prompt 胡言乱语".

## Defect B: DP+EP shared-expert reduce (thread-established on `0.10.1rc1`, model fix in vLLM `0.11.0`)

- `vllm-ascend` issue #2767 ("GLM-4.5 Accuracy Issue with DP+EP", opened by a vllm-ascend
  contributor) states:
  with DP2 + TP8 + EP the output is "chaotic, e.g. some confusing symbol", the cause is "the wrong
  implementation of **allgather** `moe_comm_method` when enable EP", and pure **TP16** avoids it.
  Its follow-up adds that `0.10.0rc1` does not show the accuracy issue, so it was introduced by the
  PRs between `0.10.0rc1` and `0.10.1rc1`. Treat the regression window as a maintainer statement,
  not a re-derived fact: both tags already switch `_get_fused_moe_state` per forward
  (AllGather / AllGatherEP / All2All / MC2), so the defect is inside those implementations, not in
  the existence of strategy switching.
- The issue points at plugin PR #2898 as a temporary patch. That PR is `state: closed`,
  `merged_at: null` ("This is only a temporary PR used for quick run"), and the file it adds,
  `vllm_ascend/models/glm4_moe.py`, is absent from `v0.10.1rc1`, `v0.10.2rc1`, `v0.11.0` and
  `v0.12.0rc1`. Do not report it as a merged fix; check `merged_at` before repeating a thread's
  "the PR fixes it".
- The formal fix named there is upstream vLLM PR #24849, commit `c15309a730` "[Model] Apply
  SharedFusedMoE to glm4_moe", merged 2025-09-17, first release tag **`v0.11.0`**. Its description
  gives the mechanism: the plugin picks an all2all-based or all-gather-based MoE strategy by
  `num_tokens`, so whether the shared experts are all-reduced together with the router experts
  changes mid-service, while `glm4_moe` fixed that decision at `__init__` through
  `must_reduce_shared_expert_outputs` — the two decisions conflict, which the PR names as an
  "all-reduce related accuracy problem": the shared-expert output is reduced inconsistently with the
  strategy actually in use (its wording, not a re-derived count). `SharedFusedMoE` moves the
  decision to forward time.
- Consequence for a 0.10.x report: no vllm-ascend 0.10.x tag has the model-level fix (each tag's
  `Dockerfile` pins `ARG VLLM_TAG`; `v0.10.1rc1 -> v0.10.1.1`, and #24849 is in no 0.10.x release).
  On 0.10.x the available route is the thread's: run pure TP without DP+EP, or move to a tag whose
  pinned vLLM is `>= v0.11.0`. Whether #24849 alone fully repairs the 0.10.1rc1 regression was not
  tested here.

## What to do with a new GLM accuracy report

1. Read the topology from the serve line. EP on with `--data-parallel-size > 1` on a 0.10.x tag is
   defect B; any tag at or before `v0.10.0rc1` also has defect A, whatever the topology. The two
   overlap in `v0.10.0rc1`+EP, which is why "upgrade one rc" is not a safe single answer.
2. Two one-liners bound the answer: `git tag --contains 93754d806` (defect A) and the tag's
   `Dockerfile` `ARG VLLM_TAG` (whether the pinned vLLM can contain #24849). A third checks a
   thread claim: `git cat-file -e <tag>:vllm_ascend/models/glm4_moe.py`.
3. For a GLM-5/5.x **W8A8** report, check the checkpoint against the tag's own weight list before
   reading quant code. `docs/source/tutorials/models/GLM5.md:16-22` at `v0.17.0rc1` offers BF16,
   `Eco-Tech/GLM-5-w4a8` and `Eco-Tech/GLM-5-w4a8-mtp-QuaRot` only, and the same page says "vLLM and
   vLLM-ascend only support GLM-5 on our main branches. you can use our glm5 docker images for
   inference" — so a `Eco-Tech/GLM-5-w8a8` conversion on an rc tag is outside the documented path.
   #7507 (W8A8 wrong instruction following on 4 machines, `1*2P1*2D` PD, reporter: "把权重换成w4a8
   正常") stayed open with only a maintainer request for scripts and prompts, and the one W8A8
   precision fix in that tag's notes (#7139, `0c659e91e`, rotary-quant MTP weights) is already
   contained, so "upgrade to 0.17.0rc1" is not a route. Report the W4A8 checkpoint as the
   documented workaround and the W8A8 numerics as the candidate.
4. Report the tags, not the PR numbers, as the upgrade boundary: "A is fixed from `v0.10.1rc1`;
   B needs a tag pairing vLLM `>= v0.11.0`". For a 5.x report with the probabilistic/capture
   shape above, check the #11774 boundary instead: only the 0.23 line and `v0.24.0rc1`+ carry it,
   and above `v0.23.0` add the ordering boundary `git describe --contains 04a4549f9`
   (`v0.23.0.post1~7`).

## GLM-5.x: stale `slot_mapping` at dummy/graph-capture time (fixed in the 0.23 line)

Symptom shape that belongs here: a GLM-5.x MoE service on 0.18-0.22 returns *probabilistic* wrong
content while the accuracy dataset still scores normally, the same request on its own is usually
right, a resend comes back correct, and the deployment captures graphs
(`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[...]}'`).
Issue #10524 (GLM-5.1 W8A8, A3, 1P1D with each role on two nodes, later reproductions through
`/v1/completions` because Claude Code's Anthropic shim calls that endpoint) has this shape, and the
reporter closed it naming plugin PR #11774.

- The defect sits in the runner's dummy-run path, not in the model or the quant scheme. At `v0.18.0`
  `NPUModelRunner._dummy_run` (`vllm_ascend/worker/model_runner_v1.py:2265`) builds attention
  metadata with `for_cudagraph_capture=is_graph_capturing` (`:2414`) but never populates
  `slot_mapping` — that happens in `_prepare_inputs` — so each KV-cache group's
  `input_batch.block_table[gid].slot_mapping.gpu` still holds the previous batch's slots. The
  builder only pads the tail (`:2098 slot_mapping[num_tokens:num_tokens_padded].fill_(-1)`, and only
  when `pcp_size == 1`).
- Fix `b9acec8bc` (#11774, merged 2026-07-11) fills every group's buffer with the pad id
  (`blk_table.slot_mapping.gpu.fill_(-1)`) on the non-capture dummy path, so the capture pass that
  follows starts from pad instead of live slots.
- Release boundaries, verified with `git tag --contains` and a per-tag `grep -c`: backport
  `7be596cd3` first ships in `v0.23.0rc1` (also `v0.23.0`, `v0.23.0.post1`); the main-line commit
  first ships in `v0.24.0rc1`. The `slot_mapping.gpu.fill_(-1)` line is absent from `v0.18.0`,
  `v0.20.2rc1`, `v0.21.0rc1` and `v0.22.1rc1`, so an upgrade inside 0.18-0.22 does not repair it and
  an "upgrade one rc" answer is wrong here.
- Mechanism (inference, not runtime-verified): the capture pass runs `reshape_and_cache` with those
  stale slots and writes dummy KV into blocks a real request owns, so only requests that later read
  those blocks are corrupted — which is why the failure is probabilistic and a resend is often
  clean. What would confirm it: dump `slot_mapping` immediately before capture, or compare the first
  N blocks' KV against a no-capture run.
- Thread hygiene for this issue: the body says `v0.20.2` showed no problem, another reporter's
  comment says `0.20.2` reproduced the same case, and the only maintainer advice was to use
  `v1/chat/completion` instead of the completions endpoint. Treat the closure (the reporter naming
  #11774) as the reporter's claim; the merged commit and its tags are the verified part.

### Second, later defect: the clear ran after the metadata build (fixed in `v0.23.0.post1`)

`v0.23.0rc1` and `v0.23.0` have `7be596cd3`'s fill but in the wrong place: the build comes first and
the clear second (`model_runner_v1.py` build `:3481` / fill `:3494` at `v0.23.0rc1`; `:3538` /
`:3551` at `v0.23.0`). Attention backends that derive their own scatter index during that build
never see the clear — DSA does exactly that in `AscendDSAMetadataBuilder.build`
(`vllm_ascend/attention/dsa_v1.py`: `slot_mapping = common_attn_metadata.slot_mapping[:n]`,
`self.slot_mapping[:n] = torch.stack([slot_mapping // self.block_size,
slot_mapping % self.block_size], -1)`), so the kernels keep the previous batch's
`[block, offset]` pairs.

- Fix `04a4549f9` (#15362, 2026-09-02) only moves the same per-KV-group
  `blk_table.slot_mapping.gpu.fill_(-1)` above `_build_attention_metadata()`; no new op, sync or
  allocation. Its PR body is the reusable statement: "A DP rank with no real requests can still
  execute a dummy batch for DP alignment ... Attention backends such as DSA may derive or copy
  backend-specific slot metadata during that build, so the later `fill_(-1)` cannot invalidate the
  already-built metadata", and its probe measured 4096/4096 finite BF16 values before a dummy
  forward and 0 finite / 4096 NaN after it.
- Boundary, `git describe --contains 04a4549f9` = `v0.23.0.post1~7`, `git tag --contains` = only
  `v0.23.0.post1`; backport `23dd2e411` (#15409) is contained in `v0.26.0rc1` and `v0.26.0rc2` only.
  So the class needs **two** gates: the clear exists (`v0.23.0rc1`+) *and* precedes the build
  (`v0.23.0.post1`+ on the 0.23 line, `v0.26.0rc1`+ on main). `v0.23.0rc1`/`v0.23.0` pass the first
  and fail the second.
- The class is not GLM-specific: the report that surfaced it is a customer sparse-attention 30B
  (`30b-bf16-dsa`) on 910B3, DP4+TP4 across two nodes, `--compilation-config FULL_DECODE_ONLY`,
  `--num-speculative-tokens 4`, garbled in graph mode and clean with graph mode off. Its thread
  names #15362 + #15409, then the reporter said applying #15362 to `v0.20.2rc1` changed nothing
  (thread statement; no maintainer diagnosis, and the follow-up attached files instead of logs).
- On 0.20-0.22 that null result follows from the code rather than from a bad port: `_dummy_run`
  there has no per-group `slot_mapping.gpu.fill_(-1)` at all (`grep -c` = 0 at `v0.20.2rc1` and
  `v0.22.1rc1`), because DSA was handled inside the builder instead:
  `_get_block_table_and_slot_mapping` carries an `elif self.use_compress:` branch (`:2797-2804` at
  `v0.20.2rc1`) that zeroes
  `slot_mapping[:num_tokens_padded]` and `blk_table_tensor[:maybe_num_reqs_padded]`, commented
  "DSA dummy/graph-capture runs do not go through `_prepare_inputs()` ... Reusing values from the
  previous real request can feed stale block-table and [block, offset] scatter indices". There is
  nothing to "move before the build" on 0.20.x, so the answer there is drop the capture mode, not
  apply #15362 (code-derived; nothing was run on hardware).
- Confirm the trigger before naming it: DP where a rank can have no real request (uneven traffic)
  plus graph capture. `--enforce-eager`, PIECEWISE instead of `FULL_DECODE_ONLY`, or the upgrade to
  `v0.23.0.post1`+ are the three workarounds; only the flag removal is reporter-confirmed here.

## Steady-state garbled output that neither defect explains

`vllm-ascend` issue #3547 (GLM-4.5-Air, vllm-ascend `0.11.0` with vLLM `0.11.0`, TP8,
`--max-model-len 131072`, `--max-num-seqs 24`, `--additional-config
'{"ascend_scheduler_config":{"enabled":true,"enable_chunked_prefill":true}}'`): the service answers
normally, then after a while every new request and conversation returns only
`!!!!!!!!!!!!!!!!...`; a restart restores it. Defect A is fixed from `v0.10.1rc1`, and `0.11.0`
pins a vLLM that contains #24849, so neither explains it; the thread carries only "vllm-v0.10.1rc 版本部署是正常的" from a commenter and "please upgrade", so this
is a third, unresolved defect — do not fold it into A or B.

Reusable observations for that class, none verified as its cause:

- The config enables chunked prefill through the Ascend scheduler. That scheduler existed exactly
  because "the npu ops didn't work well with chunked prefill" and was dropped in `v0.12.0rc1` by
  `f10acddb7` / PR #4498 ("drop ascend scheduler"), so on `0.11.0` the pairing is real. Disabling
  `enable_chunked_prefill` (and dropping the stale `ascend_scheduler_config` block) is the cheapest
  experiment; see [env-vars-and-additional-config.md](env-vars-and-additional-config.md) for the
  key's lifecycle.
- "Restart fixes it, new conversations stay broken" points at accumulated engine state (scheduler
  and KV-cache state) rather than a stateless kernel bug. Ask for the request that first shows the
  degeneration and whether the same prompt on a fresh engine reproduces.
- Commits that later touch MoE/KV numerical paths and first ship in `v0.12.0rc1` are locators, not
  evidence for this report: `2b3bfe432` ("Repair the problem of moe model accuracy caused by
  version upgrade", the added upstream `forward_impl` reduce) and `f8610b7d6`
  ("[long_seq] fix A2 accuracy problem", `prepare_finalize.py` under PCP/DCP).
