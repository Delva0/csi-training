# EP together with DP: which MoE communication method the tree selects

Scope: the reporter wants `--enable-expert-parallel` **and** `--data-parallel-size`
> 1 on the same deployment — "我希望同时开启 EP + DP", "去掉
`--enable-expert-parallel` 或把 DP 设成 1 就不报错", "前面的办法都会让
`moe_comm_type = ALLGATHER`" — with or without an engine-init failure pasted. The
deliverable is the selection table for the reported SoC, which lane *their* world
size actually lands in, and the two ablations that separate "MC2 finalize" from
"DP wiring". A **throughput** complaint in which EP was dropped on purpose ("开了 EP 会胡言乱语，所以
只用 tp+dp") is the same selector seen from the other side: answer it from the perf page
`../performance-throughput/moe-comm-path-when-ep-off.md`, which carries the older
in-line form of the selector and the recipe trap for GLM-4.5.

Not this topic: a bare `TimeoutError` with only
`multiproc_executor.py`/`shm_broadcast.py` frames
(`../hccl-and-hardware/rpc-watchdog-timeouts.md`), a KV-transfer/PD topology
question (`../pd-disaggregation/index.md`), or a capacity plan
(`../deploy-sizing/index.md`).

**Recorded #5295** — vllm-ascend v0.12.0rc1 (`42ceaf0`), GLM-4.5-W8A8, one 16-card
910B node, `--data-parallel-size 2 --tensor-parallel-size 8
--enable-expert-parallel --distributed-executor-backend mp
--compilation-config '{"cudagraph_capture_sizes": [4], "cudagraph_mode":
"FULL_DECODE_ONLY"}'`: engine init dies with `RuntimeError: Engine core
initialization failed … Failed core proc(s): {}`, the paste interleaves 16
`Worker_DP*_TP*_EP*` prefixes and its deepest frames are
`vllm/model_executor/layers/fused_moe/layer.py … moe_forward_shared` →
`torch/_higher_order_ops/auto_functionalize.py:856` →
`vllm_ascend/ops/fused_moe/prepare_finalize.py:191`, preceded by
`No available shared memory broadcast block found in 60 second(s)`. The recorded
session shipped **no answer**: 16 assistant turns, every one a tool call, ~46 K
characters of reasoning — it had the selector predicate, the "MC2 shares
`All2All.finalize`" correction and the capacity arithmetic by turn 10.

## Step 1 — read the selector, one call, before any theory

`vllm_ascend/worker/model_runner_v1.py:1373-1432` (`_select_moe_comm_method`,
verified at v0.12.0rc1 `42ceaf0`; called per batch at `:1490` with
`num_input_tokens` and at `:2152` with `num_tokens_padded`):

| Condition | Method |
|---|---|
| not a MoE model | `None` |
| `not enable_expert_parallel` | `ALLGATHER` |
| A2 (`_910B`), EP on, `num_tokens <= mc2_tokens_capacity` **and** `world_size_across_dp / pp_size >= 16` | `MC2` |
| A2, EP on, otherwise | `ALLTOALL` if the checkpoint's `moe_quantize`/`quantize` is `w4a8_dynamic`, else `ALLGATHER` |
| A3 (`_910_93`), EP on | `MC2` if `num_tokens <= capacity`, else `FUSED_ALLTOALL` when `w8a8_dynamic` and `ep_group.world_size <= 16`, else `ALLTOALL` |
| any other SoC | `ValueError: Unsupported soc_version` |

`world_size_across_dp` is `TP × PP × DP` (`vllm/config/parallel.py:323-324` at
vllm v0.12.0 `4fd9d6a`), so **DP=2 × TP=8 = 16 is exactly the A2 boundary**: EP on
at that size is the smallest configuration that can reach MC2, while DP=1 (8) and
EP-off both stay on `ALLGATHER`. That is the whole reason the reporter's two
"workarounds" agree — not a bug in the flag, and not "MC2 was never reached".

Also note the `PanguProMoE` override at `:1427-1429` (forced `ALLGATHER`) and the
DEBUG line at `:1431` that names the mode: `num_tokens: N, moe_comm_type: …`,
logged on the global first rank only. Ask for that line instead of inferring the
mode.

### Older form: v0.10.1rc1, before the selector became a function

Verified at v0.10.1rc1 (`7e16b4a7`): the decision is inline in
`set_ascend_forward_context` — `ep_size = (get_ep_group().world_size if
enable_expert_parallel else 1)` (`ascend_forward_context.py:87-88`), then
`_get_fused_moe_state` (`:26-40`): `1` → `NaiveMulticast` prefill / `AllGather` decode;
`< 16` **or any prefill** → `All2All`; `>= 16` decode → `MC2`. Dispatchers are named by
`get_dispatcher_name` (`:43-52`) and registered in
`ops/moe_dispatcher/token_dispatcher.py:48-60`. Two facts that shape an answer at that
ref: **no log line names the choice** (no `moe_comm_type`, no `[EP Rank r/N]` — both
arrive later, so ask for the serve flags and a same-request A/B instead), and the
`AllGatherEP` fusion needs `n_routed_experts == 256` (`:28-31`), so a 128-expert GLM is
not rescued by `VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP`.

### Version drift to re-grep first (verified at v0.13.0rc1 `1b5d5ab`)

- The selector moved out of the runner into the module function
  `select_moe_comm_method` (`vllm_ascend/ascend_forward_context.py:196-265`);
  `set_mc2_tokens_capacity` now sits beside it (`:160-173`). Grep the name, not the
  old file.
- The A2 row lost its `ALLTOALL` branch: at this ref it is `MC2` when
  `num_tokens <= capacity` **and** `world_size_across_dp / pp >= 16`, otherwise
  `ALLGATHER` — so "w4a8_dynamic goes ALLTOALL on A2" is a v0.12 statement only.
- `profile_run` now runs an **extra capacity-sized dummy batch** before the real
  profile run when `max_num_tokens > capacity` and the capacity-point selection
  is `MC2`/`FUSED_MC2` (`worker/model_runner_v1.py:2202-2208`). A startup
  traceback naming `npu_moe_distribute_dispatch_v2` is therefore the profiling
  batch at `capacity` tokens, not `--max-num-batched-tokens`; see
  `../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` for that fault lane and
  for the `HCCL_INTRA_*` → hierarchical-dispatch variant.

## Step 2 — the capacity number is derived, so compute it

`vllm_ascend/ascend_forward_context.py:169-183` (`set_mc2_tokens_capacity`, called
once from `worker/model_runner_v1.py:272`): with `cudagraph_capture_sizes` set,
`max_num_tokens = compilation_config.max_cudagraph_capture_size`, otherwise
`min(max_num_reqs * uniform_decode_query_len, 512)`; then
`capacity = ceil(max_num_tokens / tp_size) * tp_size`.

Consequences to state in the answer:

- The method is chosen **per batch**, so one server can run `MC2` in decode and
  `ALLGATHER`/`ALLTOALL` in prefill. A traceback inside ACL-graph capture is
  therefore an MC2-path traceback even when the prefill run of the same process is
  on `ALLGATHER`.
- `capacity >= tp_size` always, and decode batches are at most
  `max_num_seqs × uniform_decode_query_len`. On A2 at world ≥ 16 with EP on there
  is **no flag that keeps decode off MC2** — do not send the reporter hunting for
  one. Raising `cudagraph_capture_sizes` only raises the capacity.
- #5295's `cudagraph_capture_sizes: [4]` with TP=8 gives
  `ceil(4/8)*8 = 8`: any decode batch of ≤ 8 tokens takes MC2.

## Step 2b — the A2 MC2 per-card **expert** cap (24): a tiling refusal, not a hang

Verified at vllm-ascend **v0.17.0rc1** (`e20f0b1`) while triaging #7635 (two A2
nodes, 16 cards, Qwen3.5-397B-A17B-w8a8, `--data-parallel-size 2
--tensor-parallel-size 8 --enable-expert-parallel`); the mechanism is
**reporter-confirmed** in the follow-up, including the fix.

Same lane as Step 2 (the `profile_run` batch at `mc2_tokens_capacity`) and the same
"a dispatch op failed at startup" shape as
`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md`, but a different failure
kind: an **argument refusal inside the op's tiling**, back in seconds instead of a
10-minute hang, with the reason spelled out in the last line:

```text
RuntimeError: npu_moe_distribute_dispatch_v2: … NPU function error:
  call aclnnMoeDistributeDispatchV4 failed, error code is 561002
[ERROR] … ERR00100 PTA call acl api failed.
moeExpertNum is 32, in case of unlayered, it must no more than 24.
  [FUNC:MoeDistributeDispatchA2CheckAttrAndSetTiling][FILE:moe_distribute_dispatch_v2_tiling.cpp][LINE:1152]
op[MoeDistributeDispatchV2], MoeDistributeDispatchA2 CheckAttrAndSetTiling Failed
MoeDistributeDispatchV2 do tiling failed, ret is -1.
```

Read that number as **experts per card**, never as the model's expert count: it is
what the dispatch receives as `moe_expert_num`, i.e.
`len(expert_map) + global_redundant_expert_num`
(`ops/fused_moe/token_dispatcher.py:155`, packed at `:161`, called at `:215`), which
equals `global_num_experts // ep_size` (`ops/fused_moe/fused_moe.py:283`; `ep_size`
is the EP group's world size — `token_dispatcher.py:70` — i.e. TP×DP×PCP for that
parallel layout).
512 experts over 16 ranks = 32 > 24. Two artefacts settle it without arithmetic: the
engine prints `[EP Rank r/N] Expert parallelism is enabled. Local/global number of
experts: X/Y` (`fused_moe.py:284-295`), and the constant itself is **not** in
`vllm_ascend/` at this ref (`rg -n 'no more than|max_local_expert' vllm_ascend/` is
empty) — it belongs to CANN's `moe_distribute_dispatch_v2_tiling.cpp`, so its value
carries CANN-version scope, and an older tree only surfaces the raw tiling error.

The ladder is short because **no flag moves that count**:

1. Drop `--enable-expert-parallel`, or keep EP while staying under
   `world_size_across_dp/pp = 16` (DP1×TP8 = 8) → the `ALLGATHER` row, which never
   calls this op. The reporter's own working run. Price: EP weight sharding is gone,
   so the per-card MoE weight footprint goes up.
2. Enlarge the EP group until `global_experts / ep_size <= 24` (512 needs ≥ 22 ranks;
   `--data-parallel-size 4 --tensor-parallel-size 8` = 32 ranks → 16/card) — the only
   way to keep MC2 by hand.
3. Otherwise take a build with the fallback: upstream added "local experts over the
   MC2 limit → `ALLGATHER`" as **PR #7364** (postdates v0.17.0rc1;
   reporter-confirmed: service starts, load test normal, possible comm cost). It is a
   per-batch selector decision, so EP weight sharding survives (`local_num_experts`
   is computed in `FusedMoE.__init__`, independent of the method chosen per batch) and
   only routing changes: `dp_group.all_gather` of hidden states + router logits
   (padded to `max_tokens_across_dp`) and a closing `reduce_scatter` per layer
   (`ops/fused_moe/prepare_finalize.py:378-379, 441`) instead of the sparse
   dispatch/combine fused with the expert GEMM — which is exactly why the loss is
   "every layer, no overlap", not a scale change.

Refuted rows to keep in the answer: `--enforce-eager` and `cudagraph_capture_sizes`
(the profiling batch *is* `capacity`, so MC2 is selected there whatever the capture
size), the `HCCL_INTRA_PCIE_ENABLE=1`/`HCCL_INTRA_ROCE_ENABLE=0` pair (it only sets
`need_expert_scale`, `token_dispatcher.py:112` / `utils.py:967-969`, and the kwargs
carry no `comm_alg` or other layered parameter at this ref — `:158-176` — so A2 always
dispatches *unlayered*; keep the documented performance pair, but do not offer it as
an escape), `HCCL_BUFSIZE`,
`HCCL_OP_EXPANSION_MODE=AIV` and `PYTORCH_NPU_ALLOC_CONF` (the reporter's failing and
working runs both carried them; contrast `../hccl-and-hardware/aiv-mc2-env-ablation.md`,
whose signature is `…V4 … 107020` inside a hang), and the card. Also watch EPLB: the
count includes `global_redundant_expert_num`, so redundant experts can push a layout
sitting at ≤ 24 over the cap. Worth reporting upstream (the recorded session proposed
both): log the gate once when a fallback engages (at v0.17.0rc1 `moe_comm_type` is
assigned with no log at all, `ascend_forward_context.py:70-72`, so the field cannot
tell MC2 from ALLGATHER), and make the 24 a per-SoC/CANN value rather than a Python
constant that silently disables MC2 for 512-expert layouts after a CANN bump.

## Step 3 — correct the premise: MC2 inherits the All2All finalize

`PrepareAndFinalizeWithMC2(PrepareAndFinalizeWithAll2All)`
(`ops/fused_moe/prepare_finalize.py:201`) overrides only `__init__`/`prepare` and
its docstring says so ("Hence, it inherits All2All and share the same finalize
method"). The `dist.all_gather(list(split_hidden_states), hidden_states,
self.moe_config.tp_group.device_group)` at `:191` is that shared finalize — reached
by MC2 and ALLTOALL. `PrepareAndFinalizeWithAllGather.finalize` (`:399`) is the
different one, built on `dp_group.all_gather` (`:382-384`).

So a paste whose deepest frame is `:191` proves the failing layer was on the
MC2/ALLTOALL path, and `enable_shared_expert_dp`/`enable_sp` were **off** for it —
both are set from `ascend_config.enable_shared_expert_dp`
(`ops/fused_moe/fused_moe.py:250`, `:339`) and, when true, skip the split in
`prepare` and the gather in `finalize` (`prepare_finalize.py:146-155, 179-186`).
GLM-4.5 has shared experts, so its MoE layer arrives through
`torch.ops.vllm.moe_forward_shared` (`fused_moe.py:313`) — the frame in the paste —
and the gather still ran.

## What not to re-derive

- The MC2 pad target is `forward_context.padded_num_tokens`, and
  `padded_num_tokens = ceil(max_tokens_across_dp / tp_world_size) * tp_world_size`
  (`ascend_forward_context.py:147-148`), i.e. **always a TP multiple**: every
  rank's `tensor_split` slice is the same size, so an "uneven/zero-row
  `tensor_split` makes `all_gather` fail" theory is refuted arithmetically. #5295's
  session was opening the one file that settles this when it was cut — start from
  the arithmetic instead.
- `PrepareAndFinalizeWithAll2All.prepare` pads only when `num_tokens < tp_size`
  (`pad_size = tp_size - num_tokens`, `:148-155`); that is the ALLTOALL path's
  business, not MC2's.

## Step 4 — the validated EP+DP recipes in the tree (name them in the answer)

At v0.12.0rc1 the only EP+DP configurations the project ships are in
`docs/source/tutorials/Qwen3-235B-A22B.md`:

- single node: `--tensor-parallel-size 8 --data-parallel-size 1
  --enable-expert-parallel` (`:111-119`) → world 8 → `ALLGATHER`;
- multi-node: `--data-parallel-size 2 --data-parallel-size-local 1
  --data-parallel-address $node0_ip --tensor-parallel-size 8
  --enable-expert-parallel` (`:177-185`) → **one DP rank per node**.

`ep_size` is the flattened `dp × pcp × tp` of that group, so **the DP half only
widens the expert shard when the DP ranks really share a DP group**: with
`--data-parallel-size 2` alone and no `--data-parallel-size-local/-address/
-rpc-port/-start-rank`, a two-node box runs two independent TP-shaped engines and each
card keeps `experts / tp` of them. That is the 2× that turns into a load-phase OOM on a
1T-class MoE — capacity side: `../deploy-sizing/moe-load-oom-expert-shard.md`
(reporter-confirmed #7123).

No doc and no `tests/e2e/**` config runs `--data-parallel-size 2` with
`--distributed-executor-backend mp` on a single card set, which is #5295's shape
(`rg -l "data-parallel-size 2" docs/ tests/e2e/` returns only the multi-node yaml
and that tutorial). Say that: their combination is off the validated matrix, and
the answer is the ablation ladder, not a synthesized launcher script.

## Step 5 — the deliverable and its ladder

Verdict sentence first: with EP + DP≥2 on A2 the tree routes decode to MC2, the
crash site is MC2's own finalize, and the shipped DP+EP recipes use one DP rank per
node or DP=1. Then cheapest first:

1. `--enforce-eager` (or `--compilation-config '{"cudagraph_mode": "NONE"}'`): the
   pasted traceback sits inside `auto_functionalized_v2`/capture, so remove the
   compile+graph confounder before any MoE theory (`ASCEND_LAUNCH_BLOCKING=1`
   afterwards, which is refused while ACL graph is on).
2. Ask for the selector's DEBUG line (`num_tokens: N, moe_comm_type: …`) and the
   **first** ERROR of the full per-rank log — the 16-rank interleaved paste does
   not carry the exception type or message.
3. `--data-parallel-size 1` with EP kept → world 8 → `ALLGATHER`: keeps EP, drops
   DP, and confirms the boundary arithmetic.
4. Keep DP=2 and drop `--enable-expert-parallel` → ETP/`ALLGATHER`: the reporter's
   already-working run, and say plainly that it is TP-only MoE, not EP.
5. Only then the graph/compile details (`cudagraph_capture_sizes`, the AIV
   `HCCL_OP_EXPANSION_MODE` env), and file the MC2-with-DP failure with the
   selector line and both ablations.
6. On **more than one node**, compare the nodes' commands before ablating anything:
   `mc2_tokens_capacity` and the dispatch's `global_bs` are derived per rank from
   that node's own capture-size list, so an asymmetric flag makes the halves of the
   EP group declare different buffer sizes to one all-to-all
   (`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` Step 1b,
   `scripts/mc2-capacity-asymmetry.sh`).

Send back: the un-interleaved per-rank log, the `moe_comm_type` debug lines, whether
row 1 changes the outcome, the `additional_config` keys
(`enable_shared_expert_dp`, `enable_sp`), and CANN/torch_npu versions.

## Scope

Pins: vllm-ascend v0.12.0rc1 `42ceaf0`, vllm v0.12.0 `4fd9d6a`. Step 2b is pinned separately at vllm-ascend
v0.17.0rc1 `e20f0b1` + vllm v0.17.0, where the selector has moved to
`ascend_forward_context.select_moe_comm_method:199-273` and the A2 row is at `:235-243`. The selector
predicate, the capacity formula and the MC2/All2All inheritance are the parts to
re-grep at another version (`_select_moe_comm_method`,
`set_mc2_tokens_capacity`, `class PrepareAndFinalizeWithMC2`); A3 adds
`FUSED_ALLTOALL` and the `ep_group.world_size <= 16` gate, so an A3 paste is a
different row, not this one. The crash *mechanism* in #5295 was not established —
the paste has no exception text, so the page gives the lane and the ablations, not
a root cause.
