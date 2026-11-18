# FlashComm1 (= SP) crashes, and a duplicated or missing MoE TP all-reduce

## Scope

Two faults in the same collective-reduction plumbing, both routinely filed as "feature X is not
supported" or "output is garbage": (a) FlashComm V1 / sequence parallelism enabled at the reporter's
tag dies while the target graph is captured or during warm-up; (b) a MoE model with TP>1 answers
nonsense because the routed-expert output is TP all-reduced twice. A third, cheaper answer lives here
too: FC1 off in a deployment whose own official config enables it (§C). Routes in `symptom-index.md`;
worked reports in `issue-ledger/reports-later.md` (#8990, #10466, #10503) and `reports-newest.md`
(#8223, the EP-only 乱码 on `v0.17.0rc1`, §B below).

## Common rules

- **FlashComm1 and SP are the same feature in this repo.** `enable_sp()` returns
  `VLLM_ASCEND_ENABLE_FLASHCOMM1 or VLLM_ASCEND_ENABLE_FLASHCOMM` and memoises it in a module global
  `_ENABLE_SP` (`vllm_ascend/utils.py:799-816` at `v0.19.1rc1`). Through `v0.19.1rc1` the env var is
  the only switch; from `v0.20.2rc1` `additional_config.enable_flashcomm1` exists too
  (`ascend_config.py:83`, `_get_config_value`, commit 958daf832 / #9064, first tag `v0.20.2rc1`).
  A report that "disabled FLASHCOMM1" is only checked when the key spelling matches the tag
  (`../vllm-ascend/config-flag-lifetimes.md` §5).
- `enable_shared_expert_dp` forces `enable_sp()` true (`utils.py:814-817` at `v0.19.1rc1`), so a
  report can have FC1 on without ever setting the FC1 variable — read the derivation, not the env.
- A second notion exists: `enable_sp_by_pass()` (`utils.py:795`; `ascend_config.py:183`) is the
  *compilation-pass* SP from `pass_config.enable_sp`, requiring `--enforce-eager` off. Read the one the
  failing symbol uses; they are not interchangeable — but they are not disjoint either: at `v0.18.0`
  the pass flag ORs itself into the FC1 custom-op branches (`ops/register_custom_ops.py:46`
  `flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)`,
  `ops/fused_moe/prepare_finalize.py:327`/`:436`), so a deployment that never exported
  `VLLM_ASCEND_ENABLE_FLASHCOMM1` can still be running the FC1 communication path. Check both keys
  before saying FC1 is off (§D).
- FC1/SP is an Inductor pass plus custom ops (`vllm_ascend/compilation/passes/sequence_parallelism*.py`),
  so its crashes surface *inside a compiled graph*: the traceback frame is a dynamo/AOT submodule
  call with hundreds of weight arguments, not a hand-written `copy_` in the repo. Expect no useful
  Python line and do not conclude "the code is missing".

## A. FC1 + cudagraph capture: `expected src to have a size equal to the slice of self`

Report: #8990 (image `v0.19.1rc1-openeuler`, Qwen3.5-27B w8a8 + `qwen3_5_mtp` 3 spec tokens, TP2,
910B, `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'`,
`--max-num-batched-tokens 8192`, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`). Worked on `v0.18.0rc1`, fails at
`v0.19.1rc1`.

Signature: startup reaches `Capturing CUDA graphs (decode, FULL)` and dies in
`compile_or_warm_up_model`; the wrapper is `NPUModelRunner init failed` /
`EngineCore failed to start`; the leaf text is torch's copy check —

```text
expected src to have a size equal to the slice of self. src size = [64, 5120], slice size = [32, 5120]
```

**Arithmetic discriminator first.** `src_rows == tp_size × dst_rows`, with the same hidden size on both
sides (TP2: 64 vs 32; the second reporter at TP8, Qwen3.5-9B, hidden 4096: 256 vs 32). That ratio is an
all-gather of the per-rank token shard landing in a buffer/slice sized for one rank
(`tensor_model_parallel_all_gather(x, 0)` then `[:-pad_size]`, `worker/model_runner_v1.py:2137-2141`
at `v0.19.1rc1`, called from the FC1 epilogue at `:2219-2221`). The same "padding vs per-rank slice"
arithmetic appears in `_pad_for_sequence_parallelism` (`model_runner_v1.py:2223`, rounds the token count
up to a multiple of TP) and in the PP+SP override `sync_and_slice_intermediate_tensors`
(`copy_len = (num_tokens + tp - 1) // tp`, `model_runner_v1.py:2232-2250`, whose comment notes that
"the judgment logic of flashcomm1 in Ascend is inconsistent with SP in vllm"). Which of these fired was
**not** localised: the failing copy runs inside the captured graph. Do not go hunting in `torch` for
the message — it is `Tensor.copy_` rejecting a src larger than the destination slice.

- **Workaround** (reported twice, and independent of hardware): unset `VLLM_ASCEND_ENABLE_FLASHCOMM1`
  (staying on `v0.18.0rc1` is the same trade). Cost: lose the FC1 reduce-scatter/all-gather
  optimisation, so throughput drops. One commenter reports it still fails "even with the latest build",
  so this is the remedy, not a fix.
- **Status**: maintainer statement — "currently, vllm-ascend does not support Flash Comm V1 for 27B
  model", with a link to `sequence_parallelism.html#without-quantization`. The tag's own matrix
  (`docs/source/user_guide/feature_guide/sequence_parallelism.md` at `v0.19.1rc1`) claims FC1 for
  non-VL Dense **with and without** quantization (eager/graph), so the 27B carve-out is a model-family
  limit that neither doc nor code expresses — say so instead of citing the matrix. `ascend_config.py:101`
  only *warns* and rounds `max_num_batched_tokens` up to a multiple of TP×PCP; `git grep -i "not
  support"` at that tag finds no model guard for FlashComm1, so the feature runs on any model and fails
  on the shape contract instead of refusing the configuration. #8990 is open with `triaged`+`stale`
  labels, no fix commit; a stale-bot label is not a resolution.
- **Falsifying checks for the next report**: TP=1 or `--enforce-eager` no longer hits the copy check
  (names the gather/padding as the owner); `src_rows / dst_rows` should equal the TP size, and the
  ratio of the other reporter's shapes confirms it before you assert anything. A report with TP=1, or
  with the crash at a different stage (weight load, first request), is not this class.

## B. MoE + TP: garbled output from a *second* TP all-reduce

Report: #10466 (`v0.20.2rc1`, vLLM `0.20.2`, CANN `9.0.0`, MiniMax-M2.7 w8a8-QuaRot, TP8 on 910B,
`--enforce-eager`, no `--enable-expert-parallel`, no FLASHCOMM1, no fused MC2). Any prompt returns
meaningless text; the model loads and serves normally — no exception, no NaN log, no stall, no accuracy
gate. Same class later in the file's life: the **reduce** was removed by `57c582440` (#11505, first tag
`v0.23.0rc1`) and the remaining MiniMax hooks were trimmed afterwards (#10906, #13782, #13789). Read the
patch at the *reported* tag: the file `vllm_ascend/patch/worker/patch_minimax_m2.py` still **exists** at
`HEAD` (only `_patch_forward` for the attention is left), so neither `git log` nor a line-capped `ls-tree`
listing tells you what its `_patched_moe_forward` did at `v0.20.2rc1`.

Verified chain at `v0.20.2rc1`:

1. Upstream vLLM v0.20 refactored MoE through `MoERunner`, which reduces the combined output itself
   unless the backend says it already did: `_fused_output_is_reduced`
   (`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:344`, =
   `quant_method.moe_kernel.output_is_reduced()`) and `_maybe_reduce_final_output` (same file,
   `:370-392`), which all-reduces when
   `(moe_config.tp_size > 1 or moe_config.ep_size > 1) and not _fused_output_is_reduced`.
2. `vllm_ascend/ops/fused_moe/fused_moe.py:271-284` overrides the property: it returns True only for
   `ALLTOALL`/`MC2`/`FUSED_MC2`, or `ALLGATHER` with `flash_comm_v1_enabled` — so the plain
   `ALLGATHER` path makes upstream reduce.
3. `vllm_ascend/patch/worker/patch_minimax_m2.py:56-68` unconditionally does
   `if self.tp_size > 1: final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)`
   (the reduce is at `:66`), so the tensor is summed once upstream and once by the patch.

Consequence: the routed output is summed `tp_size` times per MoE layer — two reductions = `tp_size`×,
not 2×; do that arithmetic explicitly. A magnitude blow-up decodes as fluent, well-formed nonsense
rather than an error, and it is deterministic for a fixed prompt (a wrong-scale cache or a bad mask
gives position-dependent noise instead).

### Scope (verified by tag)

The manual reduce is present at `v0.20.2rc1` (`:66`), `v0.21.0rc1` and `v0.22.1rc1` (`:76`), and gone
from `v0.23.0rc1` — the affected line is v0.20–v0.22, and other v0.20.x tags were not checked one by one.
**The fix is `57c582440` (#11505, first tag `v0.23.0rc1`)**, which deletes exactly the
`if self.tp_size > 1: … maybe_all_reduce_tensor_model_parallel(…)` pair while keeping the fp32
router-logits behaviour; its own body says leaving the call in "blocks MiniMax TP8+EP startup", and it
rides inside a PR titled *"Avoid partial MiniMax parameter arguments"* — verify it by the hunk
(`../workflow/fix-discovery.md`). The reporter's guard PR **#10443** ("Fix MiniMax-M2 double TP
all-reduce on v0.20+") is **open and unmerged** (state read 2026-11-17) and was superseded; do not read
"cited PR unmerged" as "unfixed".

- **Workaround** (reported in the thread, and consistent with the code): `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`
  **and** `--enable-expert-parallel`. With FC1 on, `_fused_output_is_reduced` becomes True, upstream
  skips its reduce, and only the patch reduces — one reduce instead of two. Cost: pulls in the FC1 path
  (see A) and EP; throughput and EP/TP degrees change, so accuracy comparisons must be re-based.
- **Discriminating checks**: TP=1 gives correct output (no reduce happens at all); TP>1 with FC1 off
  gives garbage; the log's chosen `MoECommType` (ALLGATHER vs MC2) tells which branch ran.
- Do not repeat the thread's "use fusedmc2" advice as a root cause: FUSED_MC2 only moves the reduce
  into `finalize()`, where the same `_fused_output_is_reduced` guard decides whether a second one runs.
  The honest fix is the patch guard or `>= v0.23.0`.

### The same family on `≤ v0.19.x`: the guard lives in the custom op, not in `_fused_output_is_reduced` (#8223)

Before upstream's `MoERunner` refactor there is no `_fused_output_is_reduced` to read, so a report on
`0.17`/`0.18` looks like "the hook is gone" unless it is read at its own tag. Verified at `v0.17.0rc1`:

1. `vllm_ascend/patch/worker/patch_minimax_m2.py:47-62` replaces `MiniMaxM2MoE.forward` and `:57-58` calls
   `self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)` whenever `self.tp_size > 1`
   — the same unconditional second call, with no caller-side guard.
2. Whether that call reduces at all is decided by the op implementation,
   `ops/register_custom_ops.py:140-148` `_maybe_all_reduce_tensor_model_parallel_impl`: it returns
   `final_hidden_states` unchanged when `_EXTRA_CTX.moe_comm_type ∈ {ALLTOALL, MC2, FUSED_MC2}` **or**
   `_EXTRA_CTX.flash_comm_v1_enabled`, and otherwise calls `tensor_model_parallel_all_reduce`. So on this
   line the branches are inverted relative to §B's `v0.20.2rc1` chain — read both sides before deciding
   "doubled" versus "missing". FC1-on makes the *patch's* reduce the no-op and leaves upstream's (one
   reduce, not zero), while FC1-off lets two run whenever the comm type is `ALLGATHER`.
3. The comm type comes from `select_moe_comm_method`, which at this tag is in
   `vllm_ascend/ascend_forward_context.py:199-264` — **not** in `ops/fused_moe/moe_comm_method.py` (that
   module is the later home; an empty `sed` there is not evidence the function is absent). `:233-234`:
   without `enable_expert_parallel`, or with `get_ep_group().world_size == 1` → `ALLGATHER`; `:235-243`:
   A2 with EP → `MC2` only when `num_tokens <= get_mc2_tokens_capacity()` **and** `world_size_across_dp /
   pipeline_parallel_size >= 16`, otherwise `ALLGATHER`; on A3 the fused branches gate on
   `VLLM_ASCEND_ENABLE_FUSED_MC2` and `quant_type == "w8a8_dynamic"` (`:245-263`) — so on a **BF16**
   checkpoint the variable alone selects nothing, and forcing the fused path by editing that gate hangs
   the first forward inside `dispatch_ffn_combine` (`moe-ep-dispatch.md` §C, #7154). It is stamped per
   forward (`ascend_forward_context.py:70-72`), so the EP and non-EP runs of one command can take
   different branches. **EP does not imply `MC2`, so "count the reduces" cannot be decided from the flag
   list**: print the log's `MoECommType`/branch before asserting a mechanism for an "EP on = garbled, EP
   off = fine" report — a recorded run grepped the later module path, got an empty result and read it as a
   missing function.
4. `flash_comm_v1_enabled` is itself token-count dependent for the non-MoE path (`:102` requires
   `num_tokens > 1000`; the MoE path at `:95` does not), and `VLLM_ASCEND_ENABLE_FLASHCOMM1` defaults to
   `0` (`envs.py:74`).

Worked case **#8223** (`v0.17.0rc1`, MiMo-V2-Flash BF16, DP2 × TP8 = 16 cards, `--enable-expert-parallel
--data-parallel-size 2 --tensor-parallel-size 8 --max-num-seqs 1 --max-model-len 10240
--max-num-batched-tokens 4096 --enforce-eager`, no FC1; title says DP4 TP4): a single `curl` chat
completion (`你是谁`, `temperature 0`) returns 乱码 with EP on, and the same command **without**
`--enable-expert-parallel` decodes normally — a healthy service, no traceback, no metric, no accuracy gate.
The thread's one comment is the reporter's own (`ningweikang`, 2026-04-17): 「参考之前M2.5的FAQ发现：当开启EP并行时，
同时需要设置环境变量 VLLM_ASCEND_ENABLE_FLASHCOMM1，设置该环境变量后输出恢复正常」, quoting the MiniMax-M2.5
tutorial FAQ.

- **Workaround [reporter-confirmed]:** `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` and keep EP on. Cost: FC1
  pulls in the SP/Inductor pass and its own capture-time failure mode (§A), and it changes the comm type,
  so any throughput number must be re-based.
- **Verified where the cited sentence lives.** "Q: What should I do if the output is garbled in EP mode?
  A: It is recommended to keep `--enable-expert-parallel` and `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`" enters
  `docs/source/tutorials/models/MiniMax-M2.5.md` in `fb8e22ec0` (#7296, 2026-03-18), **first tag
  `v0.18.0rc1`**. At the reporter's `v0.17.0rc1` no MiniMax tutorial exists under
  `docs/source/tutorials/models/` at all, so "the docs say so" holds from `v0.18.0rc1` only — relay the
  pairing as the docs' own requirement *at the tag that has it*, not as config conformance at the
  reporter's. The page path is gone at `HEAD` (#10460 retitled it M2.7 — only
  `docs/source/locale/zh_CN/…/MiniMax-M2.5.po` keeps the msgid).
- **Mechanism stays `reported`:** the comment is the lever and the guard above is the version's code shape;
  which reduce is wrong under `MC2` is not settled here. Do not upgrade it to `verified` without the
  reporter's A/B on FC1 alone. Discriminator: the same prompt with EP off and on, one node,
  `VLLM_LOGGING_LEVEL=DEBUG` for the chosen `MoECommType`. Do **not** spend a batch on the Enum once the
  thread's lever is in hand — the recorded run read `patch_minimax_m2.py`, the op and the comm-type
  selector across five batches and shipped nothing.

## C. Garbled output where the official config enables FC1 and the reporter does not (#10503)

Report: vllm-ascend `v0.20.2rc1`, Kimi-K2.5-w4a8, two nodes × 8 NPU (A2), TP8 **with**
`--enable-expert-parallel`, internal DP2 (`--data-parallel-size 2 --data-parallel-size-local 1
--data-parallel-start-rank`), prefix caching on, `--compilation-config
'{"cudagraph_capture_sizes":[…],"cudagraph_mode":"FULL_DECODE_ONLY"}'`, `--max-model-len 262144`.
Single requests return 乱码 on both nodes; the service is healthy, no traceback, no accuracy gate.

**The cheapest check is the config diff, not the code.** At the reported tag,
`tests/e2e/nightly/multi_node/internal_dp/config/Kimi-K2_5-W4A8-A2-dual-nodes.yaml` is the upstream CI
config for exactly this deployment and its `env_common` block sets `VLLM_ASCEND_ENABLE_FLASHCOMM1: 1`
(with `VLLM_ASCEND_ENABLE_MLAPO: 1`, `HCCL_OP_EXPANSION_MODE: AIV`, `HCCL_INTRA_PCIE_ENABLE: 1`,
`HCCL_INTRA_ROCE_ENABLE: 0`) and runs `--no-enable-prefix-caching`;
`docs/source/tutorials/models/Kimi-K2.5.md` exports the same variable at `:151`, `:227`, `:290`, `:367`,
`:445` (`git show v0.20.2rc1:docs/source/tutorials/models/Kimi-K2.5.md | grep -n FLASHCOMM1`). The
tutorial's own variable list describes it as a *communication optimisation* (`:645`), while the CI
config enables it unconditionally: that gap is the point — for this deployment the reference
configuration has FC1 on, so "does my config match the official one?" is the first question, and
the #10466 lever (`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, there paired with `--enable-expert-parallel`,
which this reporter already has) is the same switch through `_fused_output_is_reduced` (§B).

- **Workaround** [reported, not reporter-confirmed]: `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` — commenter
  `ultra3g` (the same contributor whose FC1 advice resolved #10466) says it removes the 乱码 but
  "可能会变得很慢". No maintainer ever diagnosed this issue; the other comments are "Plz see this issue"
  relays.
- **Mechanism: not established.** The #10466 fingerprint is absent here: at the tag
  `patch/worker/patch_kimi_k25.py` is ViT/multimodal only and calls no
  `maybe_all_reduce_tensor_model_parallel` (`git grep -n maybe_all_reduce_tensor_model_parallel <ref> --
  vllm_ascend` returns only a `models/deepseek_v4.py:383` call, the op's own definition/wrapper and the
  MiniMax patches). With FC1 off the Ascend runner takes plain `ALLGATHER`, so the question is where the
  second reduction (or a wrong-comm metadata path) comes from for this model: count the reductions per
  "Triage order" step 1, then read the W4A8 MoE quant path and the `MoECommType` the log reports. Do not
  invent a third mechanism to fill the gap; say it is open.
- **Do not cite #9679** ("kimi-k2.5 四机 1P1D … 乱码") as the same defect: its comments describe repeated
  `think` tokens inside agent loops across DeepSeek-V4-Flash and GLM-5.1 on `v0.18.0`, and the thread's
  only advice is a logprobs + `min_p` band-aid. Different signature, different stage.
- **Falsifying checks to hand the reporter:** the same prompt with FC1 off/on on one node (not two), and
  with prefix caching off; if TP=1 or a single node also garbles it, this is not the FC1/reduce family.

## D. The `pass_config.enable_sp` route alone: output truncated mid-sentence (#12278)

Report: image `quay.io/ascend/vllm-ascend:v0.18.0-a3`, Qwen3.5-397B-A17B-w4a8-**mtp**, 910C 752T
(4 cards / 8 dies), plain `vllm serve` (no PD config in the body). A single `/v1/chat/completions` from
the body's own `curl` returns 97 completion tokens and **stops in the middle of a sentence**: the
content ends `…- 我的主要能力: 基于 Q`, `finish_reason: "stop"`, `stop_reason: 248044`,
`prompt_tokens: 15`. The same text also carries visibly corrupted tokens (`Q wen3.5`, `Q sp2.5`).

**Read the two fields before any code.** `finish_reason: "stop"` with a concrete numeric `stop_reason`
is *the engine stopping on a token*, so this is not a `max_tokens`/length cap (that reports `length`),
not a client disconnect, and not an EOS-config question: a corrupted or early-greedy logit emitted a
stop id. Corrupted words in the same answer point at a compute/reduce fault, not at the tokenizer.

**The thread's own A/B names the flag** (reporter `haidiniao`, first comment, same author as the
report): it worked on `quay.io/ascend/vllm-ascend:v0.17.0rc1-a3`; after the upgrade the deployment had
added `"pass_config": {"enable_sp": true}` to the startup script for ~5 % first-token latency; removing
it restores normal output; `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` is the reporter-named way to keep
SP. Second data point from the same comment: **another** 910C / 8-card host on the same image with the
same start parameters did *not* truncate, so in their fleet the key is necessary but not sufficient —
do not promise the flag alone explains every deployment.

**What the tag actually does with the key (`verified` at `v0.18.0`).** `ascend_config.py:163-167`
derives `enable_sp_by_pass = model_config is not None and not model_config.enforce_eager and
vllm_config.compilation_config.pass_config.enable_sp`, exposed at `utils.py:770-771`, and it feeds:

- `ops/register_custom_ops.py:46` — the `maybe_all_gather_and_maybe_unpad` impl treats
  `enable_sp_by_pass() and is_ep_comm` as `flash_comm_v1_enabled`, and `:56` returns the gathered
  tensor **without the unpad** (`# TODO: do unpad`) where the env-var path unpads;
- `ops/fused_moe/prepare_finalize.py:327` / `:436` — the MoE prepare switches from the DP group to the
  **EP group** (`_prepare_with_ep_group`) when either notion of SP is on;
- `utils.py:797` (`shared_expert_dp_enabled`) and `worker/model_runner_v1.py:1848` — the same OR.

So the pass flag is not a no-op that "should just be ignored": it re-routes the MoE communication of a
deployment whose env var is unset, and one of those branches is missing the unpad step. That is a
plausible route from a config key to the garbage/early-stop shape — label it `hypothesis` (the report
carries no plog, and nothing was reproduced on hardware), and hand the reporter the two-command check
below rather than a claimed line.

**The docs at the same tag still document the key** — `docs/source/user_guide/feature_guide/
sequence_parallelism.md:17-28` shows `--compilation-config '{"pass_config": {"enable_sp": true, …}}'`
and `:104-106` describes the derivation. The reporter's "this parameter is not supported from
`v0.18.0` on (the website does not say so; our compute team confirmed it)" is therefore a support
statement, not a code fact: the key is still read and still wired. Report both — the doc line and the
code gate — instead of repeating "unsupported" or hunting for a commit that removed it.

- **Workarounds**, in the reporter's order: (1) delete `"pass_config": {"enable_sp": true}` from the
  start script (`reporter-confirmed`; cost = the ~5 % TTFT the key bought); (2) keep SP the supported
  way, `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` (`reporter-confirmed`); (3) code-derived and untested:
  `--enforce-eager` makes `enable_sp_by_pass` False by construction (the `not model_config.enforce_eager`
  conjunct), which strips the graph-mode SP entirely — cost = ACL-graph throughput.
- **Falsifier:** the same prompt on the same host with the key removed but FC1 exported still truncating
  → the SP route is not the owner; then run §B's reduction count (a double `maybe_all_reduce_tensor_model_parallel`
  / `_maybe_reduce_final_output`) and the `--enforce-eager` A/B before naming anything else.
- **Do not answer this class with the tokenizer/EOS story.** `stop_reason` being a token id invites a
  `generation_config.json`/`eos_token_id` explanation of the same shape as #10405's 中英混杂 family; the
  discriminator is that the corruption is *inside* the emitted words and disappears with one launch-key
  change.


## Triage order for this family

1. Count the reductions: `git grep -n "maybe_all_reduce_tensor_model_parallel\|_maybe_reduce_final_output\|
   _fused_output_is_reduced" <ref> -- vllm_ascend`, plus the upstream runner at the paired vLLM tag. A
   model patch that still reduces while the runner also reduces is the fingerprint.
2. Read the patch at the reporter's tag, not at `main` and not on the upstream tag — the hook is often
   deleted later, which silently hides the defect from a HEAD read.
3. For a warm-up/capture crash, divide the two row counts before reading anything else: a ratio equal
   to the TP size is the SP token contract, not a model weight shape.

## Recheck conditions

`enable_flashcomm1` spelling, the `enable_sp_by_pass` derivation and its custom-op consumers (§D), the
`_fused_output_is_reduced` branch and `enable_sp()` memoisation change between releases; the MiniMax hooks are trimmed as they are upstreamed, while the patch *file* stays (so
check `git cat-file -e <ref>:<path>` and then the symbol, not a listing). Re-read
`vllm_ascend/ops/fused_moe/fused_moe.py`, `vllm_ascend/patch/worker/patch_minimax_m2.py`,
`vllm_ascend/utils.py` and `worker/model_runner_v1.py` at the reported tag, and re-check PR #10443's
state, before calling either mechanism fixed. Nothing here was reproduced on hardware.
