# Worked case: "Known Issues for DeepSeek-V4-Flash/Pro on A3"

vllm-ascend **#15985**, `[Bug][v0.26.0rc1]`, filed 2026-09-08 as a bulletin: two
tracked issues — #15892 (Flash, decode node dies with
`AclrtSynchronizeStreamWithTimeout` `507034`, "vector core timeout", fault kernel
`MoeDistributeDispatchV2`, after a period of sustained load) and #15930 (Pro, 1M
context PD + DSpark, four-host A3 TP16/DP2 P + TP16/DP2 D, `IndexCheck_int64_high_performance_0`
out of bounds `507035` then the same dispatch timeout on decode). No log in the
body; pins vllm-ascend v0.26.0rc1 `f2f74a16` + vllm v0.26.0 `568afb3a`.

## What the pinned release's own records establish

Read at `f2f74a16`; both are one `sed` and turn the notice into a scope
statement:

- `tests/vllm_ascend_v0.26.0rc1_test_conclusion.md:5` — "the following 4 models,
  which are the only models supported in this release" (rows at `:11` Flash, `:12`
  Pro); the table validates
  **DeepSeek-V4-Flash w8a8c8 on A3, 4 nodes, large DP, 1P1D** and
  **DeepSeek-V4-Pro w8a8c8 on A3, 2 nodes, large EP, 1P1D** for performance +
  GPQA. So the two models are in scope; the notice is talking about the release's
  own validated path.
- `…_test_conclusion.md:53` (§3, 18 rows) — the release's unresolved
  known-issues table, with issue ids. `507034` appears there as row 15
  (#15268, Ascend 950, C16 + `dsa_cp` start hang) for a scenario with nothing to
  do with DeepSeek-V4 — the code is a generic stream-sync/collective timeout, so
  it must be paired with the named fault kernel to mean anything.
- `docs/source/tutorials/models/DeepSeek-V4-Pro.md:408` "A3 series with DSpark":
  `--data-parallel-size 2`, `--tensor-parallel-size 16`, `--max-model-len 135000`,
  `--max-num-batched-tokens 4096`, `--max-num-seqs 16`, `HCCL_BUFFSIZE=1024`,
  `--speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}'`.
  The reported Pro run is **1M context** — ~7.4× that `--max-model-len` — which
  makes "outside the validated envelope" a first-class candidate next to any
  code fault, and gives a one-launch ablation at the documented length.
- `vllm_ascend/ops/fused_moe/token_dispatcher.py:120` (`enable_dispatch_v2 =
  hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")`), `:241-243` (the v2 /
  v1 call), `vllm_ascend/ascend_forward_context.py:35`
  (`_MC2_TOKENS_PER_RANK_LIMIT = 512`) — the dispatch op in the notice is a
  torch_npu op selected at runtime, so the owning layer is the MC2 all-to-all
  capacity/path (`references/hccl-and-hardware/mc2-dispatch-profile-run-hang.md`),
  not a vllm-ascend kernel bug; the `IndexCheck` OOB is a separate fault class
  whose candidates are the int64 index writes
  (`references/dsa-sparse-attention/indexer-cache-scatter-fault.md`).

## The reply shape this shape needs

1. First paragraph: the notice is accurate as a **risk statement** and not yet a
   diagnosis; name where each signature localizes at the pin; name the cheapest
   next action (plog for the failure minute on every node, both serve commands,
   one ablation that separates the family).
2. Point out that two events are being reported as one chain, and that the
   notice's own ordering (index OOB first, dispatch timeout after) is the most
   informative clue available — the timeout is the *aftermath* on a wedged
   stream, which is why the earlier fault kernel is the one worth chasing.
3. One ranked list per tracked issue, ≤3 causes, each with its single check and
   its confidence label; the growth/leak and process-placement precedents are
   support, not causes, when the reporter's symptom is a timeout.
4. State that #15892 / #15930 are outside the history corpus (earlier-month ids
   only), so no precedent body can be read for them; ask for the trackers'
   content if the reply must cite them.
5. Scope block: version, hardware, scenario, what was verified only by reading
   this tree, and what only the reporter's plog can decide.
