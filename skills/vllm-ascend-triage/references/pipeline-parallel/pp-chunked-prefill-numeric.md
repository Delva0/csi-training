# PP × chunked prefill: a *numeric* first-token / logit defect

Scope: `--pipeline-parallel-size` > 1 and the report is numeric, not a crash and not
a rate — "pp+chunked prefill 场景下首token不对", "不触发 chunked prefill 首token正常",
"不开 PP 首token也正常", a first generated token or logit delta against the TP-only
run, or the capability form "PP 和 chunked prefill 能一起开吗". Weaker variants: a
decode-only difference in a 1P1D deployment where the P side is PP, or an eval score
that only moves when the batch is long enough to be chunked. Not this page: a PP
launch that dies (`index.md` routes), a throughput or TTFT number
(`../performance-throughput/index.md`), and not a TP-axis logprob gap
(`../accuracy-benchmarks/cross-platform-logprob-parity.md`).

Inherit SKILL.md and `index.md`. Five rules decide it.

1. **On an rc image the release lane is rung 1, and it is the one rung no source
   walk can replace.** Recorded **#12060** (reporter-confirmed): official image
   `quay.io/ascend/vllm-ascend:v0.20.2rc1` = vllm-ascend `v0.20.2rc1`
   (`367b8e62da79`) + vllm `v0.20.2` (`bc150f502991`), 4 cards, Qwen2.5-7B,
   PP2+TP2 — 首token wrong with chunked prefill, correct without it and correct
   without PP. The recorded run spent 21 turns and 43 K characters of reasoning on
   the upstream scheduler and the downstream runner fork, shipped a truncated
   zero-coverage verdict with four source hypotheses, and the reporter's own
   resolution (later comment, 2026-07-17) was one line: *"tried the latest version,
   it works fine"*. So the deliverable owed on the classification turn is: this is
   a defect of that rc at a combination the pin never tests (rule 2), install the
   next rc/final and re-run the *same* prompt at `temperature=0` with a fixed
   expected first token — then the mechanism only if the upgrade does not clear it.
   This is SKILL.md rule 16's shape (a fix after the pin is invisible in the
   snapshot) reached through accuracy rather than a hang.

2. **The coverage sentence is one grep and it is citable as coverage, never as
   cause.** At `367b8e62da79`: the PP e2e is
   `tests/e2e/multicard/4-cards/test_pipeline_parallel.py` — DeepSeek-V2-Lite-Chat,
   DP2/TP1/PP2, `prompts` = `"Hello, my name is"` / `"The future of AI is"`
   (`:35-38`), i.e. 5-6 tokens, so **the prefill is one chunk whatever
   `--max-num-batched-tokens` is** and that file cannot reach a multi-chunk PP
   prefill. The only multicard chunked-prefill e2e is
   `tests/e2e/multicard/4-cards/long_sequence/test_chunked_prefill_cp.py`
   (context parallel, not PP) and `test_profiling_chunk_performance.py` is a
   performance case. "PP is CI-backed" therefore says nothing about PP ×
   multi-chunk accuracy; re-grep per pin.

3. **The documented PP+chunked-prefill recipe at this pin is *dynamic chunk*, not
   plain chunked prefill.** `docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md:16-22`
   carries the support matrix with the `**CPP**` row ✅ across Eager / Graph /
   Prefix Cache / Chunked Prefill, enabled by
   `--additional-config '{"profiling_chunk_config": {"enabled": true}}'`
   (`docs/source/user_guide/configuration/additional_config.md:90`; it swaps in
   `ProfilingChunkScheduler` at `vllm_ascend/platform.py`), with constraints at
   `:141-148` — PP>1 required, `--enable-chunked-prefill` required, incompatible
   with `VLLM_ASCEND_BALANCE_SCHEDULING`. Upstream enables chunked prefill by
   default (`vllm/config/scheduler.py:84 enable_chunked_prefill: bool = True`), so
   a plain `vllm serve --pipeline-parallel-size 2` is on the intersection the docs
   never cover unless the reporter passed `--no-enable-chunked-prefill`. Ask which;
   do not assume either way, and say "the documented path is the one with the ✅".

4. **At this pin there is no AscendScheduler fork to blame.** `vllm_ascend/core/scheduler.py`
   is absent at v0.20.2rc1 (only `scheduler_dynamic_batch.py`,
   `scheduler_profiling_chunk.py`, `recompute_scheduler.py`), so the request path is
   the **upstream** V1 scheduler plus the downstream runner fork — do not carry the
   v0.11.0rc0 "platform force-enables the fork" fact into this ref. The two
   scheduler-side anchors that are real: upstream `vllm/v1/core/sched/scheduler.py:248`
   (`self.use_pp = pipeline_parallel_size > 1`) and `:1079-1099`, which sends the
   per-request `new_token_ids` slice only when `use_pp and not async_scheduling`
   (with `--async-scheduling` the PP path consumes them over a GPU broadcast). That
   makes `--async-scheduling` a genuine 2×2 axis at PP>1 and the downstream
   `sync_and_slice` / `sync_and_gather_intermediate_tensors` override in
   `vllm_ascend/worker/model_runner_v1.py` the other. Both are hypotheses about
   *where*; neither can be confirmed from the snapshot.

5. **Ladder — one axis per launch, in this order.** (a) next rc/final, same prompt;
   (b) `--pipeline-parallel-size 1` with TP absorbing the 4 cards (removes the
   pipeline and the `max_concurrent_batches = pp_size` queue depth, so it is the
   reporter's own PP-off control); (c) `--no-enable-chunked-prefill`, then the
   inverse — keep chunked prefill, set `--max-num-batched-tokens` *below* the prompt
   length so 2/3/4 chunks are forced: an error that first appears from the 2nd
   chunk is a chunk-boundary state carry, not a TP numerics gap; (d)
   `profiling_chunk_config.enabled: true` (rule 3 — a good answer here says the
   plain path is the broken one); (e) `--no-enable-prefix-caching`
   (`#2943`: "do not enable chunked prefill with prefix cache when running with
   Ascend scheduler", an older release's known issue — precedent, not this pin);
   (f) `--async-scheduling` on/off; (g) graph off / `--enforce-eager`
   (npugraph_ex default-on since v0.18.0, `release_notes.md:23`) — a graph captured
   at one chunk size reused for another.

Ask back, in the first message: both serve commands, `enable_chunked_prefill=` /
`pipeline_parallel_size=` / `max_num_batched_tokens=` from the startup config dump
(`vllm/config/vllm.py:1791` prints the first), the prompt's token count, whether
`profiling_chunk_config` is set, `--async-scheduling`, and expected/actual first
token plus the TP-only reference answer.

Gap to state: at `367b8e62da79` `docs/source/user_guide/release_notes.md` starts at
`## v0.18.0` (`:3`), so the pinned rc's own release notes and Known-Issues section
are **missing from the export** — "no known issue mentions PP+chunked prefill" read
off that file is a gap, not evidence. Limits: no NPU here, so a verdict is
source-verified at that commit plus the reporter's own ablations.
