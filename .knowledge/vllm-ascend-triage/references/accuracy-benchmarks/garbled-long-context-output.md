# Garbled / degenerate long-input output

Symptom family: "a ~400K-token prompt returned 乱码 / garbage / repeated or
meaningless text", on a deployment where short prompts are fine. This is a
corruption report, not a score gap: no harness is being compared and no metric is
being recomputed. Do not route it through `harness-vs-engine-attribution.md`.

Before using this page, read `usage.prompt_tokens`: if the prompt is only a few
thousand tokens, the length-dependent owners below cannot be the cause — use
`short-prompt-corruption.md` instead. And if the prompt is short while the
*completion* runs to tens of thousands of tokens and the symptom is sentences
repeating rather than token soup, use `long-output-repetition.md`: the owners there
are reuse and draft bookkeeping, not the sparse-attention / DCP math below.

## Grade the symptom before theorising

| Observed | Class | First owner to suspect |
|---|---|---|
| only the first request after start is garbled, a **different-prompt** later one clean | warm-up / init | connector, graph capture, c8 reshape buffers — precedent #12905 |
| every request at long length, same prompt | length-dependent corruption | sparse-attention indexer / DCP block remap / interleave math |
| fluent but wrong or looping | decode/reuse defect | prefix cache, KV offload/eviction, MTP draft — for long *outputs* see `long-output-repetition.md` |
| empty, `null` or cut mid-sentence | budget/truncation | `max_model_len`, `max_tokens`, `reasoning-output-routing.md` |
| present **only** with `stream: true` (the same body with `stream: false` is clean), clustered where the script changes (ascii ↔ 中文 ↔ 箭头), sometimes losing leading characters and repeating a fragment | streamed **text delta**, not generation math | the incremental detokenizer — `../tool-call-streaming/detokenizer-text-delta.md`; one grep for `resetting decode stream` in the server log decides it |

Ask which one it is; the four have different owning code. A one-line repro with
`temperature=0` and `max_tokens` well inside `max_model_len` removes the sampling
and truncation rows cheaply.

**A repeated identical prompt is not the warm-up control — it is a cache probe.** Every
"首次请求乱码 / 第二次 curl 正常" report must be re-sent with a **fresh prompt** (or a
nonce appended), because with prefix caching on (the default) or a KV pool attached
(`AscendStoreConnector`/memcache, `MooncakeConnectorV1` + `AscendStoreConnector` in a
`MultiConnector`) the identical second request can be served from a hit and never re-run
the prefill/write path that produced the corruption. Recorded #12905 filed this shape
without saying whether the second curl used a different prompt, set no
`--no-enable-prefix-caching`, and had memcache attached — both confounds are unresolved
by the report, which is why the control must be run deliberately. So the first row of the
ladder is "new prompt, nothing else changed" — if that one is garbled too, the warm-up/init
row above is refuted before any source read.

## First move when both serve commands are pasted: diff them against the model tutorial

Every documented model ships runnable commands in
`docs/source/tutorials/models/<model>.md` plus a `Recommended Configurations`
section. When a report pastes its **P and D** commands and the output degenerates
only at long context + concurrency, that diff is the answer skeleton: each
deviation is one ablation row, and it takes two calls to build. Verified at
vllm-ascend **v0.23.0rc1** (`f4a08bd`), `docs/source/tutorials/models/Kimi-K2.6.md`:

| item | validated P node | validated D nodes | where |
|---|---|---|---|
| `--no-enable-prefix-caching` | present | present | `:172,357,437,517,597` (`:190`: it is in every example) |
| `HCCL_BUFFSIZE` | 800 | 800 / 1100 | `:337,:417,:497,:577` |
| `--max-model-len` | 32768 | 32768 | `:355,:435,:515,:595` |
| `--max-num-batched-tokens` | 16384 | 32 or 4 | `:356,:436,:516,:596` |
| eagle3 `num_speculative_tokens` | 1 | 3 | `:360,:440,:521,:601` — but §9.1 Table 2 (`:850-859`) prints `MTP Speculation Num 3` for the PD rows too, so quote the runnable command and do not spend a turn reconciling the two |
| `--max-num-seqs` | 4 | 8 | `:354,:434,:514,:594` |
| decode-only features | `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, no MLAPO | `VLLM_ASCEND_ENABLE_MLAPO=1`, `recompute_scheduler_enable: true` | `:625-627` (explicit: MLAPO and the recompute scheduler on decode only) |

§9.1 (`:831-864`) states it was "validated in specific test environments", and
its Table 1 has **no PD row for 128K**: long context appears only as
"Single-Node Mixed", while every 1P1D/2P1D row is the 16K/multimodal scenario and
Table 2 gives those PD rows `Max Model Len 17K`. So a PD deployment asking for a
131072 context is outside the documented envelope — say that as a caveat next to
the matrix, not as a root cause, and note that a 23K prompt still sits inside the
32768 of the example commands.

Applied to #13002 (Kimi-K2.6 W4A8, 1P1D + `MooncakeConnectorV1`, eagle3,
≥16 concurrency, 23K prompt, 6-10 % degenerate answers), the five deviations were:
`--enable-prefix-caching` on **both** nodes, `--max-model-len 131072`, P
`HCCL_BUFFSIZE=256`, D `--max-num-batched-tokens 256`, eagle3
`num_speculative_tokens 3` on P. Those are the rows to hand back; none of them is a
proven cause. The first recorded session never got as far as saying so — it spent
21 turns and its whole output budget inside
`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` (2600+ lines,
read in eight slices) and shipped no answer at all. The replay that had this page
shipped the complete ranked answer in 11 assistant turns / 17 tool calls and opened
that connector exactly once — for the single `dp_size`/`tp_size` grep the P-vs-D
asymmetry needed — which is the cost this page is meant to save. **A still later
replay with this page shipped nothing, and the cut point is textual**: it found this page
on its 3rd assistant turn, verified the table above plus one parser gate in the 4th (two
calls), wrote "the deliverable is that page's recipe-diff ladder. Let me verify its two
version-specific claims" — and spent the remaining ten turns (the parser split, the
connector's `engine_id` assert, the recompute gate) until `output_tokens` ended the session
with **no message text at all**: 14 assistant turns, 18 tool calls, ~1 K characters of
status-line text against 54.8 K of reasoning, 1.08 M cumulative prompt tokens. A turn whose
text *names* the deliverable and then opens a verification is the turn to send it: type the
five deviations as the ranking, then let each later read correct one named row — the two
owning lines below arrived that way and belong in a follow-up message, not in front of the
first one.

Two report-side traps from the same case, both cheaper than any source read:

- **A `</think>` leak into `content` is a routing question, and the key that reaches
  the parser is now settled.** At vllm v0.23.0 (`0fc695f`) the request's own kwargs *do*
  reach it — the parser is constructed per request with
  `chat_template_kwargs=self._effective_chat_template_kwargs(request)`
  (`vllm/entrypoints/openai/chat_completion/serving.py:180-190,243-249`) — and `kimi_k2`
  reads the key **`thinking`** (`kimi_k2_reasoning_parser.py:39-47`), so the Qwen-family
  body key #13002's repro sends, `chat_template_kwargs: {"enable_thinking": false}`, is a
  **no-op**: `thinking` stays True, no identity parser is installed, and the tag is treated
  as a routing marker — `extract_reasoning` splits at the **first** `</think>`
  (`:157-177`), so text before it becomes `reasoning` and text after it `content`. The body
  key that turns the parser off is `{"thinking": false}`. A tag still visible inside
  `content` therefore means the split already happened (a second tag, or the streaming path
  at `:190-245`); one `stream: false` response printing both `content` and `reasoning`
  separates that from a model-emitted tag, and a client that concatenates only
  `delta.content` cannot (`reasoning-output-routing.md` carries the neighbouring shapes).
  A startup that raises `KimiK2ReasoningParser could not locate think start/end tokens in
  the tokenizer!` (`kimi_k2_reasoning_parser.py:62-66`) is the tokenizer, not the request:
  that check runs at parser construction.
- **The reporter's own mechanism is a hypothesis.** #13002 asserted that
  `HCCL_BUFFSIZE=256` "overflows and corrupts KV"; a too-small HCCL buffer is a
  collective-side complaint, and the value is worth A/B'ing because the validated
  command uses 800, not because the mechanism is established.
- **A `completion_tokens` figure far above the returned text is a routing
  measurement.** #13002's shape 2 reported `completion_tokens: 9665` against 831
  characters of `content`; with a reasoning parser active and a client that
  concatenates only `delta.content`, the missing bulk is in `reasoning`, not lost.
  Compute the tokens the text should have cost before calling it collapse, and ask
  for one `stream: false` response with both fields.
- **Re-read the reproduction table before adopting the reporter's own axis.**
  #13002 filed "long context **+** ≥16 concurrency" and its table agrees only on
  the length: 16-concurrency on a single instance was 0/32 at 23K, while the
two-instance runs degraded 8-9 % — i.e. the pressure that moved was the aggregate
  load on the shared P node, and the reporter separately noted that production
  degrades at concurrency 3 and that `max_num_seqs=8` changes nothing. An axis the
  reporter asserts but their own numbers do not support is a ranking input, and the
  cheapest row is the one that halves the load.
- **When the decode node carries `recompute_scheduler_enable: true` together with
  `cudagraph_mode: FULL_DECODE_ONLY`, rank the recompute/pad row high.** #8320
  (Qwen3.5-397B, mixed deployment, DP8·TP2, KV cache saturated) is the closest
  recorded mechanism: a recomputed request turns one card into a long prefill while
  the others are padded to full decode length, the padded tokens were not masked
  out of MoE routing, they concentrated on a few experts and exceeded the fused
  dispatch/combine capacity, and the output came back empty or garbled; the
  reporter's own A/B was batch size down, `skip_all_reduce=True`, `--enforce-eager`
  and a larger `max_output_size`. That is a reporter-side claim, but it names the
  rows for this deployment shape (`recompute_scheduler_enable: false`,
  `--max-model-len` small enough that KV is never saturated, `--enforce-eager`) and
  it explains a 3-minute streaming gap better than any connector story. At
  vllm-ascend v0.23.0rc1 (`f4a08bd`) that row also has an owning site:
  `attention/utils.py:372-392` sets `treat_short_extends_as_decodes = True` whenever
  `is_pd_decode_recompute_scheduler_enabled()` is true — true only for a **kv_consumer**
  with `recompute_scheduler_enable` (`vllm_ascend/utils.py:1148-1168`, docstring: "After
  KV recv, RecomputeScheduler sets num_computed_tokens to N-1 so the decode node
  recomputes the last prompt token before MTP decode. Worker metadata must not treat that
  step as prefill.") — i.e. on exactly this D-node shape a recomputed request is
  classified **decode** in the mixed-batch split (`:381-392`). Ask for one degenerate
  step's `max_query_len` / `num_decodes` line beside that row.

## The deliverable is an ablation matrix

In a parallelism-rich deployment (PD disaggregation, DCP/PCP, sparse c8
attention, KV offload, MTP) the report names several enabled features and no
ablations, and every feature is a plausible owner; a source walk cannot separate
them. Hand the reporter this table instead (each row is one restart, same failing
prompt, `temperature=0`), and keep it in the first answer:

| Row | Change | Corrupt here means | Clean here kills |
|---|---|---|---|
| length sweep 32K / 128K / 256K / failing length | none, only prompt length | corruption starts at one length → block-count/capacity arithmetic; corrupt at every length → sparse-selection shape | "the weights/quantization are bad": length-tracking corruption is plumbing |
| `--decode-context-parallel-size 1` | DCP off, tp and all sparse flags unchanged | DCP KV split/remap owns it | DCP size as the trigger |
| `enable_sparse_sfa_c8: false` (keep `enable_sparse_li_c8`) | sparse SFA path off | sparse SFA prefill path owns it | sfa_c8 as the trigger |
| `enable_dsa_cp: false` on the **prefill** node | DSA context parallelism off | prefill-side CP gather/store owns it | dsa_cp as the trigger |
| same request repeated 3x, `--max-num-seqs 1` | isolate first-request and reuse | warm-up/prefix-cache class | steady-state corruption |
| optional: sparse KV offload / prefix caching off | only if enabled | offload/eviction mis-loads blocks | KV offload as the trigger |
| re-run at the documented tested length (e.g. 198k) | shrink into the tested envelope | — | "outside tested envelope" becomes the finding, not a regression |

Constraints that make a row un-runnable must be stated, not discovered by the
reporter: with sparse SFA models (GLM-5.x), `decode_context_parallel_size > 1`
requires `dcp_size == tp_size` (`vllm_ascend/platform.py:723-729` in v0.26.0rc1),
so "dcp=8 with tp=16" refuses to start — the meaningful DCP row is off vs
`dcp == tp`. PD disaggregation with CP requires
`cp_kv_cache_interleave_size == block_size` (`platform.py:731-741`).

Ask for the artefacts with the matrix: exact `vllm serve` command of **both** P
and D nodes (they are configured separately and often differ), prompt token count
vs completion text, and the request's log lines. A garbled-output report without
the two commands cannot attribute anything to a component.

## Check the tested envelope before calling it a defect

Verified at vllm-ascend v0.26.0rc1 (`f2f74a16c3c50a76f4349d807918e83edec1e35c`):

- `docs/source/user_guide/support_matrix/supported_models.md:42-43` lists GLM-5/5.1
  and GLM-5.2 with `max-model-len 200k`; the tree contains no GLM-5.3 entry at all
  (`grep -rn "GLM-5.3"` → nothing), and #15088 is the request to adapt GLM-5.3.
- `tests/vllm_ascend_v0.26.0rc1_test_conclusion.md:5` — "the following 4 models,
  which are the only models supported in this release" (GLM-5.2 w8a8c8 A3 among
  them); `:53` starts the release known-issues table (issue ids + impact).
- `docs/source/tutorials/models/GLM5.2.md:1796-1805` — every reference performance
  config, including the rows titled "high concurrency(1M)", was measured at
  **Max Model Len 198k**; `:1140-1141` is the PD-disaggregation command with
  `--decode-context-parallel-size 16 --cp-kv-cache-interleave-size 128`;
  `:1817` notes sparse SFA C8 is "Experimental in v0.23.0" and, on w4a8c8
  weights, combining it with DCP has known issues (the reporter's w8a8c8 is the
  combination the doc does endorse).

So a 512K / 400K-input report against this release is outside the validated
envelope; say that explicitly and put the ablation result next to it. Do not
translate "untested" into "unsupported root cause".

## Precedent, and what it does not prove

- #12905: v0.23.0rc1, A3 4-node PD disaggregation, GLM-5.2 with
  `enable_sparse_sfa_c8: true`, DCP=8 (tp=8) — *first* curl request answered
  garbled. Same feature cluster, different release; the issue body carries the
  full command and no cause. Use it to justify the first-request/repeat row, not
  as the answer.
- #14320: v0.26.0rc, GLM-5.2 w4a8 PD + DCP=8 + MTP — decode EngineCore **crashed**
  in `LightningIndexerQuant` (MTE invalid GM address); the reporter's instrumented
  claim is that the DCP replicated block table produced block ids above `k_cache`
  capacity after the PCP-removal refactor. A crash, not silent garbage, and a
  reporter claim rather than a maintainer confirmation — but it is the strongest
  pointer that DCP block-id remapping is capacity-sensitive in this release line.

## Case note (#16015, reporter-confirmed)

GLM-5.3 w8a8c8, 0.26.0rc1, PD disaggregation, 512K scenario, `sfac8 && dcp16`,
400K+ input → 乱码. The original reporter later added, in the same thread,
"dcp==16 的场景正常" (the dcp==16 scenario is normal). Read narrowly, that
exonerates DCP=16 on its own and leaves the sparse-c8 path, the length/capacity
math and the PD KV transfer/offload as the live rows; read broadly it is only a
one-line statement with no command attached, so ask which configuration it refers
to before closing the DCP row. Either way the next evidence is the matrix above,
not a source walk.
