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
| only the first request after start is garbled, later ones clean | warm-up / init | connector, graph capture, c8 reshape buffers — precedent #12905 |
| every request at long length, same prompt | length-dependent corruption | sparse-attention indexer / DCP block remap / interleave math |
| fluent but wrong or looping | decode/reuse defect | prefix cache, KV offload/eviction, MTP draft — for long *outputs* see `long-output-repetition.md` |
| empty, `null` or cut mid-sentence | budget/truncation | `max_model_len`, `max_tokens`, `reasoning-output-routing.md` |

Ask which one it is; the four have different owning code. A one-line repro with
`temperature=0` and `max_tokens` well inside `max_model_len` removes the sampling
and truncation rows cheaply.

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
| eagle3 `num_speculative_tokens` | 1 | 3 | `:360,:440,:521,:601` |
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
proven cause, and the recorded run never got as far as saying so — it spent 21
turns and its whole output budget inside
`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` (2600+ lines,
read in eight slices) and shipped no answer at all.

Two report-side traps from the same case, both cheaper than any source read:

- **A `</think>` leak into `content` is not corruption until `reasoning_content`
  is captured.** At vllm v0.23.0 (`0fc695f`),
  `vllm/reasoning/kimi_k2_reasoning_parser.py:40-46` selects the identity
  fallback from `chat_template_kwargs["thinking"]`, not from a request's
  `enable_thinking`, so whether the tag is stripped depends on which key reaches
  the parser; and a client that concatenates only `delta.content` cannot tell a
  routing decision from a model-emitted token. One request printing both fields
  decides it (`reasoning-output-routing.md` carries the neighbouring shapes).
- **The reporter's own mechanism is a hypothesis.** #13002 asserted that
  `HCCL_BUFFSIZE=256` "overflows and corrupts KV"; a too-small HCCL buffer is a
  collective-side complaint, and the value is worth A/B'ing because the validated
  command uses 800, not because the mechanism is established.

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
