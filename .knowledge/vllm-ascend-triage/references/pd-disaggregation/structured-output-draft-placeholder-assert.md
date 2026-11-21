# Decode-node `assert accepted` with `-1` draft tokens under structured output

## When this page applies

A **P/D deployment** with speculative decoding (`mtp`, `deepseek_mtp`, `dspark`,
eagle) **and** structured output (`response_format`, `guided_*`,
`--structured-outputs-config`) where the **decode** node's EngineCore dies at the
scheduler step, before any model execution:

```
[..] /project/cpp/grammar_matcher.cc:428: Warning: The token id -1 is out of range [0, 129280). Rejecting the token.
ERROR .. [backend_xgrammar.py:180] Failed to advance FSM for request chatcmpl-… for tokens -1. Please file an issue.
ERROR .. [patch_core.py:70] EngineCore encountered a fatal error.
  File ".../vllm/v1/core/sched/scheduler.py", line 1048, in get_grammar_bitmask
  File ".../vllm/v1/structured_output/__init__.py", line 288, in grammar_bitmask
    assert accepted, (token, req_id, scheduled_spec_decode_tokens)
AssertionError: (-1, 'chatcmpl-…', {'chatcmpl-…': [-1, -1, -1]})
```

Three readings of the paste, before any other theory:

- **The `-1`s are placeholder draft slots, and the count is the launch's
  `num_speculative_tokens`** — `[-1, -1, -1]` beside a decode
  `--speculative-config '{"num_speculative_tokens": 3, …}'`. A real drafter never
  emits `-1`, so the ids were seeded, not produced.
- **The assert is upstream and backend-agnostic** (`vllm/v1/structured_output/__init__.py:288`);
  `guidance`/`outlines` reach the same line. The C++ `out of range … Rejecting`
  warning is the matcher doing the right thing, not the fault, so "change the
  structured-output backend" is not a lever.
- **Nothing ran on device**: `get_grammar_bitmask` is called from `EngineCore.step`
  before `execute_model`, so no plog, `npu-smi` or HCCL read belongs in the reply.

Recorded **#6421** (vllm-ascend `v0.13.0rc2` `a17059a2` + vllm `v0.13.0`
`72506c98`, DeepSeek-V3.2-w8a8, P/D + MTP + a JSON schema). The reporter's own
later comment on the issue: **"Solved: Inconsistent configuration between
prefilling and decoding nodes."** Which key differed was not disclosed — so the
first action is the diff, and it costs one command.

## Step 1 — diff the two serve commands (the recorded resolution)

This crash sits in the same class as the confirmed #16177 (`kv-group-count-mismatch.md`):
a decode node whose spec/bookkeeping config does not match its prefiller. Compare,
in this order:

| Key | What must hold |
|---|---|
| `--speculative-config` | present on **both** sides with the same `method`, the same `num_speculative_tokens` and the same draft path. One side only is #16177's reporter-confirmed cause; the pinned recipes state the rule in words (Kimi-K3 §5.2 / FAQ `:886`, and `DeepSeek-V3.2.md:433` ships it on the decode block) |
| `--additional-config` → `recompute_scheduler_enable` | the recipes enable it on the **decode** node only; a key present on the wrong side is a deviation to name, not a fix (it is also the gate for Step 2) |
| `--kv-transfer-config` | complementary `kv_role`, and `kv_connector_extra_config` `prefill{dp,tp}` / `decode{dp,tp}` equal to each node's real `--data-parallel-size`/`--tensor-parallel-size` (`recipe-contract-diff.md` invariants 1-4) |
| everything else the recipe pins on both sides | `--block-size`, prefix caching, `--no-disable-hybrid-kv-cache-manager`, `--max-model-len`, `--quantization`, `--tokenizer-mode` / `--reasoning-parser` / `--tool-call-parser`, the serve name and model path |

A tell inside the paste itself: the seeded list is `[PLACEHOLDER] * self.num_spec_tokens`
of the **decode** side, so the length always matches the decode command. If it
matches the decode value and **not** the prefill one, the two commands already
disagree — say so and stop ranking code.

## Step 2 — the source path that seeds `-1`, and the ablation that splits it

Verified at the pin above (`rg -n` re-checks in the last section):

| Fact | Anchor |
|---|---|
| Gate: spec config **and** `kv_transfer_config` **and** `is_kv_consumer` **and** `vllm_version_is('0.13.0')` | `vllm_ascend/core/recompute_scheduler.py:101-104` |
| `add_request` seeds the draft slots: `request.spec_token_ids = [PLACEHOLDER_TOKEN_ID] * self.num_spec_tokens` — comment "to enable full graph when decode nodes pull the KV cache of one request from prefill nodes" | `:107-111` |
| `PLACEHOLDER_TOKEN_ID` is **-1** | vllm `vllm/v1/sample/rejection_sampler.py:22` |
| The seeded ids are trimmed into `scheduled_spec_decode_tokens[req_id]` on the request's first scheduled step (`num_scheduled_spec_tokens = num_new_tokens + num_computed_tokens − num_tokens − num_output_placeholders`) | `:289-298` (running queue), `:519-532` (waiting queue, `is_mtp_kv_consumer` branch) |
| Upstream hands that dict to the manager untouched — the only filter is `req.use_structured_output` | vllm `vllm/v1/core/sched/scheduler.py:1030-1057` (`:1043`) |
| The manager advances the FSM **once per scheduled draft token** and asserts | vllm `vllm/v1/structured_output/__init__.py:264-292`, assert at `:288` |
| xgrammar rejects `-1` as out of vocab → the two log lines | vllm `vllm/v1/structured_output/backend_xgrammar.py:170-186` (message at `:181`; the paste's `:180` is line drift, not another site) |
| `recompute_scheduler_enable` defaults to `False` and is a plain `additional_config` key | `vllm_ascend/ascend_config.py:90-91`, pulled in by `platform.py` |

The repo's own DeepSeek-V3.2 P/D **decode** recipe ships both halves of the gate —
`--speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}'`
(`docs/source/tutorials/DeepSeek-V3.2.md:433`, again `:512`) and
`--additional-config '{"recompute_scheduler_enable" : true}'` (`:463`, again
`:542`) — so a reporter following the tutorial reaches this state without inventing
anything.

**The one-launch ablation that decides the lane:** at this pin the injection is
unconditional once `is_mtp_kv_consumer` is true, so Step 1's parity fix is *not*
predicted to remove it. Remove `"recompute_scheduler_enable"` from
`--additional-config` on **both** nodes, keep MTP and the same schema, re-send one
request:

- assert gone → the placeholder path is the mechanism; Step 3 applies, and the
  parity diff is secondary (the same launch also removes the recompute-retry
  behaviour on decode — say what it costs).
- assert still there with `-1` → the ids came from somewhere else, Step 1 owns the
  case, and the missing artefact is the two serve commands plus each side's
  `--speculative-config`.

Report both readings in the first message instead of picking one: the recorded
resolution (config parity) and the source-verified placeholder path are competing
explanations of the same paste, and this single ablation separates them.

## Step 3 — only if the ablation says "placeholder path"

The repair must keep the bitmask rows aligned with the sampled positions —
`cumulative_index` advances once per scheduled token, so a request keeps
`1 + len(spec tokens)` rows. So:

1. Do not seed placeholders for a request that carries
   `sampling_params.structured_outputs` (`recompute_scheduler.py:109`); that request
   takes a different spec length on its first decode step (graph-bucket miss, eager
   fallback, no correctness change).
2. Or skip `token < 0` before `accept_tokens` **while still advancing
   `cumulative_index`** for that row.
3. Deleting the `assert` (or `try/except` around it) is not a fix: the request's
   masks then advance fewer states than the sampler consumes, so enforcement is
   silently wrong instead of crashing.

Acceptance check for any patch: the same schema returns JSON that parses, an
invalid continuation is still blocked (compare against the MTP-off run), and
`bitmask.shape[0] == Σ(1 + drafts)`.

## Version scope and recheck

Verified at vllm-ascend `v0.13.0rc2` (`a17059a2`) + vllm `v0.13.0` (`72506c98`);
the seeding is gated on `vllm_version_is('0.13.0')`, so it is specific to that
pairing. Line numbers drift between releases — re-grep on the reporter's tree:

```
rg -n 'PLACEHOLDER_TOKEN_ID|is_mtp_kv_consumer' vllm_ascend/core/recompute_scheduler.py
rg -n 'assert accepted' $(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')/v1/structured_output/__init__.py
```

A tree with no `PLACEHOLDER_TOKEN_ID` hit under `recompute_scheduler.py` retires
Step 2 and leaves the parity lane.

## What to ask for

Both serve commands verbatim (the `--speculative-config`, `--additional-config`,
`--kv-transfer-config` and `--structured-outputs-config` blocks), per-node
`vllm`/`vllm-ascend` versions, and the result of the Step 2 ablation. Nothing about
plog, HCCL or rank tables applies: the crash is a host-side scheduler step.
