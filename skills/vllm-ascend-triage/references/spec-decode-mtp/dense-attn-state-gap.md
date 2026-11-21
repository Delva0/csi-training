# `assert attn_metadata.attn_mask is not None`: the spec-decode state the dense backend never learned

Signature (recorded verbatim, #2071, vllm-ascend v0.9.2rc1):

```
(VllmWorker rank=0 pid=15500) ERROR … [multiproc_executor.py:522]   File "…/vllm_ascend/attention/attention_v1.py", line 413, in forward
(VllmWorker rank=0 pid=15500) ERROR … [multiproc_executor.py:522]     assert attn_metadata.attn_mask is not None
(VllmWorker rank=0 pid=15500) ERROR … [multiproc_executor.py:522] AssertionError
```

on a launch carrying `--speculative-config '{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_max":4}'`.
Every rank reports it in the same second; before it the engine served requests
normally for tens of seconds; the traceback may pass through `<eval_with_key>.N`,
`torch.fx.graph_module` and `torch.ops.vllm.unified_ascend_attention_with_output`
(`attention_v1.py:440-459` at that ref) — that is the graph wrapper calling the
same eager Python body, not a capture artefact.

## What the assertion is

`attention_v1.AscendAttentionBackendImpl.forward` is a state machine over
`attn_metadata.attn_state` with three explicit branches — `PrefillNoCache`
(:330-353), `PrefillCacheHit` (:356-374), `DecodeOnly` (:375-388) — and then one
`else:` commented `# Normal V1 situation.` (:389) that serves **both**
`ChunkedPrefill` and anything else. Inside that `else` the `head_size == 192`
sub-branch (:395-408, `vanilla_chunked_prefill`) tolerates a missing mask; every
other head size reaches `torch_npu._npu_paged_attention_splitfuse` and asserts the
mask first (:413, kernel call :420-432).

The mask comes from `_make_attention_mask` (`model_runner_v1.py:697-714`), which
returns a tensor for `ChunkedPrefill` / `PrefillNoCache` / `PrefillCacheHit` and
`None` for everything else (:712-714, `# Decode-only situation.`). So a state that
reaches the `else:` and is not `ChunkedPrefill` gets `None` and dies at :413.

## Which state reaches it, and why ngram is the trigger

State selection, `model_runner_v1.py:971-989`:

```
np.array_equal(seq_lens, num_scheduled_tokens)  -> PrefillNoCache
np.all(num_scheduled_tokens == 1)               -> DecodeOnly
np.all(num_valid_tokens == 1)                   -> ChunkedPrefill if self.use_eagle else SpecDecoding
```

`num_valid_tokens[i] = num_scheduled_tokens[i] - len(scheduled_spec_decode_tokens[req_id])`
(:911-912), and `self.use_eagle` is set **only** for `method in ["eagle", "eagle3"]`
(:221 default `False`, :234 `True`). The drafter choice is a separate field in the
same `if/elif` chain: `ngram -> NgramProposer` (:232), `deepseek_mtp -> MtpProposer`
(:240), and neither touches `use_eagle`.

Consequences worth stating in the answer:

- `SpecDecoding` is chosen for **every** non-eagle spec method the moment a request
  carries draft tokens and the batch is not uniformly one token — ngram today, and
  a dense model with `deepseek_mtp` by the same branch (unverified by run).
- It fires **after** a healthy period, on the first decode step in which the
  proposer actually returns a draft. With `--dataset-name random` an n-gram hit is
  rare, which is exactly why the paste shows ~40 s of `Running: 1 … 3 reqs` with
  normal tok/s first. That is a timing fact about the *proposer*, not a
  concurrency or capacity clue — do not rank load, KV or the sampler on it.
- `head_size == 192` models (DeepSeek-family) take the `vanilla_chunked_prefill`
  escape and never read the mask, so the same flag looks model-dependent. Whether
  that path is *numerically* correct for a verify batch is not established here.

## One grep decides it

```sh
rg -n 'SpecDecoding|spec_attn_mask' vllm_ascend/attention/ vllm_ascend/worker/model_runner_v1.py
```

At v0.9.2rc1 the counts are the diagnosis:

| Site | Evidence |
|---|---|
| The enum value is declared | `attention_v1.py:117` `SpecDecoding = 4` — and it is the **only** occurrence of the name in that file |
| The runner allocates the mask for *any* spec config | `model_runner_v1.py:220-229` `self.spec_attn_mask = torch.triu(torch.ones(2048, 2048, …), diagonal=1)` |
| Only the MLA backend consumes it | `mla_v1.py:343`, `:498` put `self.runner.spec_attn_mask` into decode metadata; `:957-968` selects `sparse_mode = 3` + `atten_mask=spec_attn_mask` under `SpecDecoding` (:1003) |
| The dense backend never reads it | zero hits for `spec_attn_mask` in `attention_v1.py` |

So the state and its mask were built for MLA; the dense backend has no branch. That
is a downstream **coverage gap**, not install/CANN/driver, and not a flag problem.

## The tree states it too

`tests/e2e/singlecard/spec_decode_v1/test_v1_spec_decode.py:62` is
`test_ngram_correctness`, and `:72` is `pytest.skip("Not current support for the
test.")` — before its first line of body. The `eagle3` test in the same file
(`:139` ff.) carries no such skip. Cite the skip as the project's own statement of
non-support at that ref; it is cheaper than any further source walk and it dates
the claim (`rg -n 'def test_ngram_correctness' -A12 tests/`).

## Fix ladder

1. Drop the method: remove `--speculative-config` and re-run the same benchmark.
   That is the ablation that ties the crash to the verify batch.
2. If speculation is needed on a dense model: `eagle3` with its head
   (`{"method": "eagle3", "model": "<EAGLE3 head>", "num_speculative_tokens": 2}`)
   — the branch CI exercises; or an MLA model where `spec_attn_mask` is consumed.
3. Not the levers, and say so explicitly: `--enforce-eager` (the failing statement is
   plain Python), `--compilation-config` / capture sizes, `--quantization ascend`,
   `VLLM_ASCEND_ENABLE_NZ`, `ASCEND_LAUNCH_BLOCKING` — none of them touch
   `attn_state`.
4. As a feature request, the fix has two candidate shapes and neither is verified
   to serve: build a splitfuse-style mask for `SpecDecoding` in
   `_make_attention_mask`, or treat non-eagle spec decode as `ChunkedPrefill`
   (drop the `use_eagle` split at :977-981 — `SpecDecoding` is otherwise read only
   by MLA's graph-mode q-reshape, `mla_v1.py:957`). Label them as proposals, and
   name the existing skipped test as the test to enable.

## Scope and recheck

Verified at `vllm-project/vllm-ascend` **v0.9.2rc1** (`b5b7e0e`) with vllm
**v0.9.2** (`a5dd03c`). The state table, `use_eagle` gate and
`_make_attention_mask` are version-specific: re-run the one grep above at any later
ref before reusing the verdict — a later release may implement the branch, and the
snapshot cannot see one.

### Recheck at v0.19.1rc1 (`da421af`): the table was rewritten, the assert is gone

Do **not** carry the #2071 verdict to a later ref. At v0.19.1rc1:

- `_make_attention_mask` no longer exists (zero hits in the tree), and the dense
  `attention_v1.py build_for_graph_capture` now accepts `DecodeOnly`,
  `ChunkedPrefill` **and** `SpecDecoding` (`:335-352`), so the "state with no dense
  branch" is closed.
- State selection moved into `_build_attn_state` (`model_runner_v1.py:1171-1200`):
  `PrefillNoCache` / `DecodeOnly` (+ `SpecDecoding` only when
  `method == "mtp"`) / `SpecDecoding` when `np.all(num_valid_tokens == 1) and
  speculative_config` / `ChunkedPrefill` / `PrefillCacheHit`. `use_eagle` is now the
  upstream `speculative_config.use_eagle()` (`:354-355`), not a local flag.
- A **new** split appears at `:1195-1198`: for any `method != "mtp"` the returned
  local stays `SpecDecoding` (and `with_prefill` at `:644` reads that) while
  `self.attn_state` is stored as `ChunkedPrefill`, and `:2495` puts the *stored* value
  into the live attention metadata; `_dummy_run` pins `DecodeOnly` at `:2744-2750`.
  That is a suffix/ngram/eagle-only live-vs-capture disagreement, not the #2071 assert.
  At this ref it is **not** a fault mechanism to rank: the consumers treat both states
  alike (`mla_v1.py:1387-1392`, `sfa_v1.py:1106`). A device dump on such a launch is the
  capture/sampler family instead — `draft-graph-capture-device-fault.md`, section
  "The non-MTP variant". Unverified: whether `deepseek_mtp` on a dense model fails
identically (same branch, no run), and whether any ngram request without draft
tokens ever completed multi-turn on that release.

Worked case #2071: 13 assistant turns / 22 tool calls / 871 K prompt tokens,
answered correctly but with most of the calls re-reading `model_runner_v1.py`
around the state table; the grep in the table above and the skip marker would have
replaced that walk.
