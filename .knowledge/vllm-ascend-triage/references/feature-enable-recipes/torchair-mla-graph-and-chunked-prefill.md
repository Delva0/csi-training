# Torchair graph mode on a DeepSeek/MLA model: `UnboundLocalError: 'decode_hs_or_q_c'` and the chunked-prefill contract

## Scope

An offline `LLM()` or `vllm serve` on a DeepSeek-family MLA checkpoint with
`additional_config={"torchair_graph_config": {"enabled": true}}` passes weight load and
graph capture (`Capturing torchair graph …` then `Graph capturing finished in N secs`),
and the **first request** kills every worker:

```text
… vllm_ascend/models/deepseek_v2.py:489 in forward … self.mla_attn.impl.forward(self.mla_attn,
… vllm_ascend/attention/mla_v1.py:1099 in forward … self._q_proj_and_k_up_proj(decode_hs_or_q_c)
UnboundLocalError: local variable 'decode_hs_or_q_c' referenced before assignment
```

Read this page for that signature, for its follow-ups (`decode_k_pe`, and
`TypeError: 'NoneType' object is not subscriptable` at `k_pe = k_pe[:num_actual_toks, ...]`
after a hand patch), and for "how do I turn torchair graph mode on for DeepSeek at all".
Not this page when the **capture** dies (`../startup-failures/torchair-dynamo-compile-unsupported.md`)
or when a graph mode that printed its banner and captured cleanly then hangs the first
request with no Python frame (`../hccl-and-hardware/aclgraph-full-decode-runtime-hang.md`).

## Verdict first (the G1 sentence)

At v0.9.1rc1 torchair graph mode does not work with chunked prefill, and the pinned tree
says it twice — in the doc (`docs/source/user_guide/graph_mode.md:50`
`# TorchAirGraph is only work without chunked-prefill now`, with the DeepSeek example at
`:51` setting **both** keys: `additional_config={"torchair_graph_config": {"enabled": True},"ascend_scheduler_config": {"enabled": True,}}`,
online form `:58`) and in CI (`tests/e2e/multicard/test_torchair_graph_mode.py:46`
`# torchair is only work without chunked-prefill now`, whose `VllmRunner` sets both keys
plus `refresh: True` at `:52-60`, TP4, and the same four prompts as the report at `:38-43`).

`ascend_scheduler_config.enabled` is the half that disables chunked prefill:
`AscendSchedulerConfig.enable_chunked_prefill = False` and
`initialize_from_config` forcing `scheduler_config["enable_chunked_prefill"] = False`
(`vllm_ascend/core/schedule_config.py:26,43`), applied from
`vllm_ascend/platform.py:215-221` before the scheduler is built. So the deliverable is a
config contract plus one re-run — not a batch-mechanism investigation.

## Why the traceback lands in `mla_v1`

`AscendMLAImpl.forward` (vllm-ascend v0.9.1rc1 `c30ddb8`) binds the decode slice on two
branches only:

- `:1040-1042` `self.running_in_graph = self.torchair_graph_enabled and attn_metadata.attn_state in [DecodeOnly, SpecDecoding]`;
- `:1065-1075` under `if not self.running_in_graph:` … `if not self.torchair_graph_enabled:` binds `decode_hs_or_q_c` (`:1069`), otherwise the `else:` binds it from the whole tensor (`:1075`);
- `:1099` consumes it under `if has_decode:`.

A step with `num_decodes > 0` whose `attn_state` is `ChunkedPrefill`/`PrefillCacheHit`
therefore has `running_in_graph == False` **and** `torchair_graph_enabled == True`, takes
neither binder, and dies at `:1099`. `attn_state` is chosen in
`worker/model_runner_v1.py:1012-1027`; `has_decode`/`has_prefill` come from the builder's
per-request `num_scheduled_tokens` classification (`mla_v1.py:229-264`).

Nothing rejects the pair at startup: `check_ascend_config` gates only eager × graph
(`ascend_config.py:131-136`, `Can't enable graph mode and eager mode at the same time`)
and a non-DeepSeek model under torchair (`:143-147`). The crash *is* the enforcement,
which is why such a report carries no config error to read.

**Do not try to derive which scheduler step mixed decode with prefill.** The paste carries
no per-step `num_decodes`/`num_prefills`, so `max_num_batched_tokens` × prompt-size
arithmetic cannot settle it, and the answer does not need it. Recorded #1369 spent a
19 K-character block on exactly that (`np.all(num_scheduled_tokens == 1)`,
`min(max_num_seqs * max_model_len, …)`) plus 46 messages and shipped no answer, while the
two doc/test comments above were already in hand at turn 10.

## The "I patched the line and got another error" follow-up

Patching one binder moves the crash one variable down, because the same
`not torchair_graph_enabled` guard owns the **KV projection** side too: `kv_c_normed`
(`:1044-1051`), `k_pe`/`decode_k_pe`/`prefill_k_pe` (`:1070-1073`), and `decode_k_pe` is
*consumed* at `:1112` in the eager `rotary_emb` branch. In torchair mode the caller
supplies none of them: `deepseek_v2.py:480-492` passes `k_pe = None` and the **raw**
`hidden_states` as the `hidden_states_or_kv_c_normed` argument, because the graph path
projects KV inside the attention (`exec_kv` `:857`, `exec_kv_prefill` `:889`, called at
`:1136`). Releasing the guard without that projection raises
`TypeError: 'NoneType' object is not subscriptable` at the `k_pe` slice. So the guard must
be keyed on `running_in_graph` (a torchair-enabled eager step has to behave exactly like a
non-torchair run, KV projection included), never on `torchair_graph_enabled` alone.
Reporter-stated outcome of #1369: fixed in PR #1378 — not in the snapshot, so it is
quotable as the reporter's claim only, not as source.

## "I already enabled it and it still fails" — check the key, not the mechanism

`additional_config` is read by exact literal key and unknown keys are **silently
ignored**: `AscendConfig.__init__` does `additional_config.get("ascend_scheduler_config", {})`
(`vllm_ascend/ascend_config.py:34-37`) with no validation, and the small
`AscendSchedulerConfig.__init__` merely `setattr`s what it finds (`:80-87`). #1369's own
script carried the key as `"asacend_scheduler_config"`, which yields `enabled=False` and a
byte-identical rerun, with no warning. Two consequences:

- grep the reporter's own snippet for the literal string before accepting "the fix does
  not work"; ask for the launch line, not a description of it.
- `init_ascend_config` caches one global and returns it unless `refresh` is set
  (`:92-101`) — that is why the CI test passes `refresh: True`. A notebook/REPL run that
  flips the key and rebuilds `LLM` in the same process keeps the old value.

The cheap observable that the key *did* take effect is the state name: with the scheduler
enabled, a prefill-carrying batch no longer takes the `ChunkedPrefill` arm at
`model_runner_v1.py:1024` (`PrefillCacheHit` at `:1027` instead), and decode steps stay
`DecodeOnly`. A follow-up that still reports `ChunkedPrefill` proves the key never landed.

The maintainer's later position on #1369 is this contract verbatim ("torchair graph mode is
not work with chunked prefill currently, plz refer to this doc to enable `AscendScheduler`,
which disables chunked prefill by default … we will support this after the required CANN
and pta is available"), and the reporter's next message after being pointed at
`mla_v1.py#L1075` was "I was careless".

## Ladder

1. Add `"ascend_scheduler_config": {"enabled": true}` beside `torchair_graph_config`, with
   that exact spelling (plus `"refresh": true` if the process built an engine before), and
   re-run the same script. Expect no `UnboundLocalError`.
2. If a serving need forbids turning chunked prefill off, `enforce_eager=True` — or
   removing the `torchair_graph_config` key, since `check_ascend_config` refuses both at
   once — is the **working control, not the fix**: it drops graph mode.
3. Only if it still fails with the scheduler demonstrably on: a graph-enabled prefill or
   mixed step is reaching a torchair-eager branch, i.e. the code defect above. The artefact
   to request is the failing step's `attn_state` + `num_decodes` + `num_prefills` (one
   DEBUG line in `_process_reqs`) and `pip show vllm vllm-ascend` from inside the image.

Pins: vllm-ascend v0.9.1rc1 `c30ddb8`, vllm v0.9.1 `b6553be`. The pair was scheduled to
gain support once the CANN/PTA pair allowed it, so at another ref re-grep the two
`only work without chunked-prefill` comments and the `decode_hs_or_q_c` block before
repeating the verdict — a later release may carry a real fix, and then this page becomes
version history rather than a workaround.
