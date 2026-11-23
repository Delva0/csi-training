# Layerwise PD: a *shorter* TTFT beside garbled output — the send-side wait

Worked case: #15402 — MiniMax-M2.5-w8a8, 40-60 K context (typical 50 K), vllm-ascend
v0.23.0rc1 (`f4a08bd`) / vllm `0.23.0+empty` / torch_npu 2.10.0.post2 / CANN 9.0.1,
PD with `MultiConnector` wrapping `MooncakeLayerwiseConnector`
(`use_layerwise: true`, `use_ascend_direct: true`, prefill dp1/tp8, decode dp1/tp8).
Symptom as filed: **偶现 TTFT 变短，并且输出乱码** — intermittently the TTFT drops
(~1.2-1.3 s against a ~2.7-3 s baseline) *and that answer is garbled*. Nothing
crashes; both engines serve on.

## Scope

This page owns one mechanism question: on a layerwise p2p transfer, **can a layer be
sent before its KV write is complete/visible**, so that the decode side consumes a
half-written layer. Its signature is the TTFT *moving the wrong way* (down) together
with corrupt tokens, and the reporter's own ablation usually shows that one wait
idiom is stable and another is not.

Not this page: the deployment never connects (`transport-topology.md`,
`mooncake-adxl-connect-103900.md`); the prefiller dies inside `start_load_kv`
(`layerwise-split-metadata-crash.md`); a hybrid model whose output alternates good/bad
with the *connector choice* as the variable (`hybrid-state-transfer.md`); a
version-to-version TTFT regression (`../performance-throughput/upgrade-vs-older-baseline.md`).

## Step 1 — read "shorter TTFT" as premature transfer, not speed

A layerwise transfer sends one small piece per layer, overlapping the prefill compute.
A layer submitted before its KV write is complete/visible buys two symptoms at once:
the prefiller frees and reports earlier (TTFT falls), and the decode node receives a
partial layer (every later token is wrong). So this is a **correctness** report, and
"TTFT got better" must be said out loud to be wrong — it is the same bug.

The discriminating check needs no code change: pair each request's latency with its
output, `(req_id, TTFT, prompt_tokens, first 16 output tokens)`. #15402's report
already had the two classes (short + garbled vs slow + correct). If the short class is
exactly the garbled class, this page is the answer; if garbled responses also appear in
the slow class, the wait is not the whole story — demote it and read Step 4.

## Step 2 — the branch: three wait idioms inside one block

Verified at v0.23.0rc1 (`f4a08bd`),
`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`,
in `_transfer_kv_cache`:

```python
if send_task.k_quant_cache is not None:            # :482  quant / c8 layers
    self.resharding_stream.synchronize()
elif self.pd_head_ratio == 1:                      # :483
    """Note: Due to a bug in ADXL, calling current_event.synchronize() may
    occasionally hang. This issue will be fixed in CANN version 8.5.rc1. ..."""   # :484-489
    send_task.wait_event.synchronize()             # :490  host-side wait only
elif self.pd_head_ratio > 1:
    self.resharding_stream.synchronize()           # :492
for session_id, transfer_meta in session_meta.items():
    ... self.engine.batch_transfer_sync_write(session_id, src, dst, length)       # :497
```

- The event is created and **recorded on the compute stream** at `:1720`
  (`reshape_cache_event = torch.npu.Event(); reshape_cache_event.record()`) and handed
  to the send task at `:1810-1811` (`SendTask(wait_event=reshape_cache_event, …)`);
  `SendTask.wait_event` is declared at `:129` under the comment `# pd_head_ratio == 1 use`
  (`:128`).
- In the two `resharding_stream` branches the same event was already installed as a
  **device-side dependency** on that stream before the buffers are read:
  `save_kv_layer` does `with npu_stream_switch(self.resharding_stream):
  reshape_cache_event.wait()` (`:1736-1737`). The `pd_head_ratio == 1` branch switches
  no stream and inserts no dependency — the only ordering is a host-side
  `Event.synchronize()` executed in the **send thread**
  (`KVCacheSendingLayerThread`, class `:204`, `run()` `:266-274`, `torch.npu.set_device` `:269`).
- `batch_transfer_sync_write` takes no stream argument (`:497`): the C++ transfer engine
  does not inherit the torch current stream, so nothing else orders the copy behind the
  KV write.

**Which branch a given deployment takes** is `ascend_config.pd_head_ratio`
(`vllm_ascend/ascend_config.py:211`), not a per-call derivation: it becomes
`prefill_tp_size // decode_tp_size` only when `kv_transfer_config` is set **and** the
model is **not** `is_deepseek_mla` and `pd_tp_ratio > 1` (`:214-234`). Consequences, all
source-verified: P TP == D TP leaves it at `1`; any MLA checkpoint skips the whole block
and also stays at `1`. So the host-event branch is the *ordinary* path, not an exotic
config — and the "Layerwise" recipe for DeepSeek-V3.1 in the docs
(`docs/source/tutorials/DeepSeek-V3.1.md:260`) runs on it too. Print it rather than
infer it from the TP sizes:
`python -c "from vllm_ascend.ascend_config import get_ascend_config; print(get_ascend_config().pd_head_ratio)"`.
The send thread's only error surface is `Failed to transfer KV cache. layer_idx=…`
(`:275-283`) — no such line in P's log means the sends "succeeded", which is exactly
what a visibility race looks like.

Also worth one line before theory: the outer `use_layerwise` key in
`kv_connector_extra_config` is read only by the KV-pool connector
(`kv_pool/ascend_store/ascend_store_connector.py:86`); on the Mooncake p2p path the
behaviour comes from the chosen class, so "the flag was ignored" is not a lead here.

## Step 3 — the checks, cheapest first

1. The `(TTFT, output)` pairing of Step 1, on data the reporter already has.
2. **A/B the wait idiom on their own build** — same image, same weights, same quant,
   same prompts, one line changed at `:490`, stream created once per send thread (e.g.
   in `KVCacheSendingLayerThread.run()` next to `torch.npu.set_device`):

   ```python
   self.wait_stream = torch.npu.Stream()          # once, in run()
   self.wait_stream.wait_event(send_task.wait_event)
   self.wait_stream.synchronize()                 # replaces :490
   ```

   That is the form the two neighbouring branches already use (`:1736-1737` + `:492`),
   which is why it is the fix shape rather than a guess.
3. **Do not use "turn KV quantization on" as the discriminator.** It does move the run
   to `:482`, but it also changes the KV dtype, the registered buffers and the copy path
   (`enable_kv_quant` / `enable_c8_quant` at `:1184-1193`, `create_kv_buffer`
   `:1219-1228`), so a vanishing symptom would not isolate the wait — it is confounded
   with the numerics change.
4. `ASCEND_LAUNCH_BLOCKING=1`: ordering vs visibility. If it disappears, the ordering
   is the suspect; if it persists, the wait is not the whole story.
5. Then the plog of one failing request (per-layer transfer start/end, event
   record/complete), and only then a HIXL/CANN question — that layer is not answerable
   from the export.
6. **Acceptance is token-for-token**, not "no exception": one 50 K request's full output
   against a pure-P/direct serve of the same body.

## Step 4 — what the pin excludes, and the second hypothesis

- The `:484-489` comment is about a **hang** in `current_event.synchronize()` (claimed
  fixed in CANN 8.5.rc1), i.e. a different failure mode from an early return, and
  #15402's environment was CANN 9.0.1 — quote the comment as "this path has a
  host-event history", never as the explanation.
- **Coverage, not cause:** at this pin the MiniMax-M2 tutorial covers M2.5/M2.7 and has
  a PD section (`docs/source/tutorials/models/MiniMax-M2.md:395,459`) whose connector is
  `MooncakeConnectorV1` with `use_ascend_direct`, `enable_cpu_binding` and eagle3
  (`num_speculative_tokens` 1 on P, 3 on D); `rg -c -i layerwise` on that file returns
  0. The `MultiConnector` + `use_layerwise` combination is therefore not a documented
  recipe at v0.23.0rc1 — say that, and say it does not make it unsupported.
- **Hybrid lane if the wait fix does not end it:** M2.5 carries a linear-attention patch
  (`vllm_ascend/patch/worker/patch_minimax_m2_linear_attn.py`), and the layerwise
  connector moves at most one block for a `MambaSpec` layer, branching on
  `mamba_cache_mode == "align"` (`:297-299`); the connector/mode questions are
  `hybrid-state-transfer.md`. #15402's own A/B (only the wait changed → stable) ranks
  this below the wait, not equal to it.

## Step 5 — the delivery

First paragraph: "TTFT 变短不是变快" — one cause, two symptoms — plus the branch
(`:483-490`) and whether the reporter's TP pair selects it. Then the ranked list with
one check each, the ladder above, and the scope: verified at v0.23.0rc1 `f4a08bd`;
line numbers move between releases, so give the recheck
`sed -n '478,496p' vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`
and ask for the two serve commands plus `pd_head_ratio` / `enable_kv_quant` /
`enable_c8_quant` printed from their build. Version rung: closed — the reporter later
confirmed publicly that both sides ran the current `0.23.0+empty` / `0.23.0rc1` pair, so
do not spend a rung on "upgrade the image".
