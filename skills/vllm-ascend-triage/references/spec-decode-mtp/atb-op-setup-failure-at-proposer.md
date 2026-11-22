# `SelfAttentionOperation setup failed!` at a proposer frame

## Signature

An eagle/eagle3 serve starts clean and captures its graphs; the **first** request
kills every rank in the same second. Recorded shape `#4323` (vllm-ascend
v0.11.0rc1 `c5fe179c` + vllm v0.11.0 `b8b302cd`): Qwen3-32B target at
`-tp 4`, `--speculative-config '{"method":"eagle3","model":<AngelSlim/Qwen3-32B_eagle3>,
"num_speculative_tokens":4,"draft_tensor_parallel_size":1}'`, `--max-model-len 4096`,
`--max-num-batched-tokens 4096`; client gets `500`, title says 乱码/无法启动.

Immediately after the paste's **single** `INFO … [acl_graph.py:187] Replaying
aclgraph` line, each rank prints:

```text
[rank1]:[Ennnn hh:mm:ss.xxx] SelfAttentionOperation setup failed!
Exception raised from OperationSetup at build/third_party/op-plugin/op_plugin/CMakeFiles/op_plugin_atb.dir/compiler_depend.ts:203
frame #1: c10::detail::torchCheckFail(...)
frame #2: atb::OperationSetup(atb::VariantPack, atb::Operation*, atb::Context*)  (torch_npu/lib/libop_plugin_atb.so)
frame #3: <unknown>  (libop_plugin_atb.so)
frame #4-7: <unknown>  (libtorch_npu.so)
```

then the worker traceback, whose deepest `vllm_ascend` frame is the **proposer's
slicing line** (`spec_decode/eagle_proposer.py:476`,
`sample_hidden_states = last_hidden_states[last_token_indices]`), wrapped as
`RuntimeError: The Inner error is reported as above … the current working operator
name is SelfAttentionOperation` + `[ERROR] … ERR00100 PTA call acl api failed`,
followed by torch_npu's own note that the stacktrace may be inaccurate and that
`ASCEND_LAUNCH_BLOCKING=1` gives the real one. EngineCore then reports
`Worker failed with error …` (one per worker, all four at the same second).

The frames above that victim line are the whole story, and they are all
**drafter**-side (verified verbatim at `c5fe179c`): `worker_v1.py:276 execute_model`
→ `model_runner_v1.py:2144 execute_model` (`self._draft_token_ids =
self.propose_draft_token_ids(`) → `:1767 propose_draft_token_ids`
(`self.drafter.generate_token_ids(`) → `eagle_proposer.py:200 generate_token_ids`
(`self._propose(`) → `:476`. The target forward, the prefill branches and the
multi-step draft loop are in none of them. The reporter's own follow-up shrinks the
trigger to one short request — `{"model":"Qwen3-32B","messages":[…{"content":"Where
is New York?"}],"max_tokens":16,"temperature":0}` — so prompt length, prefix cache,
batch size and sampling are all already excluded; no control on them is worth a
reply, and the long-prompt sibling section below cannot apply.

Not this page when the paste says only **`RuntimeError: setup failed!`** with no
`<Op>` name and no `op_plugin_atb` frames: that is a call site in the deepest
`vllm_ascend` frame handing its own mask/size arguments to the op —
`../custom-ops-runtime-faults/setup-failed-call-site.md` (recorded #393, a V0
prefix-cache-hit prefill). Nor when the op is named `SelfAttentionOperation` but no
`--speculative-config` is in play and a dense model gets one long prompt: that is the
V1 one-shot prefill branch handing a `[prompt, prompt]` mask to
`torch_npu._npu_flash_attention` (`attention_v1.py:346`, #2239) — same page, its second
section, cause left open. The five surfaces and which one owns each are tabulated
there.

## Read it in three layers

1. **The Python frame is a join point.** `last_hidden_states[idx]` is an eager
   op that cannot be an ATB self-attention; the async-attribution caveat is in the
   paste itself, and history `#447` carries the same "current working operator name
   is SelfAttentionOperation" wording over a `torch_npu._npu_reshape_and_cache`
   frame. Do not answer "the slicing at :476 is the bug" — that is the reporter's
   own read in `#4323`, and it sends them to a host-side index tensor.
2. **The ATB block is this process's, not a replayed older record.** It is stamped
   at the incident time, so a *host-side op setup* did fail (contrast the
   replayed-block gate in `../hccl-and-hardware/error-report-timestamps.md`). A
   setup failure is a shape/dtype/attribute complaint — not device health, not a
   collective, not an OOM.
3. **Which op reaches ATB is not answerable from the snapshot.** `ATB`
   (ascend-transformer-boost) and op-plugin-atb's sources are not exported; one
   grep bounds the question instead of a walk:
   `rg -n 'atb\.|npu_fusion_attention|SelfAttention' vllm_ascend/`. At `c5fe179`
   the only model-path ATB entry is `torch_npu.npu_fusion_attention`
   (`attention/attention_v1.py:633-635`, gated on `AttentionType.ENCODER_ONLY`);
   the decode/spec path uses `npu_fused_infer_attention_score` (`:411`, `:539`).
   But `_register_atb_extensions()` runs at worker init
   (`worker/worker_v1.py:88`), so `op_plugin_atb` can replace ops vllm-ascend never
   names — write that as the open question plus its one check, not as a theory.

## Attribution recipe (the report's own advice does not work as written)

- `ASCEND_LAUNCH_BLOCKING=1` **alone is refused** while graph mode is on:
  `ValidationError: ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1. Please
  unset ASCEND_LAUNCH_BLOCKING or set it to 0.` (reporter-confirmed on `#4323`
  after the recording, and their plog under `~/ascend/log/debug/plog/` only repeats
  the same line). Same gate from the other side:
  `../startup-failures/graph-mode-not-applied.md`.
- So the order is `--enforce-eager` (or `cudagraph_mode: NONE`) **then**
  `ASCEND_LAUNCH_BLOCKING=1`. `eagle_proposer.py:46-48` gates the draft graph on
  `compilation_config.level == PIECEWISE and not enforce_eager`, so eager also
  removes the `Replaying aclgraph` step that immediately precedes the failure. If the
  fault **survives** eager, the ATB-calling op is on the eager path too and blocking —
  now permitted — is the only step that names it.
- If the error moves to a named op under eager+blocking, that op owns it. If it
  disappears, graph capture/replay of the draft is the rank-1 story.
- The single `Replaying aclgraph` line does **not** on its own prove the drafter
  (target and draft both replay through `acl_graph.py`) — the **frames** do, since
  every one of them is inside `propose_draft_token_ids`. Inside `_propose` the only
  work between the metadata build and `:476` is the mask/metadata build
  (`:448-450`), three buffer copies (`:464-466`) and the draft model forward
  `self.model(...)` under `set_ascend_forward_context` (`:471-475`) — so the launch
  window under the async attribution is the **draft forward**, and the open split is
  its captured-graph replay versus an eager op inside it. `use_cuda_graph` is
  `compilation_config.level == PIECEWISE and not enforce_eager` (`:46-48`), which is
  exactly what `--enforce-eager` flips — and why that flag is the rung that separates
  the two halves, while `num_speculative_tokens` is not (see the ladder).

## The reporter's own later analysis: grade it, do not adopt it

The reporter's follow-up comment on `#4323` is a 根因分析 report that fixes the
blame on `:476` and lists three candidates. It is a hypothesis table, and every row
is gradeable against their own paste plus the pin:

- **"the index needs `SelfAttentionOperation`" (the premise)** — refuted by the
  paste: the raising call is `atb::OperationSetup` inside `libop_plugin_atb.so`
  (frame #2), and a tensor index selects rows of a host-shaped tensor — it cannot
  invoke an ATB self-attention; torch_npu's own note three lines below says the
  stack is not the launching op. Do not answer this line as the bug.
- **"tensor shape / index / 4-card TP split"** — dead twice: an index overrun is an
  `IndexError` in Python, not an ATB host setup refusal, and
  `draft_tensor_parallel_size` moves nothing (the drafter runs at target TP, one
  grep below). A TP-sharding shape story would also have to explain four ranks
  failing in the same millisecond with an identical message.
- **"memory layout / transpose mismatch"** — nothing in the paste reports a layout
  or NZ cast, and the ATB refusal is a `torchCheckFail` from `OperationSetup`, i.e.
  attribute/shape validation before any data moves. No rung.
- **"ATB / CANN compatibility"** — the only surviving candidate, and it is the page's
  own open question (which op reaches ATB). Keep it open with its one check, the
  eager-then-blocking stack, instead of ranking it.
- **Their `error code 100000`** is not a code at all: the paste's own line is
  `[ERROR] … (PID:<pid>, Device:<n>, RankID:-1) ERR00100 PTA call acl api failed.`,
  and `ERR00100` lives in ATB/op-plugin, which no export carries. Nothing in this
  snapshot's `cann/` export defines `100000` either, so
  `scripts/cann-error-lookup.sh 100000` is a dead end — quote `ERR00100` from the
  paste, and label it ATB-side rather than an ACL `ACL_ERROR_RT_*` name.

## Eliminate cheaply (one grep each, both verified at this pin)

- **`draft_tensor_parallel_size` is a no-op.** `SpeculativeConfig` validates it and
  builds `draft_parallel_config` (`vllm/config/speculative.py:409-426`,
  `create_draft_parallel_config:493-516`; downstream mirror
  `patch/platform/patch_config.py:73,214-231`), but nothing under `vllm/v1/**`
  reads that object at v0.11.0 (`rg -n draft_parallel_config vllm/v1/` is empty),
  and `EagleProposer.load_model` builds the draft with the **target**
  `vllm_config` (`vllm/v1/spec_decode/eagle.py:809-824`; downstream
  `spec_decode/eagle_proposer.py:77-82`), so the drafter runs at target TP. A
  "draft TP vs target TP" mechanism has no code path here — say so instead of
  ranking it.
- **The validated matrix does not cover this combination.** The eagle/eagle3 e2e
  at rc1 is `tests/e2e/singlecard/spec_decode_v1/test_v1_spec_decode.py`:
  single card, Llama-3.1-8B drafts, `num_speculative_tokens` 2-3. TP>1 with a 32B
  eagle3 draft is unvalidated — an honest verdict, not "proposer bug".
- **The target-family gate is not it.** `eagle3_target_supported = ["llama",
  "qwen", "minicpm", "gpt_oss"]` (`vllm/config/speculative.py:540-547`) accepts
  `qwen3` and otherwise raises a `ValueError` at engine start, so a wrong target
  family cannot reach a request.

## Ladder

1. `--enforce-eager` (or `cudagraph_mode: NONE`) — the only one-variable launch that
   changes the launch under the window, because it clears `use_cuda_graph`
   (`:46-48`) and with it the draft graph replay. Read the result against the three
   predicted outcomes in *Attribution recipe*.
2. `--enforce-eager` **plus** `ASCEND_LAUNCH_BLOCKING=1` — permitted only after (1),
   and the only step that names the ATB-calling op if the fault survives eager.
3. `--speculative-config` removed entirely (this topic's row 1) — the control that
   the drafter is even required.
4. TP=1 with a smaller same-family target and the same draft config — separates
   TP sharding from the draft checkpoint.
5. The image pair: this report's own evidence is that the **same** config produced
   乱码 on v0.11.0rc0 and this crash on rc1. That flip is the strongest single
   discriminator the reporter can supply; ask for both image tags (only one tree is
   in the snapshot, so the rc0 side is unverified here).

**`num_speculative_tokens: 1` is not a rung for this shape**, even though it looks
like the obvious one: the early return is at `eagle_proposer.py:480-483`, i.e.
*after* the draft forward (`:471-475`) and the victim line (`:476`), so it cannot
remove the failing launch and it reproduces on step 1 either way — it splits the
multi-step draft loop only, and reading it as the answer costs a reporter round trip
(#4323's recorded answer led with it anyway).

## Delivery note

The recorded run had the draft-TP no-op, the single-card coverage and the `#447`
precedent by its middle turns and still shipped **no message text on any of its
26 turns**, hitting the 16 384-token cap inside a reasoning block that already held
the ranked causes. The ATB ownership hunt (a component no export contains) is
exactly the loop that never converges; see `../evidence-workspace/delivery-discipline.md`.
