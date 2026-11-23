# DeepSeek-V2-Lite (MLA, `q_lora_rank: null`) answering a tiny prompt as one repeated token

Scope: a DeepSeek-V2-Lite serve whose completion is **one token repeated** to
`finish_reason: length` — recorded #894: `"prompt": "Alice is "` (4 prompt tokens),
`temperature: 0`, `max_tokens: 50` → `"...............................…"`, no
traceback, engine healthy, TP4 on one 910B1 node. Not the few-thousand-token profile
of `short-prompt-corruption.md`, not the VL vision-tower family
(`vl-vision-tower-degeneracy.md`), not first-call-only state
(`first-generate-degenerate.md`): every request is wrong, and nothing crashes. The
same symptom on a Qwen3-Next (GDN) checkpoint is `qwen3-next-gdn-numeric-path.md`.

## What the recorded case settled, and what it did not

Reporter-confirmed, 2025-06-03 on #894: moving **both** repos to paired main —
vllm `f32fcd944430603ebcbbf04454b4e15754168ef4` and vllm-ascend
`f24375f318536ad18bce0d3f44758c54af3b29ea`, which **includes #905**, the PR a
maintainer had offered as the fix — left the output wrong ("accuracy is still not
correct"); on 2025-06-05 the issue was still open ("solve this problem").

- **The stack-pairing / nightly-vs-image hypothesis is dead for this shape.** The
  reporter ran main-of-both inside an `v0.8.5rc1-openeuler` image, and pairing them
  did not change the result. Do not spend the session on `pip show`, image tags or
  "which vllm is vllm-ascend built against": the defect survives in the code at that
  date or **below** it (torch_npu / CANN kernels / the checkpoint).
- **The mechanism stayed open** in the visible window. Say that, and deliver the
  matrix that localises it — do not name an MLA or MoE defect as a cause.
- Delivery: the recorded replay of this report spent its whole 16 K output budget on
  `models/deepseek_v2.py`, `ops/fused_moe.py` and `platform.py` (35 tool calls,
  ~43 K characters of reasoning) and shipped 69 characters of message text — one
  "I'll start…" line — before the cap, i.e. no answer. A report with no serve
  command, no log and no `collect_env` body is answered by ≤4 ranked leads plus the
  rows below; the model-source reads decide nothing on their own.

## Checks, in this order

0. **Which checkpoint, which prompt path.** The validated documentation row is
   `DeepSeek-V2-Lite-Chat` (`docs/source/tutorials/multi_node.md:143-151`, `:159-164`),
   while the reporter's model directory is `DeepSeek-V2-Lite` (base). A base
   checkpoint driven through `/v1/completions` with a 4-token prompt is not an
   accuracy test, and the project's own sample output on the documented path is
   already malformed English on a 6-token prompt (`multi_node.md:187`:
   "heavily influenced by neuroscience and cognitiveGuionistes…"). So the *prose
   quality* is weak evidence in this model family — the **repetition of a single
   token** is the signal. One request against `-Chat` with a chat-shaped prompt, same
   deployment, is the cheapest discriminator and costs a restart, not a source read.
1. **TP=1 vs the reported TP.** V2-Lite bf16 is ~31 GB, so it fits one 64 GB card:
   a one-card run is a real row here (unlike V3/R1). Clean at TP=1 ⇒ the sharded MLA
   latent / proposed-expert path owns it; still repeated ⇒ checkpoint or engine-wide.
2. **The same directory on CPU** through `transformers`, same prompt,
   `do_sample=False` — the checkpoint and tokenizer hypothesis, one command, no NPU.
3. **Below vllm-ascend.** The project has already fixed a *repeated-symbol* output
   from the other end of the stack: v0.7.3rc2 "Bump torch_npu version to
   dev20250320.3 to improve accuracy to fix `!!!` output problem"
   (`docs/source/user_guide/release_notes.md:81`, #406), with #374 as the report
   (same model fine under vllm, `!!!` indefinitely under vllm-ascend). Ask for
   `torch_npu.__version__`, the CANN package version and the exact image tag before
   concluding the engine is clean.
4. **`--enforce-eager` is not a lever at this pin.** `vllm_ascend/platform.py:124`
   forces `enforce_eager = True` and `:133` logs "Compilation disabled, using eager
   mode by default", so ACL-graph capture cannot own the symptom at v0.8.5rc1. Cite
   the line and skip the row instead of asking for it.

## The two V2-Lite-only surfaces (one grep each, if rows 1-3 point below vllm)

V2-Lite is the model the tutorial serves but has no support-matrix row
(`docs/source/user_guide/supported_models.md:5-6` list only DeepSeek v3/R1/Distill,
`:34` says "DeepSeek v2.5 | | Need test"), so "unsupported" is not an answer — but it
is true that V3/R1 are the *validated* DeepSeek models, and every branch they never
take is a place a V2-Lite-only defect can live:

- `vllm_ascend/models/deepseek_v2.py:296-312` and `:375` — `q_lora_rank is None`
  (V2-Lite) selects `q_proj`, where V3/R1 build `q_a_proj`/`q_b_proj`. The attention
  backends store `self.q_lora_rank` (`attention.py:879`, `mla_v1.py:303`) and never
  branch on it, so the divergence is confined to the model file: one read, not three.
- `vllm_ascend/ops/fused_moe.py:467-468` — the AscendC gate
  `if global_num_experts == 256:` sends V3/R1 to `torch_npu.npu_moe_gating_top_k`
  and V2-Lite's 64 experts down the fallback `select_experts`. The validated models
  are the ones on the device kernel; V2-Lite's routing is the untested path.

Name these as two greps on the suspect list, each with the row that would implicate
it — not as causes. `native_grouped_topk` with `n_group=1`/`topk_group=1` is
arithmetically the plain `topk` (the masking selects the single group, then the
top-k is unchanged), so the routing branch refutes itself on inspection; leave it
labelled.

## Precedents (one line each)

- `#894`: V2-Lite TP4 910B1, 4-token prompt → 50 × `.`; a paired-main bump including
  #905 did not fix it (reporter-confirmed).
- `#374`: `!!!` indefinitely under vllm-ascend, the same model fine under vllm.
- `#406`: repeated-symbol output fixed by a torch_npu dev bump (release note above).

## Open

Whether this case ended in a torch_npu/CANN move, a checkpoint problem, or a later
MLA/MoE fix: the visible record stops at "still not correct". Say so, and ask for the
per-rank log, `torch_npu`/CANN versions, the checkpoint id (base vs `-Chat`) and the
TP=1 result.
