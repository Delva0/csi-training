# A ❌ row in the pinned support matrix is the verdict — and the reporter will send more errors

## When this page applies

"离线推理 <model> 报错", "<model> 在 <ver> 上跑不通", "通用环境代码可以跑通 qwen3 模型"
with a device-side paste (an `ERR33133 PTA call acl api failed` / a
`PagedAttentionOperation` / `EZ9999` block) and only screenshots as evidence. The
reporter is asking *can this model run here at all*; the answer is a release-scoped
capability verdict, not an operator-level root cause.

## Step 1 — one grep of the release's own claim table

```bash
rg -n -i '<family-token>' docs/source/user_guide/supported_models.md \
                          docs/source/user_guide/support_matrix 2>/dev/null
```

Run it against the **directory**, not a guessed filename: the layout moved. At
v0.9.2rc1 (recorded #6257) the matrix is
`docs/source/user_guide/support_matrix/supported_models.md` (+ `supported_features.md`
beside it); older refs use a flat `docs/source/user_guide/supported_models.md`. A
grep for the wrong path returns **nothing**, which reads exactly like "no row" — the
recorded run spent four turns on head counts and attention branches after that empty
result, and found the `Gemma-3 | ❌` row (Note column: tracking issue `#496`) at
`:26` only when it grepped `docs/source` instead.

Three row states, three different answers:

- `✅` — the model is in the release's claimed set; keep investigating the op error.
- `| Need test |` — a **coverage gap**, never a refusal: say "not validated here",
  and quote the precedent that did run.
- `❌` + a tracking issue — the verdict. The reported device error is a
  *manifestation* of an unvalidated path; do not spend the session deriving which
  kernel shape it came from beyond naming the branch (see step 3).

## Step 2 — the control the reporter already gave you

"同一套环境能跑通 qwen3" plus a ❌ row pointing at a ✅ neighbour (`Qwen3 | ✅` at
`:12` at the same pin) is the whole attribution: the install, the driver and the
launch script are exonerated, and the model is the variable. Say that in the first
paragraph, before any log reading.

## Step 3 — name the branch, then stop

One grep names the op that faulted and its owner: the paste's
`current working operator name is PagedAttentionOperation` with a
`super().replay()` frame is `torch_npu._npu_paged_attention`, reached from the
decode branch `vllm_ascend/attention/attention_v1.py:379` (DecodeOnly; the mixed
path is `:420` `_npu_paged_attention_splitfuse`), and it lives in `torch_npu`/ATB
(`op-plugin`'s `PagedAttentionAtb`), not in `vllm_ascend/` — so a source walk in
the plugin repo has nothing to read. `rg -ni '<family>' vllm_ascend/` returning
nothing (Gemma: nothing at v0.9.2rc1) is the *coverage* statement: the family runs
the generic upstream model file with no downstream branch, which is why an
unvalidated shape reaches an op nobody exercised.

## Step 4 — the one arithmetic that can still change the verdict

A `tensor_parallel_size = torch_npu.npu.device_count()` launch (screenshot code,
`NPROC_PER_NODE=8` in the env block) against a small-head model is worth one check,
because it decides whether the failure is even reachable: upstream
`vllm/model_executor/models/gemma3.py:112` is
`assert self.total_num_heads % tp_size == 0` with the KV rule at `:115-123`
(replicate when `total_num_kv_heads < tp_size`, `num_kv_heads = max(1, total // tp)`).
So a *construction-time* assert bounds nothing, but a fault seen inside
`llm.generate(...)` proves TP divides the head count — which means the reporter's
`device_count()` is not what the paste suggests. Ask for `device_count()` and the
model's own `num_attention_heads` / `num_key_value_heads` / `head_dim`; do **not**
recall them (label any recalled config as unverified, as the recorded answer did).

## Step 5 — expect a second batch of *different* errors, and pre-empt it

An unvalidated model path fails repeatedly in unconnected places. Recorded #6257:
the ❌-row answer (with TP=1 / `enforce_eager` ablations) was followed by the
original reporter posting "其他错误：" with **four more** failure screenshots. The
deliverable must therefore say up front: *each workaround you try will surface a
different error; that is what an unsupported row means* — ask for all of them in
one thread, point at the tracking issue, and name the one check that could still
move the verdict (step 4). Do not offer a ladder that reads as "then it will work".

## Step 6 — say what a newer version would change, and what you cannot check

The snapshot usually carries one vllm-ascend pin, so "fixed in a later rc" cannot be
confirmed locally: say that, and ask the reporter to run
`rg -n -i '<family>' docs/source/user_guide/support_matrix/supported_models.md` (or
the older flat path) on their own newer tree. Precedents are quoted per issue, each
with its version and its failure point: `#496: gemma-3-4b-it 输出乱码`;
`#1183: gemma3-27b on 0.9.0rc1, V1 error`; `#359: [New Model] request`.
