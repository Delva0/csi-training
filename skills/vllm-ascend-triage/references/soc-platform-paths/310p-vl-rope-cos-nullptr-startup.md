# 310P + VL checkpoint: engine start dies in `npu_apply_rotary_pos_emb` with `Check cos != nullptr failed`

Verified at vllm-ascend **v0.23.0rc1** (`f4a08bdd`) + vllm **v0.23.0** (`0fc695fc`)
while triaging #13522 (MiniCPM-V-4.6, `Ascend310P3`, CANN 9.1.0-beta.1, torch_npu
2.10.0.post2, `--enforce-eager --dtype float16 --max-model-len 2048`). Line numbers
move between releases — the shape is two `_310p` rope lines plus **one gate** in
`vllm_ascend/ops/rotary_embedding.py`; re-grep, do not re-derive.

## When this page applies

The paste is `EngineCore failed to start`, and its frames run
`determine_available_memory` → `profile_run` → `_dummy_run` → `_model_forward` →
model forward → `self.rotary_emb(positions, q, k)` → `custom_op.py forward` →
`vllm_ascend/_310p/ops/rotary_embedding.py:NNNN in forward_oot` →
`:NNNN in _rope_forward_oot` → `torch_npu.npu_apply_rotary_pos_emb(q_rot, k_rot,
cos, sin, rotary_mode=…)`, ending:

```
RuntimeError: npu_apply_rotary_pos_emb:../third_party/op-plugin/op_plugin/ops/opapi/ApplyRotaryPosEmbNpuOpApi.cpp:58
NPU function error: call aclnnApplyRotaryPosEmbV2 failed, error code is 161001
[ERROR] … ERR00100 PTA call acl api failed.
AclNN_Parameter_Error(EZ1001): Check cos != nullptr failed
```

Weights loaded, no quant/dtype refusal, and the model is **VL/omni** (a text-only
model on the same box serves). `161001` is the op-plugin error code and
`EZ1001 / Check cos != nullptr` is aclnn's parameter check — the tensor handed in
was a Python `None`, so nothing in the checkpoint, the vision tower or CANN is
the owner.

Not this page: the same checkpoint **serves** and answers garbage
(`310p-vl-ocr-garbage-output.md`); `EZ3002 … TransData … not found in this op
store` / `EZ9999 … Parse dynamic kernel config fail`
(`atlas-300i-310p.md`, `kernel-package-vs-card.md`); `aclnn<Op> … not in
libopapi.so` (`../custom-ops-runtime-faults/opapi-symbol-absent.md`); the MRoPE
slices' own `RuntimeError: MRoPE cos/sin slices are not initialized` (a different
class, `:203` at this pin).

## The answer, from the paste alone (one message, no source walk)

Five lines, all at the pin:

1. `utils.py:781,800` — `is_310p()` swaps `"RotaryEmbedding"` for
   `AscendRotaryEmbedding310`; its `forward_oot` is a one-line forwarder
   (`_310p/ops/rotary_embedding.py:262-273`), so the paste's class is the plain
   (non-MRoPE) rope.
2. `_rope_forward_oot:139` — `cos, sin = get_cos_and_sin_slice()`, which returns
   the module globals `_cos_slice` / `_sin_slice`
   (`ops/rotary_embedding.py:149-150`). They are written **only** by
   `update_cos_sin` (`:129-146`).
3. `update_cos_sin:135-136` — `if _cos_sin_cache is None or _cos is None or
   _sin is None: return`, i.e. it never writes them while `_cos` is unset.
4. `set_cos_and_sin:78` — `elif not is_vl_model(vllm_config) and
   has_rope(vllm_config):` allocates `_cos`/`_sin` **only for non-VL models**. A
   multimodal config therefore never allocates them, the globals stay `None`
   forever, and both branches that consume them hand aclnn `cos=None`:
   `:146` (`head_size == 128` and `cos_sin_cache.shape[-1] == 128`) and
   `:158` (`self.rotary_dim == 64`, the frame in #13522).
5. Owner: the 310P rope override reads module globals the VL gate never fills.
   The structural repair is ladder item 4, and no serve flag reaches the gate
   (ladder item 3).

The env dump's `torch` / `torch_npu` lines are not a lane here — the crash is
the **first** forward (`profile_run`), so nothing about the request path, the
sampler or the vision tower is in scope; if the pins do look wrong on their box,
that is `../image-dependency-faults/pip-index-visibility.md`, answered separately.

## Why the runner's `update_cos_sin` calls do not save it

Both exist and both are innocent: `worker/model_runner_v1.py:2215`
(`execute_model`) and `:3523` (the parent `_dummy_run`, comment `# update
global cos, sin`), while 310P's `_model_forward` only prepares the **MRoPE**
slices (`_310p/model_runner_310p.py:657-668`, gated on `uses_mrope`). Every one
of them ends in the same early return at step 3.

**The trap for the reader:** inside `_rope_forward_oot:136-137` the only
`update_cos_sin(positions)` sits behind `if getattr(self,
"_is_drafting_update_enabled", False)`, a flag the spec-decode proposer sets
(`_310p/spec_decode/llm_base_proposer_310.py:43`). Stopping there produces the
wrong theory ("the globals are only populated when drafting, so this is a
spec-decode path") and turns the answer into a code walk. The decisive line is
`:78`'s `is_vl_model` gate, and it is one grep in **their** tree:
`rg -n 'not is_vl_model' vllm_ascend/ops/rotary_embedding.py`.

## Which branch: the paste's line, not the model family

- `:146` — full rotary with `head_size == 128` (cache carries 128 values);
- `:158` — partial rotary with `rotary_dim == 64` (#13522's frame);
- `:162-169` — `torch_npu._npu_rotary_embedding(positions, …, self.cos_sin_cache)`,
  which needs **no** globals and cannot produce this error. So a paste carrying
  this traceback is in branch (a)/(b) **by construction** — no probe needed.

The branch is set by the checkpoint's `rope_parameters`, not by the family:
upstream builds rope with `get_rope(head_size=self.head_dim,
rope_parameters=config.rope_parameters)` (`vllm/model_executor/models/qwen3_next.py:260-265`),
and `scripts/310p-rope-branch.py <model dir | config.json>` derives
head_size/rotary_dim and prints the predicted branch. #13522's text tower is
qwen3_next (`vllm/model_executor/models/minicpmv4_6.py:917` remaps
`model.language_model.` → `language_model.model.`, `:1223` forwards through it);
its checkpoint is not in the snapshot, and the paste already fixes
`rotary_dim == 64` with `rotary_dim < head_size` — ask the script (or their
`config.json`) only for the `head_size` × `partial_rotary_factor` split, and do
not infer it from the family.

## The same gate's other outcome — and how it reads

#11936 (v0.21.0rc1, 310P, GLM-OCR / MinerU2.5) **launched and inferred** garbled
text on the same kind of VL checkpoint, i.e. there `cos=None` did not reach an
aclnn nullptr check (or the model sat in branch (c)). So the two recorded
reports bracket both outcomes of one gate, and **the paste's symptom is the
discriminator**: a `profile_run` traceback with `Check cos != nullptr` is this
page; a serve that answers wrong text is
`310p-vl-ocr-garbage-output.md`. Do not spend turns deciding between them from
source — only one of the two shapes carries a traceback.

## Ladder for the reporter (ranked, cheapest first)

1. **The pair is not a documented combination — say that first.** At this pin
   `docs/source/user_guide/support_matrix/supported_models.md:106` is
   `| MiniCPM-V | 🔵 | | A2/A3 |` (no 310P column entry), and the 310P page's
   own weights table is Qwen3 Dense `W8A8SC` 8B/14B/32B only
   (`docs/source/tutorials/hardwares/310p.md:203-220`), with `:524` naming
   Qwen3-VL as early access behind #7394. So: run the checkpoint on A2/A3, and
   file the crash for the 310P lane — that is the no-code rung.
2. **Confirm the nullptr is theirs** (one ~2 s call, bounds the claim to their
   CANN/torch_npu pair rather than to the source chain):
   `python -c "import torch, torch_npu; q = torch.zeros(1,4,2,64, dtype=torch.float16, device='npu:0'); torch_npu.npu_apply_rotary_pos_emb(q, q, None, None, rotary_mode='half')"`
   must print the same `161001` + `Check cos != nullptr failed`.
3. **No serve flag reaches it.** The branch comes from `head_size`/`rotary_dim`,
   the globals from `is_vl_model`; `--enforce-eager`, `--dtype float16`,
   `--max-model-len`, `--limit-mm-per-prompt` are already in the paste, and
   `compilation_config.custom_ops` cannot move a class override.
4. **Repair (one file), then validate numerically.** Derive the slice locally in
   `_rope_forward_oot` from `self.cos_sin_cache` — the same arithmetic
   `update_cos_sin:139-146` performs — instead of reading the globals:

   ```python
   n = positions.size(0)
   cos_sin = self.cos_sin_cache.index_select(0, positions)      # (n, rotary_dim)
   cos, sin = cos_sin.view(n, 2, -1).repeat(1, 1, 2).chunk(2, dim=-2)
   cos = cos.reshape(1, n, 1, -1)
   sin = sin.reshape(1, n, 1, -1)
   ```

   **Do not** "fix" it by deleting `not is_vl_model(vllm_config)` at `:78`
   alone: that global is shared with the MLA path, and the header comment
   (`ops/rotary_embedding.py:42-50`) records that MLA/SFA models pass cos/sin
   through `attn_metadata` and GQA models take other routes — widening the
   non-VL allocation changes what the other consumers see. Acceptance is the
   engine reaching ready **and** one image request whose text matches the same
   request on a working stack: an exception-free start leaves the numerics
   unvalidated (SKILL.md rule 3).

## Recorded case (#13522)

Title `使用v0.23.0rc1-310部署MiniCPM-V-4.6模型报错：RuntimeError:
npu_apply_rotary_pos_emb:…ApplyRotaryPosEmbNpuOpApi.cpp:58 … error code is
161001`; body = full env dump + the serve command + the traceback quoted above.
The paste alone carries the owner, the branch (`:158`) and the gate. The
recorded session spent 14 assistant turns / 24 tool calls inside `_310p/`
re-deriving which class the `:273` frame belongs to and whether the parent
`_dummy_run`'s `update_cos_sin` call rescues it, wrote **693 characters** of
message text against 44 K characters of reasoning, and was cut at the 16 384-token
cap with no answer (`../evidence-workspace/delivery-discipline.md`). The
mechanism it reached in its reasoning (VL gate → `update_cos_sin` early return →
`cos=None` → `161001`) is the one written above; the ship was lost because the
answer was never typed.

## Verified dead ends

- **The op's source.** `ApplyRotaryPosEmbNpuOpApi.cpp` is in neither pinned
  export: `Ascend/op-plugin` ships 103 files at this pin and `rg -l
  ApplyRotaryPosEmb` is empty, and `Ascend/pytorch` has only a meta-device test
  (`test/test_fake_tensor.py:1525-1543`). The wording is aclnn's own parameter
  check — one grep, then stop.
- **`_npu_rotary_embedding` (`:162-169`).** Takes `positions` + `cos_sin_cache`
  and needs no globals; it is the branch that works, not the one that fails.
- **The MRoPE class.** `AscendMRotaryEmbedding310.forward_oot:203` raises a
  plain `RuntimeError` when its slices are unbuilt, and the runner builds them
  whenever `uses_mrope` (`_310p/ops/rotary_embedding.py:68,239-252`,
  `_310p/model_runner_310p.py:666-668`); the paste's class is the plain one, so
  MRoPE merging (`merge_mrope_cos_sin_for_apply:51`) is not in play.
- **Phase.** The early return at `update_cos_sin:135` is independent of
  `profile_run` vs a real decode batch: any VL config on this branch fails on
  the first forward it runs, inside or outside the dummy run.
