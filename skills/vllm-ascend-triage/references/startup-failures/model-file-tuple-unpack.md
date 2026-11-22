# `linear(): argument 'input' must be Tensor, not tuple` — a model file feeds a norm's tuple to a linear

Scope: engine start dies inside `profile_run` (`determine_num_available_blocks` →
`model_runner.profile_run`) with

```text
TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple
```

and the traceback runs over the **pinned upstream model file**
(`vllm/model_executor/models/<model>.py`) → `torch/nn/modules/module.py` →
`vllm/model_executor/layers/linear.py` (`forward` → `quant_method.apply` → `F.linear`), the
only `vllm_ascend/**` frames being `worker/worker.py` / `worker/model_runner.py`. Take this
route on that shape.

Not this route: a `TypeError`/`AttributeError` whose **failing** frame is in
`vllm_ascend/{ops,patch,models}` (`../upstream-downstream-drift/index.md`); a raise while
the config is built, before `Starting to load model` (`index.md` rows); a device dump
(`../hccl-and-hardware/index.md`). Recorded lane: startup, component = the model
implementation, not the platform.

## 1. Read the layer's norm calls before any hypothesis

`RMSNorm` returns a **2-tuple** whenever it is given a residual, on every platform:

- upstream vllm v0.8.4 `vllm/model_executor/layers/layernorm.py:111-147`
  (`forward_native`: `if residual is None: return x` else `return x, residual`);
- vllm-ascend rebinds `RMSNorm.forward_oot` (`vllm_ascend/ops/layernorm.py:24-40`) with the
  same contract from `torch_npu.npu_add_rms_norm`, so **an Ascend/downstream contract
  mismatch is not the cause** — check it once (`git -C … show`/`rg -n 'forward_oot'`), then drop it.

So one line of the form

```python
hidden_states = self.post_attention_layernorm(hidden_states, residual)   # binds (x, residual)
hidden_states = self.mlp(hidden_states)                                  # → gate_up_proj → F.linear
```

binds the tuple and the next linear raises. The error names `F.linear`'s `input`; the owner
is the **assignment above it**. `rg -n 'layernorm\(.*residual' vllm/model_executor/models/<model>.py`
and look for a single-name target where the sibling lines unpack two.

## 2. Why it fires at startup on the first decoder layer

The layer-0 call arrives with `residual=None`, and the layer's own first branch
(`if residual is None: residual = hidden_states`) sets it before the post-attention norm — so
the fatal line is on **layers 0..N-1, batch 1**, i.e. inside `profile_run`, before the server
ever binds its port. Nothing about `--max-model-len`, dtype, TP or graph mode changes it;
`--enforce-eager` cannot help either. That is why the report is "启动即崩" with no request.

## 3. The unpack alone is often not the fix — restore the fused-residual contract

Recorded (#587, GLM-Z1-32B-0414, vllm v0.8.4 `dc1b4a6` + vllm-ascend v0.8.4rc1
`086423dc`, `glm4.py:203-204`): the same decoder forwards **three** mistakes at once —

```python
hidden_states = self.post_self_attn_layernorm(hidden_states)      # :199
hidden_states = residual + hidden_states                          # :200 explicit add
hidden_states = self.post_attention_layernorm(hidden_states, residual)  # :203 fused add AGAIN
hidden_states = self.mlp(hidden_states)                           # :204 tuple
hidden_states = self.post_mlp_layernorm(hidden_states)            # :205
hidden_states = residual + hidden_states                          # :206
return hidden_states, residual                                    # :207 unchanged pre-layer residual
```

Unpacking at `:203` alone double-counts the residual at `:200`/`:203`, and returning the
**unchanged** `residual` makes the next layer's `input_layernorm(hidden_states, residual)`
add it a third time. Report the tuple as the crash and the arithmetic as the follow-on defect,
in one answer — a reporter who patches only the unpack gets wrong tokens, not a clean run.

The repaired upstream form is in the snapshot when a newer `vllm` pin exists — cite it instead
of deriving the math: vllm v0.24.0 (`ee0da84`) `glm4.py:211,214-217`

```python
hidden_states = self.post_self_attn_layernorm(hidden_states)
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
hidden_states = self.post_mlp_layernorm(hidden_states)
return hidden_states, residual        # post_mlp output + the pre-MLP sum
```

i.e. the ordinary vLLM contract: the layer returns the un-added MLP branch plus the running
sum, and `LlamaModel`'s loop / final `hidden_states, _ = self.norm(hidden_states, residual)`
(`llama.py:368` at v0.8.4) add the last residual. Any patch must keep returning a 2-tuple —
`residual=None` would make that final line unpack a tensor.

Recheck condition: `rg -n 'post_attention_layernorm\(hidden_states, residual\)' vllm/model_executor/models/<model>.py`
at the reporter's ref. A single-name assignment = broken; `hidden_states, residual = …` = repaired
(at v0.24.0 the same grep returns the repaired line). Line numbers and class names move between
pins; the *shape* of the assignment is the version-independent tell.

## 4. Coverage decides the verdict's wording

Model-file support is per-release. Recorded #587: at vllm-ascend v0.8.4rc1 the supported-models
table lists `ChatGLM ❌ (#554)` and `GLM-4v | Need test`
(`docs/source/user_guide/supported_models.md:24,:34`) and `rg -ln 'glm' vllm_ascend/` is empty —
so GLM-4/GLM-Z1 dense is **not** a validated arch at that ref. Say both: the tuple is a real
upstream model-file defect (platform-independent, one line, quotable), and this arch is outside
what the release validated, so the next error may be a different model-file deviation.

## 5. A follow-up that quotes a foreign workaround is not a verdict on §3

The reporter's #587 follow-up is
«https://github.com/vllm-project/vllm/issues/16818#issuecomment-2814931469 使用该办法解决上述问题，但还是报错»
plus two `user-attachments` screenshots and no text. It settles **none** of the three readings
below on its own, so the answer must not pick one for them: (a) the link is a **vllm-side issue
comment**, so "该办法" need not be §3's four-line repair — do not report §3 as refuted, and do not
re-send §3 as the answer either; (b) a screenshot URL carries no traceback (this case view keeps
no attachments at all), so the new error is unread — ask for it as **text** together with
`sed -n '196,210p'` of the installed `glm4.py`; (c) at that pin §3's site has **no second
occurrence**: `glm4.py:203` is the file's only norm call that passes `residual` into a single
name (the other four — `:190`, `:192`, `:199`, `:205` — pass no residual or unpack two), so if
they did apply the repair, the next failure is a *different* class, not another tuple in the
same block. Frame that in §4's coverage verdict (the arch was never validated at that ref) and
rank from the new traceback: the next owner is a different line or subsystem.

## Recorded case

#587 — title `TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple`,
vllm 0.8.4 + vllm-ascend 0.8.4rc1, GLM-Z1-32B-0414, 910B, `-tp 8`, death at
`determine_num_available_blocks → profile_run`. `glm4.py:46` aliases the MLP
(`from .llama import LlamaMLP as Glm4MLP`), so the frame pair `glm4.py:204 → llama.py:92`
(`x, _ = self.gate_up_proj(x)`) is one layer, and `glm4.py:203` is the single-line owner. The
session read that line on its 4th assistant turn, then spent its remaining three turns and all of
its 16 384-token output cap re-deriving §3 (52.6 K characters of reasoning, 800 characters of
message text, **no answer delivered**). §3's repaired form and the "not validated at this ref"
verdict are what the 4th turn already owed.
