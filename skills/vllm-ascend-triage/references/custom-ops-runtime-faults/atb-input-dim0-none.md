# `SelfAttentionOperation setup failed!` with `dimNum 0` in the ATB log: an uninitialised tensor reached the pack

The fifth `setup failed!` surface, and the only one where the pack's *content* is
missing rather than wrong: a caller passed `None` (or a 0-element tensor) where
the ATB op expects a tensor. Verified at vllm-ascend `v0.10.1rc1`
(`7e16b4a7cdb1`) + vllm `v0.10.1`, recorded for **#2779** (Qwen2.5-72B, TP4,
4×910B3, CANN 8.2.RC1, `torch-npu 2.7.1.dev20250724`, `--enable-prefix-caching`,
`additional_config={"torchair_graph_config":{"enabled":true},
"ascend_scheduler_config":{"enabled":true}}`).

## The signature, and the line that separates it from the other four

```text
[rank2]:[E905 06:51:44.673826892 compiler_depend.ts:429] SelfAttentionOperation setup failed!
Exception raised from OperationSetup at build/third_party/op-plugin/op_plugin/CMakeFiles/op_plugin_atb.dir/compiler_depend.ts:151
frame #2: atb::OperationSetup(atb::VariantPack, atb::Operation*, atb::Context*)
cat $HOME/atb/log/*.log
[error] [tensor_check.cpp:28] tensor dimNum 0 is invalid, should >0 && <= MAX_DIM(8)
[error] [operation_base.cpp:182] SelfAttentionOperation_1 inTensor [1] CheckTensorShape failed. ErrorType: 8
[error] [operation_base.cpp:625] SelfAttentionOperation_1 invalid param, setup check fail, error code: 8
```

The `atb::OperationSetup` / `libop_plugin_atb.so` frames only say *which* phase
failed — every row of the table in `setup-failed-call-site.md` prints them. Two
things in the ATB log make this row:

- **`dimNum 0`** — a tensor with no dimensions at all. A real operand of a wrong
  shape has `dimNum` 2-4 and fails a size/stride check instead, so `dimNum 0`
  means the slot was never filled (Python `None`, `atb::Tensor()` or an empty
  `torch.Tensor()`), not a size limit, a mask length or a tiling key.
- **`inTensor [N]`** — the N-th input of that op, counting from 0 in the caller's
  own argument order. Only the first bad input is reported, so a message quoting
  `[1]` alone is consistent with `[2]` being unset as well.

`_npu_flash_attention_qlens` is the call that carries this pack on the prefill
path (`key_cache` is argument 1), which is why the index lands on `[1]`: treat
the mapping as consistent-with, not proof — no snapshot export carries ATB's or
op-plugin's argument table.

## Mechanism (source-verified at the pin): the fork dropped one binding line

| Fact | Anchor |
|---|---|
| The eager impl declares the two attributes and **binds them in `forward`** | `vllm_ascend/attention/attention_v1.py:265-266`, `:534-536` (`if self.key_cache is None: self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]`) |
| The torchair impl declares the same two attributes and **never binds them** (its `forward` only binds locals: `key_cache, value_cache = kv_cache[0], kv_cache[1]`) | `vllm_ascend/torchair/torchair_attention.py:305-306`, `:367` |
| The `PrefillCacheHit` branch passes the never-bound attributes straight to the op | `vllm_ascend/torchair/torchair_attention.py:410-424` (`key_cache=self.key_cache, value_cache=self.value_cache`) |
| Reaching that branch needs a prefill whose `seq_lens != num_scheduled_tokens`, i.e. a **prefix-cache hit**, with the Ascend scheduler on and chunked prefill off | `vllm_ascend/worker/model_runner_v1.py:1307-1328` (`_build_attn_state`) |

That is why the report reads the way it does: the engine starts, captures and
serves for minutes (`Prefix cache hit rate: 0.0%` in the samples *before* the
crash is not a contradiction — the first cached prompt is the trigger, and the
next 10 s metrics line never prints), then one rank dies in ATB setup on the
first partially-cached prefill.

## The check that settles it in one command

```bash
uv run --no-project python <skill_root>/scripts/init-none-attr-probe.py \
    vllm_ascend/torchair/torchair_attention.py vllm_ascend/attention/attention_v1.py
# torchair_attention.py:305: self.key_cache = None in __init__, never reassigned; read at line(s) 416
# torchair_attention.py:306: self.value_cache = None in __init__, never reassigned; read at line(s) 417
```

Run it on the failing impl **and its sibling impl in the same tree** (eager vs
torchair, graph vs non-graph, MLA vs non-MLA): the sibling's extra assignment is
the fix, and the diff needs no upstream tree. In the reporter's image the same
two files are at
`<site-packages>/vllm_ascend/torchair/torchair_attention.py`; one
`grep -n 'self.key_cache'` there is enough to confirm which side of the fix their
build is on.

Do **not** start with the `torch_npu`/op-plugin/ATB source hunt: those trees are
not in the snapshot (`Ascend/op-plugin` exports no `atb/` sources), and the
crash's own `dimNum 0` already points at the caller.

## Ladder

1. `--no-enable-prefix-caching` (or drop it from the serve line): with no cache
   hit the branch is unreachable and the report's other symptoms
   (`Prefix cache hit rate: 0.0%`) are explained at the same time. Removes the
   fault; does not repair the branch.
2. Drop `torchair_graph_config` (or `--enforce-eager`): the non-graph impl binds
   the attributes at `attention_v1.py:534-536`. A valid A/B for "is it the
   torchair fork", not a deployment recommendation.
3. The patch is the one line the fork dropped — mirror `:534-536` inside the
   torchair impl's `forward` before the `kv_cache` branch and keep the local
   aliases. Reproducer: one request with a shared system prompt sent twice.
4. If the ATB log instead carries a *dtype/format* key or `CheckTensorShape` on
   a dimensioned tensor, this page does not own it — go back to the table in
   `setup-failed-call-site.md`.

## What this snapshot cannot settle

- ATB's own input order (so the `[1]` → `key_cache` mapping above is
  consistent-with, not proof) and the exact `torch_npu` binding that turns
  `None` into a 0-dim tensor: neither tree is exported.
- The reporter's image is `vllm-ascend:v0.11.0rc0`, not this pin; grep the two
  attribute lines on their build before quoting line numbers.
- Whether other forked impls carry the same dropped binding — the probe above is
  the sweep (`vllm_ascend/torchair/*.py`, `vllm_ascend/attention/*.py`).
