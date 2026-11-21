# The traceback names a library method that does not exist: a downstream typo

Symptom: a request-time `AttributeError` in a `vllm_ascend/` frame whose missing
attribute is a **method of a built-in library object**, not of an Ascend class:

```text
(VllmWorker rank=1 …) File ".../vllm_ascend/attention/attention.py", line 140, in get_splitfuse_attn_mask
(VllmWorker rank=1 …)     right_tensor.mask_fill_(
(VllmWorker rank=1 …) AttributeError: 'Tensor' object has no attribute 'mask_fill_'
```

Recorded #1716: vllm-ascend **v0.7.3.post1** (`c69ceaca`), vllm v0.7.3
(`ed6e9075d`), Qwen2.5-14B-Instruct-1M, 8×910B, TP=8, `max_model_len` 204800,
chunked prefill on; every rank raises in the same second after a long prompt, and
the vLLM `collect_env` block's first line is an unrelated
`Warning: Failed to register custom ops, all custom ops will be disabled` (its own
lane — `../custom-ops-runtime-faults/` — and **not** the cause of a `Tensor`
attribute error; say so explicitly so the reporter does not chase it).

This is *not* drift: no upstream class, signature or state machine is involved.
The downstream tree calls a `torch.Tensor` method that has never existed, so no
release of torch, torch_npu or CANN can make it work.

## The three facts, in the order they are cheap

1. **The message itself exonerates the stack.** The named object is `Tensor`, the
   named attribute is a library method, and the correct spelling is
   `masked_fill_`. A version/ABI/compatibility hunt is a wrong turn; so is
   "`torch_npu` monkey-patched `Tensor`". One message, one call:
   `rg -n '\.mask_fill_' vllm_ascend/` — in the recorded tree it returns exactly
   `vllm_ascend/attention/attention.py:140`, i.e. one typo and no sibling copies
   to sweep. (`masked_fill_` does not contain the substring `mask_fill_`, so this
   grep cannot match the corrected line and is safe to re-run after a fix.)
2. **The enclosing branch's gate explains the prompt-length dependence.** At
   `attention.py:112` the builder takes a pre-built-mask slice when
   `max_seq_len <= self._seq_len_cached`, and only the rebuild branch below
   (`:124-144`, `device="cpu"`, then `.to(device, non_blocking=True)`) contains
   the typo. `_seq_len_cached` starts as
   `min(max_model_len, int(os.getenv("PAGED_ATTENTION_MASK_LEN", 10000)))`
   (`worker/model_runner_v1.py:265-268`), so the switch is **the longest
   request's full sequence length** crossing 10 000 — not the chunk size:
   `--max-num-batched-tokens 4096` leaves `seq_lens` unchanged and cannot be a
   workaround. It is also reached only for
   `AscendAttentionState.ChunkedPrefill` (`model_runner_v1.py:425-431`; prefill-
   no-cache, prefill-cache-hit and decode never call it) — which is why a short
   prompt test on the same image passes.
3. **The fix is a rename, and the semantic argument is one line.** The branch
   already assigned `splitfuse_mask_value` (-10000) over
   `[context_len:]`, then restores the window's lower triangle to 0; the values
   being restored are exactly `-10000`, so `masked_fill_` is the author's intent
   and in-place write-back through the row view is preserved:
   `sed -i '140s/right_tensor\.mask_fill_(/right_tensor.masked_fill_(/' <tree>/vllm_ascend/attention/attention.py`.

## What not to propose

- **`PAGED_ATTENTION_MASK_LEN=<max_model_len>` is not a workaround.** It
  pre-builds an `(L, L)` mask on the host: at L = 204800 that is ≈78 GB per rank
  (32 768 ≈ 2 GB). The arithmetic, the `attn-mask-budget.sh` script and the
  engine-start OOM variant are `../runtime-memory/attention-mask-host-allocation.md`.
- **Do not fold the neighbour warnings in.** A `Failed to register custom ops`
  first line, a `npu is experimental on VLLM_USE_V1=1` line, or a
  `pydantic` notice are separate lanes with their own pages.
- **Do not predict the mechanism from the size of the pasted log.** Eight ranks
  failing together reads like a collective only until the message names a Python
  attribute; the shared-second signature here comes from all ranks building the
  same mask.

## After the patch: the next complaint is this branch's cost

Once the name is fixed, the rebuild branch allocates `(total_q_len,
max_seq_len)` in host memory every step and copies it to the device
(`attention.py:124-144`). At a 200 K `max_model_len` the reporter's next report is
likely "runs but very slow / host RSS climbs", which is this design and not their
environment — name it in the same answer so the follow-up is already answered.

## Scope, and the one claim to leave open

Verified at **v0.7.3.post1 `c69ceaca`** only (the snapshot's single vllm-ascend
pin for the case); line numbers shift. The other two versions in the paste
(`vLLM Version: 0.7.4.dev0+ged6e9075d`, `vllm-ascend 0.7.3.post1`) are the
reporter's, so state the fix as "one line, on your tree" and give the recheck
`rg -n '\.mask_fill_' vllm_ascend/` for the version they actually want to run —
a later release that carries no hit has the rename. Do **not** claim which release
fixed it: a single-pin export cannot show that, and PR/comment access may be
absent. The attribute error is reporter-confirmed; the gate arithmetic is
source-verified at that commit; "how much this costs tokens/s" is unverified.
