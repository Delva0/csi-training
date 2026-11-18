# Kernel launch limits: `coreDim ... can't be greater than UINT16_MAX`

Family: the service starts, weights load, and a *larger* step fails with
`EE1001 ... The argument is invalid. Reason: coreDim=<N> can't be greater than UINT16_MAX` followed by
`current working operator name is <kernel>`, plus `rtKernelLaunch execute failed, reason=[invalid
value]` at the PTA/ACL layer. The named kernel is a **Triton** kernel launched from Python (a
`..._kernel[grid]` launch call); the limit is the launch geometry the Ascend triton launcher can
address for one launch (65 535); and the Python traceback printed around the error is whatever next
synchronised, not the launch site. Input length, batch size or KV occupancy is the trigger, so the
report usually says "works at 4096, fails at 8192".

## Discriminator against `device-kernel-faults.md`

| observation | this class | device fault |
| --- | --- | --- |
| error text | launch-parameter check: `coreDim=…`, `can't be greater than UINT16_MAX`, `The argument is invalid` | device runtime code (`507015`), `L0C read/write conflict`, DFX line with a `kernel hash` |
| kernel name | a Python-level Triton kernel (`layer_norm_fwd_kernel`, …) | ACLNN/AICore operator (`QuantBatchMatmulV3_…`) |
| owner of the fix | vllm-ascend (grid arithmetic or a vendored kernel) | CANN release |
| trigger | a size threshold (tokens, batch, occupancy) | a shape/layout the compiled kernel cannot execute |

Both classes print a plausible-looking Python traceback. Decide on the error text and the kernel name,
then ignore the frames. `coreDim=... UINT16_MAX` is host-side: there is no device log for it, so a
DFX/plog hunt is wasted work.

## Where the number comes from (verified at the paired ref)

At `v0.11.0rc0`, `vllm_ascend/ops/fla.py` `_layer_norm_fwd` computes `M, N = x.shape` (`:27`),
`ngroups = N // group_size` (`:31`) and launches the upstream kernel with **`grid = (M, ngroups)`
(`:58`)** — one program per (row, group). `M` is the number of rows after `x.reshape(-1, hidden)`,
i.e. the scheduled token count, so the program count grows with the prompt and `coreDim` is that
product.

The kernel itself is vLLM's `vllm/model_executor/layers/fla/ops/layernorm_guard.py` at the paired tag
(`v0.11.0rc3`), `layer_norm_fwd` (`:130` head, `:170` launch):

```python
M, N = x.shape
ngroups = N // group_size            # group_size defaults to N
grid = (M, ngroups)
layer_norm_fwd_kernel[grid](x, out, weight, bias, z, mean, rstd, ...)
```

It is reached from Qwen3-Next's `RMSNormGated` (gated-delta-net layers, via
`vllm/model_executor/models/qwen3_next.py` at the same tag); the vllm-ascend tree only supplies the
linear-attention/ops overrides.

**Do not factor the reported number in prose.** `68888 / 8192 = 8.41` is not an integer group count,
so either the step carried more tokens than one prompt (batch or padding) or a flat product was
launched under the reporter's `TRITON_ALL_BLOCKS_PARALLEL=1` — exactly the kind of question a one-line
command settles and a paragraph does not:

```bash
uv run --no-project python -c "n=68888; print([(k, n/k) for k in (8,16,32,64,128)])"
git -C /workspace/repos/vllm-project/vllm show v0.11.0rc3:vllm/model_executor/layers/fla/ops/layernorm_guard.py | sed -n '130,175p'
```

Whether that env var is what turned the grid into a single flat dimension is `hypothesis`: the check
is to re-run 8192 with `TRITON_ALL_BLOCKS_PARALLEL` unset and print the grid.

## Worked case: #3971 / #3548 — Qwen3-Next, `layer_norm_fwd_kernel`

- Reported: vllm-ascend `0.11.0rc0`, paired vLLM `0.11.0rc3`, CANN `8.2.RC1`, A3 (8×910B4), TP4,
  Qwen3-Next-80B-A3B. 4096- and 6144-token prompts answer; an 8192-token prompt fails with
  `coreDim=68888`. The reporter also exports `TRITON_ALL_BLOCKS_PARALLEL=1` and
  `ASCEND_LAUNCH_BLOCKING=1` — the second synchronises the PTA/aclnn path, which is why an operator
  name is reported at all, and did **not** improve the attribution.
- The reported Python traceback instead ends in `ops/common_fused_moe.py` →
  `ops/moe/experts_selector.py` (`npu_moe_gating_top_k_softmax`), with one worker ending in
  `torch_npu.npu_moe_gating_top_k_softmax` while the operator named in the error is the layer-norm
  kernel: the async-synchronisation artefact. Do not "fix" the frame the traceback happens to show.
- Fix (verified): `534f32d27` / PR **#3549** ("Fixed a triton kernel bug of layer_norm_fwd_kernel for
  Qwen3-next", merged 2025-10-21) vendors the kernel into `vllm_ascend/ops/fla.py`, adds
  `MAX_CORES = 65535`, loops the rows inside the kernel
  (`BLOCK_ROWS = M if M < N_CORES else N_CORES`) and launches
  `grid = (M if M < MAX_CORES else MAX_CORES, ngroups)`. Ancestry: **not** in `v0.11.0rc0`, in
  `v0.11.0rc1`+ (and `v0.12.0rc1`, …). The twin report #3548 (`v0.10.2rc1`, same error) is closed by
  it; #3971 was closed `not_planned` by the stale bot (2026-08-11) after a "please use the latest
  release or rc version" reply, so the thread is not the answer.
- Workaround: move to `v0.11.0rc1` or later (contains #3549), or on the reported tag lower the rows per
  step (`--max-num-batched-tokens` under the point where `tokens × groups` crosses 65 535), which is a
  diagnostic-grade mitigation, not a fix. Verify any release-branch claim with
  `git merge-base --is-ancestor 534f32d27 <ref>`.

## Levers that actually move it

1. **Tokens per step.** The launch dimension scales with the rows in the step, so the lever is the
   scheduled tokens, not `--max-model-len`. Verified at `v0.11.0rc0`: `vllm_ascend/platform.py:137-139`
   disables chunked prefill for non-MLA LLMs (Qwen3-Next included) and `:158-163` *rewrites*
   `max_num_batched_tokens` up to `max_model_len` on that path — so a small
   `--max-num-batched-tokens` is silently defeated there. Check the platform hook at the reported ref
   before offering the cap (`../vllm-ascend/config-flag-lifetimes.md`, Trap D).
2. **Keep the input under the threshold.** The reporter's own 4096/6144 requests pass, so bounding
   prompt length is a real, if unsatisfying, workaround; say that it caps the symptom.
3. **Unset the Triton block-parallel env var** if the report sets one (`TRITON_ALL_BLOCKS_PARALLEL`),
   and compare.
4. **Runtime build, not a device kernel.** Grid handling belongs to triton-ascend / torch_npu and to
   the upstream vLLM kernel, while the vendoring/cap is a vllm-ascend commit; name the layer honestly.

## Discriminators

| observation | class |
| --- | --- |
| `coreDim=... UINT16_MAX`, operator name is a snake_case `*_kernel` | this page (Triton launch geometry) |
| DFX `KernelName_...` + `kernel hash` + `runtime error 507015` / "L0C read/write conflict" | `device-kernel-faults.md` (CANN build) |
| `Tried to allocate N GiB` on an `n²` mask | `attention-mask/` |
| failure only when the prompt crosses one length, regardless of weights/batch | this page |

## Reuse

- `git grep -n "\[grid\]" <ref> -- vllm_ascend/ops` lists every Triton launch at the ref. Suspect the
  ones whose grid is built from a token count, a token×group product or a slot count.
- `git grep -n "MAX_CORES\|UINT16_MAX\|65535" <ref> -- vllm_ascend` shows which kernels already carry a
  cap — a kernel with no cap at the reported tag and a cap later is the fingerprint of a fixed bug.
- Divide the reported `coreDim` by the model's group / v-head count with a command
  (`uv run --no-project python -c`), not in prose, and quote the quotient next to its uncertainty.
- The kernel file and its grid move with vLLM (read it at the paired tag from
  `../vllm-ascend/version-pairing.md`), and the UINT16_MAX ceiling is a device property that can change
  with a CANN/torch_npu generation; `platform.py`'s rewrite was verified only at `v0.11.0rc0`. Nothing
  here was reproduced on hardware.
