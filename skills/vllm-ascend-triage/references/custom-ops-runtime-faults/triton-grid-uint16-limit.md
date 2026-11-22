# A Triton launch grid over UINT16_MAX: `coreDim=<N> can't be greater than UINT16_MAX`

## Scope

The engine serves small requests and dies on a *long* one, and the paste carries:

```text
[ERROR] … (PID:38630, Device:1, RankID:-1) ERR00100 PTA call acl api failed.
EE1001: [PID: 38630] … The argument is invalid.Reason: coreDim=68888 can't be greater than UINT16_MAX.
    The argument is invalid.Reason: rtKernelLaunch execute failed, reason=[invalid value]
RuntimeError: The Inner error is reported as above. … the current working operator name is layer_norm_fwd_kernel.
```

**The `[WARNING] Please DO NOT tune args …` lines above that traceback are not part of
it.** They are the triton autotune notice on the same paste (present in #3548's own
body) — attribution and the body-search route in
`../observability-and-logs/triton-autotune-tune-args-warning.md`. Reading them as a
cause costs rule 2's arithmetic.

This is **lane C, a host-side launch-geometry refusal** — the same family as
`gather-index-out-of-range.md` (an argument number out of range), not a CANN/driver
fault, not an aclnn registry or tiling rejection, not HBM. It is also not the
"failed to compile custom operators with Triton" lane
(`../image-dependency-faults/triton-ascend-pair.md`): the kernel here **compiled fine**
and ran for every shorter request.

## Rules

1. **The refusal is arithmetic; read the number, not the prose.** `coreDim` is the
   number of programs in a `@triton.jit` kernel's grid, and the Ascend backend
   launches one block per program, so the grid has a 65535 ceiling. Nothing about
   this number is a card, driver, CANN or precision fact.
2. **Divide it back into tokens and you have the shape.** The grid of a fused norm
   kernel multiplies the row count by a group count, so `coreDim ÷ (per-rank head
   count)` is an integer only for the right factor. Recorded #3971 (v0.11.0rc0,
   Qwen3-Next, TP4): `68888 / 8 = 8611` tokens in that forward; precedent #3548
   (v0.10.2rc1): `66976 / 8 = 8372`. Both quotients are exact — that is the
   confirmation, and it also states the threshold the reporter will hit next:
   `tokens × factor ≤ 65535`.
3. **The "current working operator name" line is the owner here, and the MoE frames
   belong to another rank.** A `(Worker_TPn pid=…)` paste interleaves two ranks'
   tracebacks line by line, so `npu_grouped_matmul` / `npu_moe_gating_top_k_softmax`
   frames sit *inside* the same RuntimeError and read like the failing call. Take
   the `current working operator name` as the owner and check it against rule 2's
   arithmetic instead of chasing the grouped matmul. `ASCEND_LAUNCH_BLOCKING=1`
   changes nothing: a rejected launch is already synchronous, and the reporter in
   #3971 had it exported.
4. **On Qwen3-Next the owner is the plugin's own layer-norm, at a known line.**
   `vllm_ascend/patch/worker/patch_common/patch_triton.py:16` rebinds
   `vllm.model_executor.layers.fla.ops.layernorm_guard.LayerNormFn` to
   `vllm_ascend/ops/fla.py:82`; `LayerNormFn.forward` reshapes the activation to
   `(-1, head_v_dim)` (`fla.py:100`) and `_layer_norm_fwd` launches
   `layer_norm_fwd_kernel[grid]` with `grid = (M, ngroups)` (`fla.py:58`). The
   gated-delta-net norm is built with `group_size=None`
   (`vllm/model_executor/models/qwen3_next.py:294-300`, called at `:606`), so
   `ngroups = 1` and `M = tokens × local_value_heads`. At TP4 on the 80B config
   (`linear_num_value_heads=32`) the factor is 8 — hence 8192 tokens is exactly one
   row over the ceiling (8 192 × 8 = 65 536 > 65 535), while 6144 × 8 = 49 152 and
   4096 × 8 = 32 768 pass, which is the working/failing pair the reporter already
   measured (4096 and 6144 serve, 8192 fails).
5. **Why the reported boundary is the *prompt* length: chunked prefill is off.**
   `vllm_ascend/platform.py:140-163` logs `Non-MLA LLMs forcibly disable the chunked
   prefill feature…` and, with the Ascend scheduler's `enable_chunked_prefill`
   defaulting to `False` (`vllm_ascend/core/schedule_config.py:28`), raises
   `max_num_batched_tokens` to `max_model_len` (`:163`). So the whole prompt is one
   forward, and `--max-num-batched-tokens` alone is silently rewritten. Verify from
   the reporter's own log before advising anything: that warning line, then the
   `Calculated maximum model length` / engine-config echo.
6. **Ladder.** (a) Same request under the token ceiling — a shorter prompt, already
   known good at 6144 — proves the arithmetic without a code change. (b) Keep the
   long prompt but enable chunked prefill **in the Ascend scheduler** and cap the batch
   **together**: `--max-num-batched-tokens 4096` plus
   `--additional-config '{"ascend_scheduler_config": {"enabled": true, "enable_chunked_prefill": true}}'`
   (the key is documented at
   `docs/source/user_guide/configuration/additional_config.md:72` with an example block
   under it). Without the flag `platform.py:163` restores the value; expect the plugin's
   own "would lead to performance degradation" warning. Two ways the cap already holds
   without the flag, both cheaper to check first: a multimodal model, a non-`auto`
   structured-output backend, `policy != fcfs`, `send_delta_data` or
   `scheduler_delay_factor > 0` skips the outer block at `platform.py:140` entirely, so
   the user's own `--max-num-batched-tokens` survives.
   (c) A newer release: a fix that lands after the pin is invisible in the snapshot —
   say so instead of deriving it.
7. **The repair is a tiled launch, and the acceptance check is the pair of requests.**
   One program per row cannot exceed 65535 rows, so the kernel must take a rows-per-
   program factor (`rows = cdiv(M, 65535)`, grid `(cdiv(M, rows), ngroups)`, inner
   loop over `rows`) — or the caller must bound `M`. A `try/except`, a silent
   `min(grid, 65535)` or a "shorter prompts work" answer removes the exception and
   leaves the long prompt numerically unvalidated (`SKILL.md` procedure step 3): the
   check is that the 8192-token request returns the same tokens as the 6144-token
   run of the same body.
8. **Check the release for the env escape hatch before promising one.**
   `TRITON_ALL_BLOCKS_PARALLEL=1` and `_ASCEND_TRITON_GRID_LIMIT` belong to a later
   tree (`ops/rotary_embedding.py`'s `query_shape[0] > 65535` branch,
   `../startup-failures/env-write-in-compiled-forward.md`). At v0.11.0rc0 the only
   `65535` in the whole tree is `csrc/kernels/pos_encoding_kernels.cpp:348`
   (`maxParallelSize`), so that export is inert here even though the reporter's
   launch script carries it — a version-correspondence point, not a lever.

## Answer shape for this signature

Verdict first: `real, a launch-grid limit in <kernel> at <path:line>, not hardware;
prompts at or below <threshold> tokens are fine`. Then the arithmetic (the reporter
can verify it from their own two working lengths), the classification (lane C,
request-time, first long prefill), the ladder of rule 6, and the repair of rule 7
with its acceptance check. Quote the precedent by id for the same signature at
another release, and say plainly that the pinned tree cannot show whether a later
release fixed it.
