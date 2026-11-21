# 310P hybrid (Qwen3.5/GDN) with MTP **and** prefix caching: abnormal accuracy

Scope: MTP and prefix caching switched on **together** on a **310P / Atlas 300I**
hybrid model — from the one-line version ("the accuracy of the Qwen3.5 series model
with MTP and prefix cache superimposed is incorrect", empty `collect_env`, no log, no
request, no serve command) to the fully pasted one (#17338: serve command, acceptance
numbers, corrupt text); same verdict path either way. Model family is
**hybrid** (gated-delta-net linear attention + full attention), so both features
mutate the same mamba/SSM state, and the 310P lane replaces that state machinery
with its own fallbacks. This page is the mechanism map, the coverage verdict and
the three launches that decide it. The sibling pages are
`hybrid-block-size-corruption.md` (a knob-pair 乱码 report with logs) and
`qwen3-next-gdn-numeric-path.md` (short digit prompt, no scheduling knob).

## Grade it before reading any source

No symptom detail means nothing is attributable yet — but the *verdict class* is
decidable from the pinned tree in four reads, and that verdict **is** the answer:
"each feature is covered alone on 310P, the superposition is not, and these three
launches place the fault". Do not spend the session on kernel internals. Every
`path:line` below was verified at vllm-ascend v0.23.0rc1 (`f4a08bddd0cc`, the pin
for the first report — its title's `v0.23.0` names no release) and upstream vllm
v0.23.0 (`0fc695fc`), and **re-checked at the release pin vllm-ascend v0.23.0
(`5cb98caa`)**: `patch_mamba_utils.py:22-23`, `:178-196` and `gdn_310.py:149-151`
are unchanged, while `patch_mamba_config_310.py`'s align branch and its
`HybridAttentionMambaModelConfig.verify_and_update_config` registration still sit at
the end of the file (grep the symbol there, not `:84-85` — the interior lines drift).

## What the two features do to one state machine

| Step | Verified fact |
|---|---|
| prefix caching on a hybrid model | upstream `vllm/model_executor/models/config.py:349-379` — with `enable_prefix_caching`, `mamba_cache_mode` goes `"none"` → `"all" if supports_mamba_prefix_caching else "align"` (`:353`), logs `Mamba cache mode is set to '%s' for %s by default`, and `align` **asserts `enable_chunked_prefill`** (`:365-367`) and sets `mamba_block_size = block_size` (`:377-379`). Qwen3.5 declares no mamba-prefix-caching support, so a plain `--enable-prefix-caching` launch is in `align`: per-block SSM state, copied on hits. |
| 310P re-implements that config step | `vllm_ascend/patch/platform/patch_mamba_config_310.py:84-85` (`align` ⇒ `mamba_block_size = cache_config.block_size`), `:68-73` (`"all"` ⇒ `attn_block_size`), `:86-100` (pad the mamba page to the attention page), registered on `HybridAttentionMambaModelConfig` at `:103`. The 310P-only patch can drift independently of the A2 one. |
| MTP drives the same state | every hybrid request reserves `num_speculative_blocks` extra blocks (`core/single_type_kv_cache_manager.py:45`, `_310p/model_runner_310p.py:1000`), and `preprocess_mamba` copies the previous step's state into the last `1 + num_speculative_blocks` block (`patch/worker/patch_mamba_utils.py:227-266`). The postprocess then decides, per request, `accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state` and rewrites `num_accepted_tokens` to 1 when the running state stays in the same block. |
| the 310P-only difference in that postprocess | `_can_launch_triton_batch_memcpy()` is `return not is_310p()` (`patch_mamba_utils.py:22-23`), so on 310P the bindings at `:185-196` swap `collect_mamba_copy_meta`, `do_mamba_copy_block` and `postprocess_mamba_align_gpu` for torch/numpy fallbacks (`:56-176`, the align fallback at `:108-176`) and make `batch_memcpy` raise (`:178-183`). The accepted-token/block-align logic every other SoC runs in a Triton kernel runs as **Python pointer arithmetic over numpy views** here. |
| the 310P state dtype | `_310p/ops/fla/gdn_310.py:149-151` — `_310p_get_state_dtype` returns `torch.float16` for the SSM state (bound at `:185`), because `npu_recurrent_gated_delta_rule_310` has no fp32 state support (`:205-207`); the same call casts `q/k/v/beta` to fp16 (`:134-139`). Upstream's guidance is that fp32 is the dtype without accuracy issues, so 310P is the one lane with a half-precision recurrent state. |
| the value the fallback reads is 310P-produced | `_310p/model_runner_310p.py:283-305` maintains `num_accepted_tokens` / `num_accepted_tokens_cpu`, including an `use_async_scheduling` branch that remaps through `prev_positions` and fills new requests with 1. That CPU array is exactly what `_postprocess_mamba_align_gpu_cpu_fallback` consumes. |

Two consequences to state as hypotheses, not causes: (a) fp16 recurrent state is
re-used across cached blocks and advanced by *several* accepted tokens per step
instead of recomputed — drift-shaped divergence; (b) the align fallback is the only
implementation of that bookkeeping outside the Triton kernel, so a branch/offer error
there is corruption-shaped. The repair history also matters: the fallback carries
comments about matching the fused kernel's `num_accepted_tokens` default, i.e. this
path has already been patched for exactly this interaction.

## Coverage verdict (the citable part)

At v0.23.0rc1 the 310P PR lane has **two separate tests and never the
combination**:

- `tests/e2e/pull_request/one_card/_310p/test_dense_model_310p.py:38-70,140-150`
  (`test_qwen3_5_dense_prefix_mamba_cache_tp1_fp16`) runs Qwen3.5-4B with
  `enable_prefix_caching=True, mamba_cache_mode="align", max_num_batched_tokens=2048,
  mamba_ssm_cache_dtype="float16", enforce_eager=True` and compares against
  `enable_prefix_caching=False` — **no `speculative_config`**.
- `tests/e2e/pull_request/one_card/_310p/test_spec_decode_mtp_310p.py` runs
  `speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 1}` —
  **no `enable_prefix_caching`**.

So "each feature alone passes CI on 310P" and "the product of the two is untested on
any SoC that uses the fallback" are both true, and both belong in the first answer.
Note what else the two tests pin, because it bounds the reporter's envelope: 4B
weights, fp16, `--enforce-eager`, 8 generated tokens, 2048 context.

## The three launches

Same prompt set, greedy (`temperature: 0`), same weights on the same box; record the
first divergence against a run with both features **off**.

| Row | Change | Corrupt here ⇒ |
|---|---|---|
| 1 | both features on (reproduce) | — |
| 2 | `--no-enable-prefix-caching` (MTP stays) | the align-mode state copy / fp16 state type owns it, not MTP |
| 2b | MTP + prefix cache, the *same body twice* (the second request hits the cache) against a unique random prefix per request | corruption only on the cached path ⇒ the align-mode **per-block state copy** owns it (a hit copies the SSM state, a miss recomputes); corruption on both ⇒ the MTP state advance (row 4), not the copy |
| 3 | MTP off (`--speculative-config` omitted, prefix cache stays) | prefix caching alone owns it (compare with the CI row above) |
| 4 | both on, `num_speculative_tokens` 1 vs the reporter's value | the accept-bias/multi-token advance |
| 5 | same checkpoint, both features on, on A2/A3 or a GPU host | the 310P fallback and fp16 state, not the model |

Rows 2 and 3 are the discriminators the reporter will not think of; without them
"both features are wrong" is a slogan. Ask for the serve command too — a launch that
does not set `mamba_cache_mode` is already in `align` (upstream `:353`), and an
`AssertionError: Chunked prefill is required for mamba cache mode 'align'` means the
launch never reached this path.

## Second recorded report: #17338 (vllm-ascend v0.23.0 `5cb98caa`, 310P3 ×4, TP4)

Richer paste of the same shape, and it moves one boundary: Atlas 300I Duo 310P3 44 GB
×4 at TP4, `Qwen3.8-27B-w8a8-310p` (`model_type=qwen3_5`, 64 layers, 3:1 GDN:full,
`mtp_num_hidden_layers=1`), `--mamba-ssm-cache-dtype float16 --dtype float16`,
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,4]}'`,
`enable_npugraph_ex: false`, `{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 1}`
plus `--enable-prefix-caching`: no error, **MTP acceptance 70-100 % (normal)**, output
corrupt at token level (repeated insertion, 千问→谦, 500→50); dropping only
`--enable-prefix-caching` fixes it with acceptance still >90 %.

Two things this row adds. (1) **Normal acceptance with corrupt text is still this
page**: the draft/verify math is healthy, so the fault is in the SSM state or the block
boundary, not in the sampler — do not steer to `../spec-decode-mtp/acceptance-collapse.md`.
(2) The two arms differ **only** by `--enable-prefix-caching`, so graph mode, the W8A8
checkpoint, TP4 and the 64-layer 3:1 geometry are held constant: they cannot explain
the arm difference, though they sit outside both 4B/fp16/eager CI rows and can modulate
severity. They are rows 4-5, not causes. The hit/miss split (row 2b) is the cheapest
sharpening available, since nothing in either report separates a cache-hit copy from a
cache-miss recompute.

## Stop rule

The recorded run (#14339) read 14 files across `patch_mamba*`, `_310p/ops/fla`,
`_310p/model_runner_310p.py`, the upstream `config.py`/`cache.py` and the two tests,
spent ~42 K characters of reasoning over 22 tool calls, emitted **no message text on
17 turns**, and was cut at the output cap inside ranked cause 2 — the ladder, the
scope block and the "what to send back" list never shipped. The reads above are the
whole evidence base: after the coverage verdict (two test files) and the fallback
grep (`_can_launch_triton_batch_memcpy`), the answer is owed as one message, and rows
1-3 are the next step. What is *not* established here: which of (a)/(b) is real, and
whether the reporter's own stack differs from both CI rows (a W8A8 checkpoint, a
non-default `--max-model-len`, or graph mode would each be a separate confound).
