# Qwen3-Next (hybrid GDN) on Ascend: digit-heavy prompts answered as soup

Worked case #4057, plus its same-version precedent #3689. Verified at vllm-ascend
v0.11.0rc0 (`00ba07102212c7c7a40de427f09848f2e203c498`) + vllm v0.11.0rc3
(`8ce5d3198d00631a76e1aa02a57947b46bc7218c`).

## Scope and symptom grading

- Model: hybrid linear-attention MoE (3 Gated-DeltaNet layers : 1 full-attention
  layer) — Qwen3-Next-80B-A3B-Instruct and its siblings.
- Symptom: a *short* prompt (tens to a few hundred tokens) comes back as a run of
  digits, binary chatter or a phrase loop; ordinary chat on the same server looks
  fine; no traceback, process alive, `finish_reason: length`.
- The recorded trigger is not length but **precise digit work**: #4057's prompt is
  a 19-token comparison ("9.11 和 9.8 哪个数更大"), #3689's is a multiple-choice
  question whose stem carries `2000`/`3000`. In #3689 the output *loses the
  prompt's own numbers* (it re-reads 2000/3000 as 1024/1010/1101) and then loops
  on a repeated sentence until `max_tokens`.
- Grade before investigating: `usage.prompt_tokens` (short ⇒ here, not the
  long-context page), `finish_reason`, and whether the degradation starts at the
  first tokens. Sampling-driven collapse on a pathological prompt is
  `short-prompt-corruption.md`; this page is the model's *numeric path*. If the
  reporter's working and failing commands differ by `--block-size`,
  `--enforce-eager`, `--enable-prefix-caching` or `--mamba-cache-mode`, that confound
  is graded first and the page is `hybrid-block-size-corruption.md`.

## What vllm-ascend replaces on this path at v0.11.0rc0

The whole linear-attention prefill kernel is rewritten downstream, so a numeric
divergence from the training-time kernel is expected by construction. This is a
mechanism map, not a proven cause.

| Site | Fact |
|---|---|
| `vllm_ascend/patch/worker/patch_common/patch_triton.py:12-16` | rebinds five upstream symbols: `causal_conv1d_fn` / `causal_conv1d_update` → `vllm_ascend/ops/casual_conv1d.py`, `fused_recurrent_gated_delta_rule_fwd_kernel` → `vllm_ascend/ops/sigmoid_gating.py`, `LayerNormFn` and **`chunk.chunk_gated_delta_rule` → `torch_chunk_gated_delta_rule`** |
| `vllm_ascend/ops/fla.py:124-218` | the replacement is a pure-torch reference: all inputs `.to(torch.float32)`, sequence padded to a multiple of `chunk_size=64`, the chunk recurrence rebuilt with an explicit `for i in range(1, chunk_size)` loop, `scale = 1/sqrt(head_dim)` computed internally |
| same, `fla.py:146-151` | `query`/`key` get `F.pad(...).repeat_interleave(2, dim=1)` — a **hard-coded ×2 head expansion**, consistent only while linear value heads = 2 × key heads (Qwen3-Next-80B: 32 vs 16; any TP keeps the 2:1 ratio, so this config is inside the assumption) |
| `vllm_ascend/models/qwen3_next.py:337-355` | the prefill branch loops over sequences and calls `chunk_gated_delta_rule(query=…, key=…, value=…, g=…, beta=…, initial_state=…, output_final_state=True, use_qk_l2norm_in_kernel=True)` — keyword names that match the replacement, and **no `cu_seqlens`**, which the replacement does not accept (upstream's varlen call passes it, `vllm/model_executor/models/qwen3_next.py:554-562`; upstream's signature also has `scale`/`head_first`, `vllm/model_executor/layers/fla/ops/chunk.py:112-121`) |
| `vllm_ascend/models/qwen3_next.py:310,383` | the **decode** branch calls `fused_recurrent_gated_delta_rule`, i.e. the patched Triton kernel — prefill and decode run different implementations of the same recurrence |
| `docs/source/tutorials/multi_npu_qwen3_next.md:4` | the project states the model "is using Triton Ascend which is currently experimental … behavioral changes around stability, accuracy and performance" |
| `docs/source/user_guide/release_notes.md:33` | at this release EP and MTP do not work with Qwen3-Next — both are out of scope for a v0.11.0rc0 report |

Binding caveat: the patch rebinds the module attribute, while the model does
`from … .chunk import chunk_gated_delta_rule` when it loads. In the recorded setup
the model module loads after patching, so the fp32 torch function is what runs; if
a stack shows the Triton `chunk_gated_delta_rule_fwd`, the ordering differed —
that is `../upstream-downstream-drift/patch-import-side-effect.md`, and it changes
which implementation produced the numbers.

## Reporter-confirmed narrowing (#4057)

The reporter later posted the deployment command publicly
(2025-11-07T07:41:01Z, same issue):

```
vllm serve /data2/model/Qwen3-Next-80B-A3B-Instruct --served-model-name "Qwen3-Next-80B-A3B-Instruct" \
  --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.7 --enforce-eager --port 8010
```

That single line closes four ablation rows before they are proposed: **graph mode
off** (`--enforce-eager`), **no MTP / no speculative config**, **no
`--enable-expert-parallel`**, **no quantization flag** — and, at this version, no
prefix cache to enable (see `../performance-throughput/zero-prefix-cache-hit-hybrid.md`).
Read against the pinned tutorial (`docs/source/tutorials/multi_npu_qwen3_next.md:89`,
`vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --tensor-parallel-size 4 --max-model-len 4096
--gpu-memory-utilization 0.7 --enforce-eager`) that line **is** the documented recipe —
same TP, same `0.7`, same `--enforce-eager`, plus `--served-model-name`/`--port 8010` and
`--max-model-len 8192` in place of `4096` — so "the tutorial's own command degenerates on
digit prompts at this rc" is citable rather than an inference, and the one changed length
is a valid value that is irrelevant at 19 prompt tokens (`:86` asks TP ≥ 4 for a 64 GB A2,
which TP=4 satisfies). `--port 8010` is also this thread's identification hook: the
report's screenshot is a `curl` to `…:8010/v1/chat/completions`, so the screenshot and the
later comment are one deployment — when a digit-soup paste shows a port in its OCR and a
serve line arrives in a later comment, match the two before treating them as different
runs. What remains is the linear-attention numeric path above, KV/state dtypes, and
sampling.

Because that command lives here and not in the report body, it is a **fact in hand, not
an ask**: open with it ("your later comment's `TP=4 --max-model-len 8192
--gpu-memory-utilization 0.7 --enforce-eager`"), add "if you now serve with other flags,
say so", and keep it off the *send this back* list. A later replay of this issue (same
pins) read this section, quoted the command in a mitigation bullet, and then asked for
"your actual `vllm serve` command line, the body has none" in the same message — the one
ask that could not move the case. A second replay (same pins) opened with the quoted
command and closed with only "if you now serve with other flags, say so", so this section
holds as written; its one surplus turn was a single `rg` re-confirming the four GDN sites
the mechanism table already pins — keep that grep only because *Precedents* marks those
three sites version-specific, not as a routine step.

Answer-shape guard for this case: the first sentence is *prompt-conditioned degeneration
at this pin, owner not established*, with rows 1-2 as the cheapest next action and rows
4-5 named as the rows that decide ownership. "It is the GDN path" and "no flag can fix it
on this version" are conclusions rows 4-5 would license, not the opening verdict — the
same replay asserted both in its first line while its own table labelled cause 1
*mechanism-verified, not established as a defect*.

## Ablation matrix (one restart or one request per row)

| Row | Change | Corrupt here means | Clean here kills |
|---|---|---|---|
| 1 | same prompt, `temperature: 0`, `max_tokens: 64` | deterministic degeneration inside the model | sampling/top-k collapse |
| 2 | same-length prose prompt, no digits | — | "short prompt" as the trigger; digit recall is |
| 3 | `return_token_ids: true` on the failing request | compare the prompt token ids the engine saw | tokenizer/template corruption |
| 4 | a non-hybrid model on the same host/flags (e.g. Qwen3-30B-A3B, TP=4) | platform-wide numerics | this model's GDN path |
| 5 | the same checkpoint on a GPU-hosted vllm, or `transformers` greedy for 32 tokens | model/weights | Ascend kernels |
| 6 | TP=2 or TP=8 (if cards allow) | per-rank shapes into the fp32 chunk rule | TP=4-specific sharding |

Rows 4-5 are the ones that decide ownership; rows 1-3 cost one request each and
should be in the first answer as the reporter's next step.

## Precedents and what is *not* established

- #3689 — same model, vLLM 0.11.1.dev0 + vllm-ascend 0.11.0rc1.dev103, TP=4,
  `enforce_eager=True`, `max_model_len=32768`, `temperature 0.7 / top_p 0.8 /
  top_k 20`: "gets many repetitive, redundant tokens", output drifting through
  digit/binary chatter into a phrase loop.
- #3571 — same tutorial, reporter-asserted that `Qwen3NextAttention.attn_output`
  is all zeros. At the pin that class matches upstream's gated-attention forward,
  so treat it as a check to run (instrument `attn_output` once) rather than as the
  cause.
- Neither precedent carries a confirmed owner, and no source in the snapshot
  settles whether the fp32 re-implementation, the Triton-Ascend decode kernel, or
  a checkpoint/sampling interaction is responsible. Say that explicitly; do not
  promote "the two kernels differ" to "the kernels are wrong".
- Recheck conditions: the patch list, the `repeat_interleave(2)` expansion and the
  per-sequence prefill loop are v0.11.0rc0 facts. Re-grep
  `patch_triton.py`, `ops/fla.py` and `models/qwen3_next.py` at any later pin — the
  same three sites moved in later releases (answered for v0.12.0rc1 in the next
  section).
- The `patch_common/` directory in the row above is itself version-specific: at
  v0.11.0 (`2f1aed9`) and v0.11.0rc2 (`a2e4c3f`) the file is
  `vllm_ascend/patch/worker/patch_triton.py` with the same five rebinds at
  `:12-16` (`causal_conv1d_update` binds to `causal_conv1d_update_npu`), and
  `torch_chunk_gated_delta_rule` is defined at `vllm_ascend/ops/fla.py:205` with
  `chunk_size=64`, calling into `vllm_ascend/ops/sigmoid_gating.py:45-46` for the
  decode kernel. Cite the path you grepped, not this one.

## v0.12.0rc1: the same path, one release later, on triton-ascend

Verified at vllm-ascend v0.12.0rc1 (`42ceaf08a1ef`) + vllm v0.12.0 (`4fd9d6a8`),
worked case #5065. The mechanism map above does **not** carry forward — the fp32
torch reference is gone:

- `vllm_ascend/patch/worker/patch_triton.py:2-16` now rebinds the same five
  upstream symbols to `vllm_ascend/ops/triton/fla/*` (`chunk_gated_delta_rule`,
  `LayerNormFn`, `fused_recurrent_gated_delta_rule_fwd_kernel`, `causal_conv1d_fn`,
  `causal_conv1d_update_npu`). So from v0.12 the divergence to explain is
  *triton-ascend kernel vs training kernel*, not *fp32 reference vs training
  kernel*; re-derive from `ops/triton/fla/`, not from `ops/fla.py`. The self
  maintained modelling file is also gone (Upgrade Note, `release_notes.md:37`), so
  the model body is upstream vllm's.
- **Quote the release's own statements before arguing from a passing test.** At
  this pin `docs/source/user_guide/release_notes.md` carries
  `:42` Known Issue "DeepSeek 3/3.1 and Qwen3 doesn't work with FULL_DECODE_ONLY
  graph mode. We'll fix it in next release. [#4990]";
  `:46` "bf16/fp16 model doesn't perform well, it's mainly because that
  `VLLM_ASCEND_ENABLE_NZ` is enabled by default. Please set
  `VLLM_ASCEND_ENABLE_NZ=0`"; `:38` Upgrade Note "CANN is upgraded to 8.3.RC2,
  Pytorch and torch-npu are upgraded to 2.8.0" (a host still on CANN 8.2.RC1 is
  outside the release's contract — say so, do not call it the cause). The
  Qwen3-Next graph case exists (`tests/e2e/multicard/test_qwen3_next.py:46-61`,
  `FULL_DECODE_ONLY`, capture sizes `[1,8,24,48,60]`) but it only runs
  `generate_greedy(..., 5)` with no correctness assertion, so "CI has the test"
  does not refute an accuracy report.
- **What the project says it did not verify at this ref**: `:109`
  `# TODO: will conduct accuracy verification after the subsequent version becomes
  stable` on the W8A8 Qwen3-Next case (TP4 + EP + `HCCL_BUFFSIZE=1024`,
  `max_tokens=5`), and `:64` `# TODO: Fix the accuary of batch chunked prefill`. A
  W8A8 accuracy report at v0.12-v0.13 therefore sits inside an explicitly
  unverified path: cite both lines and put a bf16 run of the same prompt as the
  first discriminator.
- **NZ is not a bf16-only lever**: `quantization/w8a8.py:196-198` casts the int8
  weight to `ACL_FORMAT_FRACTAL_NZ` under `is_enable_nz()`, and `:386-391` does the
  same for the MoE `w13_weight`/`w2_weight`; the default is `1`
  (`envs.py:127-128`) and `worker/worker_v1.py:161` prints "FRACTAL_NZ mode is
  enabled. This may cause model parameter precision issues". The release note names
  bf16/fp16 only, so label the W8A8 consequence *mechanism-verified, default-on*,
  not *stated defect*.
- The GDN `in_proj` is quantized through a per-model fusion map
  (`quantization/quant_config.py:223-230`, `in_proj: ["in_proj_qkvz",
  "in_proj_ba"]`), so a bad fused packing surfaces as fluent semantic drift
  (#5065: `who are you?` answered as a paraphrase task, a Chinese prompt answered
  with an English essay) rather than as digit soup. Grade it with the same
  ablation matrix above plus one extra row — the same prompt on the bf16
  checkpoint, which separates the quantized path from the triton GDN path — but on a
  **quantized** serve the installed triton-ascend build is checked before that row
  (next section).

### The triton-ascend build owns the quantized `!` run (recorded #5606)

#5606 is the same model, checkpoint and pin (`vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8`,
`--tensor-parallel-size 2`, `--quantization ascend`, `--max-model-len 128`), and its
symptom is a run of `!` to `finish_reason: length` on a 16-token prompt. Here the owner
is stated in the thread, not derived from the tree:

- Maintainer statement (InSec, 2026-01-05, same issue): "the currently known
  quantization accuracy issue is caused by the **triton-ascend** version. We have found
  that the **triton-ascend 1107** version will led to the occurrence of **NAN** problems
  for some operators in the quantization scenario, just like all the answers here are
  full of exclamation marks." Ask for the installed build and move it, per the model's
  own tutorial.
- Reporter-confirmed next comment (2026-01-07): after changing that wheel the model
  **deployed and served**, and the next failure was a different, unrelated error on the
  API call. The paste garbles the two version strings, so record it as "the build was
  changed", never as a specific from→to pair.
- `1107` is the pinned tutorial's own wheel: `docs/source/tutorials/Qwen3-Next.md:58-59`
  installs `triton_ascend-3.2.0.dev2025110717-cp311-cp311-manylinux_2_27_aarch64.whl`.
  So the release's documented build **is** the one the maintainer calls known-bad for
  quantization, and "update per the official tutorial" means a build newer than that
  pin — not "reinstall the pinned wheel".
- The check is one command, and it is not in the env dump: `pip show triton-ascend`
  (or `pip list 2>/dev/null | grep -i triton`). `collect_env` filters `pip list` by the
  substring `triton` (`vllm/collect_env.py:81-92`, applied at `:540-543`), yet #5606's
  "Versions of relevant libraries" block lists only numpy/pyzmq/torch/torch_npu/
  torchvision/transformers — so ask for the build instead of reading the paste, and keep
  its absence an open question (not installed vs filtered), not a finding.
- Why the symptom looks like a sampler collapse: a NaN from a quantized operator does
  not raise, it collapses the logits to one token, so the text is one punctuation mark
  repeated to the token cap. Grade it as NaN, not as sampling, and put this row before
  the sampling/temperature rows.
- Ordering for a **quantized** serve: triton-ascend build → `VLLM_ASCEND_ENABLE_NZ=0` →
  bf16 checkpoint. A bf16 run on the same bad wheel can be clean (the maintainer scopes
  the NaN to the quantization scenario), so taking the bf16 row first can mis-rank the
  rest.
