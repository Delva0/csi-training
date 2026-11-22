# A checkpoint `config.json` key selects an upstream CUDA-only path

Lane B, model-construction phase. Recorded #4309: vllm `v0.11.0`
(`b8b302cde434`) + vllm-ascend `v0.11.0rc1` (`c5fe179cef7f`), 8×910 64 GB,
`Qwen/Qwen2.5-7B-Instruct-1M` served with `--max-model-len 200000 --dtype float16`:

```
INFO [model_runner_v1.py:2642] Starting to load model .../Qwen2.5-7B-Instruct-1M...
File ".../vllm/model_executor/layers/rotary_embedding/__init__.py", line 70, in get_rope
  rotary_emb = DualChunkRotaryEmbedding(head_size, rotary_dim,
File ".../vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py", line 37, in __init__
  self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
File ".../torch/cuda/__init__.py", line 363, in _lazy_init
  raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
```

Every frame below the loader is upstream `vllm/**` and `site-packages/**`; the only
`vllm_ascend` frames are `worker_v1.load_model` → `model_runner_v1.load_model`. The
reporter had already found the workaround ("runs ok by deleting
`dual_chunk_attention_config` in `config.json`") — so the job is to *own* the
failure, not to hunt it.

## Read the environment line correctly

`torch==2.7.1+cpu` in `collect_env.py` is **normal on an NPU host**: `torch_npu`
supplies the device and `cuda`-built torch is not expected. It is not the cause, and
"reinstall torch with CUDA" is the wrong next step. What the assertion proves is
narrower: this process called a `torch.cuda.*` API, and `torch_npu` does not patch
`torch.cuda.current_device()`, so any unguarded `torch.cuda` in code the model loads
aborts the engine.

## Why this is not a vllm-ascend, CANN or flag defect

- The failing class is upstream: `dual_chunk_rope.py:37` runs unconditionally in
  `__init__`, with no `current_platform` guard anywhere in the file. Its siblings at
  the same commit are the counter-example — `deepseek_scaling_rope.py:64,83` use
  `device=current_platform.device_type`, and the rest of the package builds no device
  at all (`bash <skill_root>/scripts/cuda-only-path-audit.sh <vllm-root>` prints exactly
  one hit for the whole `rotary_embedding/` package).
- The selector is the checkpoint, not the launch command:
  `rotary_embedding/__init__.py:64-73` chooses `DualChunkRotaryEmbedding` purely on
  `dual_chunk_attention_config is not None`, forwarding only `chunk_size` /
  `local_size`. `vllm/config/model.py:1018-1033` also fires
  `verify_dual_chunk_attention_config()` whenever the key exists (it only adds
  `sparse_attention_config`). **No CLI flag or env var reaches or bypasses the
  branch.**
- Nothing downstream replaces it: `vllm_ascend/ops/rotary_embedding.py:24-26`
  imports and patches only `RotaryEmbedding`, `MRotaryEmbedding`,
  `YaRNScalingRotaryEmbedding`, `DeepseekScalingRotaryEmbedding`. `rg -n dual_chunk
  vllm_ascend/` hits **only** `vllm_ascend/torchair/models/qwen2.py:83,96,156,180`
  (the torchair model wrapper passes the arg through) — there is no OOT
  `dual_chunk_rotary_embedding` implementation, so upstream's CUDA-only class is what
  runs.
- Nothing downstream consumes what the class produces either. Its `forward_native`
  (`:114`) concatenates **five** query variants along the last dim (`torch.cat` at
  `:154-161`, i.e. `query` of width `5 × head_size`), and
  `AscendAttentionBackendImpl.__init__` (`vllm_ascend/attention/attention_v1.py:
  292-304`) absorbs `dual_chunk_attention_config` in `**kwargs` while no
  `chunk_size` / `local_size` / `q_inter` reference exists anywhere under
  `vllm_ascend/attention/` — there is no Ascend dual-chunk attention. Treat the
  "patch line 37 to `torch.device(\"npu\")` and ship" idea as a **hypothesis** until
  it is run: the forward path is portable but the 5×-wide query has no consumer, so
  the expected next failure is a reshape/`view` error or silently wrong numbers.

**The same `-1M` family has a second, unrelated recorded failure — do not let the
family match pull the answer.** A `history/` search for `dual_chunk|Instruct-1M`
returns two hits, of which only **#1716** (Qwen2.5-14B-Instruct-1M on vllm-ascend
v0.7.3.post1) is this checkpoint family: it starts and serves fine, then dies at
*request* time on `AttributeError: 'Tensor' object has no attribute 'mask_fill_'` at
`vllm_ascend/attention/attention.py:140` — a downstream typo for `masked_fill_`
(`../upstream-downstream-drift/library-method-typo.md`), a different phase and a
different owner. Cite it as distinct if the reporter raises it; it is not evidence
about the startup abort and must not make attention, `--max-num-batched-tokens` or
CANN suspects here.

## Answer shape for this class

1. **Verdict first:** yes, the model cannot be served as-is on this pin; the owner is
   upstream vLLM (`dual_chunk_rope.py:37`, CUDA-only), not vllm-ascend, the card or
   the image; the reporter's config edit is the supported workaround.
2. **Cheapest action in the same paragraph:** serve from a copy of the model
   directory with `dual_chunk_attention_config` removed from `config.json`, leaving
   `max_position_embeddings` and `rope_theta` untouched (removal falls through to
   plain `RotaryEmbedding`, `__init__.py:74-76`).
3. **Accuracy caveat, not a length-budget one:** this key *is* the checkpoint's
   long-context mechanism, so the stripped config is plain RoPE — expect degradation
   on long-context QA versus the HF/GPU reference (the recorded answer put the onset
   beyond ~32K-64K as a hypothesis; label it that way and ask for one long prompt
   comparison). `--max-model-len 200000` still validates because
   `max_position_embeddings` is ~1M.
4. **Say what not to burn runs on:** `--enforce-eager`, `--dtype`,
   `--trust-remote-code`, TP/DP flags, CANN/driver versions, allocator env — none is
   read before line 37 executes.
5. **If real long context is required:** it is two upstream items, not a flag — make
   the class device-agnostic, then add a downstream OOT implementation plus attention
   support for the widened query. File it as a feature request.

## Recheck conditions

- `bash <skill_root>/scripts/cuda-only-path-audit.sh <vllm-root> [subpath]
  [<vllm-ascend-root>]`
  re-decides the upstream half at any ref: unguarded `torch.cuda` hits, whether the
  file has any `current_platform` reference, which `forward_*` implementations exist,
  and whether the downstream tree mentions the class or its registered op name.
- Fixed upstream when `device = ` in that file uses `current_platform.device_type` or
  no device at all; fixed downstream when an OOT/`register_oot` entry for the op name
  appears **and** an attention backend handles the multi-chunk query.
- Snapshot limit in the recorded case: vllm-ascend's release test-conclusion file for
  the pinned rc was absent, and no doc, test or e2e yaml in the tree named the `-1M`
  checkpoint, so "validated or not" was left as an open question — say that rather
  than citing a support matrix row that does not exist.

## Generalisation

Any checkpoint key that selects a specialized `get_rope` / attention / quantization
class can pick a CUDA-only or NPU-unimplemented implementation. Three checks settle
it, in this order: (a) is the selected class device-agnostic (audit script), (b) is
its op name in the downstream OOT registry, (c) does anything downstream consume the
tensor it emits. A "yes, yes, yes" is the only combination that makes a device-patch
proposal credible.
