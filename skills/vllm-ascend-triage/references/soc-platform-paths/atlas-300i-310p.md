# Atlas 300I / 310P: what the platform changes, and what the docs admit

Verified at vllm-ascend **v0.11.0rc3** (`b6d63bbd`) with vllm **v0.11.0**
(`b8b302cd`); the `-310P` image is built from `Dockerfile.310p.openEuler`
(`FROM quay.io/ascend/cann:8.3.rc2-310p-openeuler24.03-py3.11:18`,
`SOC_VERSION=ASCEND310P3:51`, `VLLM_TAG=v0.11.0:37`). Re-grep per version: the
gate list and line numbers move, the shape does not.

For "there is no 310P artefact for my machine" (arm64 host, an arm64 tag that pulls
amd64, an aarch64 wheel) the owner is the image pipeline, not this page:
`../image-dependency-faults/image-arch-and-tag-availability.md`.

## The documented status (quote it, then move on)

- `docs/source/faqs.md:18`: "[Experimental] Atlas 300I inference series (Atlas
  300I Duo). Currently for 310I Duo, the **stable version is vllm-ascend
  v0.10.0rc1**."
- `docs/source/tutorials/single_node_300i.md:3-7`: 300I is "currently
  experimental", "only supports eager mode and the float16 data type", known
  issues `#3316` / `#2795`, "You can use v0.10.0rc1 version first."
- **The same file contradicts itself**, and this is the trap in #4812: its
  Qwen2.5-VL-3B tab (`:126-131`) ships exactly the failing command
  (`vllm serve … --tensor-parallel-size 1 --enforce-eager --dtype float16
  --compilation-config '{"custom_ops":["none","+rms_norm","+rotary_embedding"]}'`)
  while the image line at `:17` still says `v0.10.0rc1-310p`. When a reporter
  pastes a command that is verbatim from a tutorial, say so — it removes "user
  error" from the ranking and it is a documentation finding in its own right.
- Environment the tutorial requires and reporters routinely omit (`:43-49`):
  `VLLM_USE_MODELSCOPE=True` and
  `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`. That value is the project's
  300I fragmentation workaround, not general tuning, and from v0.18.0 on it also
  switches off vllm-ascend's own `expandable_segments:True` default
  (`vllm_ascend/platform.py:518-528` at v0.18.0; no such injection exists at
  v0.11.0rc3) — state both effects
  (`../runtime-memory/hbm-growth-and-oom.md`, step 3).

## The 310P gates that change behaviour

| Gate | Code (v0.11.0rc3) | Consequence to check |
|---|---|---|
| SoC detection | `vllm_ascend/utils.py:66-71` | every branch below keys off the build stamp, not the device name |
| Custom ops not activated | `vllm_ascend/platform.py:287-288` — `# Activate custom ops for v1, except on 310P` / `if not is_310p(): compilation_config.custom_ops = ["all"]` | on 310P the reporter's `--compilation-config custom_ops` list is the only one in play; on other SoCs vllm-ascend overwrites it. A "custom op choice made no difference" observation is expected here |
| Global NZ format | `vllm_ascend/worker/model_runner_v1.py:146-154` — `torch.npu.config.allow_internal_format = True`; `set_compile_mode(jit_compile=False)`; `ACL_FORMAT = ACL_FORMAT_FRACTAL_NZ` on 310P (ND otherwise) | every `_convert_torch_format` call site becomes a real conversion on 310P and a no-op elsewhere |
| Weight NZ conversion after load | `model_runner_v1.py:2685-2694` converts every `MergedColumnParallelLinear` / `QKVParallelLinear` / `RowParallelLinear` weight | load-time failures on 310P can be format casts, not op bugs |
| Cache NZ conversion | `model_runner_v1.py:2716-2721` (`_convert_torch_format`) called from the KV-init paths (`:2826`, `:3076` for `k_cache`) | see `../runtime-memory/kv-init-and-format-cast-alloc.md` — this frame is an allocation site |
| dtype gate on NZ | `vllm_ascend/utils.py:74-84` — `is_enable_nz()` returns **False for float16/bfloat16**, else `VLLM_ASCEND_ENABLE_NZ` (default 1, `vllm_ascend/envs.py:168-169`) and False for `qwen3_next` | at this pin an fp16 cache is returned unchanged, so a traceback whose deepest frame really executed `torch_npu.npu_format_cast` implies a non-fp16/bf16 cache (int8 / quantised) **or** a tree whose gate differs from this pin. Say which you verified |
| Attention mask NZ | `vllm_ascend/attention/attention_v1.py:253-262` (`nd_to_nz_2d` / `nd_to_nz_spec` + `npu_format_cast`) and further `is_310p()` branches at `:72,347,403,531` | mask-shape/format errors on 310P belong to this branch, not to the model's attention code |
| Separate 310P patch module | added later than this pin: at v0.18.0 `patch/platform/__init__.py:21-26` imports `patch_mamba_config_310` when `is_310p()` (the file does not exist at v0.11.0rc3) | a 310P-only patch can drift independently of the A2 one, and the gate list itself moves between releases — re-run the grep |

Two of these gates combine into the 310P LoRA answer: there is no graph mode
to amortise small launches (300I is eager-only) **and** `PunicaWrapperNPU` binds
the PyTorch `vllm.lora.ops.torch_ops` callables whenever the device type is
310P — at v0.9.2rc1 unconditionally, at v0.13.0rc2 by that explicit predicate
even though the AscendC `vllm_ascend/lora/lora_ops.py` exists in the same tree
(`vllm_ascend/lora/punica_npu.py:27-39`). So "LoRA is 4× slower here" is
structural on this SoC, and the lever is merging the adapter or moving the
release, not a flag — `../lora-adapters/decode-overhead-vs-base-model.md`.

## At v0.22.1rc1: the `_310p/` subpackage, the doc's validated table, two traps

Verified at vllm-ascend **v0.22.1rc1** (`5f6faa0`) + vllm **v0.22.1**
(`0decac0d`) while triaging #12015 (`v0.22.1rc1-310p-openeuler` image); five
distinctive log strings of that paste appear in the pinned tree with line numbers
within ±1 of the log's own `[file:line]` tags (`platform.py:60-61` breakable
cudagraph, `:526` eager-mode default, `:759` allocator injection, `:1036-1037`
GPU-only parameter, `ascend_config.py:772` Dynamic EPLB), so the pin is citable
for that release even though the image is not byte-verified.

- **The 310P code is now a subpackage.** `vllm_ascend/_310p/` holds
  `worker_310p.py`, `model_runner_310p.py`,
  `attention/{attention_v1,attention_mask,metadata_builder}.py`,
  `kv_block_zeroer.py`, `quantization/modelslim_config.py`, `ops/…`. The
  substitutions are `platform.py:651-652` (`worker_cls =
  "vllm_ascend._310p.worker_310p.NPUWorker310"`), `:807-810`
  (`AscendAttentionBackend310`) and `:197-201`
  (`AscendModelSlimConfig310`, which also prints the
  `quantization method 'ascend' … will be overwritten` warning). The v0.11-style
  `worker/model_runner_v1.py` line numbers in the gate table above no longer
  exist at this ref — re-grep inside `_310p/`.
- **The doc's validated table is narrow.** `docs/source/tutorials/hardwares/310p.md:95-176`
  validates only `Eco-Tech/Qwen3-{8B,14B,32B}-w8a8sc-310-vllm` (TP1, TP1, TP4),
  every row carrying `--dtype float16 --quantization ascend --load_format
  sharded_state --no-enable-prefix-caching --max_model_len 16384` (20480 for 32B)
  `--gpu_memory_utilization 0.90 --max_num_seqs`. `:86-87` warns that the
  attention mask is `O(max_model_len^2)` and asks for an explicit, conservative
  `--max-model-len`. A paste serving anything else — a VL checkpoint, an fp16
  dense model, prefix caching left on (the doc passes
  `--no-enable-prefix-caching`; #12015's own config echo says
  `enable_prefix_caching=True`) — is outside the validated set: say so, and make
  "run the doc's row on a small model" rung 1.
- **Trap: the fusion flag in those very doc commands is inert on 310P.** All three
  rows pass `--additional-config '{"ascend_compilation_config":
  {"fuse_norm_quant": false}}'`, but
  `vllm_ascend/compilation/graph_fusion_pass_manager.py:54` and `:69` read that
  key with `and not is_310p()`, so the pass is never added here. The
  `Enabled custom fusions: norm_quant, act_quant` line is **upstream**
  (`vllm/config/compilation.py:313`) and reports
  `compilation_config.pass_config`, which the upstream *optimization level* fills
  (`vllm/config/vllm.py:99-107` `enable_norm_fusion`, the O2 table at `:188-…`,
  default level `:353`, applied at `:1110`) — not the `additional_config` key. Under
  `--enforce-eager` the mode is `NONE` (`platform.py:525-528`) and no pass runs at
  all. So "the flag I set is not what the log lists" is a two-registry artifact,
  not a smoking gun: do not walk it (#12015 spent ~8 calls on it and ran out of
  output before answering).
- **Trap: the 310P mask allocation is real, but it is sized by `max_model_len`.**
  `_310p/attention/metadata_builder.py:80` constructs
  `AttentionMaskBuilder310(self.device, max_model_len)` and
  `_310p/attention/attention_mask.py:68` allocates
  `torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)`
  (class default `max_seqlen = 16384`, `:38`). That is what the doc's `:86-87`
  warning is about, and it is an argument for an explicit `--max-model-len` — never
  a licence to read a **CPU** `torch.zeros` top frame in a segfault dump as this
  allocation (`../hccl-and-hardware/worker-segfault-no-python-frames.md`,
  `## The startup variant`).

## Ranked actions for a 310P startup report

1. Confirm the SoC and the real versions: `python -c "from vllm_ascend import
   _build_info; print(_build_info.__soc_version__)"`, `pip show vllm vllm-ascend`,
   `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`,
   `cat /usr/local/Ascend/driver/version.info`, `npu-smi info`. The image pins
   CANN `8.3.rc2-310p`; a host driver older than that pairing is a real candidate.
2. Apply the 300I environment (`PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`),
   `--enforce-eager`, `--dtype float16` — the tutorial treats all three as
   mandatory, and the reporter usually has only the last two. Write an explicit
   value: `unset` re-enables vllm-ascend's `expandable_segments:True` injection, and
   on 310P (v0.23.0rc1 `-310p` image, CANN 9.1.0-beta.1, torch_npu 2.10.0.post2)
   that value crashed every rank in `torch.npu.empty_cache()` with `unmapHandles …
   507899` while `expandable_segments:False` deployed (reporter-confirmed #12534 —
   `../runtime-memory/allocator-expandable-segments-unmap-crash.md`). Also quote the
   graph budget when TP > 1: `FULL_DECODE_ONLY` is supported but "the number of
   capturable graphs is limited and depends on the model depth … Qwen3-32B can
   capture and replay 2 graphs", with no limit at TP=1
   (`docs/source/tutorials/hardwares/310p.md:20-30` at v0.23.0rc1) — so a
   "2 cards fine, 4 cards stall" report takes `--enforce-eager` and
   `cudagraph_capture_sizes: [1]` before any HCCL theory.
3. Shrink the budget (`--max-model-len`, `--gpu-memory-utilization`) before
   blaming the platform: 310P cards are small and the documented KV/profile
   arithmetic in `../startup-failures/max-model-len-budget.md` applies unchanged.
4. Then the documented-status sentence: v0.11.0rc2-310P is experimental, the
   project itself points at v0.10.0rc1 for 300I, and `#3316` / `#2795` are the
   open known issues the tutorial cites. Say explicitly that this is the
   *documented* state and not a diagnosis of the reporter's host.
5. Only then propose code ownership, and name the gate you think is responsible
   from the table above.

## Limits

- The snapshot carries **one commit** per repo (`mode: one_commit_git`), so
  "which commit introduced this between v0.10.0rc1 and v0.11.0rc2" is
  unanswerable here; say so instead of guessing. `SOURCES.json` also marks the
  reported tag `v0.11.0.rc2-310P-openeuler` as `no_named_release` — the pinned
  reference is v0.11.0rc3.
- Title/command mismatches (7B in the title, 3B in the command) are common in
  this family; ask which one failed before reasoning about memory.
