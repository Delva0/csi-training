# 310P: MTP + `FULL_DECODE_ONLY` capture dies in the splitfuse mask (`107030`)

Scope: Atlas 300I Duo (310P) engine start prints
`Capturing CUDA graphs (decode, FULL)` and one rank dies **inside** the capture,
with three error layers stacked in one block:

```
RuntimeError: copy_between_host_and_device_opapi:.../CopyKernelOpApi.cpp:55
  NPU function error: aclrtMemcpy, error code is 107030
[ERROR] ... ERR00100 PTA call acl api failed.
[PID] Not_Supported(EE1006): StreamSynchronize execution failed because stream 27
  during the capture stage is not supported.
  Possible Cause: 1.The current CANN software version is not supported. ...
  Solution: 1.Upgrade the CANN software version.
rtMemcpy execution failed, reason=operation not permitted when a stream is
  capturing and the specified capture mode is not relaxed ... mode ... is 0
synchronized memcpy failed, kind = 2, runtime result = 107030
```

Deepest frames: `_310p/attention/attention_v1.py:293 forward_chunked_prefill_310`
→ `_310p/attention/attention_mask.py:90 get_splitfuse_mask` →
`attn_metadata.query_start_loc.to("cpu", dtype=torch.int32)`. Recorded #11308,
verified at vllm-ascend **v0.21.0rc1** (`80610e44`) + vllm **v0.21.0** (`ad7125a`),
300I Duo, Qwen3.5-35B-A3B-w8a8 + MTP, hand-upgraded CANN 9.1.0.beta3.

## The verdict, and why the upgrade advice is not it

1. The `Upgrade the CANN/driver` block is `EE1006`'s canned text; it prints for any
   capture-stage refusal (`SKILL.md` rule 11). The reporter had already installed
   CANN 9.1.0.beta3 by hand and nothing changed — say so in the first sentence.
2. `get_splitfuse_mask` is a host readback **inside** capture: `:90`
   `query_start_loc.to("cpu")` and `:93` `seq_lens.to("cpu")`. ACL graphs are
   entered with `aclmdlRICaptureBegin` mode 0, so `MemCopySync` is refused
   (`107030`); only RELAXED mode would allow it, and vllm-ascend does not set it.
3. The readback only runs on the **fallback** branch.
   `_310p/attention/attention_v1.py:110` sets
   `support_compressed_mask = is_compressed_mask_supported()`; `:273` is the
   compressed branch — a fixed `[2048,2048]` fp16 mask built once and cached, no
   host read (`attention_mask.py:105-128`) — and it `return`s; `:293` is the
   fallback that calls `get_splitfuse_mask`. A traceback at `:293` therefore
   *proves* `is_compressed_mask_supported()` was False on that box.
4. That gate is a **torch_npu binding** check, not a toolkit one
   (`attention_mask.py:28-29`):
   `hasattr(torch_npu, "_npu_flash_attention_v3") and
   hasattr(torch_npu, "_npu_paged_attention_splitfuse_v2")`. Installing CANN
   ops/nnal/toolkit packages never flips those attributes, so "0.21.0 needs CANN
   9.1.0beta3" is half the story: kernel side yes, Python side is the torch_npu
   (op-plugin) build.
5. `synchronized memcpy failed, kind = 2` is the one-glance discriminator: ACL kind 1 is
   host→device, kind 2 is device→host, so the refusal is a **readback** (`.to("cpu")`,
   `.item()`) issued inside capture, matching the `.to("cpu")` frame below it. Read the
   kind before theorising about loads or the driver.
6. **MTP is what makes it fatal**, which explains "drop `--speculative-config`
   and it starts". `_310p/model_runner_310p.py:599-607` computes
   `mtp_spec_dummy_capture = uniform_decode and not is_profile and
   speculative_config is not None and method == "mtp" and not use_mla`, and
   `:191-193` then forces `attn_state = AscendAttentionState.SpecDecoding` for the
   dummy capture run. Without MTP the same capture is `DecodeOnly` and goes
   `forward_paged_attention`, which never builds the splitfuse mask.

## Answer shape (owed on the first message)

Verdict line: not CANN, not the driver — a capture-unsafe fallback mask plus the
torch_npu v2-op binding. Then the one command, on the reporter's box (~12 s):

```bash
python -c "import torch_npu; print(hasattr(torch_npu,'_npu_flash_attention_v3'), hasattr(torch_npu,'_npu_paged_attention_splitfuse_v2'))"
```

Then the working combinations on 310P: eager + MTP (verified by the reporter, no
decode graph), no MTP + `FULL_DECODE_ONLY` (graph, no spec gain), and the two ways
to have both — a torch_npu build carrying both `_npu_*` ops, or patch the fallback
so the mask is precomputed outside capture (the metadata builder already carries
`query_start_loc_cpu`; `metadata_builder.py` `set_query_lens_cpu` is the pattern).

Eager working is *confirmation*, not a workaround to be explained away: the same
`.to("cpu")` outside capture is an ordinary D2H, which is exactly why the split
between eager and graph reproduces.

## Coverage gap that is quotable (and is the bug report)

At v0.21.0rc1 the 310P tutorial states `FULL_DECODE_ONLY` is supported
(`docs/source/tutorials/hardwares/310p.md:20-22`) and, elsewhere, that the
chunked-prefill path has "no compressed-mask path" (`:83`) — i.e. the documented
310P path *is* the capture-unsafe fallback. The 310P UTs force the flag off
(`tests/ut/_310p/attention/test_attention_v1_310.py:97,144,180`,
`test_attention_mask_310.py:32`), so the compressed branch has no coverage, and no
`tests/e2e` case pairs 310P with MTP + `FULL_DECODE_ONLY`. Title suggestion:
`[Bug][310p] FULL_DECODE_ONLY + MTP crash 107030: fallback splitfuse mask does
host readback inside capture`.

## Not local, and version scope

Whether a given wheel carries the two `_npu_*` ops is decided only by the
`hasattr` line: the snapshot's `Ascend/pytorch` export is truncated and has no
`splitfuse` hit, so do not assert it either way. The `107030` family on A2/A3
(`MemCopySync` from a framework-side blocking copy) is the capture-mode copy lane
in `../startup-failures/acl-graph-capture.md` Step 2; the `107027` stream-sync
sibling is `../startup-failures/captured-stream-host-sync.md`. Re-grep
`is_compressed_mask_supported`, the `.to("cpu")` sites and the
`_mtp_spec_dummy_capture` gate on any other tag — all three are version-specific.
