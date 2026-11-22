# Torchair graph enabled: `IndexError: index 0 is out of bounds for dimension 0 with size 0` at `torchair_attention.py`'s `kv_cache[0]` in the memory-profiling dummy run

Lane B, every rank in the same second, in the **memory-profiling** window (weights
already loaded, no `Capturing torchair graph` line, no capture, no request), on a serve
whose `--additional-config` carries `{"torchair_graph_config": {"enabled": true}}`:

```text
(Worker_DP3_TP0_EP12 pid=27270) ERROR ... [multiproc_executor.py:671] Traceback (most recent call last):
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...   File ".../vllm_ascend/torchair/torchair_worker.py", line 34, in determine_available_memory
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...     available_kv_cache_memory = super().determine_available_memory()
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...   File ".../vllm_ascend/worker/worker_v1.py", line 227, in determine_available_memory
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...     self.model_runner.profile_run()
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...   File ".../vllm_ascend/worker/model_runner_v1.py", line 2530, in profile_run
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...   File ".../vllm_ascend/torchair/torchair_model_runner.py", line 195, in _generate_dummy_run_hidden_states
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...     hidden_states = super()._generate_dummy_run_hidden_states(
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...   File ".../vllm_ascend/worker/model_runner_v1.py", line 2314, in _generate_dummy_run_hidden_states
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...   File ".../vllm_ascend/torchair/models/qwen3_moe.py", line 362, in forward
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...   File ".../vllm_ascend/torchair/models/qwen3_moe.py", line 260, in forward
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...   File ".../vllm_ascend/patch/worker/patch_attention_layer.py", line 67, in forward
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...   File ".../vllm_ascend/torchair/torchair_attention.py", line 365, in forward
(Worker_DP3_TP0_EP12 pid=27270) ERROR ...     if kv_cache is not None and kv_cache[0].numel() > 0:
(Worker_DP3_TP0_EP12 pid=27270) ERROR ... IndexError: index 0 is out of bounds for dimension 0 with size 0
```

Recorded #4847 (image `quay.io/ascend/vllm-ascend:v0.11.0rc2` = pin `a2e4c3f`,
vllm v0.11.0 `b8b302cd`; Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot, two nodes of
8 cards, `--data-parallel-size 4 --data-parallel-size-local 2` + `--tensor-parallel-size
4` + `--enable-expert-parallel`, `--max-model-len 8192`, `--gpu-memory-utilization
0.95`). The reporter's own stack matches the pin line for line
(`torchair_worker.py:34`, `worker_v1.py:227`, `model_runner_v1.py:2530`, `:2314`,
`torchair_model_runner.py:195`, `qwen3_moe.py:362`/`:260`,
`patch_attention_layer.py:67`, `torchair_attention.py:365`), so quote the pin's numbers
as their image's numbers too.

## Three tokens, then answer

1. **The message wording names the Python object type.** `index 0 is out of bounds for
   dimension 0 with size 0` is a *tensor* index error; indexing an empty **list** says
   `list index out of range`. So `kv_cache` here is the upstream per-layer
   **placeholder** `torch.tensor([])` that the layer fills in at init (`vllm/attention/
   layer.py:230-233` at vllm v0.11.0: "use a placeholder kv cache tensor during init,
   which will be replaced by bind_kv_cache"), not a real pool and not a `[]` list.
   Nothing about the KV budget, block count, card, collective or checkpoint can be the
   cause of an index into a placeholder.
2. **The phase is peak-memory profiling.** `determine_available_memory` →
   `profile_run` → `_dummy_run(self.max_num_tokens, with_prefill=True)`
   (`worker_v1.py:219-231`, `model_runner_v1.py:2526-2531`). Both
   `torchair_worker.py` and `torchair_model_runner.py` in the stack mean the **torchair**
   worker/model-runner pair was selected — i.e. the `torchair_graph_config` key reached
   the platform. This is not the ACL-graph capture lane, not `compile_or_warm_up_model`,
   and not a compile/dynamo failure.
3. **The guard asymmetry is the defect.** The same `forward` tests
   `kv_cache is not None and len(kv_cache) > 0 and kv_cache[0].numel() > 0` at `:335-336`
   (the KV-quant probe) and only `kv_cache is not None and kv_cache[0].numel() > 0` at
   `:365` (`torchair_attention.py` at v0.11.0rc2). The second index has no length check,
   so the size-0 placeholder raises.

## Why only the torchair start reaches `:365` (verified at the two pins)

- `_dummy_run` hard-codes the metadata builder's `with_prefill` to **False** even when
  the run it is preparing is a prefill dummy run: `model_runner_v1.py:2457-2464`
  `self._build_dummy_attn_metadata(False, num_reqs=…, aclgraph_runtime_mode=…,
  force_attention=…)`, called from `profile_run`'s `_dummy_run(..., with_prefill=True)`
  (`:2526-2531`).
- The generic builder then returns **None** (`model_runner_v1.py:2244-2255`: the local
  starts as `None` and is only filled under `force_attention or runtime mode == FULL`),
  and the generic impl returns before any kv-indexing at `attention_v1.py:615-616`
  (`if attn_metadata is None: return output…fill_(0)`). Graph-off is therefore safe by
  an early return, not by a different kv_cache.
- The torchair override takes its **else** branch for `with_prefill=False` and returns a
  real decode metadata from `build_torchair_graph_dummy`
  (`torchair_model_runner.py:160-186`, `attn_state=DecodeOnly` at
  `torchair_attention.py:163-171`). With `attn_metadata` non-None, the same dummy
  forward passes the `attn_metadata is None` early return (`torchair_attention.py:352`)
  and indexes the placeholder at `:365`.

So the mechanism is a *torchair-only* metadata/forward mismatch during profiling; the
reporter's DP/EP/TP arithmetic is not needed to explain it, and no per-card, HCCL or
checkpoint fact is involved.

## Ladder

0. **Diff the pasted `docker run` against the pin's own block first.** A container whose
   mount list is missing a line from `docs/source/quick_start.md` (`:30`, `:59` at
   v0.11.0rc2 carry `-v /usr/local/dcmi:/usr/local/dcmi`) fails **earlier**, at weight
   load, and the mount has to be added before this crash is even reachable — in the
   recorded case the reporter added it and the start then moved here
   (reporter-confirmed, one variable). A responding `npu-smi info` inside the container
   does not clear the mount list → `no-npu-in-container.md` §*A responding `npu-smi
   info` is not a mount check*.
1. **Remove the `torchair_graph_config` key from `--additional-config` and relaunch
   otherwise unchanged.** Reporter-confirmed in #4847: 「最后起服务时取消掉
   `--additional-config '{"torchair_graph_config":{"enabled":true}}'` 这行参数，就成功拉起来了」.
   Do not add `--enforce-eager` beside the key — `check_ascend_config` raises
   `Can't enable graph mode and eager mode at the same time…` (`ascend_config.py:277-286`),
   so "keep the key and add eager" is one launch that refuses to start.
2. **If graph mode is required, use the configuration the pin documents for it.**
   `docs/source/user_guide/feature_guide/graph_mode.md:39` scopes TorchAirGraph to the
   DeepSeek series and `:46` states "TorchAirGraph is only work without chunked-prefill
   now" — and the paste's own engine line says `chunked_prefill_enabled=True` (upstream
   turns that on by default at these `--max-num-batched-tokens`/`--max-num-seqs` values even
   though no `--enable-chunked-prefill` appears in the command), so the served combination
   is not the documented one. The *model* gate is not the blocker: `TORCHAIR_MODEL_LIST
   = ["deepseek", "pangu", "kimi_k2", "qwen"]` is matched by substring
   (`ascend_config.py:20-27`), so `qwen3_moe` passes and the failure is downstream of
   it — say that explicitly, because "torchair is for DeepSeek" is *not* the error the
   user saw. Note also that `enable_shared_expert_dp` is force-disabled under torchair
   (`:62-72`, `:298-301`), so that key cannot be the lever either.
3. **A source fix exists and is small** — add the `len(kv_cache) > 0` term at `:365`
   (mirroring `:335-336`), or make the torchair dummy-metadata override return None when
   the forward it will feed is the prefill dummy. Label it a proposal: a fix landing
   after the pin is invisible in the snapshot, so never say "fixed in the next rc".
4. `torchair_graph_config.use_cached_kv_cache_bytes` and cached `kv_cache_bytes` files
   are not rungs: the crash precedes any KV sizing (`torchair_worker.py:31-54` runs
   *after* `super().determine_available_memory()`).

## What this is not

- Not `NPU out of memory`, not a `--max-model-len`/KV-block budget: the phrase is an
  index into a placeholder and no block was ever allocated.
- Not the `E89999` GE infershape or the `DelayGraphBreakVariable` dynamo lanes — both
  arrive after `Capturing torchair graph, this usually takes 1.0~3.0 mins.` at
  `compile_or_warm_up_model` (`torchair-ge-infershape-rejection.md`,
  `torchair-dynamo-compile-unsupported.md`); this paste has neither line.
- Not a GE/`cann` question and not a missing export: the failing frame is Python inside
  `vllm_ascend/torchair/`, so `cann/ge` and the `torch_npu/dynamo/torchair` package
  being outside the snapshot changes nothing.

## Deliver

Headline: graph mode is the variable, this is a torchair-backend guard reading the
unbound placeholder during memory profiling, and one launch without the
`torchair_graph_config` key is the next action (reporter-confirmed). Then the ladder,
the evidence table with the pin's `path:line`, and the ask: both nodes' serve commands
unchanged, the full per-rank block (this one is `logger.exception` under
`[multiproc_executor.py:671]`, so the stack is below the message, not above it), the
`docker run` of both containers, and whether a non-graph serve of the same checkpoint
works (that is the control that closes the checkpoint/DP question).

## Scope and limits

Verified at vllm-ascend v0.11.0rc2 (`a2e4c3f`) + vllm v0.11.0 (`b8b302cd`) only: the
guard text and asymmetry, the hard-coded `with_prefill=False` at the metadata call, both
dummy-metadata builders, the model gate list, the eager refusal, and the graph-mode doc
lines. Not verified: whether a later release adds the length check (one grep at the
reporter's ref settles it), and which dispatch branch (`qwen3_moe.py:249` impl call vs
`:260` `self.attn(q, k, v)`) a given rank takes — the recorded frame is `:260`, both
reach `:365`, and `_sync_metadata_across_dp` all-reduces the prefill flag across the DP
group (`torchair_model_runner.py:121-158`), so with DP>1 the flag a rank sees is the
group's and the branch is not derivable from the paste alone.
