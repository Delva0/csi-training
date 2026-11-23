# DP group gloo `Connection closed by peer`: engine mode first, then lockstep

Worked for #541 (vllm-ascend v0.7.3rc1 / vllm v0.7.3, 2×910B2, one host).

## The shape

The report is an **offline data-parallel demo**: one `multiprocessing.Process` per
DP rank, each setting `VLLM_DP_RANK`, `VLLM_DP_RANK_LOCAL`, `VLLM_DP_SIZE`,
`VLLM_DP_MASTER_IP`, `VLLM_DP_MASTER_PORT`, `ASCEND_RT_VISIBLE_DEVICES`, then
`LLM(..., device=f"npu:{local_dp_rank}")` and `llm.generate(prompts, sampling_params)`.
Sometimes the title says "dp attention failed when use hccl backend in dist.all_reduce"
and asks for an HCCL fix. The failure text is

```
File ".../vllm_ascend/worker/model_runner.py", line 1136, in execute_model
  with set_forward_context(model_input.attn_metadata, ...):
File "/workspace/vllm/vllm/forward_context.py", line 92, in set_forward_context
  dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534]
  Connection closed by peer [127.0.0.1]:<port>
```

with `parallel_state.py … Adjusting world_size=2 rank=0 … for DP` and
`rank 0 in world size 2 is assigned as DP rank 0, PP rank 0, TP rank 0` earlier.

## Step 0 — the engine banner decides it, not the collective

Read the banner the paste already carries: `Initializing a V0 LLM engine (v0.7.3)`
versus `Initializing a V1 LLM engine`. **External multiprocess DP
("multiprocess_mode") is a V1-engine feature**: reporter-confirmed on #541
(2025-04-18) — "经过调试，demo可以运行，但 multiprocess_mode推理只在V1版本支持".
Asking whether the CPU group should use HCCL, exporting `GLOO_SOCKET_IFNAME`, or
raising a timeout all miss the point; a V0 build has no DP coordinator to keep
ranks in step.

Then confirm the feature exists in the pair the report names, because the
traceback frequently comes from a *different* tree than the version block:

```bash
# vllm v0.7.3 = ed6e907 (the ref these reports pin)
grep -rn "get_dp_group\|VLLM_DP_RANK\|data_parallel_size" --include=*.py vllm/   # empty
wc -l vllm/forward_context.py            # 101 lines, no all_reduce; :92 is inside the
                                         # batchsize-forward-time stats block
grep -rn "for DP\|dp_rank" vllm/distributed/parallel_state.py                   # empty
ls examples/offline_inference/ | grep -i parallel                               # empty
# vllm-ascend v0.7.3rc1/rc2 = f025df05 / 00459aea
grep -rn "dp_group\|data_parallel\|DP_RANK" --include=*.py vllm_ascend/         # empty
```

So at this pair `forward_context.py:92` **cannot** be the reported line: the
reporter's `/workspace/vllm` is newer than its own `0.7.3` version string or is
locally patched, while the installed `vllm_ascend` does match the pin
(`worker/model_runner.py:1136` calls the 3-argument
`set_forward_context(attn_metadata, vllm_config, virtual_engine)`). Say that in the
answer: the two halves of the paste fingerprint to different generations, so the
first ask is `git -C <vllm> log -1 --oneline` plus
`python -c "import vllm, vllm_ascend; print(vllm.__version__, vllm.__file__)"`,
not a backend experiment. Deliver this verdict plus the V1 route in the first
message — the recorded run spent its whole 16 K budget reconstructing the version
correspondence and shipped no answer.

Supporting evidence that this pair predates DP: `docs/source` at v0.7.3rc1 has no
data-parallel recipe at all, and `docs/source/tutorials/multi_node.md:42,51`
exports `GLOO_SOCKET_IFNAME` only for multi-node `torch.distributed` — which is
why the `Unable to resolve hostname … using the loopback address` warning in the
paste is noise on a single node.

## Step 1 — once DP exists, the gloo text names a *peer that left*

`num_tokens_tensor` is a CPU int32 tensor, so the all_reduce is deliberately on the
gloo `cpu_group`, which `GroupCoordinator` creates beside the device group
(`vllm/distributed/parallel_state.py:180-183`,
`cpu_group = torch.distributed.new_group(ranks, backend="gloo")`). DP attention
steps every rank in lockstep and blocks there on every forward, so
`Connection closed by peer` means the peer's process closed the socket — not that
the NIC, HCCL or `GLOO_SOCKET_IFNAME` failed.

Two lockstep breakers are visible in the demo script itself, so check them before
reading any vllm-ascend code:

- a **per-rank budget**: `max_tokens=[16, 20][global_dp_rank % 2]` is copied from
  the upstream example; the rank with fewer tokens returns from `generate()` first;
- a **process exit / init skew**: rank 0 prints `DP rank 0, Prompt: …` after
  `Processed prompts: 100% … 2/2`, `sleep(1)`, `main()` returns, and the
  `multiprocessing` child ends — while rank 1's bar still reads `0/2`; the two
  `init engine … took 14.98 seconds` / `17.27 seconds` lines show the init skew.

Same error text as `worker-segfault-no-python-frames.md`'s messenger, opposite
owner. Discriminate with the peer's own log block, not with the collective:

- peer printed its outputs and `Processed prompts: 100%` before the traceback → a
  **finished** rank, owner is the launch shape (this page);
- peer shows `Worker proc … died unexpectedly (exit code: None)`, a sigquit dump or
  a negative core-proc value → a **killed** rank, go to that page;
- no peer log at all → data gap; ask for the full per-rank log before theorising.

Fix for the launch shape: identical per-rank `max_tokens`, both ranks in the stream
until the end (no early exit, no `sleep`-then-exit), or move to the V1 engine /
in-process DP where the ranks are scheduled together.

## How to read a "works after debugging" follow-up

#541's later public log shows the demo running with V0 banners, but its
`parallel_state.py` prints `test current_platform.device_name`,
`config.parallel_config.data_parallel_size is 2` and
`init_distributed_environment start adjust.....` — debug lines that exist in no
release of this pair. Such a run is a **patched** tree: it shows the mechanism is
patchable, not that the V0 path is supported. Keep the verdict on the V1 engine and
say the working screenshot came from unreleased local changes.

## Evidence table (all re-read at the pinned commits)

| Claim | Anchor |
|---|---|
| no DP anywhere in upstream 0.7.3 | `vllm/forward_context.py` (101 lines, no `all_reduce`), `vllm/distributed/parallel_state.py` (no `for DP`/`dp_rank`), no `examples/offline_inference/data_parallel.py` |
| no DP in vllm-ascend 0.7.3rc1 / rc2 | `grep -rn "dp_group\|data_parallel\|DP_RANK" vllm_ascend/` → empty |
| installed vllm_ascend matches the pin | `vllm_ascend/worker/model_runner.py:1136` → `set_forward_context(attn_metadata, vllm_config, virtual_engine)` |
| the CPU group is gloo by construction | `vllm/distributed/parallel_state.py:180-183` |
| the DP all_reduce is a CPU-side lockstep barrier | `vllm/forward_context.py:92` in the reporter's tree only (not in the release) |

## Open

The snapshot holds `vllm v0.7.3` + `vllm-ascend v0.7.3rc1/rc2` only, so the exact
V1 release where multiprocess DP starts working is **reporter-stated, not
source-verified**; #541's own patch, and whether the later log's binary was a
release or the reporter's build, are unverified. Do not promote V0, and do not rank
an HCCL/driver cause on this text.
