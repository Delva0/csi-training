# Collective error chain: from a vLLM traceback to the real failing op

## Step 1 — assume the Python frame is wrong

Ascend operators run asynchronously. A collective failure is reported by the
next operation that forces a synchronization (often a device-to-host copy), so
the traceback shows the copy, not the collective. Torch_npu says so explicitly in
the message:

- `RuntimeError: The Inner error is reported as above. The process exits for
  this inner error, and the current working operator name is HcclAllreduce.`
- `Since the operator is called asynchronously, the stacktrace may be inaccurate.
  If you want to get the accurate stacktrace, please set the environment variable
  ASCEND_LAUNCH_BLOCKING=1.`

Recorded instance (vllm-ascend issue #7477, v0.17.0rc1, TP=4, Qwen3.5-27B): the
traceback ended at
`vllm_ascend/ops/mm_encoder_attention.py:108  seq_lens_cpu = torch.diff(cu_seqlens).to("cpu")`
inside the vision encoder, while under blocking mode the same failure landed in
`vllm_ascend/ops/vocab_parallel_embedding.py:192 _forward_origin` →
`maybe_pad_and_reduce` → `tensor_model_parallel_all_reduce`. The vision frame was
an artifact of the sync point. (Both file/line references come from the
reporter's logs against the installed vllm-ascend v0.17.0rc1.)

## Step 2 — re-run to get attribution

```
vllm serve ... --enforce-eager            # removes graph capture as a confounder
ASCEND_LAUNCH_BLOCKING=1 vllm serve ...   # synchronous ops: real failing op/frame
```
Unset `ASCEND_LAUNCH_BLOCKING` after debugging (documented performance penalty).

## Step 3 — read the plog

`~/ascend/log/debug/plog/plog-<pid>_<timestamp>.log` carries the HCCL-side story.
Shapes seen in the recorded case:

```
[ERROR] HCCL(...): [all_reduce_opbase_pipeline.cc:327] [AllReduceOpbasePipeline][Prepare]rank[0]
        streamNotifyMain_ size [2] error, is smaller than,intraRankSize_[4]
[ERROR] HCCL(...): [coll_all_reduce_mesh_opbase_pipeline_executor.cc:116] [KernelRun]call trace: hcclRet -> 4
[ERROR] HCCL(...): [coll_all_reduce_executor.cc:302] ... errNo[0x0000000005000004]kernel run error,
        tag[AllReduce_group_name_3], count[75497472], dataType[11], reduce op type[0]
[ERROR] APP(...): ExecFuncOpApi:452: "[PTA]:Custom hand fail! name=HcclAllreduce, ret=4"
[ERROR] APP(...): ReadQueue:437: "---Thread---...: device = 0, write_idx = 627, read_idx = 602, status = 1, ret = 4"
```

How to read that:

- `tag[AllReduce_group_name_N]` names the process group; check its rank set
  against the devices you exposed (`ASCEND_RT_VISIBLE_DEVICES`) and the
  `device = N` field in the PTA line. A group whose ranks include a faulty card
  fails as a group.
- `streamNotifyMain_ size [2] ... intraRankSize_[4]` is an HCCL-side resource
  inconsistency inside its opbase pipeline executor (notify objects per
  intra-node rank vs the node's rank count). It is not evidence about model
  shapes or which layer issued the collective.
- `errNo 0x...05000004` / `hcclRet -> 4` is CANN's internal error number; the
  human-readable cause string is what you quote, because the numeric meanings are
  version-specific and the pipeline sources are not in the public exports.
- `ERR00100 PTA call acl api failed` (blocking mode) and `ERR01100 OPS call acl
  api failed` (with `HcclAllreduce, error code is 4`) are PTA/runtime-level
  wrappers of the same HCCL failure.

## Step 4 — decide where the fault lives

Collective failures at the very first collective of `profile_run` mean the group
cannot communicate, which is environment-level. Take the failing `device = N` id
and go to `device-health-checks.md` before proposing any model or operator patch.
Also confirm that single-card work on each device still succeeds — HCCS link
faults commonly break only intra-node multi-card collectives.
