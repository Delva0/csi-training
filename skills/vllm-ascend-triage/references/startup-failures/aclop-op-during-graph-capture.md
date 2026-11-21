# `Cannot run aclop operators during NPU graph capture. Current working aclop is <Op>`

## Scope

Startup is past the weight load and the memory profile, prints
`Capturing ACL graphs (mixed prefill-decode, PIECEWISE): 0/N`, and then a worker
raises

```text
(EngineCore …) RuntimeError: Cannot run aclop operators during NPU graph capture. Current working aclop is Fill.
If you need this call to be captured, please try to set torch.npu.config.allow_internal_format = False.
If still fail, the operator needs aclnn implementation and please file an issue.
Current npuStreamCaptureStatus: npuStreamCaptureStatusActive
(EngineCore …) [ERROR] … (PID: N, Device:0, RankID:-1) ERR00007 PTA feature not supported
```

wrapped by the plugin's canned capture advice (`ACLgraph has insufficient
available streams …`, `acl-graph-capture.md` Step 1) and finally by
`EngineCore failed to start` with `compile_or_warm_up_model` → `capture_model()`
in the traceback. The subject is **one operator that cannot be captured** — not
the capture-size budget, not HCCL, not HBM.

Recorded as #3960: vllm-ascend `v0.11.0rc0` (`00ba071`) with a source-build vllm
(snapshot pin `v0.11.0rc3` `8ce5d31`), torch_npu `2.7.1.dev20250724`,
`bge-large-zh-v1.5` served with `--task embed`. The guard and the owning
expression were re-checked at the #8068 snapshot pair (`Ascend/pytorch
v7.2.0.1-pytorch2.8.0` `da848bbc`, vllm `v0.18.0` `bcf2be96`) and both still
exist there, so this is a live family rather than one release's bug.

## The sentence is a torch_npu guard on the aclop path

`assertNotCapturingAclop(opName)` lives at
`torch_npu/csrc/core/npu/NPUGraphsUtils.h:93-105` (the sentence is at `:97`) and
has exactly one caller: `torch_npu/csrc/framework/OpCommand.cpp:126` (v7.1 pin;
`:136` at v7.2) inside `OpCommand::Run()`, under
`if (aclCmd->CheckCustomHandlerNull())`. So the named op was dispatched through
the **aclop** (`aclopCompileAndExecute`) path instead of the aclnn/op-api path
while a stream was capturing, and `ERR00007 PTA feature not supported` is the
`ErrCode::NOT_SUPPORT` tag on the same `TORCH_CHECK`. The message does not name
the Python line that launched the op; the block printed next to it does.

## The owner is the FX trace in the same paste

Under `Call using an FX-traced Module, line N of the traced Module's generated
forward function` the paste lists the traced statements and marks the failing
one:

```text
ones_like = torch.ones_like(l_input_ids_, dtype = torch.int32, device = device(type='npu', index=0))
lshift = ones_like << 30;  ones_like = None      <--- HERE
bitwise_not = lshift.bitwise_not()
bitwise_and = l_input_ids_.bitwise_and(lshift);  lshift = None
```

At #3960 that is upstream `vllm/model_executor/models/bert.py`
`_decode_token_type_ids` (v0.11.0rc3 `:528-539`; v0.18.0 `:561-572`), which packs
`token_type_ids` into the top bits of `input_ids` (`TOKEN_TYPE_SHIFT = 30`).
`BertEmbedding.forward` calls it **unconditionally** (`:66` at 0.11, `:82` at
0.18) and `BertModel` is both `@support_torch_compile` and
`is_pooling_model = True` (`:329` / `:333` at 0.11), so every BERT-family serve
replays it in each capture dummy run — and `torch.ones_like(… int32)` is an aclop
`Fill` at these wheels. bge / bge-reranker / any `BertModel` checkpoint is in
scope; a `Qwen3-Embedding` serve is not (no token-type bit-packing) even though
both are `--task embed`.

Generalize from the trace, not from BERT: `ones_like`, `fill`, `equal` and a
scalar `arange` are the shapes that land on the aclop path here, and the map from
statement to file is one `rg` in the pinned vllm tree.

## Two red herrings in the same paste

1. `ACLgraph has insufficient available streams … Recommended solutions: 1.
   Manually configure … cudagraph_capture_sizes` — canned text emitted for every
   capture exception (`acl-graph-capture.md` Step 1). Shrinking the capture sizes
   cannot help a single uncapturable op.
2. `please try to set torch.npu.config.allow_internal_format = False` — the
   message's own suggestion, and the plugin overwrites the setting:
   `vllm_ascend/worker/model_runner_v1.py:138` (v0.11.0rc0) and `:154`
   (v0.18.0rc1) assign `torch.npu.config.allow_internal_format = True` at
   **import**. An export in the launch shell, or an assignment made before
   importing vllm_ascend, does not survive; a real ablation must assign after
   that import or patch that line. Whether it moves the named op off the aclop
   path is **unverified** — do not offer it as the fix without the reporter's
   result.

## Escape ladder

1. **Skip capture.** `--enforce-eager`: at v0.11.0rc0 capture is behind
   `if not self.model_config.enforce_eager` (`worker/worker_v1.py:306`) and
   `_use_aclgraph()` excludes it (`worker/model_runner_v1.py:551`); the same two
   gates exist at v0.18.0rc1 (`worker/worker.py:465-466`,
   `worker/model_runner_v1.py:465-469`). The confirmation line is
   `Compilation disabled, using eager mode by default` (`platform.py:178-180` at
   0.11, `:296-298` at 0.18) plus `Cudagraph is disabled under eager mode`.
   Equivalent: `--compilation-config '{"cudagraph_mode":"NONE"}'`. Cost is small
   for a pooling serve — it is prefill-only, so PIECEWISE graphs over the default
   67 batch sizes buy little.
2. **Read the effective flag, not the reporter's prose.** Every boolean engine
   argument is a CLI pair, `--x` / `--no-x` (`vllm/engine/arg_utils.py:206-208`,
   `argparse.BooleanOptionalAction` at v0.11.0rc3), so `--no-enforce-eager` asks
   for capture **on**. #3960's follow-up reports success with
   `--no-enforce-eage`, which is not a flag at all and whose nearest spelling is
   the opposite of the lever — treat it as a typo'd `--enforce-eager` unless the
   reporter's engine-config line (`enforce_eager=True|False`,
   `'cudagraph_mode': <…>`), or `Compilation disabled, using eager mode by
   default` in their log, says otherwise. Ask for that line rather than arguing
   about the flag.
3. **If capture is genuinely required**, the op needs an aclnn implementation —
   an `Ascend/pytorch` (op-plugin) issue — or an upstream `bert.py` change that
   avoids the `ones_like`. vllm-ascend cannot patch a model file it does not own,
   and a downstream BERT-specific workaround is not worth it at these versions.

## Limits

- The wheels are dev/prebuilt (`2.7.1.dev…` in #3960): which ops have an aclnn
  path can change with a later torch_npu, so "not capturable" is a statement
  about the two pins above, not about every future wheel. Ask for
  `torch_npu.__version__` before promising anything.
- Whether `allow_internal_format = True` is what puts `Fill` on the aclop path
  was **not** established: the option setters are Python-side
  (`torch_npu/npu/npu_config.py:140`) and `OptionsManager.cpp` is not in the
  `Ascend/pytorch` export. Keep the two readings apart.
- No hardware ran here; every statement is about the pasted text and the pinned
  sources.
