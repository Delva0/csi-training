# A plog `fault kernel_name=` that resolves to a Triton-Ascend JIT kernel

## When this page applies

The plog carries a real device fault — `[DFX_INFO] Aicore kernel execute failed …
fault kernel_name=<name>_0`, `aivec error`, `The DDR address of the MTE instruction
is out of range`, `CheckSmmuFault:Get isSmmuFault is 1`, `retCode=0x31, [vector core
exception]`, `rtEventSynchronize … ErrCode=507035` — and `<name>` is **not** a
`csrc/` AscendC operator and **not** a CANN `aclnn*` op: it is a `@triton.jit`
function in a Python tree. When `<name>` *does* look like a `csrc/` AscendC operator
(or greps to nothing in either Python tree), Step 1b decides which build produced it
before anyone blames a plugin. The vLLM side of the report is then only
`Request <id> failed (engine dead)` plus process teardown.

Two neighbours this page does **not** own: a Triton kernel that fails to *compile*
(`MLIRCompilationError`, `bishengir-compile … non-zero exit status`) is the
installed wheel/compiler pair (`../image-dependency-faults/triton-ascend-pair.md`),
and one that runs and returns wrong values is
`../image-dependency-faults/triton-extension-op-silent-miscompute.md`.

## Step 1 — resolve the name to one file (one call)

The plog drops nothing: `fault kernel_name=token_bin_counts_and_mask_kernel_0` is
the Python function `token_bin_counts_and_mask_kernel` plus Triton's grid index.

```
rg -n 'def <name>\b' /workspace/repos/<case>/vllm-project/vllm-ascend /workspace/repos/<case>/vllm-project/vllm
```

Grep both trees: upstream `vllm` and downstream `vllm_ascend` both port sampling
kernels, and the same behaviour can exist in either. At vllm-ascend v0.20.2rc1
(`367b8e62`) the two penalty/bincount implementations are distinguishable **by the
name alone**, which also says which worker generation produced the paste:

| `fault kernel_name` | implementation | active when (v0.20.2rc1) |
|---|---|---|
| `token_bin_counts_and_mask_kernel` | `vllm_ascend/ops/triton/bincount.py:32`, reached from `ops/triton/penalty.py:96` (`apply_penalties_triton`) ← `sample/penalties.py:25` (`apply_all_penalties`) ← `sample/sampler.py:46` (`AscendSampler.apply_penalties`) | the v1 sampler, i.e. **vllm 0.20.2 and older** — `patch/worker/__init__.py:22-23` sets `_V2_MODEL_RUNNER_SUPPORTED = not vllm_version_is("0.20.2")` under the comment "v2 model runner is only supported on vllm > 0.20.2", so the v2 patches are skipped on exactly that upstream version |
| `_bincount_kernel` | `vllm_ascend/worker/v2/sample/penalties.py:168`, bound by `patch/worker/patch_v2/patch_triton.py:11` | the v2 model runner (upstream newer than 0.20.2) |

A name that greps to **nothing** in either tree is a data gap (rule 10), not a
downstream/upstream claim — say which two trees you searched and at which commits.

## Step 1b — the name looks like a `csrc/` AscendC op: compare the template arguments

An AscendC fault name is the operator plus its template instantiation, so the
**number and order of the flags in the name** identify the build:

- vllm-ascend `v0.7.3` (`779eebb`) `csrc/kernels/pos_encoding_kernels.cpp:303-343`
  declares `rope_custom_<NEOX>_<TYPE>` through
  `ROPE_CUSTOM_KERNEL_TYPE_DECLARE(TYPE, NEOX)` — **one bool** plus the dtype — and
  instantiates it for `half` and `bfloat16_t`, so that tree's own names are
  `rope_custom_true_bfloat16_t` / `rope_custom_false_bfloat16_t`.
- The reported `fault kernel_name=rope_custom_false_false_bfloat16_t_0` carries
  **two** bools. A different template-argument list cannot be that tree's
  instantiation: the kernel came from another build of vllm-ascend (a later kernel
  grew the second flag), from a wheel that ships its own `rope_custom`, or from the
  CANN op package. Say which candidates remain instead of "resolves to a plugin I
  cannot see".
- Reachability decides the owner, and at that ref it points away from vllm-ascend's
  own kernel: `vllm_ascend/ops/rotary_embedding.py:40-53` and `:56-90` call
  **`torch_npu._npu_rotary_embedding`**, while the `csrc/` kernel is only bound as
  `torch.ops._C.rotary_embedding` (`csrc/torch_binding.cpp:32-105`) and needs
  `COMPILE_CUSTOM_KERNELS=1` at install time (`docs/source/user_guide/release_notes.md:33-35`,
  v0.7.3rc2 highlights: custom-op compilation "is disabled by default"). An image
  built with the default therefore has no vllm-ascend `rope_custom` at all.
- One command in the *reporter's* image settles it, because the instantiation list
  lives in the binary, not in any source export:
  `strings $SITE/vllm_ascend/lib*.so $SITE/torch_npu/lib/*.so | grep -o 'rope_custom_[a-z0-9_]*' | sort -u`
  (add the plugin's `.so`). Nothing in a case export can answer it: at #965 the
  `Ascend/op-plugin` view is a 107-file partial with no `rotary_embedding` at all.
- The paste's deepest **Python** frame is usually innocent: #965 dies in
  `vllm_ascend/sample/sampler.py:123` on `logits_sort.softmax`, reported late by
  `AclrtSynchronizeStreamWithTimeout(copy_stream)` — a sync point (`SKILL.md` rule
  11), while the fault kernel named in the same block is the real owner.

## Step 2 — what the plog does and does not pin

- It pins **which kernel** faulted and that the fault was in *addressing*, not
  arithmetic: `aivec`/`vector core exception` + MTE DDR out of range means a
  memory access from the kernel left the mapped window.
- `CheckSmmuFault:Get isSmmuFault is 1` (the runtime's own check, printed as
  `device_error_core_proc.cc:399`) says the access went through the SMMU — i.e.
  the address belonged to a host/IOMMU-mapped region rather than the device heap. That is consistent with a
  pointer to **host** memory (or a freed/recycled host block) reaching a device
  kernel; it is not proof, because the plog does not name the operand.
- The `[AIC_INFO] args(0 to 9) after execute: …` block is a **packed argument
  blob**, not the Python launch order (`totalLen=(10*8), argsSize=80` while the
  kernel takes 10 runtime params of mixed 4/8-byte width plus a constexpr), and
  `blockDim=`, `program id=` are not the Python grid. Do **not** publish "arg0 is
  `tokens_ptr`, so the tokens tensor was on the host" from that line; the packing
  scheme is in the runtime, which the snapshot does not export. Use the pointer
  only as a hint that the addresses in one launch were not all in the same region,
  and label it a hint.
- `program id=430` with `blockDim=2` also does not prove a grid bug: this kernel's
  grid is `(progs, cdiv(total_blocks, progs))` with `SEQ_BLOCK = 256`
  (`bincount.py:127-131`), which legitimately reaches ids in the hundreds for a
  200 K-token prompt.

## Step 3 — the two decoys this shape always carries

1. **The teardown block is not a second defect.** After the device fault the
   EngineCore dies, so the log continues with `Request … failed (engine dead)`,
   `INFO: Waiting for application shutdown`, a run of
   `terminate called without an active exception`, and (on a PD node)
   `!!!!!!! Segfault encountered !!!!!!!` with frames in
   `llm::LlmDatadistTimer::ProcessTimerInContext()` / `TimerThreadLoop`. That is
   the llmdatadist timer thread being torn down inside the dying process — there
   is no KV-transfer decision in it, and a PD deployment does not make it a
   connector bug. Answer it as teardown in one line and move on.
2. **The upstream tensor-construction chain is a producer-site candidate, at
   most.** Reading how `SamplingMetadata.prompt_token_ids` /
   `output_token_ids` are built (e.g. `vllm/v1/worker/gpu_input_batch.py:930-946`
   at vllm 0.20.2 — `[num_reqs, max_prompt_len]` int64, `pin_memory=…`, every
   element past `num_prompt_tokens[i]` set to `self.vocab_size`, then
   `.to(device, non_blocking=True)` at `:845-850`; and
   `vllm_ascend/sample/penalties.py:13-22` doing the same for output ids) tells
   you which tensors the kernel can be handed. It cannot tell you which pointer
   was bad, so it is one labelled hypothesis with a decider, not the answer.

## Step 4 — the ablation ladder (each row is one launch)

1. **Send the same request with penalties at their defaults**
   (`presence_penalty=0`, `frequency_penalty=0`, `repetition_penalty=1`).
   `sample/sampler.py:55-56` returns before `apply_all_penalties` when
   `sampling_metadata.no_penalties`, so the whole chain is skipped: a crash that
   disappears names the penalty/bincount chain as the owner — and this is often
   the reporter's own conclusion anyway, because the bad-case harness is what sets
   those parameters.
2. **Same penalties, a short prompt.** Isolates the batch-width/length regime (a
   max-length prompt sets `max_prompt_len` for every row of the padded tensor)
   from the chain itself.
3. **`additional_config.enable_reduce_sample`** (`ascend_config.py:272`, read at
   `ops/triton/bincount.py:133-137`): with the flag on, `vocab_start_idx =
   tp_rank * vocab_size` and the guard compares *global* ids against a *sharded*
   offset, so the per-rank window moves with TP. It is an `--additional-config`
   key, not an env var — read it out of the serve command. Compare one prompt, not
   a benchmark, because the flag also changes the sampled logits path.
4. **Version.** If the report predates the repair, that is the answer. Recorded
   #10522: the reporter confirmed on the issue that commit
   `f95a31ec266552a73cd8443810884085ae2e3880` fixes it — comment `2026-06-16T08:45Z`,
   about five hours after the report, twelve days after v0.20.2rc1 was published
   (`2026-06-03T15:24Z`) — so a source walk inside the
   v0.20.2rc1 tree cannot find the repair, and "upgrade past the pin" is a
   deliverable, not a shrug.

## Worked case: #10522 (GLM-5.1, PD, vllm-ascend v0.20.2rc1 `367b8e62`)

Two A3 nodes as the P instance, `PP4 TP8 EP32`, a bad-case harness that floods the
service with over-length prompts. The paste contains, in order: the API server
rejecting a 202753-token request (`VLLMValidationError: This model's maximum context
length is 202752 tokens…`, by design), then a request whose `SamplingParams` carry
`presence_penalty=0.5, frequency_penalty=0.5, repetition_penalty=1.1,
max_tokens=1, min_tokens=1` plus the PD proxy's `kv_transfer_params`, then
`failed (engine dead)`, teardown, and the plog block above naming
`token_bin_counts_and_mask_kernel_0` with `chipId:7, dieId:1`.

What the paste therefore decides without any further evidence: the crash is a
device addressing fault inside the **downstream** penalty bincount Triton kernel
(Step 1 table), reached only because that request carries a non-default penalty
set; the 202752-token rejection one line earlier is a different request and not the
cause; the `LlmDatadistTimer` segfault is teardown; `npu-smi` shows 16 dies `OK`,
so no card-health story.

What it does not decide — and where the recorded run spent its budget — is *why*
that address was bad. Read the kernel before ranking: at this pin
`get_token_bin_counts_and_mask_triton` allocates `bin_counts` `[n_rows, vocab_size]`
int32 with the **same** `vocab_size` it passes as the bound
(`ops/triton/penalty.py:107-113`), masks out-of-range and padding lanes with
`token_in_range`, and writes through `safe_local_token` (`bincount.py:84-88`), so a
plain element-range overflow is not reachable from the Python arguments. The
candidate that survives that read is the *address* (`isSmmuFault=1`) handed in by
one of the two pinned-host `.to(device, non_blocking=True)` hops, with the batch
width of ablation row 2 as the length regime that makes it likelier — both labelled
hypotheses, each with the row-1/row-2 decider.

Coverage note, and the reason "the nightly op test passes" cannot refute this
report at this ref: the path's three tests are all skipped —
`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_penalties_triton.py:43`
(`skip("Probabilistic failure, need zengtian after fix")`, the v1↔Ascend
equivalence grid), `triton/test_penality.py:179-182` (the v2 kernel,
`skip(reason="The test case failed and took one hour…")`), and
`triton/test_bincount.py:39` (`skip(reason="atomic_or operator hangs in current
npu_ir version")`, which imports the **v2** kernel, not this one). The project's own
comments say the downstream penalty path is unverified at this pin — quotable as
coverage, never as the cause.

## What to send back / what to state as open

Ask for: the P node's plog window around the named `task_id`; both P nodes' full
serve commands (`--additional-config` included) and whether the harness sets
penalties; `collect_env.py` from the same image; and the answer to the row-1
ablation. State plainly that the mechanism is not derivable from the paste — the
`args` blob does not name the operand — and that the reporter-confirmed fix is a
commit after the pin.

Recheck at another ref: the file/line anchors above are vllm-ascend v0.20.2rc1
(`367b8e62`) and vllm v0.20.2 (`bc150f5`). The kernel *name* is the stable part;
grep it before quoting any line.
