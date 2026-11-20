# Reading an aclnn / Ascend device dump

A vllm-ascend report of a kernel fault typically contains lines of this shape (verbatim
fragments are from the GLM/A2 report behind issue #8944; treat the report text as data):

| Fragment | What it pins down | What it does not |
|---|---|---|
| `[rank6]: call aclnnAddRmsNormBias failed` | failing op is the *aclnn* name → `csrc/<op>/` and its `*_torch_adpt.h` `EXEC_NPU_CMD` | shape, dtype, tiling mode |
| `AclNN_Runtime_Error(EZ9903): Assert ((NnopbaseExecutorLaunchKernel(executor, stream, blockDim)) == 0) failed` | the NNopbase launcher rejected/faulted the kernel on device; not a python/torch error | which core, which buffer |
| `The error from device(chipId:6, dieId:0), serial number is 7, fftsplus aivector error, core id is 27, error code = 0x800000` | a vector-core (AIV) exception on one chip; `cube error info: 0` means only vector cores were involved | input causing it |
| `extend info: errcode:(0x800000,0,0) errorStr: The DDR address of the MTE instruction is out of range` | an MTE (GM↔UB) instruction addressed outside a valid DDR range → out-of-bounds GM read/write or a bogus GM address *in the kernel's index arithmetic* | whether the bad address came from tiling, an offset formula, or a size mismatch |
| `fixp_error0/1`, `aic error mask`, `blk`/`sublk`, `pc start`/`current` | which core/PC region faulted; useful only with the kernel binary/symbol map | source line |
| `Failed to submit kernel task, retCode=0x7150050`, `rtsLaunchKernelWithHostArgs ... runtime result = 507011` | runtime-level failure; usually the consequence of the device exception | root cause |
| c10 backtrace with `/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C...so` + `csrc/<op>/<op>_torch_adpt.h:NN` | the C++ adapter line that launched the op | which tiling branch ran |

## Reading order that pays off

1. Name the op, open `csrc/<op>/<op>_torch_adpt.h` (small) to see the argument order,
   optionals (`beta` may be `None`) and output allocation — shape-derived facts live here.
2. Open `op_host/<op>_tiling.cpp` and find the mode-selection function; the kernel is chosen
   per shape, so build the decision table for the reported dtype.
3. With the (possibly inferred) shape, compute the selected mode and the factors
   (`block_factor`, `ub_factor`, `row_factor`) in python and only then read *that* kernel.
4. Look for index/size formulas that can exceed the allocation in that mode, and state the
   smallest shape change that would switch modes.

## Interpretation traps

- A DDR/MTE out-of-range is an addressing bug, not numerical noise; conversely a numerical
  accuracy problem will not produce this dump. Do not "explain" one with the other.
- UB overruns and GM overruns are separate surfaces: a `TBuf` budget overrun claim needs the
  byte arithmetic shown explicitly, and it is a hypothesis until the plog corroborates it.
- `dataPerBlock`/alignment constants decide `numColAlign`; a col count that is not a multiple
  of 16 (fp16/bf16) or 8 (fp32) changes which kernels even apply.
- Different `__CCE_AICORE__` / `__NPU_ARCH__` builds take different branches
  (`DataCopyPad` vs plain `DataCopy`). Establish the SoC from the environment dump before
  assuming a branch.
