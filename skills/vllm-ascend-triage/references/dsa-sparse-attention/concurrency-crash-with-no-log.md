# "多并发时服务崩溃" with no error text: the config-diff answer

Worked case: #6240, vllm-ascend 0.13.0.rc1, 2×Atlas 800T A2, DeepSeek-V3.2-w8a8-mtp-QuaRot,
two nodes with `--data-parallel-size 2 --data-parallel-size-local 1` + `--tensor-parallel-size 8
--enable-expert-parallel --max-num-seqs 1 --max-model-len 40960 --max-num-batched-tokens 8192
--gpu-memory-utilization 0.96 --speculative-config '{"num_speculative_tokens": 2, "method": "mtp"}'`,
`VLLM_ASCEND_ENABLE_MLAPO=1`, `HCCL_BUFFSIZE=200`, and
`cudagraph_capture_sizes [3,6,…,48]`. Single-request serving is correct (think on/off, accuracy
fine); under load the service dies — 1017-token inputs at concurrency 100, and 8270-token
inputs at concurrency 10. **The report contains no traceback, no plog, no `npu-smi`**, because
both nodes redirect to `output.log` and only the launch commands were pasted.

## What the deliverable is, and when

This is the track where the recorded run burned its whole cap: 18 assistant turns, 28 tool
calls, 37.9K characters of reasoning, the first message text on the last turn, and that answer
cut mid-sentence by the token cap (`../evidence-workspace/delivery-budget.md`, row #6240).
Nothing in the paste can confirm a mechanism, so:

- **3rd assistant turn**: say that no error text was pasted and that the three families
  (engine process exit, device fault, HBM OOM) are indistinguishable without it — that gap is
  the headline finding — then give the config-diff table below plus ranked candidates, each
  with one command.
- **8th turn**: full answer. Two extra source greps are allowed only to correct a named claim.
- Do not build a mechanism out of the model's code path when the paste has no symptom to tie it
  to. The cheap value here is arithmetic and comparison against the project's own configuration,
  not a walk through `sfa_v1.py`.

## The config-diff table (verified at the pinned `v0.13.0rc2`, `a17059a`)

| Item | Reporter | The tree's own validated value | Evidence |
|---|---|---|---|
| MLAPO on a non-PD DP deployment | `1` | e2e sets `0` | `tests/e2e/nightly/single_node/models/test_deepseek_v3_2_exp_w8a8.py:73` `{"HCCL_BUFFSIZE": "1024", "VLLM_ASCEND_ENABLE_MLAPO": "0"}` |
| MLAPO's validated shape | mixed node, `--max-num-batched-tokens 8192`, sizes to 48 | only in the PD sections, with `--max-num-batched-tokens 12`, `cudagraph_capture_sizes [3,6,9,12]`, `--max-num-seqs 4` | `docs/source/tutorials/DeepSeek-V3.2.md:421` (D node) and `:437-440`, `:500`, `:516-520`; the non-PD A2 sections use `--max-num-batched-tokens 32550 --max-num-seqs 64` (`:285-287`) |
| `HCCL_BUFFSIZE` | `200` | tutorial `256` (`:256,330,406,485`), e2e `1024` | same lines |

**The MLAPO gate is the one code fact worth carrying.** It selects on token count, not on
phase, so a 1017-token request takes the decode-designed fused path while an 8270-token prompt's
first chunk (8192) does not:

- `vllm_ascend/attention/sfa_v1.py:763` `if self.enable_mlapo and num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS:`
  → `_sfa_preprocess_decode` (`:667`);
- `MLAPO_MAX_SUPPORTED_TOKENS = 1024` (`vllm_ascend/attention/mla_v1.py:52`).

That both load cases die *on different code paths* is the argument for ranking a cross-request
factor (HBM growth, queueing, a high-load decode-stage defect) above any single kernel.

## Ranked candidates for this shape, each with one check

1. **HBM growth until OOM.** `--gpu-memory-utilization 0.96` plus
   `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` leaves little headroom; sample
   `npu-smi info` during the run and grep the log for
   `NPU out of memory|Memory_Allocation_Failure|EL0004|207001|507013|NPUWorkspaceAllocator`
   (`../runtime-memory/index.md`). Then retry at `0.9` without `expandable_segments`.
2. **MLAPO on a shape the project never validated for this deployment** — the single-variable
   A/B is `VLLM_ASCEND_ENABLE_MLAPO=0` on both nodes, nothing else changed.
3. **`HCCL_BUFFSIZE=200` too small for the MoE dispatch/combine window** — the operators say so
   themselves (`csrc/moe_combine_normal/op_host/moe_combine_normal_tiling.cpp`,
   `csrc/moe_dispatch_normal/op_host/moe_dispatch_normal_tiling.cpp`); grep the log for
   `HCCL_BUFFSIZE is too SMALL, maxBs`, then retry at `1024`.
4. **The known high-load DeepSeek-V3.2 family — quote it with its source class.** #4996 is a
   **maintainer-filed** issue in the prior-month corpus: "DeepSeek-V3.2 may report errors during
   decode stage under high-load scenarios, such as lightning indexer ops error … We will try our
   best to fix it as soon as possible." A maintainer acknowledging the family is stronger
   evidence than a user report that merely looks similar, but it is not a diagnosis of this
   reporter's crash: it names no version, and the snapshot cannot show whether it was fixed.
5. **Excluded by the config, and worth saying so:** the mixed-batch indexer scatter of
   `indexer-cache-scatter-fault.md` needs two requests in one batch, which `--max-num-seqs 1`
   forbids. Saying which mechanism is *ruled out by the launch command* is cheaper than
   listing one more plausible one.

## Two internal contradictions to point out before any load conclusion

- `--max-num-seqs 1` with `num_speculative_tokens: 2` caps a decode step at 1 + 2 = 3 tokens, so
  every `cudagraph_capture_size` above 3 can never be used, and "concurrency 100" is a queue
  served serially, not concurrency. Ask them to re-run at the concurrency they care about
  (`--max-num-seqs 16`, sizes `[3,6,…,48]` = seqs × (1 + num_spec)) — the current numbers do not
  measure what the report claims.
- `HCCL_INTTRA_ROCE_ENABLE=0` is a typo (double `T`) and is ignored. The pair the code itself
  recommends on A2 is `HCCL_INTRA_PCIE_ENABLE=1` with `HCCL_INTRA_ROCE_ENABLE=0`
  (`vllm_ascend/ops/fused_moe/token_dispatcher.py:115-119`). It does not cause a crash, but it
  silently removes the cross-machine traffic constraint the reporter intended.

## Ask for (in this order)

`output.log` tail of both nodes around the first `Traceback`/`ERROR` (not the last line), the
plog covering that minute, `npu-smi info` sampled during the load, and the load script's
concurrency model (`max_tokens`, streaming, retries). One of those converts the ranked list into
a single cause.
