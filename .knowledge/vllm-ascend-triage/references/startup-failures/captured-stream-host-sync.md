# Capture-phase host sync: `Not allow to synchronize captured-stream` (107027)

Scope: engine start reaches graph capture, and the per-rank failure is

```
RuntimeError: operator():.../torch_npu.dir/compiler_depend.ts:47 NPU function error:
  c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 107027
[ERROR] ... ERR00100 PTA call acl api failed.
EE9999: Inner Error!
EE9999: [PID: ...] Not allow to synchronize captured-stream, stream_id=N.
  [FUNC:StreamSynchronize][FILE:api_error.cc][LINE:960]
```

Trailing frames are the capture wrapper, e.g. `vllm_ascend/compilation/piecewise_backend.py:194
__call__` → `output = entry.runnable(*args)` → `torch/fx/graph_module.py:830 call_wrapped`.

Sibling lanes: `rtMemcpy:ErrCode=107030` / `MemCopySync` in the same phase is the
*device-bound* copy refusal — `acl-graph-capture.md` Step 2; the opaque
`npuSynchronizeDevice … 507015` with a device dump is the addressing-fault lane —
`acl-graph-capture.md`, v0.21–v0.23 section.

## 1. What the code means, and what it rules out

ACL refuses to synchronize a stream that is inside an ACL-graph capture. The sync
itself was issued by **code that ran while capture was active**, so the owner is a
data-dependent readback — `.item()`, `.cpu()`/`.numpy()`, `.tolist()`, or
`torch.bincount` (whose output length is `max(minlength, max(x)+1)`) — not the
device, the link or the collective. `ERR00100 PTA call acl api failed` and `EE9999
Inner Error!` are the generic PTA/CANN wrappers for that one refusal; reading them
as a fabric fault sends the session into HCCL and `npu-smi` for nothing. A real
device fault in the same phase carries `fault kernel_name=` in the plog; this one
does not.

The wrapper frame names the *capture*, never the synchronizer: `piecewise_backend.py`
and `graph_module.py` are where the traced module was being executed. Read one
frame further into model code (the `SKILL.md` rule-11 family: where it surfaced is
not its owner).

## 2. Ranked owners — W8A8 MoE + expert parallel (recorded #2473)

Verified at vllm-ascend **v0.10.0rc1** (`4604882a3e2c`) with vllm **v0.10.0**
(`6d8d0a24c02b`): Qwen3-30B-A3B-W8A8 (msmodelslim), TP8, `--quantization ascend`,
`--enable-expert-parallel`, `--max-model-len 4096 --max-num-batched-tokens 8192
--max-num-seqs 32`, chunked prefill + prefix caching, image `v0.10.0rc1`.

1. **The quantized MoE all2all dispatch.** `--enable-expert-parallel` on TP8 gives
   `ep_size 8 < 16`, so `_get_fused_moe_state` returns `FusedMoEState.All2All`
   (`ascend_forward_context.py:25-44`, ep_size read at `:70-73`), and
   `AscendW8A8DynamicFusedMoEMethod.apply`'s `else` sends it to
   `fused_experts_with_all2all` (`quantization/w8a8_dynamic.py:1001`). There
   `expert_map is not None` (EP is on), and the
   `hasattr(torch_npu, "npu_moe_init_routing_quant")` gate (`:417-433`) picks the
   routing call: the device op, or the fallback `init_routing_quant`, which runs
   `torch.bincount(expanded_expert_idx, minlength=global_num_experts)`
   (`:384`), inside `with graph_capture(...)`. Which side of that gate their
   `torch_npu` takes is a one-liner on the reporter's box (below) — owner 2 stands
   either way.
2. **A torch_npu bump is not the fix by itself** (source-verified, same pin): even
   when the `hasattr` branch at `:417-433` is taken, the next statements in the same
   function are host syncs — `token_counts_combined.to(torch.device("cpu"),
   non_blocking=True).numpy()` (`:446-447`) and `all_tokens.item()` (`:448`).
3. **The same family exists on the unquantized tree**, which is why the reporter's
   control needs one follow-up question before owner 1 is adopted: with
   `MOE_ALL2ALL_BUFFER` at its default `0` (`envs.py:110-114`) the unquantized EP
   all2all path (`ops/fused_moe.py:1135` → `fused_experts_with_all2all`) also runs
   `torch.bincount` (`:369`) and `.cpu().tolist()` (`:378-379`). So a clean bf16 run
   *with EP on* contradicts "bincount owns it"; a clean bf16 run with ep_size 1
   (AllGather branch, `w8a8_dynamic.py:985` / `ops/fused_moe.py:1125`) explains it.
   Ask which flags that run used instead of picking the comfortable reading.

## 3. The quantization axis is the cheapest discriminator

Recorded #2473, reporter-confirmed (their own later comment, 20 min after filing):
the unquantized `Qwen3-30B-A3B-Instruct-2507` serves clean on the same box with the
same flags (`--quantization None`, `--enable-expert-parallel`, identical TP/context/
batching), and only the msmodelslim W8A8 checkpoint produces 107027. One launch,
one flag, and it splits "the quant method's dispatch syncs" from "the capture lane
or the EP machinery syncs everywhere". Propose it explicitly in the first answer:
reporters often have both checkpoints on the box and will run it themselves.

Ladder, cheapest first:

1. `--enforce-eager`, identical command — `use_aclgraph` is
   `level == PIECEWISE and not enforce_eager` (`worker/model_runner_v1.py:335-336`),
   so capture is skipped and the log prints `Skipping NPU graph capture for eager
   mode.` (`:2432`). Serving, slower; it proves the phase and gives a working
   deployment.
2. Keep `--quantization ascend`, drop `--enable-expert-parallel` (TP8 only):
   ep_size 1 → `AllGather` → `fused_experts` (`w8a8_dynamic.py:985`), which has no
   `bincount` and no `.item()`. This is the only combination in this family that
   keeps ACL graphs.
3. If EP is required: stay eager and file the capture-safe-op request — a device-side
   expert count (`npu_moe_init_routing_quant(..., expert_tokens_num_mode=2)`, already
   used at `w8a8_dynamic.py:421-428` on newer torch_npu) plus fixed-size buffers for
   the D2H counts.

`MOE_ALL2ALL_BUFFER=1` is **not** an ablation for the quantized deployment: the W8A8
`apply` has no such branch (`rg -n 'MOE_ALL2ALL_BUFFER' vllm_ascend/quantization/`
is empty at this pin), so it changes nothing there.

## 4. What to ask for

The full `vllm serve` line (recorded #2473 filed without it; `--enable-expert-parallel`
had to be inferred from the code path — which the later comment confirmed), the
per-rank log from `Starting to load model` through `PIECEWISE compilation enabled on
NPU` to the first error, `python -c "import torch_npu;
print(hasattr(torch_npu,'npu_moe_init_routing_quant'))"` on their box, and the flags of
any clean run they compare against (the quantization axis above).

## 5. Version scope

Line numbers are the v0.10.0rc1 pin. Re-grep the three sync sites
(`bincount`, `.to(cpu).numpy()`, `.item()`) and the `hasattr` gate on any newer tag
before reusing them; whether a later release fixed it is not in this snapshot.
Same reading applies to any quant method whose capture path computes a
data-dependent count — the rule is the readback, not this model.
