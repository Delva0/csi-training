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

**Lane B, not lane C**: every rank prints its own Python traceback, so classify it as a
worker that raised — the `107027` inside it is the *refusal marker* for code that
synchronised during capture, not a collective/device fault to chase in the plog. A
recorded answer wrote both "this is not a device, link or HCCL fault" and "Classification:
lane C (device-op fault)" in the same message; the second sentence is what a reporter
quotes back when arguing for a hardware explanation.

Sibling lanes: `rtMemcpy:ErrCode=107030` / `MemCopySync` in the same phase is the
*device-bound* copy refusal — `acl-graph-capture.md` Step 2 (on 310P the same code
comes from the fallback splitfuse mask under MTP + `FULL_DECODE_ONLY`:
`../soc-platform-paths/310p-mtp-graph-splitfuse-mask.md`); the opaque
`npuSynchronizeDevice … 507015` with a device dump is the addressing-fault lane —
`acl-graph-capture.md`, v0.21–v0.23 section. A block that carries **both** the
`107030` `aclrtMemcpy` refusal and this CANN sentence (plus
`synchronize stream failed, runtime result = 107027`) is one refused readback, not
two lanes: the blocking copy's own stream sync reports `107027` and the copy reports
`107030` (#15536's 310P paste). There the deepest `vllm_ascend/` frame is a
`.to("cpu")` and the discriminator is `kind = 2` — take the 310P/107030 route above,
not section 6's env lever.

**A `107030` record does not stay in its capture window**: PTA latches the runtime code per
thread (`AclrtPeekAtLastError`, peeked and never cleared) and re-prints it — with its cached
EE1016 description — as `call aclnnX failed, error code is 107030` when an unrelated op fails
hours later, so a report that pairs this block with a runtime `RuntimeError` is a
masked-status diagnosis, not a capture abort. Take
`../hccl-and-hardware/latched-runtime-error-masking.md` before reading the custom-op sources:
the blocking H2D tiling copy that mints the record is in the `csrc/<op>/op_host/` adapters,
and the op-plugin `.cpp` in the reporter's traceback is not in any export.

**Scope gate - this lane needs the CANN sentence, not just the number.** Only
`AclrtSynchronizeStreamWithTimeout(copy_stream)` **plus**
`EE9999 ... Not allow to synchronize captured-stream, stream_id=N` puts a report
here. A bare `RunOpApiV2:.../compiler_depend.ts:N ... AclrtSynchronizeStreamWithTimeout(stream),
error code is 107027` with no CANN sub-message is a stream-synchronize failure
whose owner the frame does not decide: the sync is ordinary op plumbing
(`torch_npu/csrc/framework/OpCommand.cpp:241-242` syncs after every op when
`ASCEND_LAUNCH_BLOCKING=1`; `:210` and `:251` sync unconditionally when the call
passes `sync=true`), so the code reports whatever killed the stream. That lane is
section 6.

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
batching), and only the msmodelslim W8A8 checkpoint produces 107027. That W8A8
directory (`Qwen3-30B-A3B-W8A8-vllm`) is the reporter's own export (msmodelslim
`f8ab35a`, per the user guide's quantization page), not a published quantized release,
so the checkpoint itself — not only the flag — is a variable worth re-exporting. One
launch, one flag, and it splits "the quant method's dispatch syncs" from "the capture
lane or the EP machinery syncs everywhere". Propose it explicitly in the first answer:
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

## 4. What to ask for — and what is already on record

**The launch line is on record; do not ask for it again.** #2473 filed without one, but
the reporter's own comments minutes later supplied both the container and the serve
command, verbatim:

```
docker run -d --net=host --shm-size=8g --name vllm-ascend \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 \
  --device /dev/davinci3 --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info -v /root/.cache:/root/.cache \
  -v /data:/models  vllm-ascend:v0.10.0rc

vllm serve /models/Qwen3-30B-A3B-W8A8-vllm --served-model-name Qwen3-30B-A3B \
  --load-format safetensors --quantization ascend --dtype auto --kv-cache-dtype auto \
  --trust-remote-code --gpu-memory-utilization 0.9 --max-model-len 4096 \
  --max-num-batched-tokens 8192 --max-num-seqs 32 --tensor-parallel-size 8 \
  --enable-expert-parallel --enable-chunked-prefil --enable-prefix-caching
```

(the `--enable-chunked-prefil` spelling is theirs and the run reached capture with it;
the docker tag reads `v0.10.0rc` while the environment dump reports `0.10.0rc1`, so read
the version from `collect_env`, not from the tag.)
That is where §2's flag list comes from — quote it, and **do not** put "the full
`vllm serve` line" in the ask list; a recorded replay did exactly that after listing
the same flags in its diagnosis. What is genuinely missing, and all that is worth
asking for:

- the per-rank log from `Starting to load model` through `PIECEWISE compilation enabled
  on NPU` to the first error;
- `python -c "import torch_npu; print(hasattr(torch_npu,'npu_moe_init_routing_quant'))"`
  on that box (it decides which routing call ran — the pure-Python `init_routing_quant`
  in this paste is itself the answer that the gate was False);
- the clean run's flags, only if it differs from the recorded control above — the
  **raw** `Qwen3-30B-A3B-Instruct-2507` with `--quantization None` and the same
  TP/context/batching flags (§3, including why the W8A8 export is the other variable).

## 5. Version scope

Line numbers are the v0.10.0rc1 pin. Re-grep the three sync sites
(`bincount`, `.to(cpu).numpy()`, `.item()`) and the `hasattr` gate on any newer tag
before reusing them; whether a later release fixed it is not in this snapshot.
Same reading applies to any quant method whose capture path computes a
data-dependent count — the rule is the readback, not this model.

## 6. The bare `107027` (no `copy_stream`, no CANN sentence): the blocking-env lever, both directions

Recorded #2138 - vllm-ascend **v0.9.2rc1** (`b5b7e0ec`), vllm 0.9.2 (`a5dd03c1`),
dense Qwen3-32B, TP=2 on 2x910B3, CANN 8.1.RC1, torch_npu 2.5.1.post1.dev20250619,
`--max-model-len 32768 --max-num-batched-tokens 32768 --max-num-seqs 24
--block-size 128`, graph mode at its default:

```
RuntimeError: Worker failed with error 'RunOpApiV2:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:243
NPU function error: c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream), error code is 107027'
```
raised out of `vllm/v1/executor/multiproc_executor.py:224 get_response`.

**The reporter-confirmed resolution is the opposite of the instinct: setting the env
fixed it.** The original reporter's own later comment, same day: "turns out set the
env `ASCEND_LAUNCH_BLOCKING=1`". So do **not** answer *unset it* on this shape - one
recorded run led with "the trigger is your own `export ASCEND_LAUNCH_BLOCKING=1`" and
spent its ladder telling the reporter to remove the env.

Two source-verified facts that make the direction non-obvious at that pin:
- **No incompatibility guard on the 0.9 line.** `rg -n ASCEND_LAUNCH_BLOCKING
  vllm_ascend/` at `b5b7e0ec` returns only `examples/run_dp_attention_etp16.sh`, and
  `check_ascend_config` (`vllm_ascend/ascend_config.py:126`) validates only the
  torchair/eager combinations; graph mode is default PIECEWISE -> ACL Graph
  (`vllm_ascend/platform.py:154-181`). The
  `ValueError: ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1` refusal
  belongs to **later** refs (`graph-mode-not-applied.md`) - quote it only after
  checking the reported version.
- **The frame does not prove the env was on.** `RunOpApiV2` is `OpCommand.cpp:214`;
  `:241-242` syncs *iff* `CheckBlockingEnable()` (env parsed at
  `register/OptionsManager.cpp:98-101`), while `:210` and `:251` sync
  unconditionally for `sync=true`. The same traceback appears with the env off.

Ladder, cheapest first:
1. **Env off -> turn it on**: `export ASCEND_LAUNCH_BLOCKING=1`, same command. One
   launch, reporter-confirmed to make this serve work; it re-attributes
   asynchronously reported errors and serialises the stream, so report success as a
   *workaround*, not a fix - a defect that vanishes under a device-wide sync is an
   ordering symptom.
2. **Env on and the serve still dies here -> turn it off** (`=0`) and, if capture is
   then what fails, add `--enforce-eager`; that pair is legal on every pin, the env
   alone is not legal with graph mode on newer refs.
3. **Ask which run produced the traceback.** The pasted launch block can already
   carry the env because the reporter edited in the fix (this case's block does), so
   "your env is the cause" is unprovable from the paste. The full worker log
   (`Starting to load model` -> `PIECEWISE compilation enabled on NPU` ->
   `Graph capturing finished` or the first error) plus the plog's
   `[AIC_INFO] ... fault kernel_name=` line decide whether a real device fault sits
   under the wrapper.
