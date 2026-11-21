# The log stops at `Using OOT custom backend for compilation`

Scope: every rank's last line is `INFO [compilation.py:NNNN] Using OOT custom
backend for compilation.`, nothing is raised, the HTTP port is still refused, and
the reporter adds that HBM is already occupied ("模型显存已经占用起来"). Typical
shape in the recorded case: a single node, TP=8 MoE with
`--enable-expert-parallel`, `FULL_DECODE_ONLY`, quantized weights (`--quantization
ascend`), often MTP.

The banner names "compilation" and it sits in the window where a compile stage
would sit, so it reads as *weights are down, now it is compiling*. At these pins
it is the opposite marker: it is printed **while the model object is being
built**, before the first checkpoint shard is read. Ranked as a compile stall, the
answer goes to graph capture, npugraph_ex compile time and capture sizes, and the
load phase — where the process actually is — is never examined.

Not this topic: a paste that ends at `Loading model weights took X GB`
(`silent-stall-after-weight-load.md`), or with a partial shard bar
(`weight-load-stall-in-shard-loop.md` — its phase is the same, its checks are the
ones to take from here), or any traceback (`index.md`).

Verified at vllm-ascend **v0.22.1rc1** (`5f6faa0`) + vllm **v0.22.1** (`0decac0`),
the pins of recorded #12693. Re-grep per version: `@support_torch_compile` is old,
but the line numbers and the `do_not_compile` gate are not frozen.

## Step 1 — the banner is a model-construction marker, not a capture marker

| Order | Site (this ref) | Meaning |
|---|---|---|
| 1 | `vllm/config/compilation.py:1066` | the string, inside `CompilationConfig.init_backend` |
| 2 | `vllm/compilation/wrapper.py:90` | the call, inside `TorchCompileWithNoGuardsWrapper.__init__` (`:72`) |
| 3 | `vllm/compilation/decorators.py:406` | `@support_torch_compile` splices that `__init__` into the decorated **model class**; `:390` `do_not_compile` returns *before* the call when `mode ∈ {NONE, STOCK_TORCH_COMPILE}` |
| 4 | `vllm/model_executor/model_loader/base_loader.py:55` then `:64` | `initialize_model(...)` (this is where the model `__init__`, i.e. the banner, runs) and only then `self.load_weights(...)` — the first shard is opened **after** the banner |
| 5 | `vllm_ascend/worker/model_runner_v1.py:3602` / `:3613` / `:3633` / `:3641` | `get_model(...)` → `Loading drafter model...` (spec decode only) → `Loading model weights took %.4f GB` → `ACLGraphWrapper(...)` |

So: banner printed ⇒ that class's construction finished ⇒ the checkpoint read for
that model is the phase that just started. The next line a healthy run shows is
the loader's (`Loading safetensors checkpoint shards: …` /
`Loading model weights took`), not a capture bar.

**Two different wrappers, one line apart.** The banner's caller is the *per-model*
compile wrapper, not the capture wrapper. `vllm_ascend.compilation.acl_graph.
ACLGraphWrapper` is constructed at `:3641` (and asserts `runtime_mode != NONE`
at `acl_graph.py:117`); `rg -n 'init_backend|VllmBackend' vllm_ascend/` returns
**nothing** at this ref, so that wrapper cannot be the banner's emitter. Reading the
`ACLGraphWrapper` line as "the banner is emitted at the end of `load_model`"
inverts the phase; it is the mistake recorded #12693 shipped.

## Step 2 — what the paste proves, and the inferences that fail

- **One banner per decorated class.** With `--speculative-config` the drafter is a
  second model: `Loading drafter model...` (`:3613`) precedes its construction and
  its own banner. A paste with one banner and no `Loading drafter model` line is
  inside the **target** model's load — say that, it excludes the drafter.
- **Only rank 0 shows read progress, and the other ranks prove less than they look.**
  The shard bar is gated to global rank 0 (`enable_tqdm`,
  `vllm/model_executor/model_loader/weight_utils.py:702-705`, used at `:979-982`)
  while `Filesystem type for checkpoints …` / `Auto-prefetch is disabled …` are
  `logger.info_once` (`:920-961`), i.e. once per **worker process** — so a healthy
  8-rank run shows the bar on one rank and those two lines on up to eight. A paste
  where seven ranks end at the banner and only one shows the checkpoint scan is the
  moment the paste was taken, not seven dead ranks: the banners are 4-7 s apart in
  #12693. Ask for rank 0's log for the phase, and read the others only for their
  banner times.
- **Resident HBM is the pre-load state** (`base_loader.py:53-64` allocates the
  parameters on the device before reading) — `weight-load-stall-in-shard-loop.md`
  step 2 has the arithmetic and the #11095 case where that inference cost the
  session. Checkpoint-size/TP resident beside this banner proves nothing.
- Banner timestamps per rank are construction times, so they drift apart on a big
  model: seeing rank 4 print 20 s after rank 0 does not mean rank 4 is behind by a
  phase.

## Step 3 — `--enforce-eager` cannot separate this phase

At this ref eager mode **removes the banner**: upstream `vllm/config/vllm.py:1032-1038`
sets `mode` and `cudagraph_mode` to `NONE`, so the decorator's `do_not_compile`
gate (`decorators.py:390`) returns before `init_backend`; vllm-ascend's
`platform.py:590-592` then turns `enable_npugraph_ex` off as well (its hook runs
after upstream's post-init, `vllm/config/vllm.py:1296`). Its own two lines are
`Compilation disabled, using eager mode by default` (`vllm_ascend/platform.py:526`)
and `Cudagraph is disabled under eager mode` (`vllm/config/vllm.py:1234`).

Two consequences, and they are the point:

1. The weight load runs **identically** in both lanes, so a re-run with
   `--enforce-eager` that stalls again refutes nothing about a load-phase stall.
   Recommend the ablation that removes the phase you suspect; here that is a
   completed load (`Loading model weights took`), not a graph flag.
2. A reporter's "I added `--enforce-eager`, same result" is only a data point if
   their paste carries the two lines above. **A paste that still shows the OOT
   banner is not an eager run** (wrong command, stale log, or `enforce_eager`
   arriving through `speculative_config`/a stage yaml). Recorded #12693 accepted
   "加了 `--enforce-eager` 也一样" as the lane separator for a phase that flag cannot
   touch, and the phase question stayed open.

## Step 4 — checks, cheapest first (reporter's box)

1. **Rank 0: reading or blocked?** `grep read_bytes /proc/<pid>/io` twice, 30 s
   apart. Growing ⇒ the loader is progressing (cold cache on a large checkpoint,
   quant repack); frozen ⇒ blocked. Take the pid from their `npu-smi info`.
2. **Rank 0 stack:** `py-spy dump --pid <worker pid>`. Frames inside
   `safetensors` / `weight_utils.py` / the modelslim quant path
   (`modelslim_config.py`, a `process_weights_after_loading` repack) ⇒ the load is
   the owner; frames in `npu_fx_compiler`/`npugraph_ex` ⇒ re-rank toward capture
   and read `acl-graph-capture.md`; a futex/HCCL wait on every rank ⇒
   `../hccl-and-hardware/` lane.
3. **Checkpoint source:** `bash <skill_root>/scripts/shard-source-probe.sh <model
   dir> [pid] --tp <N>` — the same checks as
   `weight-load-stall-in-shard-loop.md` step 3 (bind mount, disk, copy in flight,
   shard count vs `model.safetensors.index.json`).
4. **Cross-model control on the same image and cards.** If another model starts on
   that image (recorded #12693: DeepSeek-V4-Flash runs), the image, driver, CANN
   and NIC axes are out and the question is that checkpoint's load path
   (`--quantization ascend` w4a8 → modelslim dequant/repack) or its model code.
   Ask for that control before spending calls on driver/image version comparison.
5. **Long-startup budget on A2:** the project's own GLM-5.1 W8A8 dual-node A2
   config sets `VLLM_ENGINE_READY_TIMEOUT_S: 3000`
   (`tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-A2-dual-nodes.yaml:14`)
   — quote it as the project's budget for "slow", never as an explanation of a
   load that is not reading (check 1 decides which of the two this is).

## Step 5 — recorded case #12693 (open)

800I A2 single node, 8 cards, GLM-5-w4a8 (`--quantization ascend`,
`--enable-expert-parallel`, TP8, `FULL_DECODE_ONLY`, `multistream_overlap_shared_expert`,
`deepseek_mtp` 3), images `v0.22.1rc1`, `v0.18.0` and the `glm5` tag, drivers
25.0.2 and 25.5.2. The paste has `Starting to load model` at 08:46:39, the banner
on all eight ranks 08:47:13-08:47:20, and rank 0's
`Filesystem type for checkpoints: EXT4. Checkpoint size: 391.11 GiB` immediately
after its banner — i.e. every rank is inside `get_model()` of the target model
(step 1), rank 0 one call further. **Reporter-confirmed later:** the same image
starts DeepSeek-V4-Flash, and `--enforce-eager` did not change the outcome. No
root cause was reported; the open item is rank 0's stack and its `/proc/<pid>/io`
sample. Rank the load-side causes of `weight-load-stall-in-shard-loop.md` plus the
w4a8 modelslim path; do not rank the npugraph_ex compile or capture sizes.

## Answer shape

One paragraph: the banner is a model-construction marker, so the phase is the
target model's weight load (before `Loading model weights took`), and only rank 0
can show progress — then the one sample (`/proc/<pid>/io`, `py-spy dump`) that
tells "reading" from "blocked", the cross-model control that bounds the
environment out, and the artefacts to send back (rank 0 log from `Starting to
load model`, `npu-smi info`, the stack, the checkpoint mount). Version scope: the
banner's emitter and the eager-mode gate are ref-specific; re-grep at another
release before quoting the line numbers.
