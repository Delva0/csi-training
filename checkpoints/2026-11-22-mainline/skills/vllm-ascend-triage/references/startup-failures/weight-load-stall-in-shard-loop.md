# The shard bar freezes and nothing is raised

Scope: the paste stops at `Loading safetensors checkpoint shards:   0% Completed |
0/N [00:00<?, ?it/s]` (or any partial `k/N`) and no later line arrives — no
traceback, no `OOM`, no `EngineCore failed to start`, the HTTP port still
refused, and `npu-smi info` shows the `VLLMWorker_TP*` / `EngineCore` processes
alive. The stall is **inside** the weight load, before
`Loading model weights took X.XXXX GB`.

A paste whose last line is `Using OOT custom backend for compilation` is the same
phase one call earlier — that banner is printed while the model object is built,
before the first shard — and `compile-backend-banner-then-silence.md` bounds it
(only rank 0 ever shows the progress below).

Not this topic: a paste that ends *after* `Loading model weights took X GB`
(→ `silent-stall-after-weight-load.md`), a traceback (`→ index.md`), or a shard
that fails loudly (`HeaderTooLarge` / a `ValueError` on the first weight group
→ `checkpoint-file-not-safetensors.md`, `../pooling-embedding/embedding-checkpoint-as-generate.md`).

Verified at vllm **v0.21.0** (`ad7125a4`) with vllm-ascend **v0.21.0rc1**
(`80610e44`). Re-grep per version; the phase order and the tqdm shape hold.

## Step 1 — bound the phase in the first message

At these pins the console sits in one call chain, so the *missing* next line is
the phase bound:

| Order | Site | Line the reporter would see if it finished |
|---|---|---|
| 1 | `vllm_ascend/worker/model_runner_v1.py:3619` — `Starting to load model <path>...` | (printed) |
| 2 | `:3635` `get_model(vllm_config=…)` → upstream loader: build the model, then read shards, then `process_weights_after_loading` | `Loading model weights took X.XXXX GB` at `:3666` |

So "the bar is frozen and `Loading model weights took` never appears" means the
process is inside `get_model()`. Say that, then rank causes — the bar alone does
not name one.

## Step 2 — two inferences that do not hold (both cost recorded sessions)

**The bar's index is not a progress counter.** The iterator is
`tqdm(sorted_files)` over the **shard files**
(`vllm/model_executor/model_loader/weight_utils.py:995`), and the default
strategy opens each shard with `safe_open` and pulls one tensor at a time
(`:1042-1043`). The bar advances only when a shard's last tensor has been
*consumed* and the next file is requested, so `0/N` means "not one shard
finished". Conversely, a paste taken from a terminal shows the **last** frame
tqdm wrote, while `docker logs` keeps every `\r` frame: ask which they copied
before reading the index as a distance.

**Resident device memory is not evidence that the weights are loaded.** The same
loader builds the model under a device context *before* it reads anything —
`vllm/model_executor/model_loader/base_loader.py:53-64` does
`with target_device: model = initialize_model(...)` and only then
`self.load_weights(model, model_config)`. Every parameter is therefore already
allocated on the NPU when the first shard is opened, and `load_weights`
overwrites them in place, so HBM stays flat from the first frame to the last.
A pasted `Memory-Usage 29968/65536` / `Process memory 26618` next to a `0/11`
bar is the **normal pre-load** state — and because checkpoint-size/TP is exactly
what those numbers sum to (51.75 GiB / 2 ≈ 26.5 K MiB here), the arithmetic
looks decisive while proving nothing. Do not use "weights already resident ⇒
the stall is after the copies" to move the investigation into post-load code
(`process_weights_after_loading`, NZ/quant cast, Triton warmup): that inference
is what turned #11095 into a session with no answer.

**`Filesystem type for checkpoints: EXT4` is not a read test.** `weight_utils.py:930-944`
calls `_get_fs_type` (`:784-812`), which resolves `/proc/mounts` and picks the
longest matching mount prefix, and the "Auto-prefetch is disabled because the
filesystem (EXT4) is not a recognized network FS" line only records the
prefetch decision (`:946-961`). A `docker run -v <host dir>:/model` bind mount
reports the **backing** filesystem's type; a loop device, an overlay, a stale or
network-backed bind source, a disk in a degraded state, or a directory that is
still being copied all print `EXT4` and can still block a read. Do not close the
storage hypothesis with that line.

## Step 3 — the checks, cheapest first (each ~one command on the reporter's box)

Take the pids from the reporter's own `npu-smi info` process table; it prints
the host pid and the "-in-container" pid side by side, so use the column that
exists on the shell you are given.

1. **Is it reading or blocked?** `grep read_bytes /proc/<pid>/io` twice, 30 s
   apart. Growing ⇒ it is progressing (cold page cache on a slow/large
   checkpoint, or a cheap read path); frozen ⇒ blocked. This is the one sample
   that replaces the HBM arithmetic.
2. **Blocked where?** `ps -o pid,stat,wchan:32,cmd -p <pid>`: `D` state with
   `wchan` in `io_schedule` / `__lock_page` / `rpc_wait_bit_killable` ⇒ the
   filesystem or mount owns it; `S` on a futex/semaphore ⇒ look for a rank-level
   wait instead. Then `dmesg -T | tail -60 | grep -iE 'hung_task|blocked for more
   than|nfs|cifs|I/O error|EXT4-fs error'`.
3. **Which mount is it?** `findmnt -T <model dir in container>` and
   `grep -i model /proc/mounts` — compare with what the host has under that path.
4. **Raw throughput:** `time dd if=<one shard>.safetensors of=/dev/null bs=8M
   count=128` in the container (~1 GiB). Seconds ⇒ the read path is not the
   cause; a hang or single-digit MB/s ⇒ it is.
5. **Is the directory complete?** shard count vs the `weight_map` of
   `model.safetensors.index.json`, plus `ls -l --time-style=full-iso` (mtimes
   seconds apart ⇒ a copy in flight). A short/being-written shard can also block
   rather than raise.

`bash <skill_root>/scripts/shard-source-probe.sh <model dir> [pid] [--tp N]`
runs 1, 3, 4, 5 and prints the per-rank size arithmetic.

## Step 4 — the reporter-confirmed outcome (#11095, 2026-06-29)

Setup: `quay.io/ascend/vllm-ascend:v0.21.0rc1-openeuler`, 2×910B3 (64 GB),
TP=2, Qwen3.5-27B, `--max-model-len 133000`,
`--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'`,
`--additional-config '{"enable_cpu_binding":true}'`, container
`-v /mnt/xxx/model:/model`, serve path `/model/Qwen3.5-27B`. The reporter's own
resolution: he copied the checkpoint into a **new host directory** and re-ran
the container against that directory — `--name` changed, *"其他容器启动命令不变"*
(every other container argument unchanged) — and the model started: *"可以启动成功了
… 真的是奇奇怪怪！！！"* (issue comment 4831844713).

Read it as: **reporter-confirmed outcome, mechanism open.** What it does settle
is only the ownership: the serve command, the model code, the MTP config and
`enable_cpu_binding` were identical across the failing and the working run, so
none of them is the cause, and the difference lived in the checkpoint's source
directory. What it does not settle is *why* the first directory blocked (mount
state? a copy in flight? a degraded device? permissions are excluded — a
permission fault raises). The discriminating follow-up if it recurs: run the
probe on the original directory once, and compare `sha256sum` of one shard from
each copy.

Precedents on this model are a different fault, quote them as such:
#7310 (Qwen3.5-27B, 8×A2, v0.17.0rc1 image, deployment fails) and #7477 (the
same model, `AllReduceOpbasePipeline ERROR`) — both raise; neither is this
silent stall.

## Step 5 — what the paste cannot support

- Not "Qwen3.5 unsupported at v0.21.0rc1": `Starting to load model` printed, the
  checkpoint config parsed, and the model's own GDN line appeared
  (`[gdn_linear_attn.py:169] Using Triton/FLA GDN prefill kernel`).
- Not the MTP config or `enable_cpu_binding`: both were already consumed by the
  worker before `Starting to load model`, and both were identical in the run
  that started.
- Not an OOM: the cards held 29 968 / 65 536 MiB, and an OOM prints
  `EL0004`/`207001` rather than silence.
- Not HCCL: no collective error text and nothing in this phase needs one.

## Answer shape

One paragraph: the phase (`get_model()`, before `Loading model weights took`),
the two candidate owners — (a) the shard read blocked on the checkpoint source
(directory / bind mount / disk), (b) work after the copy inside the loader — and
the single sample that separates them (`/proc/<pid>/io` twice, or `dd` on one
shard). Then the table of checks, the #11095 note that a fresh copy of the model
directory plus a fresh bind mount fixed one such report with the mechanism
unsettled, and the artefacts to send back: `findmnt -T` of the model path, the
`dd` timing, `ps -o stat,wchan` of the worker pid, `dmesg -T` tail, and the full
per-rank log.
