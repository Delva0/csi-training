# The box dies during engine start: the eager `max_model_len²` attention mask

Symptom family (#2873, reporter-confirmed; same family as #3002): a long-context
serve (`--max-model-len 131072` / `262144`) kills the **host** a few seconds
after the workers start. There is no traceback, no plog, no `EZ9999`, no
collective line — the reporter's terminal goes "Connection closed by foreign
host" and ssh cannot reconnect ("910b machine crash and can not restart",
"已经跑挂了我两台机器"). The cheap mistake is to read the last rank lines as a
HCCL/NIC problem or the reporter's "313T" hardware as the suspect: the host ran
out of **RAM** inside one `torch.zeros`, before a single weight was loaded.

This is the no-`top` variant of `host-ram-and-per-rank-rss.md` (there: the ranks
are alive and `top` shows their RES; here: nothing is left to sample). It is
also the **peak-allocation** sibling of the container lane
(`container-working-set-growth.md`): same function, opposite time signature — a
single multi-hundred-GiB spike at engine start versus a slow climb over days.

## The signature that places it (before any source read)

Read the log's **tail**, not its head. In #2873 the last application lines are
the per-rank parallel-state assignments, then silence:

```text
(VllmWorker TP0 pid=6699) INFO ... [parallel_state.py:1134] rank 0 in world size 8 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
...                                                                                  (8 such lines)
Socket error Event: 32 Error: 10053.
Connection closed by foreign host.
```

Two facts decide it:

1. **No `Starting to load model`.** That line is `load_model`
   (`vllm_ascend/worker/model_runner_v1.py:2281-2282`), i.e. *after*
   `NPUModelRunner.__init__`. Its absence with the rank lines present brackets the
   death to worker `init_device` → `NPUModelRunner(...)`
   (`vllm_ascend/worker/worker_v1.py:148-151`), which is exactly where the host
   mask is built.
2. **The reported `--max-model-len` is ≥ 128K** (or the checkpoint's own ceiling
   was used: `INFO ... [__init__.py:1750] Using max model len 262144`). The mask
   length is `model_config.max_model_len`, never the prompt length, so a
   100-token request at 256K still pays for 256K.

If both hold, price the mask before reading anything else:

```bash
bash <skill_root>/scripts/attn-mask-budget.sh --log <serve.log>     # "Using max model len" + world size
bash <skill_root>/scripts/attn-mask-budget.sh --max-model-len 262144 --ranks 8 --host-gib 1024
```

## The code path (verified at v0.10.1rc1, 7e16b4a)

| Where | What it does |
|---|---|
| `vllm_ascend/worker/worker_v1.py:148-151` | `init_device()` → `_init_device()` (logs the rank assignment) → `self.model_runner = NPUModelRunner(...)` |
| `vllm_ascend/worker/model_runner_v1.py:234-235` | `self.attn_mask_builder = AttentionMaskBuilder(self.model_config.max_model_len, self.dtype)` — full length, eager, in the constructor |
| `vllm_ascend/attention/attention_mask.py:18-33` | `_generate_attn_mask`: `torch.ones((N,N), bool)` → `torch.tril` → `~mask_flag` → `torch.zeros((N,N))` (float32) → `masked_fill` → `.to(dtype)`. **Every buffer is CPU-side** (no `device=`), so no device memory is touched |
| `vllm_ascend/attention/attention_mask.py:36-48` | `AttentionMaskBuilder.__init__` calls it directly — nothing is lazy here |
| `vllm_ascend/attention/attention_mask.py:88-94` | `_update_attn_cache` *still* grows the mask on demand, so the eager full-length build buys nothing |
| `vllm_ascend/worker/model_runner_v1.py:829-842` | the only consumers (`get_splitfuse_attn_mask`, `get_attn_mask`), each called with the batch's own seq length |

Arithmetic per rank (N = `max_model_len`): retained `2·N²` bytes for bf16/fp16,
peak `9·N²` bytes (two bool copies + two float32 buffers live together). At
N = 262144 that is **128 GiB retained and 576 GiB peak per rank**; eight TP
workers build it simultaneously → 4.6 TB of demand on one host. TP multiplies
the footprint but does not create it (TP=1 at 256K already needs 576 GiB of peak
host RAM; 128K is 144 GiB per rank). `scripts/attn-mask-budget.sh` prints the
table and the largest fitting length for a given host.

## Version scope (this is the part that separates a regression from a budget)

- **v0.10.1rc1 contains it.** `docs/source/user_guide/release_notes.md:23` lists
  "Fix bugs and refactor cached mask generation logic [#2442]" among the
  v0.10.1rc1 bug fixes; the same refactor is `#2326` in v0.9.1rc3
  (`release_notes.md:110`).
- **v0.10.0rc1 (2025.08.07) predates the refactor** and the reporter confirmed
  "work well with 0.10.0rc1" after the crash. Neither the v0.10.0rc1 section nor
  the v0.9.1 section of the release notes mentions mask work at all.
- The older API is visible in an earlier traceback (#1382):
  `AscendMetadataBuilder._attn_mask_builder = AttentionMaskBuilder.initialize_from_len(...)`
  called from the **metadata builder** with the run's own seq length (128 in
  `profile_run`). That is a different construction — build-at-use, not
  build-at-max-length — which is why the older releases survived long
  `--max-model-len`.
- Limit of the claim: the snapshot is one commit (`7e16b4a`), so the two trees
  were **not** diffed. Mechanism = source-verified at the pin; the regression
  boundary = release-note PR ids + the reporter's version ablation
  (reporter-confirmed for 0.10.0rc1, inferred for the rest).

## A works/fails model pair is only a control after both configs are read

Reports arrive as "the same cluster serves model X fine and model Y dies", and the
temptation is to price Y's `max_model_len` and call the pair decisive. The mask
length is the checkpoint's own `max_position_embeddings` (never the prompt), so the
pair only supports the arithmetic when the two checkpoints really differ there:

- Original **Qwen3-30B-A3B** is a 32,768-token checkpoint while
  **Qwen3-Next-80B-A3B-Instruct** is 262,144: the pair does validate #2873-style
  arithmetic (9·N² per rank, 144 GiB at 128K vs 576 GiB at 256K).
- The **2507 refresh** (`Qwen/Qwen3-30B-A3B-Instruct-2507`) extends the context to
  262,144 — the same ceiling as Qwen3-Next (confirm from each checkpoint's
  `config.json`). Recorded #3098's reporter answered exactly that sibling when
  asked "2507 or the original", i.e. the model that "可以部署" carries the same
  mask budget, so on that report the pair does **not** exonerate or incriminate
  this page's mechanism: both models build the same per-rank mask and what
  separates them is the failing model's own path (hybrid GDN + triton-ascend
  kernels + no prefix caching: `../image-dependency-faults/triton-ascend-pair.md`,
  `../performance-throughput/qwen3-next-decode-throughput.md`).
- Ask for both serve commands and both `Using max model len <N>` lines before
  quoting a number. A reporter who passed `--max-model-len` on the working model
  has excluded the term for *that* model only, which is exactly the reading that
  turns a pair into a control.

## What the post-crash artefacts do and do not say

The reporter later pasted `npu-smi info` from the rebooted box: 8 × 910B3, HBM
`3443 / 65536` MB on seven cards and `3894 / 65536` on card 0, and process rows
of only ~115 MB per `/usr/local/python` on the cards. Read it as **the ranks
never got to weight load** (a 30B bf16 checkpoint at TP=8 would hold ~7 GiB of
weights per card, KV aside) — not as "the cards were fine, so it was the
model". Host allocations are invisible to `npu-smi` at any point, before or
after the crash.

## Ranked causes and their checks (#2873)

1. **Eager host attention mask at `max_model_len` (first-class).** Check: the two
   log facts above + `scripts/attn-mask-budget.sh --log <serve.log>`. Confirm on
   the host with `dmesg -T | grep -iE 'oom|kill'` (an OOM kill of the 8 python
   ranks, or nothing at all if the kernel thrashed without a kill), `free -g`,
   and swap (`0` ⇒ the box hangs rather than a clean kill).
2. **Same host, malformed launch (hypothesis, cheap to exclude).** A second
   `vllm serve` left running from the previous attempt stacks another 8 masks;
   `ps -eo pid,rss,cmd | grep vllm` before the re-run.
3. **Device-side (excluded by this evidence).** A KV-init or collective fault
   would leave a device dump / one rank's HBM high and a Python frame — neither
   exists here (`kv-init-and-format-cast-alloc.md`,
   `../hccl-and-hardware/index.md`).

## Action ladder (#2873)

1. Re-run the same command with `--max-model-len 32768` (9 GiB peak per rank at
   TP=8) — if the box stays up, the mask is confirmed as the term that killed it.
2. Or run the release before the mask refactor (`0.10.0rc1`, reporter-confirmed
   in #2873) / a build without #2442 / #2326.
3. Keep the long context and patch the constructor (hypothesis, not run
   anywhere): `AttentionMaskBuilder(2048, self.dtype)` — `_update_attn_cache`
   grows it to the real seq length on first use, so the eager full-length build
   is redundant. Label it as a workaround, not an upstream fix, and check the
   release that follows for the real repair.
4. Do not propose TP changes as the fix: TP multiplies the footprint, it does not
   create it (a TP=1 262144 serve needs 576 GiB peak on its own).

## Answer shape

- One sentence: the machine ran out of **host** RAM inside the eager
  `max_model_len²` attention mask that every worker builds before loading
  weights; it is not a card, HCCL or model problem, and `npu-smi` cannot show it.
- Then the number (`9·N²` peak per rank at their `max-model_len`, × workers),
  the version scope (0.10.1rc1 has the refactor; 0.10.0rc1 is confirmed good),
  and the first action (`--max-model-len 32768` re-run, or `dmesg -T | grep -i
  oom` if they still have the box's kernel log).
- Ask for: `free -g` / `nproc` of the crashed host, `dmesg -T | grep -iE
  'oom|kill'`, the last 50 log lines of the failed run, and whether any other
  `vllm serve` was alive.
