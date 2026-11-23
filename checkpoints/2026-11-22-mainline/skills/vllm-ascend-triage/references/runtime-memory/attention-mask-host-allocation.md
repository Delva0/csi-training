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

Same sites at **v0.11.0rc0 (`00ba071`)**, the pin of recorded #3314, with one new
gate: `worker/worker_v1.py:192-195` (`init_device` → `NPUModelRunner`),
`worker/model_runner_v1.py:328-334`, `attention/attention_mask.py:19-33`, and the
`Starting to load model` line the absence argument uses is
`worker/model_runner_v1.py:2626-2627`.

| Where (v0.11.0rc0) | What it does |
|---|---|
| `worker/model_runner_v1.py:328-334` | `if torch.version.cann.startswith("8.3")`: `AttentionMaskBuilder(self.scheduler_config.max_num_batched_tokens, dtype, device)` **else** `AttentionMaskBuilder(self.model_config.max_model_len, dtype)` |
| `attention/attention_mask.py:48,52-57` | `_generate_attn_mask` is called unconditionally (so the host mask exists on both branches); on the CANN 8.3+ gate the builder additionally allocates a fixed 2048×2048 int8 chunked-prefill mask *on the device* (`:53-57`) |

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
- **The term is CANN-conditional, not only release-conditional.** At v0.11.0rc0
  the builder is sized `max_model_len` *only* when
  `torch.version.cann` does not start with `8.3`; with CANN 8.3+ the eager host
  mask is `9·max_num_batched_tokens²` instead (a few GiB at the resolved
  `--max-num-batched-tokens`, not hundreds), so a 256K `max_model_len` stops
  deciding the budget there. Read the CANN version out of the report's
  `collect_env` before pricing anything, and label an upgrade rung as
  version-gated/untested unless the reporter runs it: the shipped images pair a
  specific CANN with their own torch_npu, so it is not a free flip.
- **Reporter-confirmed afterwards (#3324, the 8.3 branch landing).** That reporter
  rebuilt `vllm_ascend 0.11.0rc0` from source and moved to
  `Ascend-cann-toolkit 8.3.RC1.alpha003`, then posted "CANN升级到8.3似乎还是解决不了" —
  but the paste is a **different phase**:
  `RunOpApiV2 … AclrtSynchronizeStreamWithTimeout(stream), error code is 107027` out of
  `[model_runner_v1.py:3556]` (the canned capture wrapper of SKILL.md rule 11) +
  `ERR00100 PTA call acl api failed`, with "使用`--enforce-eager`倒是可以运行，但是速度非常非常慢".
  So on that pair the host mask stopped deciding the budget — the 8.3 branch
  above sizes it by `max_num_batched_tokens` — and the live lane is the bare `107027`
  (`../startup-failures/captured-stream-host-sync.md` §6). Limits: the follow-up pastes no
  serve command and no `--max-num-batched-tokens`, so neither the mask value nor the eager
  run's cost is measured; the moving part is the **pair** (newer vllm-ascend × CANN 8.3),
  so a CANN-only upgrade on an older vllm-ascend keeps the `max_model_len` mask; and
  "still not solved" is the reporter reading a new symptom, not a refutation of the
  arithmetic that matched their 100K/128K numbers.

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
after the crash. Two more things that paste settles: those 8 residual pids are
the previous attempt's workers, still listed against the dies, so **reap them
before the retry** or the next launch stacks a second set of eight masks
(ranked cause 2 below); and the same command killing a **second** machine
("It has already crashed two of my machines!!!") is the control that retires a
one-off host/DIMM flake — ask for both hosts' `free -g`, since a smaller-RAM
box lowers the ceiling before anything in the command changes.

Read those process rows by **footprint against the checkpoint**, never by pid. A row at
the driver baseline (recorded #2873: ~115 MB, with `No running processes found` beside
the clean dies) is a process that never allocated device memory — no engine got past
device attach on that die; a row at roughly
`params × bytes / TP` or above means some process **completed** engine init there — a
30B bf16 at TP=8 is ~7 GB of weights, while **Qwen3-Next-80B at TP=4 is ~40 GB** and
~59–60 GB with the KV pool (recorded #3170: four dies at 60329 MB and four clean, TP=4,
`--enforce-eager`). Do **not** attribute such a row to the run that died: `npu-smi`
prints *host* pids while a containerized service's log carries namespace-local ones
(#3170: log `APIServer pid=718` / `EngineCore_DP0 pid=1127` against rows
`2997963/2997965/2997967/2997970`), so the match is unavailable — ask for
`ps -eo pid,rss,lstart,cmd | grep vllm` and, if a full-size engine is still resident,
treat it as ranked cause 2 (a second `vllm serve` holding the cards; reap it before the
retry). What that footprint never does is argue for or against this page's mechanism:
host allocations are invisible in the table before and after the crash.

## Recorded instance where the flag was simply omitted: #3314

Qwen3-30B-A3B (the 2507 refresh, ceiling **262144**), image
`quay.io/ascend/vllm-ascend:v0.11.0rc0` (vllm v0.11.0rc3), 8 × 910B2, host
`free -h` = **1.5 TiB**, `--tensor-parallel-size 8` then `4`, both with
`--enable-expert-parallel`, **no `--max-model-len`**. The paste has the four
per-rank assignment lines at 14:53:34, then nine minutes of silence, then
`Parent process exited, terminating worker` ×3, `WorkerProc initialization
failed …` (parent only, no worker traceback) and the `TBE Subprocess …
main_process disappeared` block; `npu-smi` shows ~63 GB HBM per card (the previous
run) and no worker traceback exists. The reporter's own note: "Checking the system
status, the system memory is at 100% when the error occurs. When I set
`max_model_len=65536`, everything works properly." The later public follow-up
confirms the reading and the ceiling: "reduce max-model-len can circumvent this
problem, qwen3-30b model length is 256k".

Classification: reporter-confirmed = the derived length, the 100 % host memory and
the workaround; source-verified + arithmetic = the mask as the term that dies
(262144 → 576 GiB peak/rank × 4 ranks = 2.3 TB > 1.5 TiB, while 65536 is 36 GiB ×
4). Not confirmed: that any particular larger value fits, and that the death is
this term rather than another host allocation at the same scale — the two log
facts (rank lines present, `Starting to load model` absent) are the evidence and
the reporter's paste does not carry the line, so ask for it.

**Do not answer "the engine caps you at 65536".** There is no such cap: the ceiling
is `N ≤ sqrt(headroom · host_RAM / (9 · ranks))`, so a 1.5 TiB host at TP=4 is a
different number from the same host at TP=8
(`scripts/attn-mask-budget.sh --max-model-len N --ranks R --host-gib H` prints the
largest fitting length). Give the formula and the one value the reporter already
confirmed, and say explicitly that a bigger value is a budget experiment, not a
promise. Two rungs that do **not** move it on a pre-8.3 CANN: `--enforce-eager`
(the mask is built in `__init__`, graph or not) and `--max-num-batched-tokens`
(only the 8.3+ branch reads it).
Keep that version condition on the sentence instead of writing a flat "`--enforce-eager`
is inert": the same flag doubles as this lane's cheap **discriminator**, since it cannot
remove a constructor-time host allocation — a serve that *runs* under it has refuted this
page for **its** build, and asks for the `collect_env` `vllm_ascend`/CANN pair instead of a
debate. Recorded #3324 published the flat claim and the reporter's next public comment
showed eager running (0.11.0rc0 + CANN 8.3), i.e. report and paste were one release apart;
on an 8.3+ pair the value that now sets the mask is `--max-num-batched-tokens`.

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
5. On a **CANN 8.3+** host the term is `max_num_batched_tokens`, so the release pairing
   that carries the 8.3 branch (v0.11.0rc0 and later) keeps long context without patching —
   say in the same sentence that this pairing's next failure is the capture lane's bare
   `107027`, where `--enforce-eager` is the isolation run and not the deployment
   (reporter: "速度非常非常慢", #3324).

## Answer shape

- One sentence: the machine ran out of **host** RAM inside the eager
  `max_model_len²` attention mask that every worker builds before loading
  weights; it is not a card, HCCL or model problem, and `npu-smi` cannot show it.
- **When the paste carries no host RAM, price two sizes instead of waiting for
  `free -g`** — the collector's output in these reports carries no RAM line at all (the
  #3170 paste prints CPU model, caches and NUMA nodes only), and the term is large enough
  that the verdict usually does not depend on it (TP4 at 262144 is 225 % of a 1 TiB host
  and still 112 % of a 2 TiB one). Ask for `free -h` as confirmation, not as a
  precondition for the answer.
- A release note that tells this family to lower `max_model_len` (**#2884**, at
  v0.10.2rc1: "The HBM usage of Qwen3 Next is higher than expected … set `max_model_len`
  and `gpu_memory_utilization` to suitable value") agrees with the rung but is a
  **device-HBM** statement. Quote it as the project's own advice; never as evidence that
  the host mask is the term — the two resources are separate, and only `free -g` /
  `dmesg` speak to this one.
- Then the number (`9·N²` peak per rank at their `max-model_len`, × workers),
  the version scope (0.10.1rc1 has the refactor; 0.10.0rc1 is confirmed good),
  and the first action (`--max-model-len 32768` re-run, or `dmesg -T | grep -i
  oom` if they still have the box's kernel log).
- Ask for: `free -g` / `nproc` of the crashed host, `dmesg -T | grep -iE
  'oom|kill'`, the last 50 log lines of the failed run, and whether any other
  `vllm serve` was alive.
