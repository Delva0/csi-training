# TTFT ×k on a PD/EPD prefiller after `--speculative-config` is enabled

Scope: one model, one workload, one P/D pair; the only change is turning speculative
decoding on (usually on **both** nodes), and **TTFT alone** regresses — 64 K/1 K at
"prefix 90 %", 2.2 s → 22 s, while decode throughput, TPOT and output quality are not
part of the complaint. A P node that also runs `--enable-prefix-caching` is the normal
shape, and the report almost never states the prefiller's own hit rate — that number is
the deliverable, not a thing to derive.

Not this page: spec decode *also* collapsing acceptance
(`../spec-decode-mtp/acceptance-collapse.md`, `../spec-decode-mtp/acceptance-zero-at-high-k.md`);
throughput down with acceptance visibly healthy (`spec-decode-gain-accounting.md`);
a flat `Prefix cache hit rate: 0.0 %` on a serve with **no** spec-decode switch
(`zero-prefix-cache-hit-hybrid.md` owns the gate/version question); a PD TTFT compared
against the same model colocated, with no switch changed
(`../pd-disaggregation/pd-vs-colocated-ttft.md`).

## Recorded #16997 (2026-09-20)

A5 (Ascend 950), single machine split into P and D, DeepSeek-V4-Flash,
`MooncakeHybridConnector`, `method: dspark`, `num_speculative_tokens: 5` on both sides.
P: `--data-parallel-size 4 --tensor-parallel-size 1`, `--max-num-batched-tokens 3072`,
`--max-num-seqs 8`, `--enable-prefix-caching`, `--block-size 32`, `--enforce-eager`,
`gpu-memory-utilization 0.85`, `enforce_eager: false` *inside* `--speculative-config`.
D: same DP4/TP1, `--max-num-batched-tokens 256`, `--max-num-seqs 56`,
`--no-enable-prefix-caching`, `FULL_DECODE_ONLY`, `recompute_scheduler_enable: true`,
`gpu-memory-utilization 0.92`. Claim: 64 K/1 K at 90 % shared prefix, dspark off 2.2 s →
dspark on 22 s. Source checks below are at vllm-ascend **v0.26.0rc1** (`f2f74a16`) +
vllm **v0.26.0** (`568afb3a`); the reporter ran a `0.27.1rc1` branch that is not in the
snapshot, so re-grep every `path:line` on their tree before quoting it.

Run disposition: the verdict and its first ranked cause shipped, then the 16 384-token
output cap cut the ladder. The tutorial `rg` of Step 2 arrived at turn 33 of 42; the
source walk it did first (`refresh_block_size`, the `platform.py`/attention-layer
prefix-cache gates, `use_index_cache`, `skip_topk`, the proposer's chunked-prefill
branch) changed no conclusion. Do Step 1 + Step 2 and send the answer.

## Step 1 — the ratio is the classification, before any log

`TTFT(PD) = P queueing + P prefill + KV transfer + D's first decode step`. A workload
that reuses fraction `r` of its prompt prefills `(1−r)·L` tokens when the hit lands and
`L` when it does not, so **a factor of ≈ `1/(1−r)` is a full re-prefill**, not a slower
kernel: 2.2 → 22 s at `r = 0.9` is exactly 10.0, and the cross-check is one division —
`22 s / 64 k ≈ 2.9 k tok/s` against `2.2 s / 6.4 k`, the same rate on two workloads.

The losing reading is "the drafter made prefill 10× dearer per token": that predicts the
same slope in a prompt-length sweep, and it also owes a decode-side regression the
report does not have. Write the competing predictions into the first message and say
which single number separates them (the P node's `Prefix cache hit rate` block, then a
TTFT-versus-prompt-length sweep: proportional to length ⇒ lost hit; a constant offset ⇒
hop/queue; a changed slope ⇒ per-token cost).

## Step 2 — one `rg` in the model's own tutorial before any source read

The pin's tutorial carries the *validated* recipe for the exact feature pair, including
the env var the reporter omits. One call at v0.26.0rc1:

```bash
rg -n -i 'dspark|speculative|RETENTION_INTERVAL|no-enable-prefix-caching|max-num-batched-tokens' \
  docs/source/tutorials/models/DeepSeek-V4-Flash.md
```

What it gives, and how #16997's P node differs:

| anchor (v0.26.0rc1) | tutorial | #16997 P node |
|---|---|---|
| `:664` `export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` in the *A3 series with dspark* prefill block (`:639-700`) | set | **not set** |
| `:678` `--max-num-batched-tokens 8192` | 8192 | 3072 (22 chunks for 64 K) |
| `:682` `--speculative-config '{"num_speculative_tokens": 5,"method": "dspark","enforce_eager": true}'` | `enforce_eager: true` | `false` under a global `--enforce-eager` |
| `:684` `--block-size 32` | 32 | 32 (matches) |
| `:751` decode block `--no-enable-prefix-caching`, 480 batched tokens | — | matches |
| `:330` what the var does | "Controls the retention interval, in tokens, for prefix-cache checkpoints of hybrid attention layers … applicable to DeepSeek-V4 … **Under KV-cache pressure, it can improve the effective prefix-cache hit rate for reusable long prefixes** … 128 × `--block-size` is recommended: `4096` at `--block-size 32`, `16384` at `128`" | — |
| `:1076` | "When DSpark is enabled, Prefill and Decode must use the **same** number of speculative tokens" | equal (5/5) ✔ |
| `:1075` | `recompute_scheduler_enable: true` — "enable this configuration only on decode nodes"; a request with short D-side KV is sent back to P and re-prefilled | P has `false`, D has `true` ✔ |

The env var is real at this pin, not documentation-only: upstream default is `None`
(`vllm/envs.py:297`, reader `:1114-1119`) and vllm-ascend reads it into the DeepSeek-V4
hybrid coordinator (`vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:108`,
validated at `:109-117`). An unset value is therefore the *absence* of the documented
retention, which is the cheapest rung that can move a DSV4 hit rate.

## Step 3 — on a DSV4/hybrid model the hit is a min over groups, so "one block" can cost all of it

Verified at the same pin in `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`
(the DeepSeek-V4 coordinator patch), and the reason a spec-decode switch can zero a hit
that is otherwise 90 %:

- `find_longest_cache_hit` is an iterative fixed point and the request's hit is the
  **minimum** over attention groups (`:267-378`), so one group that stops matching takes
  the whole prefix to zero.
- With EAGLE/MTP/dspark the eagle groups match one block further and then pop it:
  `_max_length = curr_hit_length + eagle_margin` (`:328-335`, margin = the effective
  block size, or `hash_block_size` under partial hashing) and `drop_eagle_block=True`
  (`:336`, passed at `:344`) — the source's own words: "Eagle needs to match one more
  block and then pop the last."
- `use_eagle` must be propagated to **every** manager of an eagle-containing attention
  group (`:236-256`), and the comment states this report's exact failure mode: if any
  such manager keeps the default `False`, "its retained tail ends one block short of the
  eagle 'peek' boundary the read looks at, the SWA group never hits, and the
  min-over-groups hybrid hit **collapses to 0%**".
- The hit length must be a multiple of the LCM of every group's effective block size
  (`:258-265`; `_get_effective_block_size` multiplies by `compress_ratio` at `:187-197`),
  so a hit is quantised — a shared prefix just under a boundary loses the whole tail.

So on a DSV4-class serve, "the hit rate went to 0 or lost most of its length once spec
decode was enabled" is a **bookkeeping/coverage** question with an in-tree owner and the
retention lever above — not a kernel question. Rank it as such, with the check, and keep
the mechanism label honest: the recorded case never got the P-side hit rate, so the
ranking is the deliverable, not the verdict.

## Step 4 — ladder, cheapest first

1. **P node: `export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`** (tutorial `:664`;
   keep `--block-size 32`). One run. Read P's `Prefix cache hit rate` and mean TTFT.
2. **P node: `--max-num-batched-tokens 3072 → 8192`** (tutorial `:678`). If TTFT now
   scales with the chunk count, the drafter's own pass over prefill chunks is the cost —
   it does run there (`vllm_ascend/spec_decode/dspark_proposer.py:360` sets
   `AscendAttentionState.ChunkedPrefill`; `:263` shows the mixed step's
   `num_prefill_reqs`).
3. **`"enforce_eager": true` inside `--speculative-config`** (tutorial `:682`) — the
   reporter's `false` under a global `--enforce-eager` is the unvalidated row.
4. Only then: the TTFT-versus-prompt-length sweep (8/16/32/64 K, same body) and the
   profiler the launch already configures — they separate "lost hit" from "dearer
   token", and nothing before them needs a source read.
5. **Never ablate spec decode on P alone.** `:1076` requires the same
   `num_speculative_tokens` on both sides, and a P/D split spec config is the recorded
   #16177: decode dies on its first request with `IndexError` in
   `_get_kernel_block_ids` because the KV-group counts differ
   (`../pd-disaggregation/kv-group-count-mismatch.md`). The legal ablation is both-on vs
   both-off, which is the reporter's own baseline — say that instead of proposing the
   asymmetric run.
6. If the hit rate is *proven* to fall under KV pressure (draft groups added at a fixed
   `--gpu-memory-utilization`), test the capacity story directly: raise P's
   `--gpu-memory-utilization` toward D's 0.92, or drop P's `--max-num-seqs` 8 → 4, and
   watch the hit rate rather than TTFT.

## Step 5 — what to send back

- P node's metrics block with dspark **on and off**, including `Prefix cache hit rate`
  and `Mean TTFT` (a `/metrics` scrape is enough); if the reporter only has the serve
  log, the same block per interval.
- D node's log lines for the recompute path (`recompute_scheduler_enable: true`,
  tutorial `:1075`) — a recompute puts a second full prefill inside TTFT and is
  independent of P's cache.
- The prompt lengths actually driven and the reuse setup ("prefix 90 %" is a workload
  parameter, not a hit rate).
- On their `0.27.1rc1` branch: `rg -n 'VLLM_PREFIX_CACHE_RETENTION_INTERVAL|eagle_margin|drop_eagle_block' vllm_ascend/ vllm/envs.py`
  — the retention default and the eagle margin are exactly the code this page quotes
  from v0.26.0rc1 and they may have moved.

## Delivery

Verdict first, with rung 1 in the same paragraph: "the 10× is `1/(1−0.9)` — the
prefiller is re-prefilling all 64 K, not computing them more slowly; set
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` on P and read its hit rate", then the
ranked causes each with one check, then the table. #16997's own run put the verdict and
cause 1 in the first message and lost everything after it to the cap; the tutorial diff
that names rung 1 was one call, so there is no room for a source walk here.
