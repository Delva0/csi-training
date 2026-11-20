# "Same flags, TP4 comes up but TP8 OOMs" at engine start

Scope: two `vllm serve` commands that differ **only** by `--tensor-parallel-size`
(usually 4 vs 8) on one node; the smaller one serves, the larger dies during
engine start with `NPU out of memory` / `torch.OutOfMemoryError` /
`Worker failed with error '... out of memory ...'`, or with the parent-only
`WorkerProc initialization failed` lane. Often a long `--max-model-len`
(128K-256K) and a hybrid linear-attention model (Qwen3-Next / Coder-Next,
Qwen3.5/3.6 GDN, DeepSeek-V4, GLM). Sometimes nothing is pasted but the two
commands and the model path.

This is a **budget** question (`index.md` rules 1 and 4: arithmetic first), not a
TP8 defect and not something to resolve by reading the allocator. Worked case:
vllm-ascend #6572, Qwen3-Coder-Next on 910B3-64G, `--max-model-len 262144
--gpu-memory-utilization 0.80 --max-num-batched-tokens 4096
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, TP4 up / TP8 OOM.

Not this page: the failure is after a healthy serving period
(`../runtime-memory/index.md`), the OOM accompanies a collective/device error
(`../hccl-and-hardware/index.md`), or the report is about wrong output on the
card that does start (`../accuracy-benchmarks/index.md`).

## Step 1 — diff the command against the project's own envelope (one read)

The tree ships the validated serve line for each tutorial model. Read it at the
pinned commit before any arithmetic on the reporter's numbers:

```bash
rg -n 'tensor-parallel-size|max-model-len|gpu-memory-utilization|enforce-eager' \
  docs/source/tutorials/Qwen3-Next.md          # and tests/e2e/**/test_<model>*.py
```

At vllm-ascend v0.14.0rc1 (`52d4acf`) `docs/source/tutorials/Qwen3-Next.md:64`
runs Qwen3-Next-80B-A3B at **TP4, `--max-model-len 32768`, gmu 0.8,
`--max-num-batched-tokens 4096`, `FULL_DECODE_ONLY`** — the reporter's working TP4
command with the context raised 8×, and the CI file
(`tests/e2e/multicard/4-cards/test_qwen3_next.py`) is TP4 as well. So "TP4 at
256K" is already outside the documented envelope and "TP8 at 256K" was never
validated at any context. Say that first: it is the answer to the literal
question ("is TP8 broken?" — no, it is unvalidated, and the shipped recipe is TP4
with a 32K window), and it costs one file read.

## Step 2 — read the OOM line as a budget, not as a number

If the paste carries
`NPU out of memory. Tried to allocate X GiB (NPU 0; T total capacity; A already
allocated; R reserved; F free)`, then compare `A` with `gmu × T`:

```bash
bash <skill_root>/scripts/oom-budget-split.sh --line '<the OOM line>' --gmu 0.8 \
     --params 80 --tp 8            # add --dtype to price W8A8/W4A8 checkpoints
```

- `A > gmu × T` → the failing request is **outside** the profiled budget: the
  phase is after the KV pool was sized (ACL-graph capture, workspace, HCCL/NPU
  runtime), and the reserve `(1-gmu) × T` is the thing that is too small.
- `A ≤ gmu × T` → the request failed **inside** the budget the engine granted
  itself, so the accounting (total memory used in the gmu product) is wrong, not
  the model.
- `A` within a block of `gmu × T` → a coarse-granularity tail: on a hybrid model
  the pool allocates `page_size = 12 × attn_page + padded mamba page` per block
  (below), so the last block is tens to hundreds of MiB and the free remainder
  can be too small for the *next* one.

In #6572 the fellow reporter's line was `Tried to allocate 2.36 GiB (60.94 GiB
total; 58.82 GiB already allocated; 823.69 MiB free)` at gmu 0.8, i.e.
`A = 58.82 > 0.8 × 60.94 = 48.75` by ~10 GiB — an outside-the-budget failure,
which is why the touch is the reserve/phase and not "more cards".

## Step 3 — which per-rank terms actually shrink with TP

Compute the three terms per card; only the first one halves:

| Term | Scaling | Source at vllm v0.14.1 / vllm-ascend v0.14.0rc1 |
|---|---|---|
| weights | `params × bytes ÷ TP` | checkpoint size; `scripts/card-budget.sh` |
| attention KV per token | `max(1, num_kv_heads // TP)` — **saturates** | `vllm/config/model.py:1142-1153` |
| mamba/GDN state per block | `conv_dim ÷ TP` and `num_v_heads ÷ TP` | `vllm/model_executor/layers/mamba/mamba_utils.py:170-193` |

Qwen3-Next-80B-A3B declares `num_hidden_layers=48`,
`num_key_value_heads=2`, `linear_num_{key,value}_heads` 16/32
(`vllm/transformers_utils/configs/qwen3_next.py:191-193,204-213`), so TP4 and TP8 both
resolve to **one** KV head per rank: the attention half of the KV cache is
identical at both sizes, and only the (small) mamba state halves.

The hybrid page geometry then moves too, because vllm-ascend *derives* the block
size from the mamba page and only ever raises it
(`vllm_ascend/patch/platform/patch_mamba_config.py:37-88`:
`attn_block_size = 128 × cdiv(mamba_page, 128 × attn_page_1_token)`, then
`:86-93` pads the mamba page up to the attention page; `vllm_ascend/utils.py:1139`
exempts `model_type == "qwen3_next"` from the usual "block size is 128" clamp).
Arithmetic for the 80B-A3B defaults (every 4th layer full attention → 36 linear /
12 full, head dims 128, conv kernel 4, bf16 state):

| TP | KV heads/rank | mamba page | derived block size | per block (12 attn + mamba) | blocks for 262144 | KV minimum for one 256K request |
|---|---|---|---|---|---|---|
| 4 | 1 | 9.42 MiB | 19,328 tokens | 122.7 MiB | 14 | 1.68 GiB |
| 8 | 1 | 4.71 MiB | 9,728 tokens | 61.8 MiB | 27 | 1.63 GiB |

Two consequences to state: the KV a 256K context needs is **TP-invariant**
(≈1.7 GiB per rank at either size), and the block is *thousands* of tokens, so
the pool's granularity is coarse and the effective block size is a number the
reporter's own log prints (`Setting attention block size to N tokens`,
`Padding mamba page size by X%`). Quote those lines back: they either match this
table (the geometry is as computed) or they do not (the checkpoint's head counts
differ from the defaults — ask for `config.json`).

Per-rank totals for this model: TP4 ≈ 40 GB weights + ~1.8 GB KV; TP8 ≈ 20 GB +
~1.8 GB. TP8 is *smaller* on every term above, so if the TP8 command OOMs while
TP4 starts, the difference is an **unprofiled per-rank cost** (capture/workspace/
collective buffers) or a reserve that is too small at that gmu — not the KV
arithmetic. Say that explicitly; it is the finding, and it is why the fix is a
budget/phase change rather than "TP8 is broken at 256K".

## Step 4 — do not lead with the full-graph warning

Every `FULL_DECODE_ONLY` launch prints the canned block "You have enabled the
*full graph* feature … A known problem is that capturing too many batch sizes can
lead to OOM … consider reducing `gpu_memory_utilization`" (`vllm_ascend/platform.py:313-327` at
v0.14.0rc1). It is printed unconditionally; it is not evidence that capture owns
this OOM. On #6572 the reporter's later public follow-up (2026-02-10) states they
run full graph on many other models without this effect and only Qwen3-Next
behaves this way, and that they suspect the model-family-specific optimisations
(reporter judgement, not verified). So rank capture as one candidate with one
ablation — `--enforce-eager`, or `cudagraph_capture_sizes` reduced to `[1]`
(`scripts/aclgraph-size-budget.sh` prices the size set) — instead of deriving its
footprint from the warning text.

## Answer shape

1. Direct answer: TP8 was never a validated configuration for this model; the
   shipped recipe at this ref is TP4 with `--max-model-len 32768`, and 256K is 8×
   outside it. Both commands already differ from the tutorial by that one flag.
2. The two per-card numbers from step 3 for their model and both TP sizes, plus
   the block size / per-block page from step 2-3 and the log lines that confirm
   them.
3. The phase verdict from the OOM line (inside vs outside the profiled budget)
   and what it implies.
4. Ranked actions, cheapest first: keep TP4 and the 32K window the recipe
   validates; if 256K is required, drop `--gpu-memory-utilization` in ~0.05 steps
   and quote the resulting `GPU KV cache size` line; two TP4 instances instead of
   one TP8 (the framework-agnostic answer the #6572 reporter reached themselves —
   BF16 at 64 GB: TP8 ceiling ≈ 0.6, TP4 works at 0.8); then the capture ablation;
   then a quantised checkpoint.
5. Gaps: the version/image pairing, `config.json` head counts, and the full
   startup log (block size, KV size, phase of the failing allocation).

## Stop rule — this class does not need an allocator mechanism

#6572 used 20 assistant turns, never emitted an answer, and died inside a
28,405-character reasoning block that was still oscillating between
"TP-invariant reserve", "capture footprint" and "the loader materialises full
tensors" when the output cap hit. Everything needed was already in hand by turn
12: the tutorial envelope (TP4/32768), the reporter's gmu ceilings, and the OOM
line. For this class the answer is steps 1-3 plus the action ladder; the
mechanism of the last 2 GiB is **not** required, and the per-rank table above is
the deliverable that replaces it.

## Scope and recheck

Verified at vllm-ascend v0.14.0rc1 (`52d4acf`) + vllm v0.14.1 (`d7de043d`).
Re-read before reusing on another release: the tutorial's serve line, the
`patch_mamba_config` page arithmetic (it has changed shape across releases — see
`../accuracy-benchmarks/hybrid-block-size-corruption.md` for a later ref),
`get_num_kv_heads`, and whether the block-size clamp exemption for `qwen3_next`
still exists (`utils.py:1139`). The reporter's gmu ceilings are one host's
measurements on one checkpoint, not a constant.
