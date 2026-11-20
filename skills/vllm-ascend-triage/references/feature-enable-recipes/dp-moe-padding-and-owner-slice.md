# DP>1 MoE: the pad target is DP-wide, the owner slice is per-rank

Scope: `--data-parallel-size` > 1, with or without `--enable-expert-parallel`, and
the report is one of — (a) the dummy/`profile_run` batch passes but a **real
request** hangs a layer or two in, (b) a MoE layer returns a correctly shaped
tensor whose values are garbage while the pre-gather tensor printed sane numbers,
(c) the reporter's own model file computes the DP padding itself and changed
which object it reads the number from (`attn_metadata.max_tokens_across_dp` →
`get_forward_context().max_tokens_across_dp`). Deliverable: which of the two
numbers is wrong — the **pad target** (must be identical on every DP rank) or the
**owner slice** (which rows of the gathered tensor belong to this rank).

Not this topic: which MoE communication method the tree picks
(`moe-expert-parallel-with-dp.md` — read it first for the `MC2`/`All2All`/
`AllGather` table), a device dump or `ERR00100`
(`../hccl-and-hardware/index.md`), a capacity plan (`../deploy-sizing/index.md`).

Verified at vllm-ascend **v0.9.1rc2** (`b9f715d`) + vllm **v0.9.1** (`b6553be`),
the pair pinned for recorded #3107. Re-grep the four names below at another ref.

## Step 1 — the pad target has exactly one writer

`vllm_ascend/ascend_forward_context.py:91-98`, inside `set_ascend_forward_context`
(entered once per model forward, `worker/model_runner_v1.py:1144-1150`):

```python
dp_world_size = get_dp_group().world_size
if dp_world_size > 1 and forward_context.dp_metadata is not None:
    max_tokens_across_dp = forward_context.dp_metadata.max_tokens_across_dp_cpu.item()
else:
    max_tokens_across_dp = num_tokens
forward_context.max_tokens_across_dp = max_tokens_across_dp
```

Three facts follow, and they are what the reporter's question is really about:

- `forward_context.max_tokens_across_dp` is the **DP-wide max** only while
  `dp_metadata` is present. `dp_metadata` is `None` when `dp_size == 1`
  (`_get_forward_metadata_across_dp` returns `None`, `model_runner_v1.py:629-630`)
  and on any call site that passes no `num_tokens_across_dp` (`:1516`), and then
  the fallback is the caller's own `num_tokens` — one rank's number, not the group's.
- The dp>1 real forward does pass one (`:1144-1150`), so inside the MoE layer a
  disagreement is more likely a **phase** difference (`with_prefill` flipping inside
  the group, `:654-676`) or a vendored reader of the wrong object than the fallback
  itself. What `num_tokens` means there also differs by phase: in prefill
  `maybe_padded_num_tokens = num_tokens` (`:664-667`), so the local count stays
  local, while decode hands over the possibly-padded counts. Never quote one
  number for the process: `profile_run`/`_dummy_run` (`:1725-1732`) and
  `kv_connector_no_forward` (`:1516`) pass yet another value.
- The rest of the block is guarded by `if num_tokens is not None:`
  (`padded_num_tokens` `:104-106`, `global_bs` `:108-109`, the MC2 chunk fields
  `:113-124`, `mc2_mask` `:126-131`). A reader called on a forward that passed
  `num_tokens=None` gets an `AttributeError`, not a wrong number — a different
  finding from a mismatch.

Upstream supplies `dp_metadata` from the `num_tokens_across_dp` vector the runner
builds in `_get_forward_metadata_across_dp` (`model_runner_v1.py:625-679`): one CPU
`all_reduce` over the DP group carries **both** vectors — the possibly-padded
per-rank counts and the real per-rank counts — plus `with_prefill`. In prefill the
real vector is used and the local count stays local (`:664-667`); in decode the
padded vector is (`:668-669`); only torchair-graph decode rewrites the local count to
the DP max (`:673-676`). So "which number pads the all-gather" is
**phase-dependent** in this tree, and a vendored forward that computes it once at
import time is wrong for one of the two phases.

## Step 2 — who consumes it (three sites at this pin)

`rg -n 'max_tokens_across_dp' vllm_ascend/` returns exactly:

| Site | Use |
|---|---|
| `ops/fused_moe.py:153-155` | MC2 `global_bs = ceil(max_tokens_across_dp / tp_world_size) * ep_world_size` |
| `ops/fused_moe.py:1382-1395` | the `AllGather` DP branch: pad `hidden_states`/`router_logits` to `max_tokens_across_dp` **when `num_tokens <` it**, then `get_dp_group().all_gather(…, 0)` |
| `ascend_forward_context.py:104-131` | `padded_num_tokens`, `global_bs`, `mc2_mask` for the MC2 path |

The return path is the other half of the convention: `ops/fused_moe.py:1446-1452`
takes this rank's rows with
`dist._functional_collectives.reduce_scatter_tensor(e_hidden_states, "sum",
scatter_dim=0, group=get_dp_group().device_group)` and only then `[:num_tokens]`.
**The head rows of a DP all-gather are rank 0's**, so `.[:num_tokens]` on the
gathered tensor is correct only on rank 0 — on any other rank it is another
rank's data, and on rank 0 it is correct only if the pad happened.

Note `vllm_ascend/models/deepseek_dbo.py:511-513` still reads the **old** source,
`attn_metadata[i].max_tokens_across_dp`, while `AscendMetadata`
(`attention/attention_v1.py:112-142`) has **no such field** at this pin: that path
is dead unless something sets the attribute out of tree. Say so with the pin and
re-grep at the reported ref rather than "the tree does X".

## Step 3 — the two directions of the arithmetic, computed not eyeballed

- **Gather width.** Each rank contributes `P = max_tokens_across_dp` rows, so the
  gathered tensor is `dp_size × P` rows. Recorded #3107: one prefill request of 68
  tokens, DP=8, TP=1 → `544 = 68 × 8` rows (`[544, 5120]` in the reporter's
  print). That equality means the pad target **agreed** on all eight ranks — the
  remaining suspects move to the slice, or to a later collective.
- **Owner slice.** With `P` rows per rank, rank `r`'s own rows are
  `[r*P, r*P + num_tokens)` (equivalently whatever `reduce_scatter_tensor` hands
  back). Recorded #3107 printed `layer1 mlp output [544, 5120]` sane and then
  `layer1 output [68, 5120]` at ≈`-5.9e37` — bf16 garbage of the right shape.
  That is a read of rows nobody wrote this step (wrong offset, or a stale buffer
  region), not an arithmetic defect in the MoE; a wrong *pad target* instead
  shows up first as a **hang**, because the collectives no longer match.
- **Phase.** `with_prefill` travels in the same all_reduce, so a rank that flips
  prefill↔decode relative to its peers changes which vector pads the gather
  (`:654-676`). One request on an 8-rank DP server is exactly the case where seven
  ranks are idle: the differences between ranks are at their largest.

## Step 4 — `--enable-expert-parallel` changes the branch, so read the selector

`get_fused_moe_state(ep_size, with_prefill)` at `ascend_forward_context.py:25-39`
takes `ep_size = get_ep_group().world_size if enable_expert_parallel else 1`
(`:66-67`):

| Launch | `ep_size` | State |
|---|---|---|
| DP>1, **no** EP | 1 | `AllGather` — the `dp_group.all_gather` branch above |
| EP on, `ep_size < 16` (recorded #3107: DP8 × TP1 = 8) | 8 | `All2All` (or `All2AllSeq` with `VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ`) |
| EP on, `ep_size >= 16`, prefill, `VLLM_ASCEND_ENABLE_CHUNK_MC2` | ≥16 | `MC2_PREFILL` |
| EP on, `ep_size >= 16`, decode | ≥16 | `MC2` |

So on the recorded launch the framework's own MoE does **not** run the DP
`all_gather`: it takes the token-dispatcher `All2All` path. A vendored model file
that gathers all ranks itself is therefore adding a collective the framework's
dispatcher does not expect for that configuration. Two ablations separate it:
`--enable-expert-parallel` off at the same DP (state → `AllGather`, i.e. the in-tree
pad-then-gather path with the same model file) and `--data-parallel-size 1` (no DP
collectives at all).

## Step 5 — the checks, cheapest first

1. Print `dp_rank`, `max_tokens_across_dp`, the local `num_tokens` and the row
   offset used for the slice **on every rank** — a rank-0-only log cannot show
   disagreement, and disagreement is the whole failure mode.
2. Compute `dp_size × max_tokens_across_dp` and compare it with the gathered row
   count (`544` in #3107): equal → the pad agreed, look at the slice.
3. `--data-parallel-size 1` on the same node (one request): if it passes, the
   fault is in the DP bookkeeping, not in the model's math.
4. `--enable-expert-parallel` off with DP kept: changes the state to `AllGather`,
   which exercises the in-tree pad-then-gather code with the same model file.
5. Ask for the vendored file and its registration diff before proposing a fix —
   the owner is a file the release does not contain
   (`../startup-failures/environment-correspondence.md`, vendored-file section).

## Recorded case

**#3107** vllm-ascend v0.9.1rc2, Pangu Pro MoE, single node
`--data-parallel-size 8 --enable-expert-parallel --max-num-seqs 8
--max-model-len 4096 --max-num-batched-tokens 4096 --enforce-eager
--no-enable-prefix-caching --no-enable-chunked-prefill`, TP=1: the dummy run
passes, a request prints `max_tokens_across_dp 68`, gathers `[544, 5120]`, then
`layer1 output` is `[68, 5120]` of ≈`-5.9e37` and the log stops at layer 2's
`input_layernorm`. The pasted `file:line` names
`vllm_ascend/models/pangu_moe.py`, which the pin does **not** contain (see the
vendored section of `../startup-failures/environment-correspondence.md`), so the
model arithmetic is the reporter's; the framework facts above are what the answer
can pin. The mechanism was never confirmed by the reporter — the recorded session
shipped no answer (12 turns, ~49.6 K characters of reasoning, 661 characters of
message text), so this page records the arithmetic and the checks, not a cause.
