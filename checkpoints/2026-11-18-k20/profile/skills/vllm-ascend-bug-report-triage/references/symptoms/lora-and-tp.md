# LoRA on Ascend with TP>1: the expand/shrink rank contract, warm-up aborts and decode hangs

## Scope

Reports that use `--enable-lora` / `--lora-modules` / `--max-lora-rank`, with or without
`--fully-sharded-loras`, and fail in one of three ways: an abort during the dummy-LoRA warm-up carrying
a message from the Ascend LoRA kernels (`hidden in should be smaller than hidden out`); a device hang
or HCCL stall once decode starts, after the same warm-up was bypassed; or an adapter that loads and
serves but returns wrong text. Fourth, much cheaper shape: the adapter serves **correctly and is simply
slow** (2–3 tokens/s against 10–12 without it) — no abort, no hang, no wrong text — and often as a
second paragraph "with `VLLM_USE_V1=1` the service does not start". Both are decided by the kernel family
and graph mode at the release, not by rank arithmetic (§"It serves and is slow"). Worked case for the
contract: vllm-ascend #16586 (`v0.23.0` + vLLM `0.23.0`, 910B3 ×8, Qwen3.5-27B hybrid GDN, adapter r=8
with `lora_B=0`, `--tensor-parallel-size 4 --enable-lora --max-lora-rank 8 --enforce-eager`). Not the
MoE-side LoRA/all-to-all features, and not the Qwen3.5 GDN model code
(`flashcomm1-and-moe-reduce.md` §A is the SP/cudagraph family).

## Which kernel family the run actually uses

**The module moved**, so locate it before quoting a line (a `git show <tag>:vllm_ascend/lora/punica_npu.py`
at `v0.9.1rc1` dies with `fatal: path '…' exists on disk, but not in 'v0.9.1rc1'` — the worktree is `main`).
One `git ls-tree -r --name-only <tag>:vllm_ascend/lora` answers the layout first:

| refs | files |
| --- | --- |
| `v0.9.1rc1`, `v0.9.2rc1` | `lora/punica_wrapper/{__init__,punica_npu}.py` |
| `v0.10.0rc1`, `v0.10.1rc1`, `v0.10.2rc1` | `lora/punica_wrapper/{__init__,lora_ops,punica_npu}.py` |
| `v0.11.0` … `v0.23.0`+ | `lora/{__init__,lora_ops,punica_npu,fused_moe,utils}.py` |

At `v0.9.1rc1` there is no selection at all — the wrapper imports the upstream PyTorch-native ops at
`:6-8` and nothing else (see §"It serves and is slow").

`vllm_ascend/lora/punica_npu.py:23-40` at `v0.23.0` picks the implementation from the device type and
the configured rank:

```python
if get_ascend_device_type() == AscendDeviceType._310P or (
    self.lora_config is not None and self.lora_config.max_lora_rank >= 128
):
    from vllm.lora.ops.torch_ops import bgmv_expand, sgmv_expand, ...   # upstream torch ops
else:
    from vllm_ascend.lora.lora_ops import bgmv_expand, sgmv_expand, ...  # Ascend custom ops
```

So a reporter with `--max-lora-rank 8` runs the Ascend C++ kernels and their `TORCH_CHECK`s, while the
same code path with `--max-lora-rank 128` silently switches to the upstream torch ops — a rank raise is
therefore a *diagnostic* (and a slow workaround), not a tuning knob. `add_expand` itself
(`punica_npu.py:214-255`) loops `_apply_expand(y, x[slice_idx], lora_b_stacked[slice_idx],
output_slices[slice_idx])`, so the per-slice kernel receives the **output width of that sub-module** as
`slice_size` and the shrink result as `x`.

## The contract the kernels enforce (verified at `v0.23.0`)

`csrc/torch_binding.cpp`:

| op | used by | check | line at `v0.23.0` |
| --- | --- | --- | --- |
| `bgmv_expand` | decode | `TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out")` | `:476` (`:464` is the signature) |
| `sgmv_expand` | prefill / dummy warm-up | same message | `:561` (`:552`) |
| `bgmv_shrink` | decode | `TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out")` | `:435` |
| `sgmv_shrink` | prefill | same message | `:519` |

Read the message direction first: *smaller than* is the **expand** op (rank width vs output slice),
*greater than* is the **shrink** op (hidden size vs rank). They are different faults. `git grep -n` the
verbatim string at the reported ref (`git grep -n '<message>' <ref> -- csrc`); never assume the op from
the traceback, because `punica_npu.add_expand` / `_apply_expand` is the caller of both.

Why relaxing the check does not rescue the run: the same function derives the kernel's rank **from the
activation** — `int lora_rank = x.size(1);` (`bgmv_expand` `:489`, `:576` in `sgmv_expand`) — and hands
it to the AIV kernel together with `weight`, allocated with in-dim `max_lora_rank`. A wider input does
not merely trip a guard, it makes the kernel read `weight[num_loras, hidden_out, lora_rank]` with a rank
larger than the row length it was sliced with: a silent out-of-bounds read inside a fused op, which is
the plausible source of a device hang or garbage rather than a Python error. Keep both halves together
in the write-up. `x` arrives from the shrink path (`_apply_shrink`/`_shrink_prefill`, same file), so the
width question is really "was the shrink output all-gathered across TP before the expand?".

## Where the two numbers come from

- `slice_size` = `output_slices[i]` = `lora_b_stacked[0].shape[2]` (vLLM `v0.23.0`
  `vllm/lora/layers/base_linear.py:150`), i.e. the **per-rank** output width of the base layer (for a
  `MergedColumnParallelLinear` of `[48, 48]` at TP=4 that is 12; report-supplied for the GDN
  `in_proj_ba`). At the same ref `create_lora_weights` (`:99-150`) sizes a column-parallel layer's
  `lora_a_stacked` rank axis as `max_lora_rank`, or `max_lora_rank // tp_size` under
  `fully_sharded_loras`, while `lora_b_stacked` is always
  `(max_loras, 1, output_size_per_partition, max_lora_rank)` — the input dim of B stays
  `max_lora_rank` in both modes.
- `x.size(1)` = the LoRA **rank** width of the shrink buffer handed to the expand op.

## Why the rank width is `TP × rank` in the default configuration

At vLLM `v0.23.0`, `vllm/lora/layers/column_parallel_linear.py::_mcp_apply` (`:24`) is the `apply` of
`ColumnParallelLinearWithLoRA` (`:338`) **and** of all four `*WithShardedLoRA` classes (`:516`, `:558`,
`:594`, `:643`), and it all-gathers the shrink buffer unconditionally:

```python
buffers = tensor_model_parallel_all_gather(buffers)      # :64
```

Only `tp_size == 1` short-circuits to `_apply_base_forward` (`:332-337`). The gathered axis is the rank
axis, so the expand input is `TP ×` the per-rank rank width, and that width is `max_lora_rank //
tp_size` only under `fully_sharded_loras` (`base_linear.py:110-116`). With TP>1 and the **default**
(non-sharded) LoRA the expand input is `TP × max_lora_rank` wide while `slice_size` is only the per-rank
output shard, so the `TORCH_CHECK` fires as soon as `TP × max_lora_rank > output_per_rank`, which small
merged slices hit immediately.

## Workaround ladder (all `hypothesis` unless the reporter confirmed it)

1. `--fully-sharded-loras` — the gathered width becomes `max_lora_rank` instead of
   `TP × max_lora_rank` (same `base_linear.py` sites), so the check reduces to
   `max_lora_rank <= output_per_rank`. Derive both sides for the module the log names before proposing
   it: it fixes the reported TP=4 case (12 ≥ 8) and still fails the TP=8 case (6 < 8), matching the
   reporter's own statement that TP=8 is not recoverable. Cost: S-LoRA sharding of `lora_A` (extra
   communication), and this path must exist at the tag — check that the `*WithShardedLoRA` classes are
   registered there.
2. Lower `--tensor-parallel-size` until every module satisfies `width <= output_per_rank`; TP=1 is a
   diagnostic. The gathered width scales with TP, so more ranks only make it worse, and
   `--max-lora-rank` cannot help the inequality (larger ranks move the wrong side of it).
3. Raise `--max-lora-rank` past 128, which switches the run to the upstream torch ops (see the kernel
   selection above) — cheap to test, changes numerics and speed.
4. Move to a build containing the fix (below); say which tag, not "upgrade".

Falsifier for (1): run the reporter's command with `--fully-sharded-loras` at TP=4 — warm-up passes or
the same `TORCH_CHECK` fires. One command on their hardware settles it.

## Ascend-side history (provenance, not the mechanism)

- `vllm_ascend/lora/utils.py` at `v0.23.0` registers four Ascend-only LoRA classes **ahead of** the
  upstream ones in `vllm.lora.utils._all_lora_classes` (`:74-88`), gated to
  `type(source_layer) is AscendQKVParallelLinear` (`:22-71`). They inherit upstream `apply`, i.e. the
  gather above is not overridden on Ascend; the layers they do not replace keep the upstream classes.
  That file changes shape between releases (a later `patch_mcp_apply_non_fully_sharded()` monkeypatch
  was added and then deleted), so read it at the tag — a `main` read misleads in both directions.
- #11940 (`b80fa2678`, 2026-07-30) deleted that monkeypatch as redundant and, in the same commit,
  changed the shrink kernels' `DataCopy` → `DataCopyPad` because `maxLoRARank_ × sizeof(dtype)` is not
  block-aligned for every rank/dtype (`bgmv_shrink.cpp`, `sgmv_shrink.cpp`), and repaired LoRA
  cudagraph replay in `attention_v1.py`. `git tag --contains b80fa2678` ⇒ first tag `v0.27.1rc1`;
  **not** an ancestor of `v0.23.0`, and not a demonstrated fix for an expand-check abort. It is the
  only Qwen3.5 + TP + LoRA fix in the clone, so call it a lead, and do not tell a v0.23.0 reporter it
  is available without checking the ancestry of the tag they can install.
- 310P LoRA in graph mode is a separate fix: #16457 (`7c71062d7`, 2026-09-23), first tag `v0.30.0`.
- The #16586 reporter's "local patch in `punica_npu.py::add_expand`, ~lines 245-256" is their own text;
  the `add_expand` body at `v0.23.0` is the plain per-slice loop, so the quotation matches the tag but
  the patch is not upstream code. Likewise "the gathered columns are identical replicas" and the
  "non-contiguous view → illegal memory access → freeze" half are their account of their own patch:
  `reported`, to be re-derived. Prefer a fix at the class-selection / kernel-contract level.

## It serves and is slow: the pre-`v0.10.1rc1` torch-native wrapper (#1629)

A report that "Multi-LoRA inference is slow, 2–3 tokens/s" on the `0.9.x` line, usually with
`--enable-lora --max-loras 3 --tensor-parallel-size 2` and a 32B dense model, is a different class from
the abort/hang above: no kernel `TORCH_CHECK` fired, so nothing is being over-read; the cost is in *how
many* small operations one decode step issues.

Verified at `v0.9.1rc1` (tag `c30ddb8`): `vllm_ascend/lora/punica_wrapper/punica_npu.py:6-8` is

```python
from vllm.lora.ops.torch_ops import (bgmv_expand, bgmv_expand_slice,
                                     bgmv_shrink, sgmv_expand,
                                     sgmv_expand_slice, sgmv_shrink)
```

with `class PunicaWrapperNPU(PunicaWrapperBase)` below it — the **upstream torch** implementation, i.e.
per-layer, per-slice masked `bmm`/`index_select` shapes rather than one fused Ascend kernel, and no rank
or device-type switch to escape it. Every LoRA-adapted linear therefore contributes several tiny ops per
step, so the step is launch- and dispatch-bound; the reporter's own contrast is 10–12 tokens/s without the
adapter against 2–3 with it on DeepSeek-R1-Distill-Qwen-32B, TP2, `--enable-lora --max-loras 3`.

- **Thread levers** `[reported]`: a maintainer says the V1 engine uses ACL Graph by default and that graph
  does not support LoRA, so add `--enforce-eager` (the reporter's second run with `VLLM_USE_V1=1` died at
  startup — that is the graph/LoRA gate, not the slowness); the punica operator efficiency is acknowledged
  as an improvement item, with the reporter pointed at the tuning guide. The issue was closed with #2113
  and #2325 merged to `main`.
- **The V1 start failure is graph capture, not LoRA itself.** vLLM `v0.9.1` disables `torch.compile` for
  LoRA **only in V0** — `vllm/config.py:4460-4465` is guarded by `(not envs.VLLM_USE_V1)` — so a V1 run
  keeps PIECEWISE compile plus cudagraph capture and meets ops that have no graph capture at that ref. The
  maintainer's `--enforce-eager` (site spelling for `--enable-eager`) is that gate.
- **The device-selected implementation arrived at `v0.10.0rc1`** (`if is_310p(): <torch_ops> else: from
  vllm_ascend.lora.punica_wrapper.lora_ops import …`). The 310P fallback and the later
  `max_lora_rank >= 128` branch (§"Which kernel family") do **not** exist at `0.9.x`, so "raise
  `--max-lora-rank`" is not a lever there.
- **Ancestry (this is the deliverable)**: #1884 `540336edc` "Add Custom Kernels For LoRA Performance"
  adds `lora_ops.py` with **first tag `v0.10.0rc1`**; #2325 `3648d18e6` carries the identical subject but
  is later, and #2113 `c0f0b7081` ("Support capture custom ops into aclgraph") is the graph half — both
  first contained in **`v0.10.1rc1`** (`git tag --contains <sha>`), and none is in `v0.9.1rc1`. A subject
  search returns both custom-kernel PRs, so check the number the thread names. For this reporter
  `--enforce-eager` buys a *starting* server, not speed; the answer is a version move to
  `v0.10.1rc1`-or-later (cost: a release jump and its own pin set), `verified` by ancestry only — no
  throughput was measured here.
- **The 310P twin (#1812, image `v0.9.2rc1-310p`, 300I Duo, Qwen3-8B, 9 → 2 tok/s)** is the same
  answer with a different ask: the reporter wondered whether --enforce-eager "can speed up", and the thread's
  closer (paulyu12) is the #1629 sentence — 「LoRA + Aclgraph Mode has been supported since PR #2113 and #2325
  were merged into main … This issue will be closed.」 At the reported tag `punica_npu.py:6-8` still imports the
  upstream `torch_ops` and `platform.py:236` wires that wrapper for the NPU, so the lever is a version move
  (first tag `v0.10.1rc1`), and `--max-lora-rank 128` is not available on `0.9.x` (that branch first exists at
  `v0.10.0rc1`). Worth one line in the reply: on 310P the graph half (#16457, first tag `v0.30.0`) may still
  force `--enforce-eager` after the move, so the gain there is the fused kernels, not the graph.
- **Falsifier / what to ask for**: same prompt and config with the adapter removed versus loaded. A
  slowdown that tracks the number of LoRA layers while device utilisation stays low is this class; flat
  memory pressure or a stall instead sends you back to the contract section.

## Check order for a new report

1. Which kernel family: locate the module at the tag (`git ls-tree -r --name-only <tag> | grep
   punica`), then `--max-lora-rank` and the device type (`git show
   <tag>:vllm_ascend/lora/punica_npu.py | sed -n '20,45p'`; at `v0.9.1rc1` read
   `lora/punica_wrapper/punica_npu.py:1,20p` instead). A `torch_ops` import with no branch means
   every LoRA linear runs the upstream torch path.
2. Whether the failing module is a MergedColumn (QKV/GDN) and whether `--fully-sharded-loras` is set;
   read `vllm_ascend/lora/utils.py` and the pinned upstream `vllm/lora/layers/` **at the tags**.
3. Print the actual numbers instead of arguing: `x.size(1)`, the slice size, TP, rank. In a report with
   no logs, the ratio between them is the discriminator.
4. Only then look for a fix commit; LoRA fixes are frequently in `csrc/kernels/*.cpp` (CopyOut
   alignment, rank divisibility) and never appear in a Python-file pickaxe.

## Recheck conditions

The kernel-family selection, the `utils.py` subclass/monkeypatch set and the `TORCH_CHECK` text all
change between releases; re-read them at the reported tag before quoting. Nothing here was reproduced on
hardware — no NPU is available.
