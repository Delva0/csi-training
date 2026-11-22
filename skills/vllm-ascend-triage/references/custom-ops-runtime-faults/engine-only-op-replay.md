# An op that faults only inside the engine: the standalone replay is a different call

Scope: the report says the same operator run **inside** the engine faults (a device
dump, an `aclnn`/ATB error, an MTE/DDR out-of-range) while the reporter's own control
— "I pulled the inputs out into a pkl and ran the op standalone, it returns the right
output" — looks clean. The control is the thing to audit first: it is only evidence
when the replay launches the **same op with the same arguments**, and this tree has
several near-identical call sites. Not for a pre-launch refusal (`index.md` rules 1-2)
and not for numbers that are wrong with no dump (`attention-op-numeric-errors.md`).

## Step 1 — resolve the faulting call site from its argument list, not its name

At vllm-ascend **v0.7.3** (`779eebb`, the recorded reporter's pin) `torch_npu`'s paged
attention family is reached from six call sites with *different signatures* (five in
the V0 file, two in V1):

| Site | Op | Distinguishing argument | Phase |
|---|---|---|---|
| `attention/attention.py:845` | `_npu_flash_attention_qlens` | `mask=compress_mask` | pure prefill, chunked prefill **off** (`:831`) |
| `attention/attention.py:867` | `_npu_paged_attention_splitfuse` | **`mask=chunk_mask`**, `seq_len=query_lens`, `context_lens=seq_lens` | mixed prefill+decode ("Splitfuse") |
| `attention/attention.py:887` | `_npu_paged_attention` | **no mask parameter exists**; `context_lens=decode seq_lens` | `DecodeOnly` |
| `attention/attention.py:1088` | `_npu_paged_attention_mla` | MLA branch | `DecodeOnly`, MLA models |
| `attention/attention_v1.py:248` / `:261` | same two paged-attention ops | `:261` passes `mask=attn_metadata.attn_mask` | V1 `DecodeOnly` / V1 mixed |

So a plog line whose `fault kernel_name=` carries `mask`
(`paged_attention_mask_1_mix_aic` in #3263) cannot come from the decode call: that op
has no mask operand at all at this ref. The reporter in #3263 re-spliced
`block_tables` into **`torch_npu._npu_paged_attention`** — the no-mask member — so
their "works standalone" result is a statement about a different kernel and does not
refute the engine-side fault. Say that in the first message instead of deriving
address arithmetic; the two ops' arguments also differ in *meaning*
(`seq_len` is the chunk length and `context_lens` the whole context on splitfuse,
while the decode op's `context_lens` is the length the builder stores as
`seq_lens`), so a replay that keeps the decode call's naming feeds one number where
the other belongs.

**Corollary for any `<op>` family.** Before crediting any standalone repro, grep the
plugin's own call sites for that op and compare the keyword lists
(`rg -n -A12 '_npu_<op>' vllm_ascend/`); two sites in one file routinely differ by an
optional mask, a `seq_len`, or a scale tensor, and the optional operand is exactly
what a "same inputs" dump drops.

## Step 2 — what a pkl roundtrip does not carry (three checks, none provable locally)

1. **The mask.** Only the splitfuse call passes one, and it is host-built (Step 3).
   A dump of "the inputs" that has no mask cannot reproduce that kernel.
2. **The block-table width.** `attention.py:597-602`
   `make_tensor_with_pad(self.block_tables, pad=0, dtype=torch.int32)` pads to the
   longest sequence's block count and pads with block id **0** (a legal block, so the
   padding is not itself out of range). A hand-spliced per-request table has a
   different width, which changes the stride the kernel walks.
3. **The engine's cache tensors.** Both runners call
   `torch_npu.npu_format_cast(<cache>, 2)` — `worker/worker.py:315-316` (V0) and
   `worker/model_runner_v1.py:850` (V1) — **and discard the return value**. The
   non-in-place spelling returns a new tensor, so this does not modify the cache:
   grep `npu_format_cast_` (in-place) before building a mechanism on it. The recorded
   #3263 run spent two turns deciding whether `2` is `ACL_FORMAT_ND` or
   `FRACTAL_NZ`; the answer does not matter while the call is a no-op, and the enum
   lives in `Ascend/pytorch`'s `torch_npu`, which that case's export does not carry
   (no `torch_npu/npu/format.py`).

## Step 3 — the one arithmetic that is worth doing: the mask the splitfuse path builds

`attention.py:595` `max_seq_len = max(max_prefill_seq_len, max_decode_seq_len)`, then
`:614-623`

```python
attn_mask = self._attn_mask_builder.get_attn_mask(max_seq_len, dtype, device)
if attn_mask.numel() > 1 and attn_mask[0][1] > 0:
    attn_mask *= -10000
for i, seq_len in enumerate(seq_lens):
    chunk_mask_list.append(attn_mask[self.context_lens[i]:seq_len])
self.chunk_mask = torch.cat(chunk_mask_list, 0)
```

The mask handed to the kernel is a **row-slice per sequence**: width `max_seq_len`,
total rows `Σ(seq_len − context_len)`. So its size is decided entirely by the request
set — a sequence whose `seq_len` exceeds `max_seq_len` contributes fewer rows than the
launch expects, and a `context_len > seq_len` contributes none. That is a host-side
arithmetic on `seq_lens`/`context_lens`, which is why the discriminating evidence is
the **batch shape at the failure** (first request after a decode already ran? a
prefix-cache hit? a partial prefill chunk?) and not the kernel. `get_attn_mask` grows
the cached square mask when asked for more (`:81-90`), so a state where the printed
`max_seq_len` and a later `seq_len` disagree is a real reproduction target rather than
a tiling theory.

## Step 4 — stop at the component boundary

`_npu_paged_attention*` is not in `vllm_ascend/`; it is a `torch_npu` → ATB op
(`op-plugin`'s `ops/atb/PagedAttentionAtb.cpp`) and **no export carries it**: in that
case `Ascend/op-plugin` exports ~100 files and none of them is under `ops/atb/`, and
neither `Ascend/pytorch` nor any other tree in that view matches `paged_attention`.
One `rg -l` per tree is the whole budget (SKILL.md rule 12's ~6-call cap); report the
absence as a gap and do not re-run it per spelling (`PagedAttention`,
`paged_attention`, `_npu_paged…`). The kernel name is also not resolvable from the
tree — `rg -n 'paged_attention_mask'` over that view's exports returns nothing — so an
answer that explains the fault must say which operand is out of range, and that comes
from the plog/`origin_op_name`, not from the name.

## Step 5 — answer shape

1. Verdict first: the standalone replay is/is not the faulting call (Step 1), the phase
   that faulted, and what that means for the reporter's question.
2. Ranked: (a) the control is a different op — most likely when the mask-bearing site
   is the only one that can mint the kernel; (b) the batch regime that reaches the
   mask-bearing branch (mixed prefill+decode, prefix-cache hit, partial chunk);
   (c) the pin — v0.7.3 predates the analysis date by several releases, so re-running
   the same body on the current image is a labelled rung that cannot be checked from
   the snapshot; (d) host driver/CANN pairing, only after the plog names an op that is
   built in and the same `pc start` appears on several chips.
3. Ladder, cheapest first: one request at a time (a decode-only batch never reaches
   the splitfuse branch) → `--no-enable-prefix-caching` (removes `context_len`) →
   `--enforce-eager` → `ASCEND_LAUNCH_BLOCKING=1` for attribution → the current image.
4. Ask for: the plog `fault kernel_name=` / `origin_op_name` at the failure second,
   the failing step's `seq_lens` / `context_lens` / `block_tables.shape` /
   `chunk_mask.shape`, the dumped `SchedulerOutput`, and the serve line with
   `--block-size` / `--max-model-len` / the prefix-cache flag.

## Recorded case

**#3263** (vllm-ascend v0.7.3, `EZ9999 … fftsplus aicore error`, `mte error info`,
`The DDR address of the MTE instruction is out of range`, `retCode=0x26, [aicore
exception]`, `fault kernel_name=paged_attention_mask_1_mix_aic`, `synchronize stream
failed, runtime result = 507053`; the reporter re-spliced `block_tables` into
`torch_npu._npu_paged_attention` and the pkl replay was clean). The recorded session
ran **46 messages / 984 K prompt tokens / all 16 384 completion tokens and shipped no
message text**: the call sites of Step 1 were grepped on the fourth assistant turn
and `attention.py:800-920` read on the fifth, and the rest of the session went to
re-reading `attention.py` five more times, grepping `Ascend/op-plugin` and
`Ascend/pytorch` for an op neither export carries, and the `ACL_FORMAT_ND` question.
The deliverable was already available at the point the fault kernel name was matched
against the one call site that takes a mask.
