# Decode-only step: the `seq_lens` request count against the ACL-graph-padded token count

## Report shape

The engine starts, serves for a long time (recorded #2822: a 1319-sample `lm_eval`
gsm8k run reached `1064/1319`) and **then** EngineCore dies in a host-side PyTorch
reshape — no device frame, no `ERR`/`EZ` code, no plog line:

```text
File "/__w/vllm-ascend/…/vllm_ascend/attention/attention_v1.py", line 379, in _forward_decode_only
    query = query.view(batch_size, 1, self.num_heads * self.head_size)
RuntimeError: shape '[255, 1, 3072]' is invalid for input of size 786432
```

The line number is checkout-specific
(that paste came from a CI build, one line off the tag it names): grep
`def _forward_decode_only` and the `query.view(batch_size` in it, never the number.

The model is one whose HF config carries `sliding_window` (Phi-3/Phi-4-mini, Gemma-2/3,
Ministral) — but do not argue that from a recalled `config.json`: the pasted frame
*is* the proof, because the branch that reshapes is only reachable when
`self.sliding_window is not None`.

## Three numbers, one division — the classification

The message already holds both counts. Divide them.

- `H*D` = the last dim of the requested shape (`num_heads * head_size`) — 3072 here.
- `M / (H*D)` = the rows `query` actually has, i.e. the **token** count of this step
  after the runner padded it to an ACL-graph bucket — 256 here.
- `N` = the first dim the code used, `attn_metadata.seq_lens.shape[0]` — the **request**
  count of the same step, 255 here.

If `M/(H*D)` is the next capture bucket above `N`, the step ran padded and the two
numbers are the two units of the decode path; nothing was launched, so no tiling,
plog, card-health or `ASCEND_LAUNCH_BLOCKING` check can move it.

```bash
uv run --no-project python - <<'EOF'
N, M, H, D, max_num_seqs = 255, 786432, 24, 128, 256   # N, M from the message
rows = M // (H * D)
buckets = [1, 2, 4] + list(range(8, max_num_seqs * 2 + 1, 8))  # vLLM V1 default
print("rows", rows, "| seq_lens", N, "| pad target", min(b for b in buckets if b >= N))
EOF
# rows 256 | seq_lens 255 | pad target 256
```

Direction matters. `rows == N × (1 + K)` is the speculative-decoding variant, whose
owner and checks are `../spec-decode-mtp/fia-tnd-seqlen-mismatch.md`; a reshape whose
two counts are both **tokens** — `num_tokens` from `q.shape[0]` and a row count that is
the *target's* ACL-graph pad, on a `cos = cos.view(num_tokens, -1)` in
`ops/triton/rope.py` — is `../spec-decode-mtp/rope-cos-sin-slice-token-mismatch.md`;
`rows < N` means
the short side is elsewhere and this page's arithmetic has to be re-derived before it
is quoted.

## Why each number is what it is (one source pass at vllm-ascend v0.10.1rc1)

| Site | Number |
|---|---|
| `attention/attention_v1.py:194` (`AscendMetadataBuilder.build`) — `seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]` | `N` = **requests**; trimmed to `num_reqs`, never padded to the graph size at this ref |
| `worker/model_runner_v1.py:1068-1072` — `if (self.use_aclgraph and total_num_scheduled_tokens <= self.aclgraph_batch_sizes[-1]): num_input_tokens = self.vllm_config.pad_for_cudagraph(total_num_scheduled_tokens)` | `M/(H*D)` = **tokens**, i.e. `num_input_tokens` |
| `worker/model_runner_v1.py:376-377` — `_use_aclgraph()` = `cudagraph_mode != NONE and level == PIECEWISE and not enforce_eager` | the gate: eager or `cudagraph_mode: NONE` removes the pad entirely |
| `attention/attention_v1.py:491` — `num_tokens = query.shape[0]`, then `:527` `query = query.view(-1, num_heads, head_size)`; only `key`/`value` are sliced to `num_actual_tokens` (`:518`, `:538-539`) | why the padded rows reach the decode branch |
| `attention/attention_v1.py:375-378` — `if self.sliding_window is not None:` → `batch_size = attn_metadata.seq_lens.shape[0]` → `query.view(batch_size, 1, …)` | the raise; the **`else`** branch (`:401-411`) hands the padded `query` straight to `_npu_paged_attention` and derives no host reshape, which is why only the sliding-window models on that pin see this crash |

The bucket list is upstream's: `vllm/config/__init__.py:3770-3774` builds `[1, 2, 4] +
range(8, cuda_graph_sizes[0] + 1, 8)` from `cuda_graph_sizes`, default
`[min(max_num_seqs * 2, 512)]` (`vllm/config/scheduler.py:243-244`), and
`vllm/config/compilation.py:520-532` maps every size up to the next bucket — so
255 → 256 and 5 → 8. vllm-ascend then *trims* that list by model depth
(`vllm_ascend/utils.py:292-372` `update_aclgraph_sizes`, first and last preserved), so
ask for the two lines it logs at `:324`/`:341` and `:363` — `Calculated maximum
supported batch sizes for ACL graph: N` and `Adjusted ACL graph batch sizes for
<Model> (layers: L): A → B sizes` —
before quoting which counts pad.

## Ladder

1. **Re-send the same body with five concurrent requests** — the cheapest confirmation
   and no restart: `5` is not a bucket, so that step pads `5 → 8` and must raise the
   same message (`shape '[5, 1, 3072]' is invalid for input of size …`). A single-request
   smoke test can never reproduce this, and the reporter's "it worked for a thousand
   samples" is expected: saturation at `max_num_seqs` keeps the decode count inside one
   bucket (256) for most of the run and the crash arrives when the stream drains to 255.
2. **`--enforce-eager`** (or `-O.cudagraph_mode=NONE`) — one restart, removes
   `pad_for_cudagraph` from the decode path; the proof line is `Compilation disabled,
   using eager mode by default` (`vllm_ascend/platform.py:146` at this pin).
3. **Move to a build whose call site carries the guard** — this is the repair, and
   whether it is in the reporter's build cannot be answered from a snapshot that
   predates it. Reporter-stated for #2822: fixed by PR #2830 one day after filing.
4. **`--disable-sliding-window`** only when the window is ≥ `max_model_len`, where it is
   semantically a no-op and takes the paged branch (`vllm/engine/arg_utils.py:509`
   sets it; `vllm/config/__init__.py:743-746` nulls `hf_text_config.sliding_window`).
   On a real SWA model it changes the arithmetic of the model's own attention — say so.

Not levers: `--max-num-batched-tokens`, `--max-model-len`, `--gpu-memory-utilization`,
card health, a plog timeline (plog records device tasks; a Python `.view` leaves no
record), or a tiling argument — the raise happened on the host, before any kernel.

## The repair a later pin carries (quote it for "is it fixed")

At vllm-ascend **v0.14.0rc1** the body still reshapes with the request count
(`attention_v1.py:716-719`, `_forward_fia_slidingwindow`), but the call site now guards
it (`:762-766`):

```python
if (
    attn_metadata.attn_state == AscendAttentionState.DecodeOnly
    and self.sliding_window is not None
    and attn_metadata.seq_lens.shape[0] == query.size(0)
):
    return self._forward_fia_slidingwindow(query, attn_metadata, output)
```

A padded step fails the equality and falls through to the generic FIA path
(`forward_impl` at `:884-896`, which also routes `sliding_window is None` to paged
attention). That guard is the fix shape: **the branch is only valid when the two counts
agree**, not "make the resize lazy" and not a `try/except`.

## Recheck before quoting

- Another generation can pad the other side: at the v0.11-era refs the *builder*
  pads `seq_lens`/`block_table` in a `DecodeOnly and num_input_tokens >
  num_actual_tokens` branch (`attention_v1.py:228-247` as read in
  `../spec-decode-mtp/fia-tnd-seqlen-mismatch.md`), and then the request count is no
  longer the short one. Grep `seq_lens` inside `build()` first; the division stays the
  classification, only the owner of the pad changes.
- `pad_for_cudagraph`'s site drifts by ref (`vllm/config/__init__.py:3431-3436` is
  `VllmConfig.pad_for_cudagraph` at vllm v0.10.1.1, and
  `../spec-decode-mtp/fia-tnd-seqlen-mismatch.md`’s table reads the call site as
  `worker/model_runner_v1.py:588-591` at a later one).
- The `else` branch tolerating the pad is a reading of *this* pin's code, not proof the
  paged path is numerically right on a padded step; do not claim it.

## What the recorded run cost

#2822's own session computed `786432 / 3072 = 256` in its opening reasoning, then spent
the remaining ~16 K output tokens on `graph_pad_size`, DP-sync and `sliding_window`
plumbing and emitted **no message text after its first turn** — cut at the cap with the
arithmetic already in hand. The division plus rung 1 is the first paragraph; the source
table above is what a maintainer asks for afterwards (`SKILL.md` rule 1).
