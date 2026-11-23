# Top-k logprobs come back containing `-inf`, while the GPU run of the same script is finite

## Scope

An offline `LLM(...)` script or a serve request with `logprobs=k` / `top_logprobs=k`
returns `-inf` for entries **inside** the returned top-k list on ordinary prompts, while
the same code on a GPU returns every entry finite; often with "对采样得到的效果影响比较大".
Recorded #2934 (vllm-ascend `v0.9.1` `0740d10` + vllm `v0.9.1` `b6553be`,
Qwen3-8B, `SamplingParams(temperature=0.8, top_p=0.95, logprobs=20, n=2)`).

Not this page: finite logprob **deltas** against a golden (that is the platform numeric
floor, `cross-platform-logprob-parity.md`), the sampled *text* degenerating
(`constant-token-degeneracy.md`), corrupted tokens at `temperature>0`
(`sampler-side-stream-corruption.md`), or `prompt_logprobs`/`echo` on a PD decode node
(`../pd-disaggregation/prompt-side-outputs-on-decode-node.md`).

## G1: `-inf` is a mask/extreme-value question — say so, then read the ids

`compute_logprobs` is `logits.log_softmax(dim=-1, dtype=torch.float32)` on both sides
(`vllm/v1/sample/sampler.py:136-137` at the #2934 pin). For a finite logit `x`,
`x - max - log Σ exp(x-max)` is finite, so a token whose true value should be ≈ -5 cannot
print `-inf` unless the row already held `-inf` (a mask) or an extreme/NaN value. Hence
the parity page's bucket statistic (`mean|Δ log p|/(1-p)`) is **inapplicable** — undefined
against `-inf` — and "the NPU softmaxes in low precision" is not the first hypothesis when
the identical request on GPU is finite.

Two readings, decided by the **ids**, not by the values:

1. **A run of the vocabulary's lowest ids, ascending**, filling exactly the slots the
   finite entries left ⇒ `torch.topk` tie-breaking over a region of equal extreme values,
   i.e. the row the top-k read was **masked** (a reading, not source-verified: confirm
   with the `top_p=1.0` dump below). #2934's screenshot OCR is the worked read — its
   `-inf` entries decode as `!`, `"`, `#`, `$`, `%`, `&`, `'`, `(`, `)`, `+`, `/`,
   `0`…`4`, i.e. ids 0,1,2,… in order, after a varying number of finite entries (1, 5, 13,
   18 …). Read the decoded **token text**, not the printed digits: that OCR turns `0` into
   `9`/`ø`/`le` and `%` into `X`, and the id column is where it fails.
2. **Scattered or fixed high ids, or the same `-inf` set under every body** ⇒ the logits
   themselves carry the extreme values. That is the padded-vocab / `lm_head` /
   quantisation lane, not the sampler.

The reporter's own objection ("可以看到图里-inf的rank也是不一样的，不太像是mask掉的") is answered
by the source: ranks printed for top-k entries are **list positions**
(`topk_ranks = range(1, num_logprobs + 1)`, `vllm/v1/engine/logprobs.py:182-192`), and only
the first entry — the sampled token — carries a true full-vocab rank (#2934's screenshot
shows `rank=639` on that one). A masked tail therefore starts at a different position in
every row, so a varying rank trail is what masking predicts, not evidence against it.

## Where a mask could reach the logprobs at the pin — facts, then the open part

Both trees read at vllm-ascend `0740d10` / vllm `b6553be`:

- Upstream order is logprobs first: `raw_logprobs = self.compute_logprobs(logits)`
  (`vllm/v1/sample/sampler.py:37-39`), and only then `apply_temperature`
  (`logits.div_(...)`, `:82-83`, in place), penalties and `topk_topp_sampler`;
  `gather_logprobs` is `torch.topk` over those fp32 logprobs plus the sampled token
  (`:165-178`). On the upstream path a top-p mask therefore cannot appear in the returned
  list — which is why the GPU run of the same script is clean.
- Downstream has **no** `vllm_ascend/sample/sampler.py` at this ref (only
  `sample/rejection_sampler.py`). The sampler patch is
  `vllm_ascend/patch/worker/patch_common/patch_sampler.py`: `:104` installs
  `Sampler.apply_min_p` unconditionally, and `TopKTopPSampler.forward_native` is replaced
  **only when `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` is set** (`:105-106`;
  `vllm_ascend/envs.py:53-56`, default `'0'`, commented "experimental"). Its
  `apply_top_k_top_p` (`:50-84`) routes to
  `torch_npu.npu_top_k_top_p(logits, p, k)` only when **both** `k` and `p` are given, and
  otherwise masks **the caller's tensor in place** (`logits.masked_fill_(-inf)`), whereas
  upstream masks a sorted copy and returns `logits_sort.scatter(...)`
  (`vllm/v1/sample/ops/topk_topp_sampler.py:184-205`) — a new tensor. A top_p-only request
  (`top_k` unset ⇒ `top_k=None`, `vllm/v1/worker/gpu_input_batch.py:584`) never reaches the
  fused op; it takes the in-place branch. **This is v0.9.1 only**: from v0.11.0 a different
  flag (`VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION`, default `'1'`) swaps the whole
  `Sampler` and its fused branch also accepts `k is None`, so a `top_p`-only body can be on
  it — check the pin (`sampler-implementation-gates.md`) before reusing this sentence.
- `vllm_ascend/patch/__init__.py:153-167` *documents* patching
  `Sampler.apply_top_k_top_p`; upstream `Sampler` has no such method at v0.9.1 — quote the
  assignments, not that comment (SKILL.md rule 11).

What the paste cannot settle: whether the logits buffer `compute_logits` returns is reused
across steps (graph/`torchair` static output buffers would let an in-place mask survive
into the next step's `raw_logprobs`). Name it as the open question with its check and stop;
#2934's recorded run read the id pattern correctly on its 8th turn and spent the rest of
the budget on that leak path, shipping **no answer at all**.

## Action ladder (one launch each, all cheap)

1. `echo $VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` — `0`/unset closes the in-place lane above and
   the logits own it; `1` puts the downstream rewrite on the path.
2. Same body with `top_p=1.0, top_k=-1` (`logprobs=20` unchanged). `-inf` gone ⇒ the
   filter/mask lane; still there ⇒ the logits. This single launch is the discriminator, and
   it is the first thing to hand a reporter who says "我再试试吧".
3. In the same `top_p=1.0` dump, the previously `-inf` ids come out finite and ordinary
   (≈ -10…-16 at Qwen ids 0-25) — that pair of dumps is the proof, and it is what replaces
   a screenshot.
4. `enforce_eager=True` / `--enforce-eager` on the identical body — removes the ACLGraph
   path the buffer-reuse hypothesis needs; check the paste carries `Compilation disabled,
   using eager mode by default` before crediting it (SKILL.md rule 15's ablation trap).
5. Greedy control (`temperature=0`): greedy never reaches the top-p mask, so `-inf` under
   greedy moves the lane to the logits.
6. Ask for the raw `out.logprobs` **text** (ids and values, one prompt), the GPU run of the
   same script, and the env block; an OCR of a screenshot is not an id dump.

## What the deliverable says

- Verdict first: not the platform's numeric floor and not expected vLLM behaviour —
  upstream computes the top-k logprobs *before* filtering on both platforms
  (`sampler.py:37-39`), so a masked tail on NPU alone is the anomaly, and one env echo plus
  one `top_p=1.0` rerun decides which lane owns it.
- Answer the "ranks differ" control with the rank semantics above, in the report's own
  language.
- Name the lane that survives the two checks, keep the mechanism open where the artefacts
  stop, and request the next artefact instead of re-deriving how a mask reaches the
  logprobs from source.

## Version scope

Verified at vllm-ascend v0.9.1 `0740d10` + vllm v0.9.1 `b6553be`. Later refs moved the
sampler downstream — `vllm_ascend/sample/sampler.py` exists by v0.20.2rc1 with
`AscendTopKTopPSampler` (`:64-78`,
`../custom-ops-runtime-faults/custom-op-ownership-and-dispatch.md`) — so re-grep before
quoting: `rg -n 'apply_top_k_top_p|log_softmax|logprobs' vllm_ascend/sample/`.
