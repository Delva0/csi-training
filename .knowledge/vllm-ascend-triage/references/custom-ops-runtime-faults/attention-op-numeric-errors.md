# Silent numeric wrongness in an attention operator (no crash, no dump)

Scope: `npu_fused_infer_attention_score`, `npu_fusion_attention` or
`_npu_flash_attention_unpad` **runs**, returns a correctly shaped tensor, and the
numbers are wrong — e.g. the output equals the per-token mean of V (softmax
dropped), or is unrelated to an fp32 reference while `scale` still changes its
magnitude. Often filed with a minimal standalone `torch_npu` repro that loads no
framework code, and always without a device dump. A dump / `call aclnnX failed`
is a different class (`index.md`, `aclnn-error-dump-reading.md`); a
`EZ1001 does not support opType` is build/install
(`op-not-registered-for-soc.md`).

## Triage order

1. **Decide the layer from the repro.** If the reported script reproduces with
   `torch` + `torch_npu` only, the defect is at or below the operator: no amount
   of reshape/layout reading in the framework can be the primary cause, and what
   still matters is the caller's combination (step 2), the counter-example in the
   tree (step 3), the coverage gap (step 5) and the fix surface (step 7).
2. **Find which caller uses that exact combination** and whether another call
   site in the same tree avoids it. One grep:
   `rg -n 'sparse_mode\s*=\s*\d' vllm_ascend/ | grep -v <caller>`.
   Recorded #13699 (`v0.23.0rc1`, `f4a08bdd`): the vision encoder is the **only**
   non-paged, long-seq, unmasked, `q_len > 1` FIA site
   (`vllm_ascend/ops/mm_encoder_attention.py:153 _run_vit_fia`, kwargs
   `:169 atten_mask=None`, `:171 input_layout="TND"`, `:172 block_size=128`,
   `:178 sparse_mode=0`). Other FIA `sparse_mode=0` sites are paged / BNSD decode
   or draft-graph paths (`attention/attention_v1.py:748,776,836,860`), and
   non-causal *prefill* goes to `npu_fusion_attention` instead — "unique caller"
   is the fact that turns the report into an owner split, so state the
   combination, not just the flag.
3. **Look for the op the framework already prefers for that gravity.**
   `vllm_ascend/device/utils.py:27-70` (`npu_large_head_prefill_attention`) sends
   non-causal TND prefill to `torch_npu.npu_fusion_attention` with the comment
   *"A2/A3 FIA TND does not support some large head sizes"* — in-repo evidence
   that FIA is not trusted for un-paged full attention, and the cheap candidate
   workaround to benchmark (other call sites: `device/device_op.py:756,1906`,
   `attention_v1.py:1394-1400`).
4. **Check whether the operator source is in the snapshot before arguing
   mechanism.** Built-in aclnn tiling lives in `cann/ops-nn`
   (`fused_infer_attention_score_tiling.cpp`) and is usually listed under
   `unavailable` in `SOURCES.json` (`cann/ops-transformer`, `cann/ge`,
   `cann/opbase` likewise). Then a "NO_MASK is a decode-only tiling class" claim
   is **the reporter's reading of a tree you do not carry**: keep it as a
   hypothesis, cite their sentence, and say which repo would settle it.
5. **Check the caller's own coverage — that part is verifiable without ops-nn.**
   `tests/ut/ops/test_mm_encoder_attention.py:44-51` stubs the op with
   `torch.zeros_like`, asserting only shape / `input_layout` /
   `actual_seq_lengths` (capture path `:53-55` zeroes the out buffer). Numerics
   are never exercised, so "untested combination" is a framework-level fact, not
   a guess. Quote the stub line; it is the reason this shipped.
6. **Name the upstream semantics the downstream class replaced.** At the paired
   vLLM version (`v0.23.0`, `0fc695f`) upstream still has
   `vllm/model_executor/layers/attention/mm_encoder_attention.py:504
   _forward_sdpa`, `:521 vit_torch_sdpa_wrapper`, `:698 forward_native`. A
   native/sdpa fallback is therefore the semantics-preserving fix, and FIA is a
   downstream performance optimization without a guard.
7. **Count the fix sites before proposing a patch.** The same kwargs are
   duplicated across eager, capture (including
   `_npu_fused_infer_attention_score_get_max_workspace`) and the graph-replay
   rebind: `ops/mm_encoder_attention.py:153` / `:223` (kwargs repeated at
   `:258,260,261,266`) / `worker/encoder_acl_graph.py:245`. A one-site fix
   leaves encoder graph mode on the broken path. Quote all three `path:line`.

## Cheap discriminators (no source needed)

- **Uniform average ⇒ softmax dropped, not a scale/layout error.** Compare the
  op output with `v.float().mean(dim=0)` broadcast over heads/tokens; ≈0 proves
  the softmax contribution vanished. `|out| ≈ mean(|v|)` is a weaker, noisier
  version of the same check.
- **`scale` changes magnitude but no setting ever matches** ⇒ scale is applied
  and the failing axis is elsewhere; stop tuning `scale`, `block_size`,
  `pre_tokens`, dtype of the seq-length tensor.
- **Causal `sparse_mode=3` correct + an all-False mask still wrong** ⇒ the failing
  axis is "non-causal, `q_len > 1`", **not** "mask missing". Say it that way so
  the report cannot be closed with "just pass a mask".
- **Sweep one axis at a time** (`T = 64/256/2048`, `H = 1/16`, `D = 64/128`) to
  turn "broken combination" into a shape envelope the op owner can take as a
  `case_json`. A correct decode/LLM path is not coverage: masked or paged
  `sparse_mode=3/4` can be fine while the encoder path is broken, so "the model
  answers text fine" refutes nothing.
- **Performance numbers in the report are not correctness evidence.** Kernel
  microseconds and req/s tables are reporter-measured on their host; use them
  only to corroborate that an unintended path is running.

## Ownership split to state in the answer

Device op (kernel/tiling, `cann/ops-nn`) for the numerics; framework
(vllm-ascend) for choosing the uncovered combination, having no non-FIA fallback
(`forward_oot` `:316` dispatches eager/capture FIA only), and having zero numeric
coverage for it. That split is what makes the report actionable even when the
operator source is absent from the snapshot.

## Recheck conditions

Line numbers move every release, and `sparse_mode` values are not a stable
contract — re-grep both callers at the pinned commit. A second pass checked this
whole list at `f4a08bdd` / `0fc695f` (`:153`, `:169`, `:171`, `:172`, `:178`; capture
kwargs `:254-266`; `encoder_acl_graph.py:245`; upstream `:36`, `:504`, `:521`,
`:698` — all as written, and the UT stub `:44-51` returns `torch.zeros_like`), so
those are citable as they stand. If the release has a native
or `npu_fusion_attention` path for the encoder, the "no fallback" claim is stale
and the report changes shape; say which release you checked.
