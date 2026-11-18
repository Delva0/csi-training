# Guided decoding / grammar bitmask: the CPU round-trip and the site-packages skew rule

## Scope

Failures inside structured output — tool calling (`--tool-call-parser` + `--enable-auto-tool-choice`),
JSON-schema / `json_object` guided decoding, `reasoning-parser`, any client that sends tools — while the
model itself loads and serves fine. Two owners, decided by the deepest frame: a binding-signature
`TypeError: <fn>(): incompatible function arguments` in `site-packages/xgrammar` (or `outlines_core`,
`llguidance`, `safetensors`, `torch_npu`) is the installed-wheel class of §A–§C, while an
`AssertionError`/`Failed to advance FSM` out of `vllm/v1/structured_output/` is upstream's own
bitmask-loop bookkeeping and belongs in §D. Worked cases: vllm-ascend #5894 (§A), #6421 (§D).

Not for: a `KeyError`/shape error in `vllm_ascend/` reached from the sampler (then it is the sampler
class), a request that never reaches the parser (`startup-and-stall.md`), and weight-file integrity
failures in `safetensors` (`quantization-load.md` §4).

## Common rules

- **A `site-packages` frame is evidence about the *image*, not about the reported repo.** Fingerprint the
  installed dependency's version from its own `file:line` (each release shifts them), then compare it
  with the pin in `requirements/common.txt` of the vLLM tag paired with the reported vllm-ascend
  release. The deliverable is a version/image statement plus the upstream fix, not a `vllm_ascend/` line.
- **`Invoked with types: ...` means the caller passed a type the extension rejects.** The owning
  contract is the *pair* (caller version, installed extension version); neither side alone is "the bug",
  and pinning one without the other can re-break it.
- **On Ascend the third-party CPU branch is live while it is dead on GPU.** vllm-ascend deliberately
  moves tensors to CPU before some upstream calls, so a bug that only exists in a library's CPU path is
  an NPU-only bug no GPU CI can see. Read the Ascend call site before assuming the library's documented
  device dispatch.
- The clones are read-only and the failure is in an installed wheel: do not try to patch the C++
  binding. Name the version delta and the upstream PR.

## A. Worked case #5894 — `apply_token_bitmask_inplace_cpu(): incompatible function arguments`

Reported: image `vllm-ascend:v0.13.0rc1` (log banner `V1 LLM engine (v0.13.0)`), GLM-4.5-Air, 8×NPU,
TP8 + `--enable-expert-parallel`, `--max-model-len 128K --max-num-seqs 16`, prefix caching + chunked
prefill, `--tool-call-parser glm45 --reasoning-parser glm45 --enable-auto-tool-choice`, served to an
agent client that sends tools. The service runs, then under normal concurrency every rank raises at
once:

```text
(EngineCore_DP0) RuntimeError: Worker failed with error 'apply_token_bitmask_inplace_cpu():
  incompatible function arguments. The following argument types are supported:
    1. apply_token_bitmask_inplace_cpu(logits_ptr: int, logits_shape: tuple[int, int],
       logits_strides: tuple[int, int], bitmask_ptr: int, bitmask_shape: tuple[int, int],
       bitmask_strides: tuple[int, int], vocab_size: int, indices: collections.abc.Sequence[int] | None) -> None
  Invoked with types: int, tuple, tuple, int, tuple, tuple, int, torch.Tensor'
```

Deepest frames: `vllm_ascend/worker/model_runner_v1.py` (`sample_tokens`/`_sample`) →
`vllm/v1/structured_output/utils.py` → `xgrammar/matcher.py:142` →
`xgrammar/kernels/apply_token_bitmask_inplace_cpu.py:46`. The frontend shows
`EngineDeadError: EngineCore encountered an issue`, each rank `Parent process exited, terminating
worker`.

### Mechanism (all verified in the clones)

1. At `v0.13.0rc1`, `vllm_ascend/worker/model_runner_v1.py:1561-1566` applies the grammar mask on
   **CPU** on purpose: `logits = logits.to("cpu").float()` (`:1563`) then `apply_grammar_bitmask(...)`
   (`:1564`), with the comment "the apply_grammar_bitmask uses torch.compile to optimize this, ascend
   does not support it now", then `logits.to(self.device)`. So Ascend takes xgrammar's CPU path on every
   guided step, where upstream's `gpu_model_runner` keeps device logits.
2. vLLM `v0.13.0` `vllm/v1/structured_output/utils.py:114` sets
   `skip_out_indices = len(out_indices) == logits.shape[0]`; when it is **not** skipped, `:116-124`
   builds `index_tensor = torch.tensor(out_indices, dtype=torch.int32).to(logits.device)` and `:126`
   calls `xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)` — a
   `torch.Tensor`, with the comment that xgrammar "expects a python list of indices but it will actually
   work with a tensor" (true for the GPU/triton path only).
3. xgrammar `v0.1.27` (the pin in vLLM `v0.13.0` `requirements/common.txt`) passes that object straight
   through: `python/xgrammar/kernels/apply_token_bitmask_inplace_cpu.py:46` →
   `_core.kernels.apply_token_bitmask_inplace_cpu(..., vocab_size, indices)`, whose nanobind signature
   is `std::optional<std::vector<int>> indices` (`cpp/nanobind/python_methods.cc:83-91`, registered
   `cpp/nanobind/nanobind.cc:385-396`) — i.e. `Sequence[int] | None`, exactly as rendered in the error.
   The image's binding is therefore the pinned release, and the *caller* is what it rejects.
4. Upstream fix: mlc-ai/xgrammar #506 / PR #507 = `365ea8b` (2026-01-03) converts a tensor to `.tolist()`
   before the call; **first tag containing it: `v0.1.30`** (`git tag --contains 365ea8b`).
5. Pin history (vLLM `requirements/common.txt`): `v0.13.0` `xgrammar == 0.1.27`; `v0.14.0`–`v0.17.0`
   `== 0.1.29` (also broken); `v0.18.0`+ `>= 0.1.32, < 1.0.0`; `v0.21.0` `>= 0.2.0`. A deployment is
   clear of *this* crash from the vLLM `v0.18.0` line on, unless the image pins xgrammar itself.
6. Ascend also removed the CPU round-trip on its own: a Triton bitmask kernel
   (`vllm_ascend/ops/triton/v2/apply_grammar_bitmask.py`) arrives with #14525 (`f8c81e379`,
   2026-08-31), first tag `v0.27.1rc1`.

### Why this symptom and not another

The error needs a structured-output request **and** `skip_out_indices == False`, i.e. the guided/tool
rows are not the whole logits batch: a mixed batch (guided + plain requests), or spec decode / MTP
adding draft rows. The reporter's dumped scheduler output shows exactly that — one new request
chunk-prefilling 1 191 tokens beside five 1-token decodes, `total_num_scheduled_tokens=1196`, and the new
request carrying `json_object=True`, `_backend='xgrammar'`. An all-decode or all-prefill step never
passes `indices` and never fails, which is why the report reads "roughly hourly at concurrency 5–7"
instead of "every request fails". Every rank runs `_sample`, so the whole batch fails while the process
survives.

### Workaround ladder (for the reported build)

| tier | change | cost / caveat |
| --- | --- | --- |
| avoid the pair | client stops sending tools, or drop `--tool-call-parser`/`--enable-auto-tool-choice` and any guided JSON | removes tool calling, the feature they are serving; a capability loss, not a fix |
| switch backend | `--structured-outputs-config '{"backend":"guidance"}'` (llguidance is an installed alternative; the reporter's dump shows `backend='auto'`) | different grammar coverage; **hypothesis**, not verified on their build |
| align the dependency | install `xgrammar>=0.1.30` into the image (the version that converts tensor indices to a list) | outside vLLM's `== 0.1.27` pin: **proposed, unverified on their image**; recheck the vLLM API the wrapper calls, and record that pip resolves a different closure than the tag |
| upgrade | move to a release whose paired vLLM pins `xgrammar >= 0.1.30`, and check the installed version (`pip show xgrammar`), not just the product version (vLLM `>= v0.18.0`; vllm-ascend's own Triton kernel from `v0.27.1rc1`) | version change; confirm in the target image |

The maintainer's reply was the generic "resolved in the latest version, please use the latest release
…" plus a stale-bot closure — `reported`, not a mechanism; the chain above is the code-level answer.

### Falsifier

`python -c "import xgrammar; print(xgrammar.__version__)"` in the reporter's image: `>= 0.1.30` kills
this diagnosis. A second falsifier: a request with `skip_out_indices == True` (one guided request alone,
no MTP) that still raises. A *shape* error from `matcher.py apply_token_bitmask_inplace`
(`bitmask.shape`, `logits.shape`) is a different class — mask/logits row-count bookkeeping, not binding
skew.

## B. Fingerprinting an installed xgrammar from the traceback

Line numbers in `kernels/apply_token_bitmask_inplace_cpu.py` and the argument list in the nanobind
error are enough to place the wheel:

| observed in the traceback | version |
| --- | --- |
| wrapper calls the binding at `:36`; error lists **6** arguments (`int, tuple, int, tuple, int, torch.Tensor`) | 0.1.20–0.1.22 |
| wrapper calls the binding at `:46`; error lists **8** arguments, ending `vocab_size: int, indices: collections.abc.Sequence[int] \| None` | 0.1.23–0.1.27 (the traceback in §A is in this range) |
| wrapper calls the binding at `:45`/`:57` (dtype branches); error lists a trailing `logit_type: str` | 0.1.28+ |
| no error, tensor indices accepted | >= 0.1.30 (#507) |

Do this instead of grepping vllm-ascend history for the message: the string exists in no vllm-ascend
file at any ref.

## C. The same rule for the other structured-output backends

`outlines_core` / `llguidance` failures on the same code path follow the same method — installed version
against the paired vLLM pin, then the upstream project's issue tracker for the exact message. The
`safetensors` variant of "third-party frame" is a **file-integrity** error
(`SafetensorError … incomplete metadata, file not fully covered`) and lives with the load-time classes
in `quantization-load.md` §4: the deepest frame's callee lives in `site-packages`, so the defect is the
shard on disk, not a code path.

## D. `AssertionError: (-1, '<req>', {'<req>': [-1, -1, -1]})` — a placeholder draft token meets the grammar (#6421)

Worked report: **#6421** (image `vllm-ascend:0.13.0rc2`, DeepSeek-V3.2, PD disaggregation, MTP on,
structured output on). The decode `EngineCore` dies with:

```text
xgrammar: Warning: The token id -1 is out of range [0, 129280). Rejecting the token.
ERROR [backend_xgrammar.py:180] Failed to advance FSM for request <id> for tokens -1. Please file an issue.
ERROR [patch_core.py:70] EngineCore encountered a fatal error.
… File "vllm/v1/core/sched/scheduler.py", line 1048, in get_grammar_bitmask
… File "vllm/v1/structured_output/__init__.py", line 288, in grammar_bitmask
    assert accepted, (token, req_id, scheduled_spec_decode_tokens)
AssertionError: (-1, 'chatcmpl-…', {'chatcmpl-…': [-1, -1, -1]})
```

`-1` is not a vocab id: it is vLLM's **scheduling placeholder**. `patch_core.py` is vllm-ascend's
`run_engine_core` wrapper, so it only *prints* this traceback; the raise is upstream vLLM (`verified` at
`v0.13.0`).

### The assert and its version window (`verified` at vLLM `v0.13.0`)

The serial bitmask loop iterates `req_tokens = scheduled_spec_decode_tokens.get(req_id, [])` and calls
`accept_tokens` on every non-`None` token (`for i, token in enumerate(req_tokens + [None])`), then
`assert accepted, (token, req_id, scheduled_spec_decode_tokens)`. A literal `-1` in that list is handed to
xgrammar, which rejects it and the assert fires; there is no padding test at this ref. From **vLLM
`v0.14.0`** the loop uses `-1` as its own end sentinel (`for token in itertools.chain(req_tokens, (-1,))`
with `if token == -1: apply_bitmask = False`), so a placeholder is skipped and the assert cannot fire.
Present at `v0.13.0`/`v0.13.0rc2` `:288`, `v0.14.0`/`v0.15.0` `:271`, `v0.16.0`–`v0.18.0` `:268`,
`v0.20.0` `:269`, `v0.23.0` `:287`; **gone by `v0.26.0`**. Current `main` treats `-1` as padding
(`strip_speculative_padding`, then `if token == -1: padding_or_grammar_invalid = True`, after
`c4f2bb4abb`, Oct 2026). So this is not a vllm-ascend parser defect, and no release upgrade inside
`0.13`–`0.23` removes it.

### Who wrote the `-1` — two candidate fillers, discriminated by config

Both are `verified` code paths; which one produced the reporter's list is `hypothesis` until the decode
node's effective config is known.

1. **The Ascend recompute scheduler's full-graph placeholder (`vllm_ascend/`).**
   `additional_config.recompute_scheduler_enable` (default `False`, `vllm_ascend/ascend_config.py:90-91`)
   is what installs the subclass: `platform.py:333-337` swaps in `RecomputeSchedulerConfig`
   (`core/recompute_scheduler.py:56-58`). Its `add_request` (`:106-111`) says "Fill in placeholder tokens
   to enable full graph compatibility" and sets
   `request.spec_token_ids = [PLACEHOLDER_TOKEN_ID] * self.num_spec_tokens`; `PLACEHOLDER_TOKEN_ID` is
   `-1` (`vllm/v1/sample/rejection_sampler.py:22` at `v0.13.0`), so three spec tokens give exactly the
   assert's `[-1, -1, -1]`. The subclass declares the shape it assumes at `:101-104`:
   `is_mtp_kv_consumer = speculative_config and kv_transfer_config and
   kv_transfer_config.is_kv_consumer and vllm_version_is('0.13.0')` — MTP + KV consumer, i.e. the
   reporter's D node, version-gated to the `0.13.0` pair. Those ids become the scheduled spec tokens
   (`vllm/v1/core/sched/scheduler.py:1110` at `v0.13.0`). Downstream fills them where nothing marks the
   request pending, so upstream's deferral (next item) never applies.
2. **Upstream's async-scheduler placeholder.** At vLLM `v0.13.0` the async scheduler sets the same
   `request.spec_token_ids = [-1] * self.num_spec_tokens` in `_update_after_schedule`
   (`vllm/v1/core/sched/async_scheduler.py:37`) for tokens the worker will fill via
   `update_draft_token_ids` (`scheduler.py:1320-1341`), **and** raises
   `scheduler_output.pending_structured_output_tokens` for requests with
   `use_structured_output and num_output_placeholders > 0` (`:39-41`), which defers bitmask/grammar work
   while placeholders are outstanding. Its own placeholders therefore do not trip the assert.

### The check this class owes before the reply

The async-scheduler provenance needs async scheduling **on**, and the plugin's copy of
`VllmConfig.__post_init__` (`vllm_ascend/patch/platform/patch_vllm_config.py:66-90`) mirrors upstream and
sets `async_scheduling = False` whenever `speculative_config is not None` (`:72-80`). A plain MTP
deployment of that release should therefore have it off, which points at the recompute scheduler (item 1)
if its key is set, or at an image-side patch if neither is. So the reply asks for, or states, the decode
node's effective `async_scheduling`, `num_speculative_tokens` and
`additional_config.recompute_scheduler_enable`. If the `-1`s survive with all three off, the assertion
site is what is left to report.

### Levers

| tier | change | cost / status |
| --- | --- | --- |
| config | drop `additional_config.recompute_scheduler_enable` on the decode node (the subclass, hence the placeholder fill, exists only while it is on) | loses the recompute-scheduler optimisation; `proposed`, but it is the only selection path at that tag |
| config parity | make P and D node configs identical (including any `--async-scheduling`) — the reporter's own `Solved: Inconsistent configuration between prefilling and decoding nodes.` | `reported`; the follow-up asking *which* keys differ went unanswered, and the gate above is what such a mismatch reaches |
| feature class | remove MTP (`--speculative-config`), the reporter's first lever | loses the DeepSeek-V3.2 MTP speedup; also removes the reason they serve D nodes this way |
| upgrade | a stack whose paired vLLM is `>= v0.14.0` | the assert cannot fire there (confirm with `pip show vllm`); for the `main`-only padding-tolerant path the whole stack has to move, CANN/torch_npu included |

The sampler half of the same contract has its own guard: #10043 (`a3b10cb6f`) states it in its own words
— "On the first decode step of the D node, PD kv-consumer MTP must keep `PLACEHOLDER_TOKEN_ID` (`-1`) in
`spec_token_ids` to preserve the required full-graph decode shape … The sampler must not use it as an
index" — and only touches `sample/rejection_sampler.py` / `ops/triton/reject_sample.py`, not the grammar
path.

### Falsifier

`-1` in `scheduled_spec_decode_tokens` while the decode engine has **no** speculative decoding, no async
scheduling and no recompute scheduler kills both producers above — then the placeholder was written by
something else (a patched scheduler in the image), and the assertion site is all that is left to report.
A rejection of a *valid* token id (`accept_tokens` returns False for e.g. a special token) is the other
class: `backend_xgrammar.py` still prints `Failed to advance FSM`, but the reported token is in-vocabulary.

## Check before reuse

```bash
# the reporter's image
python -c "import xgrammar; print(xgrammar.__version__)"
# the plugin's host-logits path at their tag (grep, do not read the file)
git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'to("cpu")' <tag> -- \
  vllm_ascend/worker/model_runner_v1.py
# the upstream call and its pin
git -C /workspace/repos/vllm-project/vllm show <paired-tag>:vllm/v1/structured_output/utils.py | sed -n '105,130p'
git -C /workspace/repos/vllm-project/vllm show <paired-tag>:requirements/common.txt | grep -i xgrammar
# §D: which filler can exist at their tag, and is the assert's sentinel already there?
git -C /workspace/repos/vllm-project/vllm-ascend grep -n -e recompute_scheduler_enable -e PLACEHOLDER_TOKEN_ID <tag> -- vllm_ascend
git -C /workspace/repos/vllm-project/vllm show <paired-tag>:vllm/v1/structured_output/__init__.py | grep -n -e '== -1' -e 'assert accepted'
```
