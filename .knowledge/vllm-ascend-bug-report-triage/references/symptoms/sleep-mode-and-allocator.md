# Sleep mode (`CaMemAllocator`): the allocator, the documented contract, and "only the first call is wrong"

## Scope

Reports whose trigger includes `enable_sleep_mode` / `--enable-sleep-mode` — offline `LLM(...)` or serving —
and whose symptom is wrong output, a captured-replay crash or a missing/freed buffer rather than a plain
startup failure. The recorded member is the *first-call-only* class (#15560: the first `llm.generate()`
returns garbage or 100 × token id `0`, every later call is correct). The anchors below were verified at
`v0.18.0`, `v0.23.0`, `v0.26.0rc1` and `v0.12.0rc1`; recheck at the reporter's tag, since this subsystem
moves. Route here when the report says "the second `llm.generate()` is fine", "the same config served by
`vllm serve` is fine" or "tp=1 is fine". A wrong-output report whose first request is *also* wrong under
`enable_sleep_mode=False` is not this class — use `aclgraph-mode-and-capture.md` §D (graph-only garbage) or
`c8-and-kv-layout.md`.

Not this page: a sleep-time OOM or a wake-time stall (`memory-sizing-and-parallelism.md`,
`startup-and-stall.md`), a graph-mode support gap with no sleep mode in the config
(`aclgraph-mode-and-capture.md`), or the `libvllm_ascend_kernels.so` import that sleep mode needs
(`../vllm-ascend/install-and-image.md` §2).

## The control matrix is the answer, and it is cheap for the reporter

`enable_sleep_mode` × graph mode × TP is a 4-row table the reporter can run in one pass. Recorded example
(vllm-ascend #15560, Qwen3.6-27B-w8a8, 2×910B3, `v0.18.0`), from a commenter who reproduced it:

| graph mode | `enable_sleep_mode` | first `generate()` | second |
| --- | --- | --- | --- |
| `FULL_DECODE_ONLY` | `True` | `AttributeError: 'NoneType' object has no attribute 'size'` during `Replaying aclgraph` | not reached |
| `NONE` | `True` | wrong: `<think>` then `finish_reason='stop'` after 3 tokens | normal, 100 tokens |
| `NONE` | `False` | normal | normal, identical token ids |
| nightly main, `FULL_DECODE_ONLY` | `True` | normal | normal, identical token ids |

Read it as: sleep mode is the *necessary* trigger for the bad first-request state, and the graph mode only
decides whether that state becomes a wrong answer or a replay crash. The maintainer's own line on the issue
("This seems some sync problem, if you start a `LLM` object and give some time, then no error will occur",
Absoler) points the same way — a race that a delay hides.

## What the allocator actually promises

- **One singleton pool, two tags.** `device_allocator/camem.py:113-139` (`v0.26.0rc1`) documents it in the
  class docstring: allocations made inside `use_memory_pool(tag)` are pooled and carry that tag;
  `sleep` offloads the tensors of the requested tag to CPU **and discards the rest**; `wake_up` restores the
  offloaded ones while the discarded ones "have empty memory". `default_tag = "default"` and
  `sleep_persistent_tag = "sleep_persistent"` (`:135-138`, the latter commented "stay mapped across
  sleep/wake cycles"); `use_memory_pool_with_allocator` at `:102-110` installs the pluggable allocator.
- **Persistent per-layer state must be allocated under the persistent tag, and the code says so.** In the
  DSA family the fork wraps those allocations in `allocator.use_allocation_tag(CaMemAllocator.sleep_persistent_tag)`
  with the comment "sleep/wake does not treat it as KV cache" — `attention/dsa_v1.py:580-586` and
  `attention/context_parallel/dsa_cp.py:213-219` at `v0.26.0rc1`. A feature that allocates such a buffer
  outside the tag loses it across sleep/wake; a feature that *should* be discardable but is tagged
  persistent keeps memory alive. Both directions are triageable from the tag of the allocation site.
- **The platform deliberately does not set `expandable_segments` under sleep mode.**
  `platform.py:767-772` (`v0.26.0rc1`): the default `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` is
  applied only `if model_config and not model_config.enable_sleep_mode`, with the note "We should not set
  this environment variable in RL (sleep mode) scenarios". So an OOM answer that recommends
  `expandable_segments:True` contradicts the reporter's own config: name one path or the other, and say
  which release you are quoting.
- **`enable_sleep_mode_extra_cleanup` changes cleanup, never arithmetic.** `ascend_config.py:175`
  (`additional_config` key, default false) plus the feature guide's section: it clears ACL-graph attention
  workspaces, resets the runner's graph manager, destroys HCCL groups on `sleep()` and recaptures graphs on
  `wake_up()`; the doc warns it trades sleep-time memory for wake latency. Do not offer it as a correctness
  lever, and do not read a change in output after enabling it as the fix.

## The documented contract, and what it excludes

`docs/source/user_guide/feature_guide/sleep_mode.md` (present with the same skeleton at `v0.12.0rc1` and
`v0.26.0rc1`) shows the offline flow `LLM(..., enable_sleep_mode=True)` → `generate` → `llm.sleep(level=1)`
→ `llm.wake_up()` → `generate`, and asserts the two outputs are equal; level 1 offloads weights and discards
KV cache, level 2 discards both. Two consequences for triage:

- **"Enabled but never slept" is outside the documented path.** The guide only makes claims across a
  sleep/wake cycle, so a first-call-garbage report without `sleep()`/`wake_up()` is not contradicting a
  documented guarantee — quote the doc to scope the claim, then treat the mechanism as open.
- **The guide requires the custom-op build** ("build from source … if you are using < v0.12.0rc1, remember to
  set `export COMPILE_CUSTOM_KERNELS=1`"). A report on such a tag with `camem_available = False` in the log
  is an install question first: `../vllm-ascend/install-and-image.md` §2.

## What the refs show: the code boundary is `v0.18.0` → `v0.23.0`

- `v0.18.0`: `vllm_ascend/device_allocator/` holds **only** `camem.py`, and `platform.py:105-106` returns
  `True` from `is_sleep_mode_available()` unconditionally, so upstream's sleep gate passes on any Ascend
  build. `platform.py:520-522` deliberately **skips** `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` when
  `enable_sleep_mode` is set, i.e. the flag changes the allocator, not merely a sleep/wake lifecycle.
- `v0.23.0`+ adds `vllm_ascend/device_allocator/sleep_mem_optimized.py` (`SleepWakeupManager`, HCCL group
  recreate) from #9585 (`92eeab2fe`, first tag `v0.23.0`), plus
  `additional_config.enable_sleep_mode_extra_cleanup` and `patch/platform/patch_camem_allocator.py`. So the
  sleep path is *different* either side of that boundary; never transplant a v0.18 reading onto `v0.23.0`+ or
  the reverse.

The signature is a *timing* one (first iteration wrong, every later iteration right, warm-up complete), so
look for an asynchronous hand-off whose host source is rewritten by the next iteration, not for model math.
`0` repeated as every token id means the sampler saw a degenerate logits vector — read what the first step
consumed, not the sampler.

Before blaming a race, spend the ancestry command, because the recorded thread already names fixes that are
**inside** the reporter's own tags:

| expression | commit | first tag | note |
| --- | --- | --- | --- |
| `Snapshot query start locations before async H2D copies` (#12071) | `c83bf9bd1` | `v0.23.0rc1` | the 0.24+ line carries the cherry-pick `1fbad8993` (#12079), first tag `v0.24.0rc1` |
| `[v0.23.0][BugFix][GDN] Restore stateful one-token decode selection` (#12255) | `739ebda83` | `v0.23.0` | plus the `v0.24.0rc1` bundle `f3911ffcc` (#13195) |
| `[v0.23.0][BugFix][Worker] Preserve async accepted-token snapshot` (#13865) | `17ac75217` | `v0.23.0` | — |

`git tag --contains <sha>` on those three settles it: a report against `v0.23.0`/`v0.26.0rc1` already runs
them, so "the async H2D snapshot race explains it" is false for those tags even when a commenter suggests it.
The query_start_loc snapshot class is still the right *shape* to look for on older tags.

**Negative result worth keeping:** #12079 (`1fbad8993`, `copy_snapshot_to_gpu`) has `tag --contains` starting
at `v0.24.0rc1`, yet the helper **is** in `v0.23.0`'s tree (a release-branch cherry-pick) and absent from
`v0.18.0` — so it cannot explain reports still reproducing on `v0.23.0`/`v0.26.0rc1`. Check presence with
`git grep -c copy_snapshot_to_gpu <tag> -- vllm_ascend`, not with `tag --contains` alone.

## Worked case #15560 (open, 2026-09)

Config (reporter's own script, quoted): `LLM("/mnt_data/Qwen3.6-27B-w8a8/", trust_remote_code=True,
language_model_only=True, tensor_parallel_size=2, compilation_config={"cudagraph_mode":
"FULL_DECODE_ONLY"}, enable_sleep_mode=True)`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
`VLLM_ASCEND_ENABLE_NZ=0`, `SamplingParams(temperature=0, max_tokens=100, seed=42)`. First generate:
`'!!!!…'`, `token_ids` all `0`, `finish_reason: length`; the second generate is coherent. Reproduced by the
reporter on the `v0.23.0` and `v0.26.0rc1` openeuler images (firmware 25.3.rc1), while `vllm serve` and
tp=1 are fine. A commenter reproduced it on `0.18.0` with sleep mode as the trigger and could not reproduce
on a main revision, attributing a main-only fix to #12255/#12071 **without bisecting** — that attribution is
`reported`, and the ancestry table above contradicts it for the release tags.

- **Root cause:** not established. The candidate to state with its falsifier is the sleep-mode allocator
  lifetime above — a first-step buffer allocated outside `sleep_persistent_tag` (or before the pool is
  installed) reads unwritten memory on the first decode step and is repopulated afterwards. Falsify with the
  same script on a tag where the pool path is absent, or with a full-model `aclgraph` disabled run on the
  reporter's image.
- **The thread's other candidates are `reported`, not bisected:** a metadata H2D race (the maintainer's
  sync remark), the MTP/GDN metadata fixes cited as "possibly related", and the `is_multimodal_model` gap
  of #14638, which a commenter says must be fixed before `v0.18.0` is diagnosable at all. Say why each is
  not the answer here (the ancestry table, the `v0.18.0` boundary) rather than picking one.
- **Status and limits:** no `collect_env` in the body beyond the image tags, so the node/firmware half is
  theirs alone. Coverage check at `v0.26.0rc1`: `git grep -ln enable_sleep_mode <tag> -- tests` returns only
  `tests/e2e/pull_request/one_card/test_camem.py`, `tests/ut/test_platform.py` and
  `tests/ut/worker/a2/test_worker_v1.py`, and none of them pairs sleep mode with `FULL_DECODE_ONLY` — the
  reported combination is untested at that tag, and the commenter's "fixed on main" claim is unverified
  here.

## Levers, strongest first

1. `enable_sleep_mode=False` — the reporter's matrix makes it the direct test: if the first call becomes
   correct, the sleep/allocator path is implicated. Cost: no weight offload, so the deployment keeps the
   full footprint resident.
2. Remove the `cudagraph_mode` entry (keeps the default PIECEWISE) — the second row of the matrix isolates
   the graph mode as the crash-vs-garbage switch.
3. `tensor_parallel_size=1` or a `vllm serve` frontend — both were confirmed to avoid it in #15560; useful
   as bisect steps, not as production levers.
4. `--enforce-eager` — removes capture as well, at the price of every graph-mode optimisation.
5. "Call `generate()` twice and discard the first" — a diagnostic, never a workaround.
6. An image whose release contains the fix: only claim it for a build where the reporter's own matrix comes
   out clean; the thread's "fixed on nightly main" is `reported`, and it names no commit.

## Check before reuse

```bash
git -C /workspace/repos/vllm-project/vllm-ascend ls-tree --name-only <tag>:vllm_ascend/device_allocator
git -C /workspace/repos/vllm-project/vllm-ascend show <tag>:vllm_ascend/platform.py \
  | grep -n 'is_sleep_mode_available\|enable_sleep_mode'
git -C /workspace/repos/vllm-project/vllm-ascend grep -c copy_snapshot_to_gpu <tag> -- vllm_ascend
git -C /workspace/repos/vllm-project/vllm-ascend show <tag>:vllm_ascend/device_allocator/camem.py \
  | sed -n '100,145p'
```

Falsifier for the whole class: the wrong answer survives `enable_sleep_mode=False` on the same build, or
appears on the *second* call too — then it is a graph-mode/attention fault and the sleep flag was
coincident. A TP=2-only symptom is not a sleep-mode defect until the row that turns sleep mode off comes
out clean.
