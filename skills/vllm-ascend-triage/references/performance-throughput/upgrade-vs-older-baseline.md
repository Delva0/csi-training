# A newer image is slower than an old release on the same script

Symptom family: the whole body is a **version comparison** and no number —
"[Bug]: <model>相同条件下性能测试不如0.7.3版本", "升级到 vX 后吞吐掉了", "同样的脚本、
同样的卡，新镜像比老镜像慢", an env dump that pins the new image only. Recorded #6119
(910B 32G, Qwen2.5-72B BF16, vLLM 0.13.0rc1 image) is the straight form: title says
"same conditions", body carries no throughput, no concurrency, no input/output
length, no serve command, and no statement of which 0.7.3-era image/CANN/torch_npu
it was compared against.

Answer shape owed before any source read: (a) "same conditions" is **not
established** across an N-release gap and the two numbers may measure different
acceleration stacks; (b) the two artefacts that would make them comparable (both
serve commands with image tags, both metric blocks); (c) the smallest set of
one-launch A/Bs that prices the *documented* differences — do not hold this for
the mechanism.

## Step 1 — the new pin's own release notes span the baseline

`docs/source/user_guide/release_notes.md` at the new commit is **cumulative**: the
v0.13.0rc1 file still carries `## v0.7.3` (line 718) and `## v0.7.3.post1`
(line 696). So a comparison against a well-known old baseline is reconstructable
in-tree, without the reporter's old image:

```bash
cd /workspace/repos/<issue>/vllm-project/vllm-ascend
uv run --no-project python - <<'PY'
import re
sec=None
for i,l in enumerate(open('docs/source/user_guide/release_notes.md'),1):
    if l.startswith('## '): sec=l.strip()
    if re.search(r'MindIE|turbo|NZ|nz_|torchair|DENSE_OPTIMIZE|prefetch|性能|performance', l, re.I):
        print(f'{i:4d} {sec:26s} {l.strip()[:200]}')
PY
```

Two traps in that file:

- **Sections are not in date order.** At v0.13.0rc1, `## v0.11.0` (line 50) sits
  *above* `## v0.12.0rc1` (line 82). Date every hit by its own `## vX — date`
  header, never by position, or the flip gets attributed to the wrong release.
- **The same env var's advice reverses between consecutive releases** — see
  step 3. One default value, or one release note line, is not the answer.

Also read the new pin's **Known Issues** block before pricing anything: at
v0.13.0rc1 it admits "The functional break on Qwen3-Next … is fixed, but it
introduces a regression on performance … #5357". A version-vs-version report can
be answered by the project's own admission, and that line outranks every flag.

## Step 2 — the old script loses levers silently, so "same conditions" is false

Each line below is a *silent* no-op: the flag is accepted, nothing errors, the
tuning is gone. Verified at vllm-ascend v0.13.0rc1 (`1b5d5abf865f`):

| Old lever | Status at the new pin | Anchor |
|---|---|---|
| `--additional-config '{"torchair_graph_config": {"enabled": true}}'` (the 0.7.3-era graph mode) | removed — "doesn't work anymore. Please use aclgraph instead." | `release_notes.md:109` (v0.12.0rc1) |
| `--additional-config '{"ascend_scheudler": {"enabled": true}}'` (note the source's own typo) | Ascend scheduler removed | `release_notes.md:108` |
| `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1` (the Qwen3-dense flag of v0.10.2rc1, `release_notes.md:256`) | removed; `VLLM_ASCEND_ENABLE_PREFETCH_MLP` replaces it and defaults **0**, comment "only used in small concurrency" | `release_notes.md:35`, `vllm_ascend/envs.py:100-102` |
| `VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION` (an old script may set `0`) | removed, and the feature is now **on by default** — this one helps the new image | `release_notes.md:110` |

So the answer's second paragraph is a checklist the reporter runs against their
own 0.7.3 command, one line per removed/renamed switch, plus the two log lines
that say which lane actually ran: `PIECEWISE compilation enabled on NPU`
vs `Compilation disabled, using eager mode by default`.

## Step 2b — the mirror case: a lever that became **default-on** in the new image

A version comparison can also be hurt by something the new image turned *on* without
the script changing. The recorded lever is **CPU binding**: it became default on ARM
hosts at v0.18.0rc1 (`release_notes.md:273` "**CPU Binding Enabled by Default** …
global-slicing A3 policy, improving inference throughput in hostbound scenarios"
#6686; design doc `docs/source/developer_guide/Design_Documents/cpu_binding.md:5-8`),
and at v0.22.1rc1 it is read as `additional_config.get("enable_cpu_binding", True)`
(`vllm_ascend/ascend_config.py:142`) and executed once per worker after warmup
(`vllm_ascend/worker/worker.py:780-786` → `bind_cpus(self.local_rank)`,
`vllm_ascend/cpu_binding.py:606`). The pool it binds to is derived from
`npu-smi info` / its process table filtered by `ASCEND_RT_VISIBLE_DEVICES`
(`cpu_binding.py:106-190`), so a **multi-instance or PD host** (several engine
processes, or cards hidden from the process view) can bind two instances onto
overlapping CPU ranges — the bug class of release notes `:486`/`:515` ("Avoid A2 CPU
binding overlap from hidden NPUs", #8792) and `:455` ("Defer CPU binding until worker
warmup", #7829). The **multi-instance** form of that overlap (two single-card processes
computing the same pool, recorded #8600 at v0.18.0rc1) is its own page —
`cpu-binding-multi-instance-overlap.md` — and this case's prior-month corpus
(`/workspace/history/8600/`, 1115 issues) has no precedent for it: a title search for
`绑核|affinity|cpu_binding|cpuset` returns 0 and the body hits are incidental
`"enable_cpu_binding": true` launch lines, so do not spend a call looking for one.

Cheapest one-launch A/B on the **new** image, no code read: add
`--additional-config '{"enable_cpu_binding": false}'` and compare per-token time
(`index.md`'s `gen tok/s ÷ Running`), or compare `taskset -cp <worker pid>` across the
two versions. On a **PD** host that runs more than one engine process this is a
hypothesis, not a finding: pin it with the A/B or the affinity dump before ranking it
above the removed-switch checklist above, and keep it separate from the pid-level
`OMP_PROC_BIND`/`OMP_NUM_THREADS` axis (`pre-graph-mode-eager-era.md`).

## Step 3 — the old number may include an accelerator that no longer exists

At v0.7.3 the release notes name the accelerator **for exactly the family these
reports use**: "Integrate MindIE Turbo into vLLM Ascend to improve DeepSeek V3/R1,
**Qwen 2 series** performance. #708" (`release_notes.md:728`), and v0.7.3.post1
still recommends it (`:702`). From v0.9.1 the same file says it is **harmful**:
"Mindie Turbo is not needed with this release. And the old version of Mindie Turbo
is not compatible. Please do not install it. Currently all the function and
enhancement is included in vLLM Ascend already." (`:342`).

Consequence: on a Qwen2.5 dense serve, 0.7.3-with-turbo and 0.13.x are *different
acceleration stacks*, and the honest first verdict is "incomparable until the
reporter shows the old command and its log". The probe is one line, present or
absent per process: `MindIE Turbo is installed. vLLM inference will be accelerated
with MindIE Turbo.` (`vllm_ascend/utils.py:180`, ungated — mechanism and the
plugin-vs-graph crash form in
`../image-dependency-faults/mindie-turbo-graph-break.md`). Precedent #2451:
"vllm 0.9.1 + turbo 没有生效, vllm 0.7.3 + turbo 吞吐有性能提升" — same shape, and the
difference was only that INFO line.

## Step 4 — NZ weight layout is a *two-direction* A/B, not a default to quote

`VLLM_ASCEND_ENABLE_NZ` decides `FRACTAL_NZ` (29) weight layout in
`maybe_trans_nz` (`vllm_ascend/utils.py:127-142`): `0` never, `1` only for
quantized weights (**the default**, `vllm_ascend/envs.py:120-121`),
`2` also for bf16/fp16. Four release-note lines in that one file contradict each
other, in this order:

| Anchor | Section | Says |
|---|---|---|
| `:509` #1409 | v0.9.1rc2 | "Added a switch for enabling NZ layout in weights and enable NZ for GMM" |
| `:133` #4495 | v0.11.0rc3 | "Disable NZ weight loader to speed up dense model … temporary solution" |
| `:125` | v0.12.0rc1 | "bf16/fp16 model doesn't perform well, it's **mainly because** `VLLM_ASCEND_ENABLE_NZ` is enabled by default. Please set `=0` to disable it" |
| `:37` #4878 | v0.13.0rc1 | "`VLLM_ASCEND_ENABLE_NZ` is **disabled for float weight case** … Feel free to set it to `2` if you make sure it works for your case" |

So for a BF16 dense model the lever is a **three-value A/B on the new image**
(`unset`, `2`, `0`), one launch each, same script, same client. Never present one
value as "the fix" and never cite one note line as the project's position: the
`:37` line is itself the flip that makes a 0.7.3-era NZ-on run non-comparable.
Pair any speed win from `=2` with a short accuracy sample — the note's own
qualifier is "if you make sure it works for your case".

## Step 5 — what to send back, and how the two numbers are compared

One line per item: both serve commands with image tags; both metric blocks
(`Avg prompt throughput`, `Avg generation throughput`, `Running`, `Waiting`,
`GPU KV cache usage`, `Preemptions`); the `MindIE Turbo is installed` line from
each startup log; the compilation line from each; prompt/output lengths and the
client's concurrency for each run. With those, the arithmetic is
`index.md`'s — `gen tok/s ÷ Running` per request, and queueing read off
`Waiting` — never the AICore % of an `npu-smi` screenshot.

## Version scope and what this page does not settle

Verified at vllm-ascend **v0.13.0rc1 (`1b5d5abf865f`)**, vllm **v0.13.0
(`72506c98349d`)**, the #6119 pins; line numbers are that file's and shift every
release, so re-grep them. The 0.7.3-side facts here come from the *new* tree's own
cumulative notes; they are documented statements, not a run of the old build. The
recorded #6119 replay ranked the NZ flip first, the removed flags second and
mindie-turbo third on a history precedent (#2451) rather than on `:728`/`:342`,
which carry the family-level statement; the reporter never confirmed a cause, so
treat every rank here as a hypothesis with one A/B each. If the reporter returns
with numbers and both commands, the question stops being this page and becomes
`index.md` (a measured block) or an admitted regression in the new pin's Known
Issues.
