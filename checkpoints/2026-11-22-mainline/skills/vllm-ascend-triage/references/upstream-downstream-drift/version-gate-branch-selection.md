# The version gate picks the wrong API generation (`vllm_version_is`)

## Scope

The report says a *version string* decides an import: "版本号不严格是 0.11.0 时，
0.11.0-dev branch 中 mla.py 在加载 vllm.attention 时会出现 bug", or the paste is a
bare `if vllm_version_is("X.Y.Z"): … else: …` block with "不会正常加载，建议改成
try catch". The symptom is an **`ImportError` / `cannot import name …` at import
time** of a `vllm_ascend` module, on an install whose vllm is an rc, a `.devN`,
a `.postN` or a locally built version of the same release line. It is also the
answer to "why does the branch for my vllm version import something my vllm does
not have", and to the code-change request that follows it.

Route here when `rg -n 'vllm_version_is' <vllm_ascend>/` hits at the reported ref
— 54 occurrences in 20 files at vllm-ascend v0.15.0rc1 `3d43ed997e`, so this is a
systemic pattern, not one file. Not this page: a bare version *number* question
(build/SCM artifacts, `+empty`, editable installs) → `../image-dependency-faults/
installed-version-vs-checkout.md`; a `patch/**` frame whose body is a copy of
upstream → `patch-copy-vs-upstream.md`.

## Pins

Verified at vllm-ascend **v0.15.0rc1 `3d43ed997e`** with upstream vllm
**v0.15.0 `f176443446`**. The gate's *string* is bumped every release (the pin's
is `"v0.15.0"`), so re-grep the three `path:line` below at another ref; the
semantics have not changed.

## The gate is exact equality, read once, cached

`vllm_ascend/utils.py:360-376`:

```python
@functools.cache
def vllm_version_is(target_vllm_version: str):
    if envs_ascend.VLLM_VERSION is not None:      # envs.py:68 -> os.getenv("VLLM_VERSION", None)
        vllm_version = envs_ascend.VLLM_VERSION
    else:
        import vllm; vllm_version = vllm.__version__
    try:
        return Version(vllm_version) == Version(target_vllm_version)
    except InvalidVersion:
        raise ValueError("Invalid vllm version … Set the environment variable VLLM_VERSION …")
```

Three consequences, in the order they matter:

1. **`==` is PEP 440 equality, so pre/post/local builds are unequal to the final.**
   `0.11.0rc1`, `0.11.0rc3`, `0.11.0.post1`, `0.11.0.dev0+g<sha>` and `0.11.1` all
   compare **False** against `0.11.0`, while `Version(...).release` is `(0, 11, 0)`
   for the first four. A reader takes `vllm_version_is("0.11.0")` to mean "this
   release line"; it means "this exact final". Give the reporter the one-liner
   rather than asserting it: `python -c "from packaging.version import Version as
   V; print(V('0.11.0rc1')==V('0.11.0'), V('0.11.0rc1').release)"` → `False (0, 11, 0)`.
2. **So the `else` branch is the branch for every *other* upstream version**, and it
   must import a name that is correct on all of them. That breaks as soon as
   upstream renames the class more than once — which it has, at least twice for
   MLA: `vllm.attention.Attention` / `MultiHeadLatentAttention` →
   `vllm.attention.layer.MLAAttention` / `MultiHeadLatentAttentionWrapper` →
   `vllm.model_executor.layers.attention.MLAAttention`. At vllm v0.15.0 the last
   path is a **0-line `__init__.py`** (`vllm/model_executor/layers/attention/__init__.py`),
   and `vllm/attention/__init__.py` is 0 lines too, while
   `vllm/attention/layer.py:143` (`Attention`), `:496` (`MLAAttention`) and
   `vllm/model_executor/layers/mla.py:14` (`MLAModules`), `:34`
   (`MultiHeadLatentAttentionWrapper`) exist. A two-valued switch over a
   three-valued API history is the defect; an rc/dev/post install of the gate's own
   release is exactly what falls through it.
3. **`functools.cache` + import-time read ⇒ the override must be set before the
   process starts.** `VLLM_VERSION` changed in a running shell after the first call
   does nothing (the unit test pins both semantics:
   `tests/ut/test_utils.py:134-157`, which has **no** rc/dev case at all — so
   "the test covers this" is not an answer).

## What to deliver, in order

1. **Verdict sentence**: the gate is exact-equality, the install is not the exact
   final, so the block took the other generation's branch; this is a code defect
   and the fix is not on the reporter's side alone.
2. **The cheapest rung, before any patch**: `export VLLM_VERSION=X.Y.Z` (the value
   the gate compares against — the *float/string* the gate target uses, e.g.
   `0.11.0`) for the launch, or install the paired final. Both are documented: the
   env var is the project's own remedy for the sibling failure
   (`docs/source/faqs.md:135-137`, "format of `VLLM_VERSION` should be `X.Y.Z`"),
   and the release table pairs an rc of vllm-ascend with the **final** vllm
   (`docs/source/community/versioning_policy.md:35`: `| v0.11.0rc1 | v0.11.0 | …`,
   `:127` `| v0.11.0-dev | Maintained | CI commitment for vLLM 0.11.0 version |`).
   So on a *paired* install the gate never sees an rc — the reported break means
   the pairing itself is off, which is worth one `pip show vllm vllm-ascend` line.
3. **Why try/except on the import is not the fix**: the two branches are different
   *generations* of the class, not two locations of one name — constructor and
   forward signatures differ (`MLAModules` exists only in the new generation), so a
   guarded import starts the engine with a class body written against the other
   API. Say what the structural fix is: compare release tuples (`Version(x).release
   < (0, 12, 0)`), or probe the module (`hasattr(<upstream module>,
   "<symbol>")`), and keep the body consistent with the branch. State the
   acceptance check as **one real request whose output matches a working run**, not
   "the ImportError is gone" and label it by the claim ladder
   (SKILL.md rule 6: reporter-confirmed > source-verified > precedent > hypothesis).
4. **Scope of what you verified**: the snapshot carries one vllm-ascend ref and one
   vllm ref. The names on the *reporter's* branch come from their paste; ask them
   to print the block (`sed -n 'A,Bp' <file>`), their `vllm.__version__`, and the
   sweep (`rg -n 'vllm_version_is' $(python -c "import vllm_ascend,os;
   print(os.path.dirname(vllm_ascend.__file__))")`) — 20 files is a sweep, not one
   patch. Do not claim the older release's module layout from the pin.

## Not established / recheck

- The class names on the reporter's own branch (0.11-lineage) are paste-derived:
  no 0.11 tree is in the snapshot.
- Which direction the *next* upstream refactor takes (`model_executor/layers/
  attention` may gain exports again): re-grep the empty `__init__.py` at the pin in
  hand before repeating "the else branch is dead".
- `vllm_version_is` also gates behaviour, not only imports (`ops/mla.py:129`,
  `patch/**`); a report that names a *silent* behaviour difference can be the same
  defect with no traceback — check the call site's own polarity (`if not
  vllm_version_is(...)`) before assuming the else branch is the buggy one.
