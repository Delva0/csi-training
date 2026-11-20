# Cheap checks per track: the first moves after classification

## Scope

The executable half of `SKILL.md` procedure step 3: for each lane and track, the
checks to take **before** any source archaeology, each ending at the topical page
that owns the rule. `scripts-catalogue.md` is the same material one level down:
the flag synopsis and expected reading of every re-runnable script the two pages
name. It is consulted per report — place the report with the
classification table in `SKILL.md`, then read only the entry for that track.
`SKILL.md` keeps the Lane A/B/C ordering, the attribution rule and the delivery
gates; this topic keeps the per-track detail that used to sit beside them.

## Common rules

- Inherit every rule in `SKILL.md`. In particular: message text on every turn
  (rule 1), pins and `path:line` for every code claim (rule 5), labelled claim
  strength (rule 6), narrow reads (rule 7), missing artefact as the finding
  (rule 10), and the timestamp gate before any error family is named (rule 13).
- One axis per launch when you propose an ablation, cheapest first.
- A named script under `scripts/` is the check — run it rather than re-deriving
  its arithmetic by eye.
- Links on the content page are skill-root relative (`references/<topic>/…`),
  exactly as they appear in `SKILL.md`; prepend `<skill_root>` when running a
  script from them.
- If no entry matches, fall back to `SKILL.md`'s general sequence: version
  correspondence, one `history/` grep, then the owning code path.

## Routes

- `scripts-catalogue.md` — every script under `scripts/`, with its arguments and
  what its output decides. Open it when a track or topical page names a script;
  the script's own header repeats the flags and is authoritative.
- `cheap-checks-by-track.md` — the full list, in the order the tracks appear in
  `SKILL.md`: image/native-module aborts, launch-env and library-path, the
  connector/config-attribute and schema lanes, custom-op resolve/build and
  offline source builds, upstream patch drift, PD transport, single-node
  KV-pool/MemCache faults, observability/flag questions, the spec-decode and
  tool-call rows, first-response, 310P, memory (allocation, growth, host RAM),
  sizing, EP×DP, feature-enable, code-change / flag-retirement, community-test,
  pooling/embedding (crash under concurrency, wrong long-input vectors), accuracy
  and corruption, `patch/**` frames, the async-spec `KeyError` set, the
  drafter-count copy dump, mixed-batch attention crashes, and silent operator
  numerics. It also carries the two startup one-liners that resolve without source
  archaeology: the env-registry `Unknown vLLM environment variable detected`
  warning and the DSA-CP `aclnn_input_scale` `AttributeError`.
