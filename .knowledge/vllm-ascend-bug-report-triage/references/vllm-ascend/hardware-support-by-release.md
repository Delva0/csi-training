# Hardware support by release: "when does <hardware> support <feature>?"

## Scope

A `[Usage]`/`[Bug]` question of the shape "does 310 series support graph mode and quantization", "when
will A5 get feature F", "is <model> faster on <hardware>": the answer is a **support statement pinned to
the reporter's ref**, not a roadmap. The statement lives in the *hardware tutorial page* at that ref, and the
"when" half is two different tags — the tag where the **code path** first exists and the tag where the
**docs** admit it. Both are cited; a date nobody shipped is not.

Related: model-level floors are `../symptoms/model-type-support-floor.md`, image/base-CANN pairing is
`version-pairing.md` and `install-and-image.md`, flag lifetimes are `config-flag-lifetimes.md` (e.g. why
`weight_nz_mode` is a no-op on 310P). This page carries the per-hardware support status itself; the 310P
*failure* classes (capture-event exhaustion, sharded W8A8SC checkpoints, the explicit `--max-model-len`
rule) are `../symptoms/310p-capability.md`.

## Common rules

- **The hardware tutorial is the only support statement.** Feature guides
  (`user_guide/feature_guide/graph_mode.md`, `quantization.md`) describe the mechanism for supported
  hardware and often never mention the one the report is about; an empty grep there is **not** "unsupported".
  Grep the tutorial tree by hardware name and open the page's `{note}`/`{warning}` block.
- **`--enforce-eager` in a reporter's command usually came from that page.** Quote the block the reporter
  copied, at their ref, before proposing a flag change: on 310P the same page's own serve command carries
  `--enforce-eager`, so "remove `--enforce-eager`" is only an answer on a ref whose page dropped it *and*
  whose code has a graph path for that device.
- **Answer in the reporter's terms and version.** State what holds at the reported ref, then the first tag
  where it changes and how (flag, image variant, model type gate). If the escalation needs a different image
  family (`-310p`), say so — the tree has no 310P branch at all before `v0.14.0rc1` (`code-layout.md`).
- **Never infer "supported" from a hook.** A `is_310p()` branch in a weight-format helper is not a
  supported graph mode; the config gate that raises for the reporter's model type decides.

## Method (three commands)

```bash
# 1. which page exists at the ref (the path moves between releases)
git -C "$REPO" ls-tree -r --name-only "$TAG":docs/source | grep -iE '300i|310p|hardwares'
# 2. the support statement itself — the note block is at the top of the page
git -C "$REPO" show "$TAG":docs/source/tutorials/<page>.md | sed -n '1,45p'
# 3. "when": bisect the note across tags, then date the code path
for t in v0.10.0rc1 v0.16.0rc1 v0.19.1rc1 v0.20.2rc1; do
  printf '%s ' "$t"; git -C "$REPO" show "$t":docs/source/tutorials/hardwares/310p.md 2>/dev/null \
    | grep -c -i 'only supports eager mode'; done
git -C "$REPO" log --all --oneline -i --grep='<feature>'; git -C "$REPO" tag --contains <sha> | sort -V | head -3
```

If the reporter's complaint is *performance*, the answer also needs the model-type gate at their ref
(torchair is a different graph engine from ACL graph and is restricted to a model list), so read both before
writing "graph mode is available".

## 310P (Atlas 300I / 300I Pro / 300I DUO / 200I Pro): verified timeline

Doc page per ref (the path moves twice, and an absent page is not "no statement"):

| ref | page |
| --- | --- |
| `v0.9.1rc1` – `v0.11.0rc1` | `docs/source/tutorials/single_node_300i.md` — `v0.9.1rc1` and `v0.9.2rc1` have it, the `v0.9.1` tag has only the locale `.po` (still there at `v0.13.0rc1`), so a missing page at one tag is a tree gap, not "no statement" |
| `v0.13.0rc1` – `v0.14.0rc1` | `docs/source/tutorials/310p.md` |
| `v0.15.0rc1` – `v0.23.0rc1` | `docs/source/tutorials/hardwares/310p.md` (`v0.23.0rc1` title: "Atlas 300I DUO / Atlas 200I Pro") |

Support statement, verbatim:

- `v0.10.0rc1` `single_node_300i.md:5` — "Currently, the 310I series only supports eager mode and the data
  type is float16." The page's serve block (`:64-68`) is `--enforce-eager --dtype float16
  --compilation-config '{"custom_ops":["none", "+rms_norm", "+rotary_embedding"]}'` — a reporter running
  exactly that copied the doc, so the flags are not their experiment.
- `v0.19.1rc1` `hardwares/310p.md:5` — same sentence, "the Atlas 300I series". `enforce_eager=True` is still
  in every Python example (`:205`, `:248`, `:291`).
- `v0.20.2rc1` `hardwares/310p.md:17-26` — first doc that documents graph mode: "The current release supports
  `FULL_DECODE_ONLY` graph mode on Atlas 300I DUO devices", with a hardware event-id limit (TP>1: few
  capturable graphs, model-depth dependent — Qwen3-32B: 2; TP=1: no limit). Example block `:114-124` uses
  `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32]}'`
  with `--quantization ascend` and W8A8SC weights.

Code, which is earlier than the docs:

- decode-only **ACL graph** for 300I: `ddc78dbad` "[300I] support decode-only aclgraph mode (#6849)"
  (2026-03-02), first tag **`v0.16.0rc1`**; its own body names the two limits (event-id graph count, and the
  CANN version nobody outside Huawei could confirm). Follow-ups in the same area: `4ba23cc58` (#8948,
  blocktable aclgraph error), `4b1d43a62` (#9104, fused-operator ACLGraph), both first tagged `v0.20.2rc1`.
- **torchair** graph on 310P is a *different* engine and is gated by model type, not by device: at
  `v0.10.0rc1` `vllm_ascend/ascend_config.py:20` `TORCHAIR_MODEL_LIST = ["deepseek", "pangu", "kimi_k2"]`
  and `:162-168` raises "Torchair graph mode only works with following model types" for anything else. So a
  Qwen/VLM report on that ref has **no** graph mode at all, whatever the `is_310p()` hooks elsewhere suggest
  (`utils.py:173-184` converts MoE weights to NZ only when `is_310p() and use_torchair`).

Quantisation on 310P:

- Path is ModelSlim static W8A8: `--quantization ascend` + `quant_model_description.json`
  (`quantization/quant_config.py` `AscendQuantConfig.get_config_filenames`). At `v0.10.0rc1`
  `quantization/w8a8.py` carries the 310P branches (`is_310p()` at `:46`, `:105`, `:257`, `:286`, `:303`,
  `:323`, `fused_experts_310p` at `:454`), while `w8a8_dynamic.py` and `w4a8_dynamic.py` have **no** 310P
  branch — dynamic/W4A8 are not the 310P answer at that ref.
- From `v0.14.0rc1` the 310P family is `vllm_ascend/_310p/quantization/` (#5776 `1ffca8673`), and from
  `v0.23.0rc1` `platform.py:199-202` imports `AscendModelSlimConfig310` for 310P instead of the generic
  configs, i.e. a 310P quantisation report is answered by that scheme (`repo-map.md`).

`--compilation-config '{"custom_ops": …}'` is a **no-op on 310P** — `platform.py` sets
`compilation_config.custom_ops = ["all"]` only `if not is_310p()` (`v0.10.0rc1:185-187`;
`v0.20.2rc1:485-487` and `v0.23.0rc1:663-665` via `get_ascend_device_type() != AscendDeviceType._310P`). There are no
310P custom ops to enable, so a reporter tuning that list is asking the wrong lever.

### The same timeline by first tag, for the code paths a report names

Each row is `git log --diff-filter=A -- <path>` + `git tag --contains <sha>`:

| fact | first tag | commit |
| --- | --- | --- |
| `vllm_ascend/_310p/` (attention, `model_runner_310p`, `ops`, `worker_310p`) | `v0.14.0rc1` | `1ffca8673` (#5776, "Add basic 310p support. Only dense models work with eager mode now") |
| 310P W8A8 (ModelSlim) quantisation, `_310p/quantization/modelslim_config.py` | `v0.15.0rc1` | `39e77fb9e` (#6454, "[Feat.]: support 310p w8a8") |
| `_310p/sharded_state_loader_310p.py` (`--load_format sharded_state`) | `v0.16.0rc1` | `5899438a8` (#6878) |
| `_310p/spec_decode/` (Qwen3.5 MTP + graph mode on 310P) | `v0.21.0rc1` | `969baeda8` (#10309) |
| `docs/source/tutorials/hardwares/310p.md` exists | `v0.15.0rc1` | `7d4833bce` (#6501) |
| the same page's graph-mode/event-id note | `v0.18.0`, then continuous from `v0.20.2rc1` | `00ddacf4e` (#8639) |

At `v0.10.0rc1` therefore: no `_310p/` tree at all, the tag's 300I page says eager-only + float16, the
graph-mode guide lists ACLGraph/TorchAirGraph without a 310P statement, `platform.py:186-187` sets
`compilation_config.custom_ops = ["all"]` only `if not is_310p()`, and the quantisation package is
`{w8a8, w8a8_dynamic, w4a8_dynamic}` with no 310P path — so the honest answer to a `0.10.0rc1` user is
the release that adds the feature plus the lever available on their build (e.g. drop `--enforce-eager`
only once a tag that documents 310P graph mode is reached), labelled `reported` for the maintainer's
"resolved in the latest version" line.

## Answer shape for this class

1. The doc line at their ref, quoted, with the page path and ref (that is the "supported / not supported"
   verdict, and it is also the reason their flags look the way they do).
2. The first tag where it changes for their device **and** their model type — code tag and doc tag named
   separately — plus the exact flag/`--compilation-config` the newer page uses, and the image family.
3. What is left on their ref if they cannot move (model/quant choice, the doc's own env, the limits above),
   labelled `proposed`.
4. `status and limits`: what was not verified (e.g. whether their CANN generation enables the capture path);
   never a date.

## Recheck conditions

Re-derive the page path and the note for any ref outside `v0.9.1rc1`–`v0.23.0rc1`, whenever the hardware
tutorial was reorganised, and for A5/950 (their pages did not exist for most of the range above). A doc
sentence and the code that implements it can disagree for several releases — that gap is a fact worth
reporting, not a reason to pick one.
