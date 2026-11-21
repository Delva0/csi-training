# Graph-mode start on a composite (VL) config: the layer count read by `update_aclgraph_sizes`

Scope: the engine dies **at config time**, before any `Starting to load model`
line. The last informational line is `PIECEWISE compilation enabled on NPU.
use_inductor not supported - using only ACL Graph mode` (or the
`FULL_DECODE_ONLY` twin), the Python frames are `vllm/config.py`
`__post_init__` → `NPUPlatform.check_and_update_config` →
`update_aclgraph_sizes`, and the raise is an `AttributeError` naming a **model
config object**, never a `vllm_ascend` class. Eager mode never reaches the
call, so the same command with `--enforce-eager` either serves (recorded #1276) or
fails for its own, unrelated reason (recorded #2064, where eager died earlier on a
vllm/vllm-ascend pairing fault — `environment-correspondence.md`).

The failure is `update_aclgraph_sizes` deriving a layer count from a *composite*
config — a VL/audio wrapper config whose outer object carries sub-configs rather
than the scalar. `resources_per_graph = num_hidden_layers + 1` is the budget
input, so the function needs that number before any capture.

## The wordings, by release

| Release (pin) | Code | Raise |
|---|---|---|
| v0.9.0rc2 | `vllm_ascend/utils.py:134` `num_hidden_layers = vllm_config.model_config.hf_config.num_hidden_layers` | `AttributeError: 'InternVLChatConfig' object has no attribute 'num_hidden_layers'` (#1276, InternVL2_5-38B, image `v0.9.0rc2`) |
| v0.9.2rc1 (`b5b7e0ecc765`) | same flat read, `utils.py:307` inside `update_aclgraph_sizes` (`:299`), called from `platform.py:180` | the same `AttributeError` is what remains; `get_max_hidden_layers` / `to_dict` do **not** exist at this ref |
| v0.10.0rc1 (`4604882a3e2c`) | `utils.py:317-321` `if hasattr(hf_config, 'num_hidden_layers') … else: num_hidden_layers = get_max_hidden_layers(hf_config)`; `:291-292` `def get_max_hidden_layers(hf_config)` → `cfg_dict = hf_config.to_dict()` then a recursive `num_hidden_layers` walk returning `max()` | the raise moves **into the checkpoint's own remote code**: `configuration_internvl_chat.py:87 output['llm_config'] = self.llm_config.to_dict()` → `AttributeError: 'InternVLChatConfig' object has no attribute 'llm_config'. Did you mean: 'sub_configs'?` (#2064) |

## Read the three frames correctly

- **The deepest frame being remote code does not move the owner.** In #2064 the
  traceback ends inside `~/.cache/huggingface/modules/transformers_modules/.../
  configuration_internvl_chat.py` and then `transformers/configuration_utils.py`
  `__getattribute__`. That `to_dict()` is broken for that checkpoint revision
  (the config was updated to `text_config`/`sub_configs`, its `to_dict` still
  writes `llm_config`), but vllm-ascend is what called `to_dict()` on a composite
  config for a scalar it could have read through the text sub-config. Quote both
  frames and put the owner on `update_aclgraph_sizes` / `get_max_hidden_layers`.
- **The right accessor is the text sub-config, and vllm already computes it:**
  `vllm/config.py:546 self.hf_text_config = get_hf_text_config(self.hf_config)`,
  whose body (`vllm/transformers_utils/config.py:825`, vllm v0.9.2) uses
  `config.get_text_config()` and asserts `num_attention_heads` on the result. A
  transformers too old to carry `get_text_config()` is the other half of the
  same defect, so the accessor question and the transformers version are one
  check, not two.
- **The call site is the graph branch only** (`platform.py:180` at v0.9.2rc1, in
  the `PIECEWISE` `elif`): `--enforce-eager` never reaches it. So "eager starts,
  graph dies in the same second" is this page, and dropping to eager is the
  workaround — say what it costs (no ACL graph) rather than presenting it as a fix.
- **Do not chase the capture-size arithmetic here.** Nothing was captured and no
  `Calculated maximum supported batch sizes` line exists. That lane is
  `acl-graph-capture.md` step 3; this one is one config read.

## The one command to send

```bash
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('<model>', trust_remote_code=True); print(type(c).__name__, hasattr(c,'num_hidden_layers'), type(c.get_text_config()).__name__); print(sorted(k for k in c.to_dict() if k.endswith('config')))"
```

It answers the two facts this page turns on: whether the outer config exposes
`num_hidden_layers` (so which wording they will get), and whether its `to_dict()`
still emits `llm_config` or has moved to `text_config`. The checkpoint's remote
code is not in the snapshot, so this is the reporter's call, not a local read —
label it as such.

## Support statement when the reporter asks "bug or doc error"

Both. At v0.9.2rc1 the support matrix lists **InternVL2 and InternVL2.5**
(`docs/source/user_guide/support_matrix/supported_models.md:45-46`) with no
InternVL3 row, and `tests/` + `examples/` contain no InternVL case at all, so
"listed as supported" (2.x, for the launch shapes CI runs) and "this config shape
raises" are not in conflict. Give the code verdict, the graph-only scope, the
eager workaround, and the version where the read changed — then the doc gap.

## Recheck before reuse

The function was rewritten between every release above, so quote line numbers
per ref: `rg -n "def update_aclgraph_sizes|def get_max_hidden_layers|num_hidden_layers"
vllm_ascend/utils.py` at the pin. A later tree that reads
`model_config.hf_text_config` (or a vllm `get_num_hidden_layers`-style helper)
has fixed the class and supersedes this page.
