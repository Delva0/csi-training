# The connector name resolved to an upstream class, so `*_module_path` never loaded

Scope: a PD deployment that never gets past KV-cache registration, and the deepest
frame of the paste is **upstream vLLM's** connector —
`vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py::register_kv_caches`
with `AttributeError: 'tuple' object has no attribute 'shape'` (or any other
per-layer-value `AttributeError` on that line) — while the reporter's
`--kv-transfer-config` carries a `kv_connector` name *and* a `kv_connector_module_path`
they believe selects the Ascend implementation. Inherit `index.md`; this is not the
recipe-diff lane (`recipe-contract-diff.md`) and not a transport fault.

## Mechanism (verified at vllm-ascend v0.14.0rc1 `52d4acfa`, vllm v0.14.1 `d7de043d`)

Connector resolution is **name first**:

- `KVConnectorFactory._get_connector_class_with_compat` returns
  `cls._registry[connector_name]()` when the name is registered
  (`vllm/distributed/kv_transfer/kv_connector/factory.py:110-111`) and reads
  `kv_connector_module_path` **only in the `else`** (`:113-122`). The module path is
  therefore never imported for an already-registered name: no warning, no
  `ModuleNotFoundError`, and the wrong class runs silently.
- upstream registers `MooncakeConnector` → its own GPU-layout file
  (`factory.py:199-203`). vllm-ascend cannot reuse that key —
  `register_connector` raises `ValueError("Connector '<name>' is already registered")`
  (`factory.py:31-34`) — so every downstream connector lives under a **suffixed key**:
  `MooncakeConnectorV1` (→ `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py::MooncakeConnector`),
  `MooncakeConnectorStoreV1`, `AscendStoreConnector`, `MooncakeLayerwiseConnector`,
  `UCMConnector` (`vllm_ascend/distributed/kv_transfer/__init__.py:21-45`, called from
  `vllm_ascend/__init__.py:25-28` at import).
- the tuple is the Ascend layout: one `(k_cache, v_cache)` tuple per layer, three with
  `dsa_k_cache` on a sparse model (`vllm_ascend/worker/model_runner_v1.py:2510`, `:2508`),
  which upstream reads as a tensor (`cache_or_caches.shape`,
  `mooncake_connector.py:685`, after the `Registering KV_Caches. use_mla: %s` line at
  `:675`). The Ascend worker's own `register_kv_caches` unpacks exactly that tuple
  (`kv_p2p/mooncake_connector.py:1152-1163`: `first_kv_cache_tuple[0]`, `use_sparse =
  len(...) == 3`), which is why the same model starts fine with the right key.

## Which class actually ran — decide from the paste, not from a guess

- **The log sentence differs, and that is the cheapest witness.** Upstream prints
  `Registering KV_Caches. use_mla: False` (two fields, `:675`); the Ascend worker prints
  a per-branch `num_blocks: …` line first and then
  `Registering KV_Caches. use_mla: …, use_sparse: …, shape …` (three fields,
  `kv_p2p/mooncake_connector.py:1173-1214`). A two-field sentence with no `use_sparse`
  is upstream's file — and note `[mooncake_connector.py:675]` alone proves nothing,
  because Ascend's `:675` is socket-handshake code.
- the traceback's own paths settle it: `/vllm-workspace/vllm/vllm/distributed/…` cannot
  be the Ascend file, whose `register_kv_caches` sits at `:1152`.
- **Not a discriminator: `Creating v1 connector with name: <X>`** (`factory.py:64-68`).
  It prints `connector_cls.__name__`, and *both* registry keys resolve to a class named
  `MooncakeConnector`, so the line reads identically before and after the fix. Do not
  offer it as the confirmation.
- one command inside the reporter's image lists the collision:
  `python -c "import vllm_ascend; from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory as F; print([k for k in F._registry if 'ooncake' in k])"`
  → `['MooncakeConnector', 'MooncakeConnectorV1']`; the config named the first.

## Fix and acceptance

Change `kv_connector` to the downstream key that exists at *their* pin and drop
`kv_connector_module_path`. At v0.14.0rc1 the pinned tutorial and examples already say
so (`examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md:51,118`,
`examples/offline_disaggregated_prefill_npu.py:55,102`,
`docs/source/tutorials/Qwen3-235B-A22B.md:446`), and the path the reporter wrote
(`vllm_ascend.distributed.mooncake_connector`) does not exist at that ref at all — the
implementation lives under `vllm_ascend.distributed.kv_transfer.kv_p2p.`. Keep the
`prefill`/`decode` `tp_size` + `dp_size` block: the Ascend worker asserts those keys
(`kv_p2p/mooncake_connector.py:1130-1148`). Acceptance is not "the server starts":
both sides must log the Ascend `num_blocks: …` line and one real request must return the
same answer as the co-located deployment.

## Do not confuse it with

- a name that is **not** registered anywhere fails loudly and is a plain config typo:
  `ValueError: Unsupported connector type: <X>` (`factory.py:115`) or
  `AttributeError: Class <X> not found in <module>` (`:119-122`).
- PD corruption **after** registration, or a device fault at raised scale →
  `recipe-contract-diff.md`; a connect/handshake failure → `transport-topology.md`.
- a third-party connector registered under a name that happens to collide with an
  upstream one behaves the same way: the registry key decides, never the module path.

## Version scope and recheck

The three invariants to re-grep at another ref: the registry-first branch in
`factory.py`, the key set `vllm_ascend/distributed/kv_transfer/__init__.py` registers,
and the value type built by `initialize_kv_cache`. Open item: a later pin's A2
8-node GLM-5.2 recipe still prints `MooncakeConnector` **with**
`kv_connector_module_path` (`../dsa-sparse-attention/c8-flags-and-kv-store-path.md`
cites `GLM5.2.md:1040,1121` at v0.23.0rc1), which this mechanism says cannot select an
Ascend class — either that tree's resolution order/registration changed or the recipe
is wrong. Recheck before reusing this page across a major version.
