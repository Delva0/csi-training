# A `# Copied from …` body still reading the source's old attribute

## Scope

Engine start dies with `AttributeError: '<X>Config' object has no attribute '<field>'`
(or the same name as `KeyError`/`NameError`) raised inside a function whose header is
`# Copied from <path inside the same tree>`, while the named source function at the same
commit no longer reads that name. The copy is stale; the config object is correct.
Recorded #4891 (vllm-ascend `v0.11.0rc3` `b6d63bbd`, vllm `v0.11.0`): `CPUOffloadingConnector`
loaded through `--kv-transfer-config '{"kv_connector": "CPUOffloadingConnector",
"kv_connector_module_path": "vllm_ascend.distributed.cpu_offload_connector"}'` dies at
`initialize_kv_cache` → `register_kv_caches` with
`AttributeError: 'AscendConfig' object has no attribute 'use_sfa'`. The two streams in
the paste are one failure: `Worker failed with error ''AscendConfig' object has no
attribute 'use_sfa''` re-raises the worker's `AttributeError`.

Not this page: a missing attribute that a **library** type never had (a downstream
typo — `library-method-typo.md`), a copy of an **upstream model body**
(`patched-forward-api-drift.md`, rule 14 there), or the upstream
`offloading_connector.py` `KeyError` under an MLA checkpoint
(`kv-offloading-layer-coverage.md`, v0.17 shape).

## Mechanism (two greps, in this order)

1. `rg -n '<field>' <ascend-root>/vllm_ascend/` — at #4891 this returns
   `distributed/cpu_offload_connector.py:440` `use_sfa = ascend_config.use_sfa` plus four
   **commented-out** `#use_sfa=False` lines in `tests/ut/test_platform.py`. Nothing in
   `ascend_config.py` sets it: the field was dropped from `AscendConfig`, and the copy is
   its only live reader (`:447` `if use_mla and not use_sfa:` is the only consumer).
2. Read the copy header and the source function: `cpu_offload_connector.py:434`
   `# Copied from vllm_ascend/worker/model_runner_v1.py.` names
   `get_kv_cache_spec`, whose live body reads an instance attribute instead —
   `model_runner_v1.py:315-316`
   `self.use_sparse = hasattr(self.vllm_config.model_config.hf_config, "index_topk")`,
   consumed at `:3343` `use_sparse = self.use_sparse`. The flag was re-homed
   (`AscendConfig` field → model-config predicate), the duplicate kept the old read.
   That is the whole diagnosis; do not reach for a version pairing (the paste's
   `vllm 0.11.0` + `vllm-ascend 0.11.0rc3` trio is this snapshot's own pair) and do not
   read the connector's copy/load code before saying so.

## Repair line, and why "it started" is not acceptance

- Replace the read with the source's live predicate. For a worker-side connector there
  is no model-runner instance, so the honest form is to take the runner's own
  `kv_cache_groups` specs rather than re-derive them; a one-line
  `use_sparse = hasattr(vllm_config.model_config.hf_config, "index_topk")` is the
  minimal patch for a dense model, and `use_sfa = False` is valid for that reporter
  only.
- A `getattr(..., False)` guard is a start-only diagnostic (see rule 14 of
  `index.md`); the acceptance check is one real request compared with the same body
  served **without** `--kv-transfer-config`.
- Say what comes next: at `v0.11.0rc3` this file has **no test and no doc reference** —
  `rg -ln 'CPUOffloadingConnector|cpu_offload_connector'` hits only the connector and
  `cpu_offload_manager/metadata.py` — so the copy is unexercised code and the *next*
  failure is likely inside it (`metadata.py`, the block-slice copy). A stale field is
  one instance of the drift, not the end of the audit.
  `rg -n 'Copied from' vllm_ascend/` lists the sibling copies worth a glance when the
  same paste carries a second `AttributeError`.

Pins: vllm-ascend `b6d63bbd` (rc3) / vllm `b8b302cde434` (#4891, opened 2025-12-10).
Recheck per ref: the handler moved to `vllm_ascend/kv_offload/` around v0.17 and the
legacy connector family has its own contract audit
(`legacy-wrapper-contract-audit.md`); grep the field before quoting these line numbers.
