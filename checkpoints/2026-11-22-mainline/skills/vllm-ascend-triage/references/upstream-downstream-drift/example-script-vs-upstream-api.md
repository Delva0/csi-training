# Example script vs the pinned upstream API (`from_cli`, and what the V0/V1 lane really does)

Signature family:

```
File "/vllm-workspace/vllm-ascend/examples/disaggregated_prefill/disaggregated_prefill_offline.py", line 38, in run_prefill
    ktc = KVTransferConfig.from_cli(
AttributeError: type object 'KVTransferConfig' has no attribute 'from_cli'
…
All process done!
```

- The deepest frame is a **repo example** (`vllm-ascend/examples/<dir>/<script>.py`),
  usually in a `multiprocessing` child, so the same traceback appears twice, and the
  script's own closing line (`All process done!`) follows it — which is why the report
  reads like a harness/teardown problem. No engine code ran.
- Recorded #1450, pins vllm-ascend `v0.9.1rc1` `c30ddb833d` / vllm `v0.9.1`
  `b6553be1bc`, 910B3, torch_npu 2.5.1, CANN 8.1.RC1, `transformers` 4.52.4. Four
  call sites: `examples/disaggregated_prefill/disaggregated_prefill_offline.py:46,86`
  and `examples/offline_disaggregated_prefill_npu.py:49,89`.
- The snippet has a **maintainer origin**: `[Guide]: How to use disaggregated_prefill`
  (#857 in this snapshot's corpus, the first hit for
  `bash <skill_root>/scripts/history-titles.sh -b 'from_cli'`) pastes the same
  `KVTransferConfig.from_cli` + `ktc.kv_connector_extra_config = …` +
  `print("All process done!")` block. Cite it as where the pattern came from — the
  rot is upstream of the reporter, not a modification they made.

**Classification:** the shipped example is stale against the pinned upstream. The
reporter's install, cards and CANN are not implicated and there is no version pairing
to hunt. Deliverable: the corrected script, the one recipe that *is* current, and "file
the example fix upstream".

## Four calls settle it

1. **The symbol, in both trees.** `rg -n '<method>' examples/ vllm_ascend/` plus
   `rg -n 'class <Config>|def <method>' <vllm>/vllm/config.py`. Recorded: at vllm
   `v0.9.1` `KVTransferConfig` is `vllm/config.py:3641` with **no** `from_cli`; the
   only `from_cli` classmethod in the tree is `CompilationConfig.from_cli`
   (`vllm/config.py:4047`). The attribute is genuinely absent at the pin, so "wrong
   vllm installed" is refuted by the pin itself — close it for the reporter anyway
   with one line they run:
   `python -c "import vllm; from vllm.config import KVTransferConfig as K; print(vllm.__version__, vllm.__file__, hasattr(K,'from_cli'))"`.
2. **Why the author thought it existed — the CLI path is generic.**
   `vllm/engine/arg_utils.py:207-209` is
   `if hasattr(cls, "from_cli"): return cls.from_cli(val)` else
   `return TypeAdapter(cls).validate_json(val)`. That is why
   `--kv-transfer-config '<json>'` works for every `@config` dataclass while only
   `CompilationConfig` implements `from_cli`: the snippet copied the CLI's duck-typed
   path into Python, where the class is a plain dataclass.
3. **The correct constructor form, from the tree itself.** No `from_cli` is needed —
   `KVTransferConfig(kv_connector=…, kv_buffer_device="npu", kv_role=…,
   kv_parallel_size=…, kv_connector_extra_config=…)` validates the same way:
   `__post_init__` (`vllm/config.py:3704`) checks `kv_role` against `KVRole` (`:3709`)
   and the `kv_connector ⇒ kv_role` pairing (`:3713`), while `kv_buffer_device` is not
   validated at all, so `"npu"` is fine despite its `"cuda"` docstring (`:3651`). The
   tree's own idiom is the direct call
   (`tests/e2e/singlecard/core/test_ascend_scheduler.py:81`, a different connector),
   as is upstream's (`vllm/examples/offline_inference/disaggregated_prefill.py:37,81`).
   Pass `kv_connector_extra_config` **into** the constructor instead of assigning it
   afterwards.
4. **The V0/V1 lane — a check, and the direction matters.** Both Ascend connectors
   subclass the **V0** base (`vllm_ascend/distributed/kv_transfer/simple_connector.py:24,36`;
   `vllm_ascend/distributed/llmdatadist_connector.py:26,133` import
   `…kv_connector.base.KVConnectorBase`), and at this pin an **unset** `VLLM_USE_V1`
   is not a problem: the engine's own oracle declines V1 on NPU
   ("npu is experimental on VLLM_USE_V1=1. Falling back to V0 Engine.",
   `_warn_or_fallback`, `vllm/engine/arg_utils.py:1645-1659`) and then writes
   `VLLM_USE_V1=0` itself (`vllm/engine/arg_utils.py:1025-1035`: `try_v1` then the oracle,
   then `if envs.is_set("VLLM_USE_V1"): assert use_v1 == envs.VLLM_USE_V1` else
   `envs.set_vllm_use_v1(use_v1)`). That is why
   `tests/e2e/pd_disaggreate/setup_pd.sh` runs `AscendSimpleConnector` with no such
   export. An **explicit `VLLM_USE_V1=1`** is the state that breaks it: `:1647-1654`
   only warns and keeps V1, so the engine reaches
   `create_connector_v1`'s `assert issubclass(connector_cls, KVConnectorBase_V1)`
   (`vllm/distributed/kv_transfer/kv_connector/factory.py:73`), or
   `_raise_or_fallback` (`:1633-1640`) raises `NotImplementedError: VLLM_USE_V1=1 is
   not supported with device type=npu`; `pytest.ini:60` records the same limit.
   So ask "did you export `VLLM_USE_V1`?" and never hand `VLLM_USE_V1=1` over as a
   workaround. The recorded answer called an explicit `VLLM_USE_V1=0` a required
   second fix — at this pin it is harmless but unnecessary; the discriminating
   question is whether the reporter exported a **1** (the R1-era copy-paste habit,
   `examples/offline_inference_npu_v1.py:24`, `examples/run_dp_attention_etp16.sh:2`).

## Why it rotted, and the one recipe that is current

Rank the three copy sources by whether their keys still match the pin:

- `tests/e2e/pd_disaggreate/setup_pd.sh` — **current**, and wired into
  `.github/workflows/vllm_ascend_test_pd.yaml`: `AscendSimpleConnector` as
  `--kv-transfer-config` JSON (`:69`, `:122`) with `prefill_device_ips` (`:52`,
  `:102`) and `llmdatadist_comm_port` 26000 (`:54`), no `VLLM_USE_V1` export.
  Recommend this (or its `run_*_server.sh` shape, once the key is fixed).
- `examples/disaggregated_prefill/disaggregated_prefill_offline.py` — right key
  (`prefill_device_ips`, `:16`), wrong constructor (`from_cli`).
- `examples/disaggregated_prefill/run_prefill_server.sh` / `run_decode_server.sh` —
  the CLI form is right, but their extra config uses **`prompt_device_ips`** (`:31`),
  which the pinned connector no longer reads: `simple_pipe.py:64-73` requires
  `prefill_device_ips` and raises `ValueError: Please specify prefill_device_ips and
  decode_device_ips…` otherwise. Fix the key before recommending them; the `V0`
  exports they do carry (`:9`) are not what makes them work.

`kv_connector_extra_config` is load-bearing, not decoration:
`vllm_ascend/distributed/kv_transfer/simple_pipe.py:64-73` reads
`prefill_device_ips`/`decode_device_ips` (absent ⇒ `ValueError`), `:81-82` defaults
`llmdatadist_comm_port` to 26000, and `:96-101` reads `proxy_ip`/`proxy_port`
(both empty ⇒ the ZMQ proxy registration thread is skipped, which is the intended
single-pair mode). The examples' `1.2.3.x` values are placeholders; real device IPs
come from `examples/disaggregated_prefill/find_device_ips.py`, named in the
docstring at `:8`.

The two `.py` examples have no CI and no `docs/` reference
(`rg -n 'disaggregated_prefill_offline' docs/` and the same for
`offline_disaggregated_prefill_npu` are both empty at this pin) — that is the reason
they rotted, and it belongs in the answer.

## Fingerprint the paste before quoting counts

The recorded reporter's frames were `:38`/`:79` where the pin has `:46`/`:86` — a
checkout older than the pin, with the same defect. Quote the pin's lines as the
pin's, say the paste's tree is slightly older, and give the one grep that counts
their call sites (`rg -n 'from_cli' examples/`). Do not reconstruct their file from
the pin (`SKILL.md` rule 5, `index.md` rule 9).

## Delivery note

The recorded run reached the right verdict but spent 18 assistant turns on it; the
four calls above plus one extra-config read are the whole investigation, so the
verdict (example rot, not your install) belongs in message text by the second or third
turn, with the constructor patch attached.

## Pitfalls

- **Do not patch the `issubclass` assert** to "get it running": a V0 connector on the
  V1 runner is not exercised at this ref.
- **Do not answer "upgrade vllm"**: the pin already carries the working CLI form and
  the class never had `from_cli`.
- **Do not re-add `from_cli`** to `KVTransferConfig` in the reporter's copy: the
  example is not a library surface, and a monkeypatched classmethod hides the stale
  keys behind it.
- A test constructing the config directly with a *different* connector
  (`SharedStorageConnector`) is evidence for the constructor form only, not proof
  that `AscendSimpleConnector` runs on the V1 engine.
