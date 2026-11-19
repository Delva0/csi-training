# PD proxies: the example script must match the connector, not the hardware

## Scope

PD disaggregation is up — both roles report ready, `/health` answers — and the **client-facing proxy**
from `examples/disaggregated_prefill_v1/` misbehaves: the P engine shuts down after the first forwarded
request, requests reach the proxy and never return, or answers come back wrong/empty. Usual cause: the
wrong proxy for the configured connector (often because the tutorial chose it by hardware name);
proxy↔backend address mistakes are the second. Reports arrive as "following the tutorial, the P node
shuts down as soon as I send a request" or "the proxy starts but nothing comes back", and often the
reporter has already found that swapping the example script fixed it.

Decide by the **connector** in each side's `--kv-transfer-config`, never by the hardware name or by the
script the tutorial happens to print. Not this page: a mid-run freeze with no proxy in the path
(`pd-collective-hang.md`), fluent-garbage answers from a hybrid model's missing state transfer
(`hybrid-gdn-pd.md`), or a KV pool that never receives a PUT (`kv-pool-store-hang.md`).

## Two proxies, two handshakes (`verified` at `v0.23.0`, both files present in the tag)

| script | order | request it sends | callback it serves |
| --- | --- | --- | --- |
| `load_balance_proxy_server_example.py` (plain) | proxy → **P** first, then D | `build_prefill_request` (`:790-805`): `kv_transfer_params = {do_remote_decode: True, do_remote_prefill: False, remote_engine_id/remote_block_ids/remote_host/remote_port: None}`, `max_tokens = min_tokens = 1`, `stream: False`, `stream_options` dropped; reads `kv_transfer_params` **out of the P response body** and copies it into the D request | none — no `/v1/metaserver` route |
| `load_balance_proxy_layerwise_server_example.py` | proxy → **D** first | `:431-436`: `kv_transfer_params = {do_remote_decode: False, do_remote_prefill: True, metaserver: http://<proxy host:port>/v1/metaserver}` | `POST /v1/metaserver` (`:600-609`): the D-side connector POSTs the chosen prefiller's params there, the proxy sends the prefill request, P pushes KV layer by layer |

The plain proxy serves `MooncakeConnector` (V1/V2; the D side pulls the KV); the layerwise proxy serves
`MooncakeLayerwiseConnector`. At HEAD the docs state the pairing directly
(`docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:1168-1170`, and the Qwen3.5
guide's §5.7 table `:1208-1209`: a "Connector → Request Order → Proxy" table prefaced by "Use the proxy
paired with the actual connector, rather than choosing it by hardware name"). Earlier releases often
picked the proxy by **hardware**, which is the trap below.

## Why the mismatched pair kills the P instance (`verified` at `v0.23.0`)

`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`, `update_state_after_alloc`:
a request carrying `do_remote_decode` enters the "layerwise prefiller add request need send" branch and
executes `remote_cache_tokens = params["remote_cached_tokens"]` (`:990`; still
`params["remote_cached_tokens"]` at `HEAD:1067`) — an unguarded index for a key the plain proxy never
sets. A `MooncakeLayerwiseConnector` producer therefore raises `KeyError: 'remote_cached_tokens'` inside
its scheduler step on the first forwarded request (`verified` by reading both sides); that an uncaught
scheduler-step exception takes the whole engine process down is the mechanism behind the reporter's
「P节点 shutdown」 (`reported`), not a hang. The layerwise proxy's own metaserver hop is
`_access_metaserver` (`:1089-1100`), which retries and re-raises when the URL is `None`. Do not claim the
reverse direction from symmetry: the plain `mooncake_connector.py` path branches only on
`do_remote_prefill`/`do_remote_decode` (`:1809-1915` at the same ref) and was not tested with a layerwise
proxy in these traces — label it `hypothesis` if a report needs it.

## Worked case #15389 (`0.23.0`; Qwen3.5-122B-A10B deployed from the 397B guide)

- [verified] §5.4 (A3 PD) of `docs/source/tutorials/models/Qwen3.5-397B-A17B.md` at `v0.23.0`
  configures `{"kv_connector": "MooncakeLayerwiseConnector"}` on **both** roles (`:521` P
  `kv_producer`, `:597` D `kv_consumer`).
- [verified] §5.6's A3 tab (5.6 `:807-838`) says "Use the layerwise proxy script" (`:817`) and then runs
  `python3 load_balance_proxy_server_example.py` (`:823`); its follow-up `bash proxy.sh` (`:832-836`)
  names a file that does not exist at that tag (`git ls-tree -r --name-only v0.23.0 | grep -i proxy` →
  two proxies, two markdown files). The 950DT tab (`:841-850`) is the mirror image: the prose and command
  use the layerwise proxy (`:844`, `:850`) while §5.5 configures plain `MooncakeConnector` (`:703`,
  `:786`). So the two tabs' commands are swapped relative to their own connectors.
- [reported] reporter: 执行后P节点shutdown, then 使用此协议后代理服务启动成功 with
  `load_balance_proxy_layerwise_server_example.py`.
- Status: corrected on main only (`737908a1d`, #18233, 2026-10-25; `git tag --contains` → no tag in
  this clone), and later docs fix the *shape* rather than the naming: `HEAD` §5.7 (`:1204-1215`) is the
  connector→proxy table and the recipes were renumbered to `MooncakeConnectorV1`/`V2` (`HEAD:553`,
  `:668`, `:825`, `:937`). At `v0.23.0` the defect is present, so the reporter's version is the
  authority; a v0.23.0 tutorial quote must be read against the tag, not against `main`.

## Triage order

1. Read `--kv-transfer-config` on **both** roles: the connector name is the discriminator
   (`MooncakeLayerwiseConnector`, or a `multi_connector` listing a layerwise child, versus
   `MooncakeConnector`/V1/V2).
2. Read the tutorial section they copied **at their tag**
   (`git show <tag>:docs/source/tutorials/models/<model>.md`) — never `latest`, which tracks main
   (#15836's lesson) — and compare its prose sentence, its command line and the connector in the same
   page's serve block.
3. Diff the proxy they ran against the other one at the two `kv_transfer_params` sites above; the
   presence of `metaserver` versus `remote_*` is the tell for which flow a file implements.
4. Only then addresses: a proxy backend is each service's **HTTP** port (not the DP RPC port, not the
   KV-transfer port), and the proxy's `--host` must be reachable from D because the D-side connector
   calls back to `/v1/metaserver`.

## Answer shape

The lever is a config/doc line, not code: "run the layerwise proxy" (reporter-confirmed here) or "use
the plain proxy with a `MooncakeConnector` deployment". Root cause = the tutorial's
prose/command/connector disagreement at `docs/…:<line>` in the reporter's tag (for a layerwise connector,
the connector's own `params["remote_cached_tokens"]` index at `mooncake_layerwise_connector.py:990` that
only the layerwise proxy supplies); status = present there, corrected on main in #18233. Do not chase a
`vllm_ascend/` frame: nothing is wrong inside a connector paired with the proxy its own docs specify.
Falsifier: one request through the proxy with the P log open — `KeyError: 'remote_cached_tokens'` /
`Access metaserver fail` (URL `None`) confirms this pairing class; a clean handshake with the plain proxy
means the deployment configures the plain connector and the tutorial text, not the command, is the
defect.

## Recheck conditions

Both example proxies and the tutorials are rewritten between releases (a `proxy.sh` example, a
`/reset_prefix_cache` broadcast endpoint, V2 rows in the pairing table all appear and move). Re-read the
two `kv_transfer_params` sites and the §5.x proxy sentence at the reported tag before reusing any line
number above.
