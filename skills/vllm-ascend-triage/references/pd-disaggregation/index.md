# PD/EPD disaggregation: transport plane and KV-state transfer

Scope: a P/D (or EPD) deployment that starts but cannot move KV, or moves it and
produces wrong tokens — *not* the generic startup lane. Two failure families,
decided by one question first: **did the peers fail to connect, or did they
connect and the decode node then produced wrong output?**

- Connect/handshake failures, `HIXL Connect failed | HIXL Error Code: 103900`
  (or 503900 and friends), `LMCache ERROR: wait_for_save failed ... skipping
  save`, HCCL `RanktableCheck` / `'device_ip' is not set correctly`,
  `get dev list error:serverId[...]`, P/D pairs that only work with
  `hostNetwork: true`, `--net=host`, or one shared IP →
  `transport-topology.md`.
- Connected, but the answer is corrupt/empty/alternating, or the decode node's
  first token is wrong, on a **hybrid linear-attention (GDN/Mamba)** model —
  Qwen3.5/Qwen3.6, DeepSeek-V4, GLM — under PD →
  `hybrid-state-transfer.md`.

Inherit the rules in SKILL.md. Rules specific to this topic:

1. **The transport identity is a topology input, not a tunable.** Any Ascend
   transport that hands HCCL one `ip:port` string per process (HIXL/ADXL through
   LMCache-Ascend or Mooncake's asc-end-direct transport) uses that same string
   as the HCCL rank-table `server_id`. Two processes on one physical node with
   two different addresses are therefore *two servers* to HCCL, which then
   demands `device_ip` and a NIC data path. Before reading connector code,
   establish what the rank table says and run
   `bash <skill_root>/scripts/ranktable-check.sh <table-or-log>`; it prints the
   effective `server_count`, the per-server ids and the decision that follows.
2. **A transport error code is not a diagnosis.** HIXL/HCCL numeric codes
   (`103900`, `503900`, `507057`) carry almost no information and two different
   codes are usually different failure classes. The reason lives in
   `~/ascend/log/debug/plog/plog-<pid>_*.log`; ask for that file, not for a
   better Python traceback.
3. **"Not in the documented recipe" is not "unsupported".** The LMCache-Ascend
   examples state RoCE as a prerequisite and mark `hixl` experimental, while
   reporter evidence shows the same channel completing over HCCS once the rank
   table converges to one server. Keep the two claims apart: no validated path
   vs. impossible on that hardware.
4. **Name the owning repository.** vllm-ascend's `LMCacheAscendConnector` is a
   shim that re-exports upstream `LMCacheConnectorV1`
   (`vllm_ascend/distributed/kv_transfer/kv_pool/lmcache_ascend_connector.py:1-5`
   at v0.23.0); an HIXL engine-id / rank-table change belongs to
   `LMCache/LMCache-Ascend` or `cann/hixl`, not to vllm-ascend.

## Pages

- `transport-topology.md` — P/D has two Pod IPs on one node and the transport
  will not connect: how the engine id becomes HCCL's `server_id`, the plog grep
  set and rank-table decision, `HCCL_INTRA_ROCE_ENABLE` / `HCCL_INTRA_PCIE_ENABLE`
  semantics (source-verified in the `cann/hcomm` export), port-band collisions,
  and what is verifiable in the snapshot when `cann/hixl` is absent.
- `hybrid-state-transfer.md` — PD on a hybrid linear-attention model: the
  connector matrix (`MooncakeConnectorV1` / `MooncakeHybridConnector` /
  `MooncakeLayerwiseConnector` / `AscendStoreConnector`), the
  `mamba_cache_mode` gate that only the KV-store connectors satisfy, which
  runner paths that skips, and the ablation matrix to hand the reporter when the
  symptom is alternating good/bad output.
