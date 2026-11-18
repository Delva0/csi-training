# Multi-node startup: `ZMQError: Address already in use` in the message queue

Worked case: vllm-ascend #16237 — "0.23.0版本，PD分离，开启pp，多机 prefill 启动报 MessageQueue ZMQ bind
"Address already in use"". 4×A3, 2 prefill nodes (16 cards each) + decode nodes, GLM-5.3 w8a8c8,
DP=2/PP=2/TP=8/EP, `--distributed-executor-backend mp --nnodes 2`, MemCache + Mooncake
(`MultiConnector`), `--async-scheduling`; vllm-ascend `0.23.0` with vLLM `0.23.0`. Open, no comments at
read time (2026-11-17).

## Signature

```
zmq.error.ZMQError: Address already in use (addr='tcp://[NETWORK_ADDRESS_1]:60001')
  multiproc_executor.py:588 _init_message_queues
  → parallel_state.py:521 create_single_reader_mq_broadcasters
  → shm_broadcast.py:846 create_from_process_group_single_reader
  → shm_broadcast.py:434 self.remote_socket.bind(socket_addr)
```

- Intermittent, worse when both nodes' 16 workers start together; the port is not in `ss`/`netstat`
  afterwards (no lingering process), and changing the Mooncake `kv_port` does not help — it is not a
  KV port.
- The peer node fails with `RuntimeError: ... Connection closed by peer [ip]:<port>`: one worker's
  bind fails and the `dist.gather_object`/`all_gather` handshake is torn down. That is a consequence,
  not a second defect.

## Owning repo is vLLM, not vllm-ascend

The whole traceback is upstream (`vllm/v1/executor/multiproc_executor.py`,
`vllm/distributed/parallel_state.py`, `vllm/distributed/device_communicators/shm_broadcast.py`);
vllm-ascend only launches it. Fingerprint the ref in **both** clones: at vllm-ascend `v0.23.0` the
pairing file `.github/vllm-release-tag.commit` says `v0.23.0`, and vLLM `v0.23.0` has
`remote_socket.bind` on exactly `:434`, `create_from_process_group_single_reader` containing `:846` and
the caller on `:521` — lines match, so the reported build is that tag (not `main`).

## Mechanism (verified)

Pre-fix, the writer asks for a port and binds it later:

```python
remote_subscribe_port = get_open_port()          # bind(("",0)) probe, socket closed
socket_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
self.remote_socket.bind(socket_addr)             # ZMQ binds the same number afterwards
```

`get_open_port()` (`vllm/utils/network_utils.py`) is a **probe**: `_get_open_port()` binds port 0,
reads the assigned port, closes the socket, and returns it. Another process doing the same probe in
that window can be handed the same number, so two concurrent writers announce and bind the same
`tcp://ip:port`. That is why the failure is intermittent, why it scales with the number of workers
started at once, and why nothing is holding the port afterwards.

Fixed upstream by `f7ef489e93` — "[Bugfix] Fix ZMQ port TOCTOU race in shm_broadcast MessageQueue
(#50960)", 2026-08-07: bind `tcp://{connect_ip}:0` and read the port back from
`remote_socket.getsockopt(zmq.LAST_ENDPOINT)`, with a regression test
(`tests/distributed/test_shm_broadcast.py::test_remote_subscribe_addr_unique_concurrent_writers`).
First release containing it: **vLLM v0.28.0rc1** (`git tag --contains f7ef489e93`). Not in v0.23.0,
v0.27.x, or `main` before that date.

## Do not relay the reporter's own diagnosis unverified

#16237's 「初步定位」 quotes a `get_open_port()` that keeps a `_used_ports` set to avoid reuse. **No
such symbol exists** at vLLM `v0.22.1`, `v0.23.0`, `v0.24.0` … `v0.30.0`, `main`, or anywhere in the
vllm-ascend clone (`git grep -n _used_ports HEAD` → nothing); what exists is the `VLLM_DP_MASTER_PORT`
…+10 exclusion and, in `_get_open_port`, an upward scan from `VLLM_PORT` when that env var is set.
The race they describe is real without it. Report the confirmed probe-then-bind race instead of the
quoted "fix", and say explicitly that their snippet does not exist at their ref
(`../workflow/version-pinning.md`, "Reporters' code quotations").

## Workaround

- Practical on 0.23.0: retry the launch (it is intermittent), and stagger worker start-up (start the
  nodes a few seconds apart) to shrink the collision window. `VLLM_PORT=<base>` narrows the range but
  every process still probes from the same base, so it does not remove the race; the mq writer path
  does not use `get_open_ports_list()`.
- Real fix: a vLLM build containing #50960 (v0.28.0rc1+), i.e. the vllm-ascend release that pairs
  with it. Version string alone is not enough — check
  `git show <vllm tag>:vllm/distributed/device_communicators/shm_broadcast.py | grep -n LAST_ENDPOINT`.
- Do not chase `HCCL_*`/`GLOO_*` env settings for this error: the frames are ZMQ, above the HCCL
  layer.

## Check before reuse

`git -C /workspace/repos/vllm-project/vllm grep -n "remote_subscribe_port" <tag> --
vllm/distributed/device_communicators/shm_broadcast.py`, then `git log -S'LAST_ENDPOINT' --all --` in
the same repo. Line numbers 588/521/846/434 belong to that one pairing; re-fingerprint per report.
