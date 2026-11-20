# `Communication_Error_Bind_IP_Port(EI0019)` and impossible port numbers

String family (verbatim from #12487's paste):

```
Communication_Error_Bind_IP_Port(EI0019): Failed to enable listening for the host
network adapter socket. Reason: The IP address [NETWORK_ADDRESS_1] and port 65536
have already been bound.
Solution: 1. Check whether this port has been occupied by another process. If yes,
you can adjust using the environment variable HCCL_IF_BASE_PORT and use
sysctl -w net.ipv4.ip_local_reserved_ports=****-**** ...
2. Check whether the service process is started multiple times on a device ...
```

It arrives as a device error record inside a Python traceback, so it inherits the
rule-13 problem: the record keeps its own date. In #12487 the title named this port
line, while every log line was `07-21 06:27:44` and the record itself said
`2026-07-18-11:20:51` — three days before the reported crash — and the in-window
first error was an AICPU `RunAicpuRpcSrvLaunchV2_reduce_scatter` /
`execute kernel param invalid` with 204 s notify-wait timeouts on
`chipId:4`. Run `plog-timeline.sh --since <report date>` before ranking it; when
the record date is outside the window it is a re-print and contributes nothing to
this incident.

## What the number proves

The reported port can exceed the TCP range (`65536`). No socket can bind that, so
"have already been bound" is HCCL's own wrapping text around its port arithmetic,
not evidence of a foreign process holding the port. Verified anchors at
`cann/hcomm` (in-tree export):

- `src/pub_inc/hccl_common.h:170` — `constexpr u32 HOST_CONTROL_BASE_PORT = 60000;`
  ("控制面起始port"), the control-plane base.
- `src/platform/common/externalinput.cc:828-853` — `HCCL_IF_BASE_PORT` is parsed
  and validated to `[HCCL_BASE_PORT_MIN, HOST_PORT_MAX]` = `[1024, 65520]`
  (`:213`); out of range or unset falls back to the base, logging
  `HCCL_IF_BASE_PORT set by default to [60000]`.
- The consumer (`GetExternalInputHcclIfBasePort`, `externalinput.cc:1855`) and the
  per-communicator increment are **not** in the export, so how a specific rank
  lands on 65536 is a stated gap, not a fact.

So: base 60000 plus a per-comm/per-rank offset that ran past `HOST_PORT_MAX`. It
is a control-plane port-band problem, at start-up of whichever process emitted it,
and it is a real finding **only** when its date is inside the incident window
(usually the engine start). `grep -m1 'HCCL_IF_BASE_PORT set' <plog|serve log>`
gives the base actually in use.

## Levers (only when in-window)

1. Move the band down and reserve it on every node:
   `export HCCL_IF_BASE_PORT=20000` plus
   `sysctl -w net.ipv4.ip_local_reserved_ports=20000-24000` (both are what the
   message itself recommends; the reserved range keeps ephemeral sockets out of
   it).
2. `ss -ltnp | grep -E ':(6000[0-9]|6[0-9]{4})'` and `ps -ef | grep -E 'vllm|python'`
   on the affected node: a leaked process from a previous launch holding the same
   band is the second thing the message names.
3. Then reconcile the timeline: if the service ran for days after a start-time
   EI0019, the bind failure belongs to a different process/attempt, and the crash
   is elsewhere.

Do not build a mechanism out of this line when the paste's in-window error is a
device fault: in #12487 the reported rank (`DP2_TP0_EP8`, `Device:0`) was not even
the faulting chip (`device_id=8`, `chipId:4`), which is the lane-C pattern of
`collective-error-chain.md`.
