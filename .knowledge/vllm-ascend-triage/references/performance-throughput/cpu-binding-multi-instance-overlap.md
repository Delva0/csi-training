# Two single-card instances on one host, one CPU core range: `topo_affinity` binding overlap

## Scope

An ARM A2/310P host runs **more than one** `vllm serve` — one card per process
("分别执行单卡推理"), or several DP instances — and the complaint is **speed**, not a
crash: "A2相同亲和性的NPU分别执行单卡推理时绑核范围重合导致性能下降", "两个实例的绑核范围
一模一样", "开两个单卡服务比开一个慢很多". The paste usually carries two
`The CPU allocation plan is as follows:` blocks that are **byte-identical**, and
`[cpu_bind_mode] mode=topo_affinity`. Owner: `vllm_ascend/cpu_binding.py`, the
host-side affinity policy that `enable_cpu_binding` (default **True**) turns on — not
a card, HCCL, NUMA-bandwidth-only or graph-mode question (recorded #8600, v0.18.0rc1).

The neighbouring but different shapes: a stray/foreign PID on a die is
`../deploy-sizing/node-processes-and-device-attribution.md`; binding that runs *after*
warm-up and hangs a 200-GiB engram/offload serve is
`../feature-enable-recipes/engram-offload.md`; a whole-host death while long-context
masks are built is `../runtime-memory/attention-mask-host-allocation.md`.

## The pin's own admission: read `release_notes.md` before the source

At v0.18.0rc1 (`99e1ea0`) the cumulative notes fill both sides:

| Anchor | Section | Says |
|---|---|---|
| `release_notes.md:547` | `## v0.11.0` — **Known Issues** | "CPU bind feature doesn't work if more than one vLLM instance is running on the same node." |
| `release_notes.md:650` | `## v0.11.0rc2` — **Known Issues** | "cpu bind feature doesn't work for multi instance case(Such as multi DP on one node). We'll fix it in the next release." |
| `:87` (#6945), `:139` (#6889) | `## v0.16.0rc1` | "Implement global CPU slicing … ensuring non-overlapping CPU partitions" / "Fix CPU binding logic" |
| `:196` (#6785), `:182` (#6686), `:370` (#5555) | v0.15.0rc1 / v0.14.0rc1 | memory migration + IRQ cores; ARM-only A3 policy; bind-logic rewrite |

The catch that makes this the verdict rather than a footnote: every "non-overlapping"
fix is the **`global_slice`** strategy, and `cpu_binding.py:22-26` assigns
`global_slice` to **A3 only** — `A2` and `_310P` keep `topo_affinity`. So on an A2
the admitting Known-Issue row is still an accurate description at v0.18.0rc1, and the
answer opens with "real, and the project documents it for multi-instance; the A3 fix
does not cover your SoC", not with a config hunt. A later pin's notes carry
"**Avoid A2 CPU binding overlap from hidden NPUs**" (#8792, quoted at
`upgrade-vs-older-baseline.md` §*Step 2b*), i.e. an A2 fix did land after this pin —
its exact coverage is **unverified here**, so grep the reporter's own pin for it and
label an upgrade rung as untested.

## Why two processes collide (source-verified at v0.18.0rc1, `99e1ea0`)

| Where | What it does |
|---|---|
| `cpu_binding.py:22-26` | `DEVICE_BINDING_MODE`: `A2`/`_310P` → `topo_affinity`, `A3` → `global_slice` |
| `:93-124` `get_running_npus` | the NPU list is `npu-smi info`'s **process table** (mapped through `npu-smi info -m`) ∩ `ASCEND_RT_VISIBLE_DEVICES`; an empty result raises, and `worker/worker.py:302-306` catches it |
| `:331-338` topo branch | pool per **visible** npu = `npu_affinity[npu] ∩ allowed_cpus` (`npu-smi info -t topo`, `:133-146`, which parses **all** cards) |
| `:209-222` `extend_numa` | then the **whole next NUMA node** (cyclic `(node+1) % len(numa_to_cpu_map)`) is appended if those CPUs are in `allowed_cpus` — the pool doubles |
| `:340-349` + `:198-207` | NPUs whose **extended pool string is identical** are split evenly by `average_distribute` — **only among the NPUs of this process** |
| `:352-364` `allocate` | `pool[0],pool[1]` reserved for IRQ, `main = pool[2:-2]`, `acl = pool[-2]`, `release = pool[-1]` |
| `:450`, `:492-495` `bind_npu_irq` | sq/cq `smp_affinity` written to `cpus[0]`, `cpus[1]` |
| `:424-435` | stops `irqbalance` **host-wide** (and says so in a warning) |
| `worker/worker.py:302-306` | any exception inside binding is a `Bind cpus failed in rank…: … Skip binding cpu.` warning — "no error" is not "bound" |

Two consequences and that is the whole mechanism:

1. A single-card process sees exactly **one** NPU (`running_npu_list == [n]`), so its
   group has one member and `average_distribute` never fires: it takes the *entire*
   base+extension band.
2. Nothing is shared between processes — no lock, no state file, no node-wide index.
   Two processes whose cards sit in the same `npu-smi info -t topo` band therefore
   derive the same band, and their `main`, `acl`, `release` **and** IRQ cores all
   coincide. The guard that prevents this on A3 (`global_slice`, slicing by global
   logical NPU id) is exactly the code path A2 does not take.

## The arithmetic that settles it from the paste alone

`allocate()` trims the pool, so a plan line can be inverted:

```text
NPU0: main=[146 … 189]  acl=[190]  release=[191]
  ⇒ pool = [144, 145] (the two IRQ cores) + main + [190, 191] = 144..191  (48 CPUs)
  ⇒ base = the card's own `CPU Affinity` column (144-167, 24 CPUs)
     + extend_numa's neighbour node (168-191, 24 CPUs)
  ⇒ group size was 1: a 2-card group would have split the 48 cores in half
```

That single inversion distinguishes the three readings of a "慢" report — a real
overlap (identical plans, base+extension), an intra-process split that worked (two
halves of one band), and no binding at all (no plan line, plus the
`Skip binding cpu` warning). Then build the band table from the reporter's own
`npu-smi info -t topo` `CPU Affinity` column:

| Affinity column | Cards sharing it | Band after `extend_numa` |
|---|---|---|
| 144-167 | NPU0, NPU2 | 144-191 |
| 0-23 | NPU1, NPU3 | 0-47 |
| 96-119 | NPU4, NPU6 | 96-143 |
| 48-71 | NPU5, NPU7 | 48-95 |

**Same band ⇒ identical plans; different bands ⇒ disjoint plans.** That table is why
"0+1 works and 0+2 collides" on this host, and it is derived from the reporter's
hardware, not from a general rule: on another box the bands and the pairing differ,
so ask for the topo table before naming a pair.

One call does the inversion and the comparison for every plan in a paste:
`uv run --no-project python <skill_root>/scripts/cpu-bind-plan-audit.py server*.log`
(pools, IRQ cores, NUMA anchors, and a `disjoint` / `overlap N` / `identical` verdict
per pair).

## Ranked causes

1. **Same-band cards + per-process pool ⇒ identical plans (source-verified at the
   pin).** Check: the two `NPU<n>: main=[…]` lines are byte-identical and
   `[cpu_bind_mode] mode=topo_affinity` (`:320`) with
   `visible_npus=[<one id>]`; the inverted pool equals base + one whole neighbour node.
   This is a design gap for multi-instance hosts, not a wrong flag — `enable_cpu_binding`
   is the only knob (`ascend_config.py:90`), and there is no mode override.
2. **The IRQ cores collide too, and they survive a partial fix (source-verified).**
   Both instances put their sq/cq interrupts on the same two CPUs (`pool[0]`,
   `pool[1]`), and the first process stops `irqbalance` for the whole host
   (`:424-435`), so a hand-split of `main` alone leaves both cards' interrupts on one
   core pair. Check: `grep sq_send_trigger_irq /proc/interrupts` plus each instance's
   `smp_affinity`.
3. **The two cards also share the memory band, which no CPU split removes (premise
   from the reporter's table + the `[migrate] NPU:0 -> NUMA [6]` /
   `NPU:2 -> NUMA [6]` lines).** `bind_memory` anchors migration on `pool[0]`
   (`:374-400`), so two same-band cards migrate to the *same* NUMA node; even a
   perfect CPU partition leaves them sharing that node's bandwidth and PCIe root. This
   is why the pairing rung outranks every tuning lever.
4. **A cpuset that excludes the band degrades silently (source-verified).** Topo mode
   raises `CPUs available in 'Cpus_allowed_list' conflict with NUMA affinity.` when
   `npu_affinity[npu] ∩ allowed_cpus` is empty, and the worker catches it — so a
   container cpuset can produce "no binding at all" while the reporter believes the
   halves were split. Check for the `Skip binding cpu` warning before crediting a
   partition.

## Action ladder

1. **Pair cards from different bands** (here `0+1`, `2+3`, `4+5`, `6+7` instead of
   `0+2`): zero cost, no code path changed, and the plans become disjoint by
   construction. Verify in the log, not in theory: the two `main=[…]` ranges must not
   intersect.
2. **If the same-band pair is unavoidable, stop relying on the feature:**
   `--additional-config '{"enable_cpu_binding": false}'` on **both** instances, then
   partition by hand — `taskset -c 144-167 numactl --cpunodebind=6 …` and
   `taskset -c 168-191 numactl --cpunodebind=7 …` — set the IRQ affinity yourself
   (`echo <mask> > /proc/irq/<id>/smp_affinity`, needs privileges; the feature's IRQ
   half is gone with it), and keep `OMP_NUM_THREADS` matched to the cores each
   instance actually gets. A cpuset that no longer contains the card's affinity band
   is the failure mode to avoid — it silently skips binding (cause 4).
3. **Code change (proposal, not run anywhere):** group over `DeviceInfo.npu_affinity`
   (`:133-146` already parses *all* cards, no visibility filter) and give each NPU its
   index **within its affinity band**, mirroring `_slice_for_npu` (`:292-297`). Every
   process then derives the same disjoint split from the same topo table. Do **not**
   present "force `global_slice` on A2" as the fix: that policy slices the cpuset by
   global logical NPU id and its own docstring says "NUMA locality is achieved only if
   CPU numbering aligns with NUMA layout" (`:237-252`) — on this host it would move
   NPU0 off its own band.
4. **Version rung:** a release whose notes carry the A2-overlap fix (#8792 per
   `upgrade-vs-older-baseline.md` §*Step 2b*); grep the reporter's pin for that bullet
   first, and label the upgrade untested.

## Paste hygiene for this shape

- **A plan line that prints a CPU outside the host's CPU range is a transcription
  slip, not a second bug.** #8600's second block reads
  `IRQ_ID=2562 -> CPU3679, cq_update_irq IRQ_ID=3680 -> CPU145`; the emitter is
  `IRQ_ID={sq_irq} -> CPU{sq_cpu}` with `cq_irq = sq_irq + 1` and
  `sq_cpu = cpus[0]` (`:450`, `:488-495`), so the raw line can only have been
  `IRQ_ID=3679 -> CPU144, cq_update_irq IRQ_ID=3680 -> CPU145`. Ask for
  `grep -n 'cpu_binding' server*.log` with the plan blocks instead of ranking a
  phantom.
- **A one-line skew** between the paste's `cpu_binding.py:<N>` and the pin is a
  release-vs-rc difference (`print_plan` logs at `:366-372`); re-grep at the tag the
  reporter quotes rather than calling their build different.
- `visible_npus=[…]` with **more than one** id means a multi-card engine, so this page
  does not apply (that case's overlap question is `average_distribute`'s own).
- With `ASCEND_RT_VISIBLE_DEVICES` unset the list is node-wide and
  `current_npu = running_npu_list[rank_id]` (`:368`, `:408`, `:421`) — rank *r* can be
  planned against **another** instance's card. That is how "an extra process on my
  NPU" and a wrong binding show up in the same report; pin the visible set first
  (`../deploy-sizing/node-processes-and-device-attribution.md` rules 1 and 3).

## What to send back

Both serve commands; **all** ranks' plan blocks (the block is printed per rank, so a
single-rank paste hides the others); the host's `npu-smi info -t topo`; each
instance's cpuset (`grep Cpus_allowed_list /proc/self/status`); the metric that
dropped, per instance alone versus together (tok/s, TTFT); and each instance's sq/cq
interrupt CPUs.

## Scope

Source-verified at vllm-ascend **v0.18.0rc1** (`99e1ea0`), the pin of recorded #8600;
that case had **no follow-up in the snapshot**, so nothing here is reporter-confirmed
— treat cause 1 as the ranked leader from source + the paste's own arithmetic, not as
a fix the reporter validated. The band table is the reporter's own `npu-smi info -t
topo`, so re-derive it per host. The A3 `global_slice` code and the later A2 fix
(#8792) were read only as notes, never run; a newer pin may change
`DEVICE_BINDING_MODE`, so re-read `cpu_binding.py:22-26` and the notes before quoting
any rung.
