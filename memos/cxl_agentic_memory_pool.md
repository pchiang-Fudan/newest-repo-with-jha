# CXL Agentic Memory Pool Demo

## Product Pitch

Optically disaggregated commodity DDR5 memory pool for Gen5 CXL servers running
stateful agentic AI.

## Strategic Reframe

ContextPilot / Middle-Out Token Compression is not only an API middleware demo.
It is also a workload generator and system-level demo for a CXL-attached memory
appliance.

The key observation is that coding agents carry large warm state that is
valuable to keep alive but does not need GPU HBM latency:

- repo indexes
- AST and symbol graphs
- file snapshots
- terminal and test logs
- search results
- tool traces
- agent plans
- prompt and prefix caches
- prior patch candidates
- long-lived session history

This state is too expensive to recompute and too bulky to duplicate in every
local CPU node. It is also not the hottest decode path. That makes it a natural
fit for pooled commodity DDR5 behind a CXL memory fabric.

## Target Architecture

```text
coding IDE / agent clients
  -> CPU agent-control plane
       planning
       repo indexing
       retrieval
       tool orchestration
       terminal/test execution
       workflow/session state
       prompt assembly
       warm state in CXL pooled DDR5
  -> GPU inference plane
       H100/H20 or similar GPU servers
       model weights in HBM
       active decode in HBM
       hot KV in HBM
       DeepSeek-V4-Flash or another coding/agentic LLM

CPU agent-control hosts
  -> PCIe Gen5 / CXL optical links
  -> central CXL memory appliance
       FPGA Type 3 endpoint prototype
       commodity DDR5 pool
       partitioned/reassigned regions
       ASIC later if the workload validates
```

## Plane Separation

### GPU Inference Plane

The GPU plane serves the LLM. It should keep only the latency-critical model
state close to the GPU:

- model weights
- active decode state
- hot KV
- short-lived batch scheduling state

The GPU plane should not become the storage layer for every agent's warm repo
and workflow history.

### CPU Agent-Control Plane

The CPU plane runs the stateful coding-agent loop:

- planning
- repo indexing
- retrieval
- tool execution
- test execution
- prompt assembly
- session resume
- context routing

This is where ContextPilot-style state naturally lives. The CPU plane can use
local DRAM for hot state and CXL pooled DDR5 for warm shared state.

### Optical CXL Memory Plane

The CXL memory appliance exposes centrally pooled DDR5 as Type 3 memory to one
or more Gen5 CXL hosts. In the first prototype, an FPGA endpoint is enough. The
goal is not to prove an ASIC immediately; the goal is to prove that the workload
benefits from pooled, persistent, reassigned warm memory.

## Why This Workload Is Strong

Coding agents are a better first demo than generic LLM inference because the
state is large, persistent, reusable, and not all latency-critical.

Baseline servers overprovision local DRAM so each host can keep enough repo and
session state warm. In a rack with many active agents, that duplicates memory
across hosts and loses state when sessions move.

CXL pooled DDR5 can improve the system by:

- reducing local DRAM per host
- increasing active coding-agent sessions per rack
- reducing project resume time
- preserving warm repo/index/session state across workers
- allowing session reassignment without full recomputation
- keeping GPU utilization higher by reducing CPU-side cold starts

## Demo Target

Run a ContextPilot-style coding workload on CPU workers that call GPU inference
below. Compare two modes:

1. Local DRAM only
2. Local DRAM plus CXL pooled DDR5

The first demo does not need optical links or an ASIC. Start with:

```text
one Gen5 CXL-capable host
  -> CXL-enabled FPGA Type 3 endpoint
  -> FPGA-attached commodity DDR5
```

Then extend to:

```text
multiple CPU hosts
  -> optical Gen5 CXL links
  -> central memory appliance
  -> partitioned/reassigned pooled DDR5 regions
```

## Cloud Memory Pooling

Before buying CXL hardware, run a cloud software proof that emulates the future
pooled warm-memory tier.

```text
many ContextPilot agent workers
  -> local L1 cache in each VM DRAM
  -> shared L2 memory pool
       Redis / Valkey / Tair / Dragonfly / Memcached / Ray object store
  -> cold tier
       S3 / OSS / EBS / NVMe
  -> model API or GPU inference backend
```

This does not prove CXL latency or CXL enumeration. It proves the workload
thesis:

> Do fleets of coding agents working on shared repos reuse enough warm
> project/session state to justify a hardware pooled-memory tier?

Cloud Memory Pooling is useful because it is cheap, fast to deploy, and
provider-agnostic. AWS, Alibaba Cloud, or another cloud can supply
memory-optimized CPU workers plus a managed or self-hosted shared in-memory
service. The shared pool stores repo indexes, symbol graphs, file snapshots,
test logs, tool traces, prior patch candidates, session state, context plans,
and prompt/prefix metadata.

Compare:

1. isolated workers with only local VM memory
2. workers with local L1 memory plus shared cloud L2 memory

If the shared L2 memory layer does not improve resume time, duplicate indexing,
context assembly, token reuse, or active-agent density, CXL hardware is unlikely
to be a strong product. If it does, the CXL DDR5 appliance becomes a lower
latency, lower overhead physical implementation of the same memory-pooling
concept.

## Metrics

Track system value, not only token value:

- cost per active coding agent/session
- project resume time
- context assembly latency before LLM calls
- cache hit rate for repo/index/session state
- concurrent active agents per CPU/GPU rack
- local DRAM saved per host
- GPU utilization improvement from keeping CPU-side agent state warm
- recomputation avoided for repo indexes and long-lived session state
- session migration time between CPU workers

## MVP Implementation Path

### Phase 1: Software-Only Memory Tier Simulation

Add a storage tier abstraction around ContextPilot state:

- local in-process memory
- local disk/SQLite baseline
- simulated pooled memory tier

Measure object sizes, access frequency, reuse distance, and resume latency.
This produces the memory working-set profile before hardware exists.

### Phase 2: Cloud Memory Pooling PoC

Deploy several ContextPilot workers on cloud CPU instances. Add a shared memory
pool using Redis, Valkey, Tair, Dragonfly, Memcached, or a Ray object store.
Measure shared-state hit rate, duplicate recomputation avoided, resume time,
context assembly latency, input tokens avoided, and active agents per worker.

This is the first go/no-go gate for the CXL strategy.

### Phase 3: Commercial CXL DDR5 Expander Demo

Use one CXL-capable Gen5 server and a commercial CXL Type 3 DDR5 memory
expander. Map selected warm ContextPilot state into the CXL memory region and
compare against local DRAM-only and cloud-memory-pool operation.

This proves whether the workload benefits from a real CXL memory tier before
building custom FPGA hardware.

### Phase 4: Single-Host CXL FPGA Demo

Use one CXL-capable Gen5 server and an FPGA Type 3 endpoint with attached DDR5.
Map selected warm ContextPilot state into the CXL memory region and compare
against local DRAM-only operation.

### Phase 5: Optical Link Demo

Insert an optical Gen5/CXL link between the host and the memory appliance.
Re-run the same workload while tracking latency sensitivity and bandwidth use.

### Phase 6: Multi-Host Pooled Memory Demo

Attach multiple CPU hosts. Partition or reassign pooled DDR5 regions across
hosts. Demonstrate session resume or migration without rebuilding repo indexes
from scratch.

## Success Criteria

Continue if pooled DDR5 shows:

- lower local DRAM requirement per CPU host
- more active agent sessions per rack
- faster project/session resume
- stable context assembly latency
- reduced recomputation of repo/session artifacts
- measurable GPU utilization improvement from fewer CPU-side stalls

Narrow or stop if:

- CXL latency significantly hurts context assembly
- warm state is too small or too cold to matter
- local NVMe/DRAM caching is already good enough
- operational complexity exceeds the memory savings

## Bottom Line

The initial ContextPilot project reduces tokens. The larger infrastructure story
is that stateful coding agents create a new warm-memory workload.

That workload can be used to demonstrate a CXL pooled-DDR5 appliance before a
custom ASIC exists.
