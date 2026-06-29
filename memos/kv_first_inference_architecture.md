# KV-First Inference Architecture

## Breakpoint

We are explicitly setting aside:

- BitNet as the primary research vehicle
- hard-coded / fixed-weight ASICs as the primary thesis
- ternary weights as the main source of advantage

The reason is simple: for useful context lengths and many simultaneous users,
the dominant bottleneck is not weight movement. It is runtime state.

The new thesis:

> KV cache is the product. The model is constrained by runtime memory. The
> hardware value is in reducing, organizing, and serving KV state.

## Why The Prior Direction Stalled

Hard-coded weights reduce or remove one memory stream:

```text
fixed model-weight reads
```

But they do not remove the dynamic memory stream:

```text
KV cache generated from each user's prompt and generated tokens
```

KV cache cannot be hard-coded because it depends on runtime inputs. For
transformer decode, each generated token must access prior context state. Even
when weights become cheap, KV cache remains:

- capacity-bound
- bandwidth-bound
- energy-bound
- scheduling-bound

So fixed-weight inference only wins cleanly in very short-context workloads or
low-quality-tolerant use cases. For broader usefulness, the architecture must
attack runtime memory directly.

## Core Problem

For a transformer with:

```text
L layers
H_kv KV heads
D head dimension
B bytes per KV element
```

KV bytes per token are:

```text
2 * L * H_kv * D * B
```

The factor of 2 is for K and V.

Per-user KV grows linearly with context length:

```text
KV_user = context_tokens * KV_bytes_per_token
```

During decode, vanilla attention repeatedly reads the relevant prior KV state.
This makes per-token generation increasingly dominated by memory access rather
than arithmetic.

## Design Goal

The target is not simply a faster matrix engine.

The target is an inference architecture that reduces:

- KV bytes per token
- KV reads per generated token
- KV energy per generated token
- KV capacity per active user
- scheduling overhead for many users

while preserving enough task quality for commercially useful workloads.

## Architecture Directions

### 1. Latent KV / MLA-Style Attention

Store a compressed latent representation instead of full K/V vectors.

Potential benefit:

- reduces KV capacity
- reduces KV bandwidth
- keeps attention relatively dense
- hardware-friendly if latent dimension is fixed

Risk:

- likely requires training or deep continued pretraining
- quality depends heavily on compression ratio
- architecture must be co-designed with the model

### 2. Local Window + Compressed Global Memory

Keep recent tokens exactly and compress older context into summaries, blocks, or
retrievable memory.

Potential benefit:

- preserves local coherence
- reduces reads over older context
- maps naturally to a memory hierarchy

Risk:

- long-range recall may degrade
- summary quality is task-dependent
- requires careful training or at least workload-specific validation

### 3. N-Gram / Block-Compressed KV

Represent recurring token patterns, blocks, or spans with compact shared state.

Potential benefit:

- can exploit redundancy in prompts and conversations
- may reduce both capacity and bandwidth
- could be useful for enterprise/support workloads with repeated templates

Risk:

- retrieval/selection logic may be complex
- compression can be brittle without training
- quality can collapse on precise recall tasks

### 4. KV Quantization

Store KV in fewer bits.

Potential benefit:

- easiest to test
- immediate capacity/bandwidth reduction
- may work for modest compression

Risk:

- quality degradation stacks with other approximations
- not enough by itself for large reductions
- aggressive int4/int2 KV is likely fragile

### 5. KV Memory Hierarchy

Treat KV as a first-class memory-system workload:

```text
on-chip SRAM:
  hot window
  active decode state
  tile buffers

near memory:
  current session KV
  compressed global context

pooled / optical / fabric memory:
  many-user KV
  cold context
  cross-request state
```

Potential benefit:

- reduces expensive memory movement
- improves concurrency
- allows workload-aware scheduling

Risk:

- does not reduce KV by itself
- latency matters because decode is serial
- requires scheduler and memory co-design

## Research Questions

1. How many bytes of runtime state per token are actually needed?
2. How much can KV be compressed before task quality breaks?
3. Which workloads tolerate approximate or compressed memory?
4. Can local exact context plus compressed global memory preserve enough quality?
5. What is the best hardware hierarchy for hot/warm/cold KV?
6. How much does many-user scheduling improve locality and utilization?
7. Which compression methods require full pretraining versus light adaptation?

## Recommended Next Experiments

### Experiment 1: KV Arithmetic Baseline

Build a parameterized simulator for:

- dense attention
- MLA-style latent KV
- local-window attention
- compressed global memory
- KV quantization
- many-user concurrency

Output:

- KV capacity per user
- bandwidth per generated token
- energy per generated token
- TTFT estimate
- decode throughput bound

### Experiment 2: Inference-Only Stress Tests

Use existing open-weight models to test:

- sliding window
- attention sinks + recent window
- KV quantization
- block eviction
- block pooling
- retrieval-style prompt compression

Purpose:

> Measure how quickly quality degrades when runtime state is constrained.

This will not prove the final architecture, but it reveals the quality cliff.

### Experiment 3: Tiny Trained Prototype

Train a small model, not a 7B model:

```text
100M-300M parameters
local exact window
compressed global memory
short/medium context
targeted evals
```

Purpose:

> Determine whether a model can learn to use compressed runtime memory.

### Experiment 4: Hardware Memory PPA

Using real process data, model:

- SRAM area/energy
- NoC/router energy
- near-memory bandwidth
- off-chip memory energy
- optical or pooled memory if relevant
- scheduling overhead

Purpose:

> Determine whether compressed-KV inference has a meaningful hardware advantage.

## Decision Criteria

Continue only if at least one architecture shows:

- large KV memory reduction, ideally 4x-16x
- acceptable quality on target tasks
- lower joules/token than GPU inference
- lower TTFT for short and medium prompts
- clean scaling to many simultaneous users

Stop or narrow if:

- quality collapses below 4x KV reduction
- memory energy still dominates after compression
- training requirements become equivalent to frontier-model R&D
- hardware advantage depends on unrealistic bandwidth or utilization

## Current Conclusion

The main opportunity is not fixed weights.

The main opportunity is:

```text
compressed runtime memory
+ KV-aware model architecture
+ KV-aware memory hierarchy
+ many-user scheduling
```

Fixed low-bit weights may return later as an optimization. They are not the
foundation.

