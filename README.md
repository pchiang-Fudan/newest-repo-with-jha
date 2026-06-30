# Middle-Out Token Compression

Adaptive ML token-routing and context planning for stateful coding agents.

This repo contains a ContextPilot-style proxy and benchmark suite for reducing
token cost in agentic coding workflows. The current software layer sits above
model APIs, orders/reduces repo context, and measures cache/cost behavior.

## Strategic Direction

The project is now also a workload and demo vehicle for:

> Optically disaggregated commodity DDR5 memory pool for Gen5 CXL servers
> running stateful agentic AI.

The idea is to separate the system into:

- GPU inference plane: H100/H20-class GPUs serve the LLM and keep model weights,
  active decode, and hot KV in HBM.
- CPU agent-control plane: Gen5 CXL-capable hosts run planning, repo indexing,
  retrieval, tool orchestration, terminal/test execution, workflow/session
  state, and prompt assembly.
- Optical CXL memory plane: CPU hosts connect to a central CXL Type 3 memory
  appliance exposing pooled commodity DDR5.

Coding agents carry large warm state that is expensive to recompute but does
not need HBM latency: repo indexes, AST/symbol graphs, file snapshots, terminal
logs, test logs, search results, tool traces, agent plans, prompt/prefix caches,
prior patch candidates, and session history.

See:

- `cache_proxy/README.md`
- `memos/cxl_agentic_memory_pool.md`
- `memos/kv_first_inference_architecture.md`
