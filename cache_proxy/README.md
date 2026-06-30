# Adaptive-Router Context Proxy

This is a small OpenAI-compatible proxy that sits above model APIs such as
DeepSeek, OpenAI, or another OpenAI-compatible endpoint.

Its job is not to implement KV caching. Model providers already do that. Its job
is to make prompts more cacheable by putting stable repo/context blocks before
volatile task-specific text, reduce unnecessary context with the adaptive
router, and normalize provider cache telemetry.

## Strategic Direction: Stateful Agentic AI Memory

This project is also a workload and demo vehicle for an optically disaggregated
commodity DDR5 memory pool for Gen5 CXL servers running stateful agentic AI.

The near-term middleware reduces token cost by improving context routing. The
larger infrastructure direction is to use ContextPilot-style coding agents as a
memory-system workload:

- GPU inference plane: H100/H20-class GPUs serve DeepSeek-V4-Flash or another
  coding/agentic LLM, keeping model weights, active decode, and hot KV in HBM.
- CPU agent-control plane: Gen5 CXL-capable hosts run planning, repo indexing,
  retrieval, tool orchestration, terminal/test execution, workflow state,
  session state, and prompt assembly.
- Optical CXL fabric: CPU hosts connect over PCIe Gen5/CXL optical links to a
  central memory appliance.
- Central memory appliance: a CXL Type 3 FPGA prototype exposes pooled
  commodity DDR5 now; an ASIC can follow later if the workload validates.

Coding agents carry large warm state: repo indexes, AST/symbol graphs, file
snapshots, terminal/test logs, search results, tool traces, agent plans,
prompt/prefix caches, prior patch candidates, and long-lived session history.
This state is expensive to recompute and bulky to duplicate per CPU node, but it
does not need HBM latency. Pooled DDR5 can reduce local DRAM overprovisioning
and increase active coding-agent sessions per CPU/GPU rack.

Demo target:

```text
ContextPilot-style CPU workers
  -> GPU inference API below
  -> local DRAM only baseline
  -> compare with CXL pooled-DDR5 warm-state tier
```

Primary metrics:

- cost per active coding agent/session
- project resume time
- context assembly latency before LLM calls
- cache hit rate for repo/index/session state
- concurrent active agents per CPU/GPU rack
- local DRAM saved per host
- GPU utilization improvement from keeping CPU-side agent state warm

See `memos/cxl_agentic_memory_pool.md` for the full architecture and prototype
plan.

## MVP Flow

```text
IDE / coding agent
  -> POST /v1/repos/{repo_id}/context
  -> POST /v1/chat/completions
  -> proxy canonicalizes context
  -> model API
  -> proxy logs cache telemetry
```

## Run

```bash
export CACHE_PROXY_PROVIDER=deepseek
export DEEPSEEK_API_KEY=...
.venv/bin/python -m cache_proxy.server
```

Provider options:

```bash
# DeepSeek
export CACHE_PROXY_PROVIDER=deepseek
export DEEPSEEK_API_KEY=...

# OpenAI
export CACHE_PROXY_PROVIDER=openai
export OPENAI_API_KEY=...

# Any OpenAI-compatible endpoint
export CACHE_PROXY_PROVIDER=openai-compatible
export CACHE_PROXY_BASE_URL=https://your-provider.example.com
export CACHE_PROXY_API_KEY=...
```

The proxy forwards to `/v1/chat/completions`. OpenAI-style prompt cache
telemetry is normalized from `usage.prompt_tokens_details.cached_tokens`;
DeepSeek-style telemetry is normalized from `prompt_cache_hit_tokens` and
`prompt_cache_miss_tokens`.

Then call:

```bash
curl http://127.0.0.1:8000/health
```

If this sandbox blocks local port binding, run it on a normal shell outside the
Codex sandbox with the same command.

## Deploy On Render

This repo is Render-ready with `render.yaml` and root `requirements.txt`.

1. Push this folder to a GitHub or GitLab repository.
2. In Render, choose **New** -> **Blueprint**.
3. Select the repo.
4. Render will read `render.yaml` and create the web service.

The service uses:

```bash
uvicorn cache_proxy.server:app --host 0.0.0.0 --port $PORT
```

The public website and demo planning endpoint work without provider API keys.
For real model forwarding, add one of these env var sets in Render:

```bash
# DeepSeek
CACHE_PROXY_PROVIDER=deepseek
DEEPSEEK_API_KEY=...

# OpenAI
CACHE_PROXY_PROVIDER=openai
OPENAI_API_KEY=...

# Any OpenAI-compatible provider
CACHE_PROXY_PROVIDER=openai-compatible
CACHE_PROXY_BASE_URL=https://your-provider.example.com
CACHE_PROXY_API_KEY=...
```

If real model keys are configured, also set `CACHE_PROXY_PUBLIC_TOKEN` so
protected API endpoints are not open to the public internet.

## Chat Request Shape

The proxy accepts ordinary OpenAI-compatible chat requests. It also accepts an
optional `cache_context` object:

```json
{
  "model": "deepseek-v4-flash",
  "messages": [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Fix the failing test."}
  ],
  "cache_context": {
    "repo_id": "my-repo",
    "commit_hash": "abc123",
    "selected_blocks": ["README.md", "src/app.py"],
    "context_budget": {
      "enabled": true,
      "profile": "auto",
      "max_repo_chars": 24000,
      "max_full_files": 3,
      "include_repo_map": true
    }
  }
}
```

For OpenAI, use an OpenAI chat-completions model in the same shape, for example
`"model": "gpt-4.1"` or another supported chat model.

The proxy will build:

```text
stable prefix:
  original system messages
  repo metadata
  selected repo blocks in canonical order

dynamic suffix:
  user messages
  volatile request-specific data
```

## Context Planning

By default, ContextPilot uses a repo-aware planner instead of sending all stored
blocks. A lightweight classifier first chooses a planning profile from the task
and agent artifacts, then the planner emits a stable repository map for
orientation plus a small set of relevant full file bodies.

Useful knobs live under `cache_context.context_budget`:

```json
{
  "enabled": true,
  "profile": "auto",
  "max_repo_chars": 24000,
  "max_full_files": 3,
  "include_repo_map": true
}
```

Supported profiles are `auto`, `focused`, `debug`, `review`, `refactor`,
`docs`, `handoff`, and `general`. Omit `profile` or set it to `auto` for the
classifier. Explicit `max_repo_chars`, `max_full_files`, and `include_repo_map`
values override the classifier policy.

Forwarded requests include `metadata.cache_proxy.context_planning`:

```json
{
  "enabled": true,
  "policy_profile": "debug",
  "policy_reasons": ["test or failure artifact"],
  "repo_map_blocks": 5,
  "full_blocks": 2,
  "selected_paths": ["cache_proxy/server.py", "cache_proxy/context_planner.py"],
  "repo_context_chars": 18320
}
```

## Telemetry

DeepSeek responses include cache-hit accounting in `usage`:

```json
{
  "prompt_cache_hit_tokens": 1234,
  "prompt_cache_miss_tokens": 456
}
```

This proxy logs those values, plus latency and estimated hit rate.

## Benchmark

After starting the server:

```bash
.venv/bin/python -m cache_proxy.benchmark_deepseek --rounds 3
```

The second and third rounds should show whether DeepSeek is producing more
`prompt_cache_hit_tokens` for the stable repo prefix.

## Direct A/B Test

This does not require running the local proxy server. It calls DeepSeek directly
with two prompt layouts:

- `naive`: task-specific text before stable repo context
- `cache-aware`: stable repo context before task-specific text

```bash
export DEEPSEEK_API_KEY=...
.venv/bin/python -m cache_proxy.ab_test_deepseek --model deepseek-v4-flash --rounds 4
```

The script uses the official V4 Flash prices as of May 30, 2026:

- cache-hit input: $0.0028 / 1M tokens
- cache-miss input: $0.14 / 1M tokens
- output: $0.28 / 1M tokens

It prints per-request cache-hit telemetry and estimated cost for each arm.

## Messy Multi-Agent Benchmark

This benchmark simulates multiple coding agents working on the same repo while
repeating tool schemas, test logs, policies, and volatile session metadata in
different orders. The DeepSeek-native arm intentionally looks like practical
IDE traffic: task/session artifacts appear before the repo context, and wrappers
vary by agent. It reports:

- DeepSeek V4 Flash API with no internal caching, as a billing counterfactual
- DeepSeek V4 Flash API with internal caching
- upstream ContextPilot from `vendor/ContextPilot`
- fixed-general ContextPilot, our base planner with adaptation disabled
- budget-shadow-pruned upstream, which searches several conservative document
  and excerpt budgets and chooses the cheapest prefix-reuse candidate online
- adaptive ContextPilot, our base planner with classifier-selected profiles
- adaptive-router ContextPilot, which routes each request to upstream,
  fixed/adaptive, or budget-shadow based on workload bucket and confidence

Adaptive ContextPilot is kept as a baseline and fallback. The experimental
adaptive-budget-shadow ablation is available with `--include-adaptive-hybrid`,
but it is not part of the default comparison because it did not beat
budget-shadow pruning in the current blended workload.

Local dry run:

```bash
.venv/bin/python -m cache_proxy.benchmark_messy_agents --rounds 8
```

Large-repo pressure test:

```bash
.venv/bin/python -m cache_proxy.benchmark_messy_agents --corpus large --synthetic-file-count 60 --rounds 10
```

Hard required-file recall test:

```bash
.venv/bin/python -m cache_proxy.benchmark_messy_agents --corpus large --workload synthetic-targeted --synthetic-file-count 60 --rounds 10
```

Blended workload test:

```bash
.venv/bin/python -m cache_proxy.benchmark_messy_agents --corpus large --workload mixed-plus-targeted --synthetic-file-count 60 --rounds 20
```

The large corpus keeps the real repo files and adds deterministic synthetic
source/docs files. This makes upstream ContextPilot's all-doc prompt expensive
enough to test whether repo-aware pruning can beat stable full-prefix caching.
The targeted workloads add required-file recall metrics. Budget-shadow pruning
must preserve explicitly mentioned files before it is counted as a valid cost
improvement.

### Current Offline Result

Best current arm:

```text
upstream ContextPilot ordering + required-path-safe budget-shadow pruning
```

On the 20-round blended large-repo workload, the local estimator reports:

| Arm | Estimated cost | Required-file recall |
| --- | ---: | ---: |
| DeepSeek V4 Flash native, no caching | $0.26154352 | 10/10 |
| DeepSeek native internal cache | $0.14744087 | 10/10 |
| Upstream ContextPilot | $0.01869307 | 10/10 |
| Fixed-general ContextPilot | $0.00865934 | 10/10 |
| Adaptive ContextPilot | $0.00650052 | 10/10 |
| Adaptive-router ContextPilot | $0.00398779 | 10/10 |
| Budget-shadow upstream | $0.00236122 | 10/10 |

Budget-shadow is the aggressive upper-bound arm. Relative to the no-cache
DeepSeek V4 Flash baseline, it is about 99.1% cheaper in this offline workload
while preserving all explicitly required files. Relative to upstream
ContextPilot, it is about 87.4% cheaper.

Adaptive-router ContextPilot is the more realistic product arm. It is about
98.5% cheaper than the no-cache baseline, 78.7% cheaper than upstream
ContextPilot, and 38.7% cheaper than the earlier adaptive planner. In this
20-round run it chose adaptive planning for 8 broad tasks, fixed-general for 2
refactor tasks, and budget-shadow for the 10 explicit-path tasks.

Important caveat: an earlier budget-shadow variant looked much better but failed
required-file recall on targeted tasks. The current version treats explicitly
mentioned repo paths as non-prunable before counting savings as valid.

Product thesis:

```text
Do not replace ContextPilot. Add a required-path-safe budget controller on top
of ContextPilot's ordering.
```

Bottom line: ContextPilot already provides the valuable stable ordering layer.
The near-term opportunity is to add a lightweight, ContextPilot-compatible
adaptive budget controller for coding agents. It should classify each request
by workload bucket, preserve explicitly mentioned repo paths as non-prunable
guardrails, search conservative context and extractive file-slice budgets when
confidence is high, and fall back to broader ContextPilot plans when uncertainty
is higher.

### Near-Term Opportunity

Build a lightweight ContextPilot-compatible budget controller for coding-agent
workloads. The controller should:

- reuse upstream ContextPilot ordering as the stable context backbone
- classify each request into workload buckets such as explicit-path,
  test-debug, refactor, docs, handoff, review, and unknown
- search conservative context and excerpt budgets before each request
- preserve explicitly mentioned repo paths as non-prunable guardrails
- choose the cheapest cache-aware prompt shape that still preserves required
  context
- report savings by workload bucket, not only aggregate average
- report cost, cache-hit estimates, and required-file recall for every run

This keeps the MVP focused on prompt planning above existing inference APIs,
without requiring changes to the underlying model or provider cache layer.

Live DeepSeek run:

```bash
export DEEPSEEK_API_KEY=...
.venv/bin/python -m cache_proxy.benchmark_messy_agents --corpus large --rounds 8 --live
```

The dry run estimates prompt shape and prefix reuse. The live run is the real
measurement because provider-side cache-hit accounting determines the actual
billing impact.

## Local Quality Benchmark

The quality benchmark compares upstream ContextPilot against adaptive-router
ContextPilot on the same tasks. It is deterministic and local: it scores whether
the prompt contains the hidden/expected full-context files and grounding terms
needed to answer the task. This is a context-availability proxy, not a live LLM
answer-quality score.

```bash
.venv/bin/python -m cache_proxy.benchmark_quality --rounds 8
```

Acceptance targets:

- cost reduction vs upstream ContextPilot: at least 2x
- quality loss vs upstream ContextPilot: below 5%
- severe failures: zero or near-zero
- hidden-file recall: at least 98%

Current local result:

| Metric | Result |
| --- | ---: |
| Upstream ContextPilot cost | $0.01624001 |
| Adaptive-router ContextPilot cost | $0.00453821 |
| Cost reduction | 3.58x |
| Quality loss | 0.0% |
| Hidden-file recall | 100% |
| Severe failure rate | 0.0% |
| Acceptance gate | pass |

The adaptive router reaches this by adding inferred-context guardrails: for
vague tasks, it infers likely source files from symbols, paths, and task terms,
then requires candidate prompts to include those files before choosing the
cheapest valid plan.

## Adaptive-Router HTTP Service

The proxy exposes the adaptive router as a dry-run planning API. This lets an
IDE or agent inspect the chosen context plan before calling an LLM:

```bash
curl -s http://127.0.0.1:8000/v1/adaptive-router/plan \
  -H 'Content-Type: application/json' \
  -d '{
    "repo_id": "demo",
    "commit_hash": "abc123",
    "messages": [
      {"role": "system", "content": "You are a coding assistant."},
      {"role": "user", "content": "Review chat_completions and identify one cache risk."}
    ]
  }'
```

The response includes:

- compiled `messages`
- router bucket, confidence, selected strategy, inferred guard paths, and
  selected files
- local token/cache usage estimate
- prompt payload delta

To use the same adaptive-router plan in the chat proxy, include:

```json
{
  "cache_context": {
    "repo_id": "demo",
    "commit_hash": "abc123",
    "context_budget": {
      "enabled": true,
      "mode": "adaptive-router"
    }
  }
}
```
