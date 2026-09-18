# Autonomous AutoQuant Harness

**Status:** Approved  
**Scope:** Turn AutoQuant from a JSON decision generator into an autonomous, evidence-gated quantization run.

## Goal

`orka autoquant` autonomously chooses and evaluates quantization decisions using an LLM through Gratis, then emits a verified `.orka` artifact, compatible allocation data, and an audit report.

The harness can decide which internal analysis and refinement tools to use. It cannot publish an artifact unless deterministic validation and measured quality gates succeed.

## Configuration

Every run requires:

- Gratis OpenAI-compatible endpoint and model alias
- OpenRouter-backed provider route
- Maximum LLM spend
- Maximum tool steps and wall-clock duration
- Quantization objective and quality target
- Output location

Gratis is the only LLM transport. Unavailability, invalid responses, or budget exhaustion fail the run. The harness must not silently fall back to deterministic-only AutoQuant.

## Architecture

| Component | Responsibility |
|---|---|
| `RunConfig` | Validated immutable per-run settings and resource ceilings. |
| `HarnessState` | Persistent run status, candidate state, spend, step count, evidence, and terminal failure reason. |
| `GratisClient` | Structured LLM requests, response parsing, usage accounting, and provider failure normalization. |
| `ToolRegistry` | Typed inspect, probe, allocate, pack, pulse-check, refine, verify, and report actions. |
| `Reasoner` | Selects the next allowed tool based on the accumulated evidence. |
| `Validator` | Applies schema, resource, safety, allocation-compatibility, and artifact-integrity checks. |
| `AcceptanceGate` | Permits success only after measured objective success and artifact verification. |
| `TraceWriter` | Writes an audit trace with decisions, tool arguments/results, usage, and outcomes. |

```text
RunConfig
  → HarnessState
  → Gratis Reasoner
  → Typed ToolRegistry action
  → Validator + TraceWriter + resource checks
  → Reasoner refinement loop
  → AcceptanceGate
  → allocation map + verified .orka + audit report
```

## Decision Protocol

1. Inspect the checkpoint and construct structural architecture evidence.
2. Probe candidate tensors and obtain deterministic allocation candidates.
3. Ask the reasoner to select the next tool or candidate, using structured output only.
4. Validate every response before execution.
5. Pack and run quality checks.
6. Give measured results to the reasoner for further refinement until accepted or a configured limit is reached.
7. Verify the artifact and write the report.

The reasoner may propose candidate allocations and refinement actions only through the registered tools. It has no arbitrary filesystem, shell, network, or artifact-publication access.

## Safety and Compatibility

- Output heads must not use RVQ.
- Norms and biases remain FP16.
- Recurrent safety constraints remain enforced.
- AutoQuant output must be converted to the allocation contract consumed by the packer.
- Every configuration validates supported methods, bits, stages, normalization, and dense exceptions.
- A target below the allocation engine's feasible minimum is rejected before expensive work begins.
- Artifact publication requires format verification and measured objective success.

Structural classification must distinguish output heads, input embeddings, tied tensors, and recurrent blocks. It must combine `ArchProfile` evidence with AutoQuant role logic rather than replacing one with the other.

## Failure Model

The run enters a terminal failed state for:

- Gratis unavailability, malformed response, or exhausted configured spend
- Exceeded tool-step or time limit
- Invalid reasoning action or unsafe candidate
- Infeasible allocation target
- Pack, pulse-check, refinement, or artifact-verification failure
- Failure to reach the requested quality objective

Every failure records its stage, evidence, and normalized reason in the trace and report.

## Testing

- Unit tests for configuration, state transitions, Gratis response validation, budget accounting, tool permissions, safety checks, and allocation conversion.
- Deterministic fake-Gratis integration tests for successful refinement, unsafe proposal rejection, invalid response, budget exhaustion, and step exhaustion.
- End-to-end small-checkpoint test that produces and verifies an artifact.
- Model-gated integration test that validates measured quality and the final audit report.

## Non-goals

- Direct LLM-provider access outside Gratis
- Silent deterministic fallback after an LLM failure
- Unbounded agent loops
- Arbitrary model-authored shell or filesystem actions
- Multi-GPU orchestration and a UI
