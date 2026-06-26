# Autoquant — arch-agnostic auto-config-derivation for orka

**Status:** Design approved 2026-06-26
**Owner:** Kailas
**Scope:** One implementation cycle (spec → plan → build)

## Problem

Choosing a quantization config (per-tensor bits, method, normalization, what to keep
fp16) is expert, manual, and arch-specific. Done wrong it is catastrophic - e.g. RVQ on
the output head took pythia-160m from ppl 42 to 1.2M while int8 on the same head was
lossless. We want to pass **any model of any architecture** and have a controller derive
the correct config automatically, reasoning step by step over cheap measurements.

## Goal

`orka autoquant <model> --objective {min-bits|max-quality|knee}` → a validated
`allocation_map.json` + packed `.orka` artifact + an audit report, with no human picking
bits or methods.

Non-goals (v1): training/QAT, new quant primitives, multi-GPU orchestration, a UI.

## Approach

A **hybrid brain**: a deterministic policy core decides most tensors; an LLM (pi-reason
generate→verify→refine pattern) decides the hard calls. The controller **orchestrates
existing orka tools** (`allocate`, `pack --allocation-map`, `pulse-check`) - it is a brain
on top, not a re-implementation of quant. Provider-agnostic LLM transport (litellm), a
role-based router (lite model for easy tensors, strong for stubborn ones), and a global
signature-cache make it reproducible and cheap.

Chosen over: depending on the siphon repo as a library (couples orka to a separate
product) and pi-ai direct (TS/Node, friction with Python orka). We **port the pattern**
(pi-reason reasoning + siphon's router/tool-dispatch shape) into a standalone Python module.

## Module layout

```
orka/autoquant/
  roles.py        arch-agnostic tensor role classifier
  probes.py       cheap per-tensor signals
  policy.py       deterministic config core
  harness.py      pi-reason generate→verify→refine LLM loop
  escalation.py   the 4 triggers + signature cache
  orchestrator.py the chain + per-objective stopping rules
  schema.py       allocation_map + report data contracts
  priors.py       seeded knowledge
```

Each module is independently testable: `roles/probes/policy` are pure functions of their
inputs; `harness/escalation/orchestrator` are the I/O-bound parts.

## Data flow

```
[1] INTROSPECT  roles.py     any arch -> role per tensor (+confidence)
[2] PROBE       probes.py    RD curve, SQNR@bpw, sensitivity, fragility prior
[3] POLICY      policy.py    signals+priors -> config[tensor] (+confidence)
[4] ESCALATE    escalation   triggered tensors -> harness -> override config
[5] ALLOCATE    allocate.py  water-fill remaining freedom to budget (if objective needs it)
[6] PACK+CHECK  pack -> pulse-check (KL + top1, 32 prompts, seconds)
[7] REFINE      attribute regression to tensors -> escalate offenders -> bump -> back to [6]
STOP per objective:
  min-bits   : KL ≤ target & cannot drop bits without breaking it
  max-quality: bpw budget hit, quality maximized
  knee       : marginal KL-improvement-per-bit < ε
OUTPUT: allocation_map.json + .orka + autoquant-report.md
```

Inner loop = refine (6↔7), gated by cheap pulse-check; escalate to a few-chunk ppl only at
final accept. Regression attribution ranks suspect tensors via per-tensor SQNR +
leave-one-fp16 deltas, then escalates only those.

## Components

### roles.py — role classifier (arch-agnostic)
Layered: (1) name patterns, extending `classify_tensor_family` but **splitting output-head
from input-embed** (tie-detection + graph position) and resolving q/k/v/o, up/down/gate,
norm, bias; (2) shape/position heuristics for unknown names (`[vocab,hidden]` at edge =
embed/head; `[3*hidden,hidden]` = fused qkv); (3) low confidence → escalate (trigger #4).
Output `{tensor: (role, confidence)}`. Pure function of checkpoint meta.

### policy.py — deterministic core
Table-driven from `priors.py`:
```
out-head        → int8, never RVQ            (conf 1.0)
norm, bias      → keep fp16                  (conf 1.0)
in-embed        → RVQ, bits from RD knee     (conf 0.8)
attn.v/mlp.down → +1 stage vs siblings (high sensitivity)
default         → RVQ, bits = RD point hitting SQNR ≥ 30 dB
```
Emits `config[tensor] + confidence`. Confidence < threshold OR probe-conflict → escalate.

### harness.py — pi-reason loop (LLM brain)
Per escalated tensor:
```
in:  {role, shape, signals, siblings' configs, objective}
loop: generate(config) → verify(predicted ΔKL vs policy baseline) → refine
out: {method, bits, stages, norm, keep_fp16, rationale}
```
litellm transport, role-router (lite default, strong for stubborn). Verdict keyed by
**signature** = `(role, shape-bucket, sensitivity-bucket, objective)`.

### escalation.py — triggers + cache
Four triggers (all active in v1): low policy confidence | probe conflict | refine-failure
| unknown role. **Global** signature-cache at `~/.orka/autoquant-cache.json` → grows into a
learned policy reusable across all models; cache hit skips the LLM call.

### schema.py — data contract (consumed by `pack --allocation-map`)
```json
{ "tensor_name": {"method":"rvq|int8|fp16","bits":3,"stages":2,
                  "normalization":"block-max","keep_fp16":false,
                  "source":"policy|llm|cache","confidence":0.9,"rationale":"..."} }
```
`source` + `rationale` per tensor = audit trail; the report is generated from this.

## Error handling & safety

- LLM unreachable / no key → policy default + warn; never blocks a pack.
- LLM returns invalid config → schema-reject, log, fall back to policy for that tensor.
- Refine non-convergence → hard cap (5 rounds); accept best-so-far + flag in report.
- Unknown role + low confidence + no LLM → **keep fp16** (never silently low-bit an
  un-understood tensor - the lm_head lesson).
- Pulse-check below floor at min bits → report failure honestly, emit last-good config.

## Reproducibility

- Global signature-cache → same tensor-class → same verdict, deterministic once warm.
- `--no-llm` forces pure-policy (deterministic, for CI).
- Every decision carries `source` + `rationale`.

## Testing

- `roles.py`: fixtures across gptneox / llama-qwen / gpt2 → asserted role maps (the
  arch-agnostic claim is tested).
- `policy.py`: table-driven (role+signals → config); includes this session's regressions
  (lm_head→int8, head-RVQ→rejected).
- `harness.py`: mocked LLM (canned verdicts) → loop logic without network.
- `escalation.py`: each of the 4 triggers fires on a crafted input.
- `orchestrator.py`: end-to-end on pythia-160m → must derive the int8-head config found by
  hand + converge. **Integration gate.**
- Structural oracle `d73e0b19fc38f099` stays green (autoquant does not touch pack core).

## Priors seeded from prior work

- Output head (lm_head/embed_out): **int8, never RVQ** (RVQ → ppl 1.2M; int8 lossless).
- Input embedding: RVQ fine (ppl 130 vs 128).
- Norms / biases: keep fp16.
- Target per-linear SQNR ≈ 30 dB (14 dB was catastrophic at model scale).

## CLI

```
orka autoquant <model> --objective {min-bits|max-quality|knee}
                       [--target <kl|bpw|mb>] [--no-llm] [--prompts <file>]
                       [--out <artifact>]
```
Default pulse-check budget: 32 prompts. Objective flag selects the orchestrator stopping rule.
