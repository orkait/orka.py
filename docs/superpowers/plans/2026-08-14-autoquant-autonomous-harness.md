# Autonomous AutoQuant Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gratis-only autonomous controller that reasons over registered quantization tools and emits only verified AutoQuant artifacts.

**Architecture:** A typed run configuration and state machine bound the Gratis reasoning loop. The agent selects registered analysis and refinement actions; deterministic validators enforce safety and convert decisions to the packer's allocation contract before any artifact can be accepted.

**Tech Stack:** Python 3.10, dataclasses, JSON, NumPy, pytest, existing Orka CLI, Gratis OpenAI-compatible API.

---

## File structure

| File | Responsibility |
|---|---|
| Create `orka/autoquant/harness_schema.py` | Run config, state, tool request/result, terminal states. |
| Create `orka/autoquant/gratis.py` | Gratis-only structured chat client and budget accounting. |
| Create `orka/autoquant/validation.py` | Decision safety and allocation compatibility validation. |
| Create `orka/autoquant/tools.py` | Registered inspect, derive, allocate, pack, pulse, refine, verify actions. |
| Create `orka/autoquant/runner.py` | Bounded reasoner loop and trace/report emission. |
| Modify `orka/autoquant/schema.py` | Convert `TensorConfig` maps to the pack allocation schema. |
| Modify `orka/autoquant/roles.py` | Accept structural architecture evidence while preserving input-embedding distinctions. |
| Modify `orka/cli/parser.py`, `orka/cli/commands.py` | Expose and run autonomous mode. |
| Create `tests/autoquant/test_harness_schema.py` | State/config contract tests. |
| Create `tests/autoquant/test_gratis.py` | Client, structured-output, and budget tests. |
| Create `tests/autoquant/test_validation.py` | Safety and conversion tests. |
| Create `tests/autoquant/test_runner.py` | Fake-Gratis autonomous-loop tests. |
| Extend `tests/autoquant/test_cli.py` | End-to-end CLI artifact/report assertions. |

### Task 1: Define bounded run contracts

**Files:**
- Create: `orka/autoquant/harness_schema.py`
- Test: `tests/autoquant/test_harness_schema.py`

- [ ] **Step 1: Write failing configuration tests**

```python
import pytest
from orka.autoquant.harness_schema import RunConfig

def test_run_config_rejects_non_positive_limits(tmp_path):
    with pytest.raises(ValueError, match="max_steps"):
        RunConfig("http://gratis", "gratis-auto", 0.01, 0, 60, "knee", 0.02, tmp_path)

def test_run_config_requires_gratis_endpoint(tmp_path):
    with pytest.raises(ValueError, match="Gratis"):
        RunConfig("", "gratis-auto", 0.01, 3, 60, "knee", 0.02, tmp_path)
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/autoquant/test_harness_schema.py -q`  
Expected: FAIL because `harness_schema` does not exist.

- [ ] **Step 3: Implement immutable config and serializable state**

```python
@dataclass(frozen=True)
class RunConfig:
    gratis_base_url: str
    model: str
    max_usd: float
    max_steps: int
    timeout_seconds: int
    objective: str
    target: float
    output_dir: Path

    def __post_init__(self):
        if not self.gratis_base_url:
            raise ValueError("Gratis endpoint is required")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
```

Define `RunStatus` with `RUNNING`, `SUCCEEDED`, and `FAILED`; define `HarnessState` with `step`, `spent_usd`, `trace`, `failure_reason`, and `candidate`.

- [ ] **Step 4: Run the contract tests**

Run: `uv run pytest tests/autoquant/test_harness_schema.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/harness_schema.py tests/autoquant/test_harness_schema.py
git commit -m "feat(autoquant): add bounded harness run contracts"
```

### Task 2: Add Gratis-only structured client

**Files:**
- Create: `orka/autoquant/gratis.py`
- Test: `tests/autoquant/test_gratis.py`

- [ ] **Step 1: Write failing client tests**

```python
import pytest
from orka.autoquant.gratis import GratisClient, GratisError

def test_client_rejects_response_without_tool_action():
    client = GratisClient("http://gratis", "gratis-auto", 1.0, post=lambda *_: {"choices": []})
    with pytest.raises(GratisError, match="tool action"):
        client.next_action([], [])

def test_client_rejects_budget_overrun():
    client = GratisClient("http://gratis", "gratis-auto", 0.01, post=lambda *_: {"usage": {"cost": 0.02}})
    with pytest.raises(GratisError, match="budget"):
        client.next_action([], [])
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/autoquant/test_gratis.py -q`  
Expected: FAIL because `GratisClient` does not exist.

- [ ] **Step 3: Implement OpenAI-compatible request and strict action parsing**

```python
class GratisClient:
    def next_action(self, state: list[dict], tools: list[dict]) -> dict:
        response = self._post("/v1/chat/completions", {
            "model": self.model, "messages": state, "tools": tools,
            "tool_choice": "required",
        })
        self._charge(response)
        calls = response.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        if len(calls) != 1:
            raise GratisError("Gratis response must contain exactly one tool action")
        return calls[0]
```

Use only the configured Gratis URL. Normalize HTTP, timeout, malformed response, and missing cost data into `GratisError`. Never import direct provider SDKs.

- [ ] **Step 4: Run client tests**

Run: `uv run pytest tests/autoquant/test_gratis.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/gratis.py tests/autoquant/test_gratis.py
git commit -m "feat(autoquant): add Gratis-only reasoning client"
```

### Task 3: Enforce decisions and convert allocation contracts

**Files:**
- Create: `orka/autoquant/validation.py`
- Modify: `orka/autoquant/schema.py`
- Test: `tests/autoquant/test_validation.py`

- [ ] **Step 1: Write failing safety and conversion tests**

```python
import pytest
from orka.autoquant.schema import TensorConfig, to_packer_allocation
from orka.autoquant.validation import validate_config

def test_output_head_rvq_is_rejected():
    cfg = TensorConfig("rvq", 4, 2, "block-max", False, "llm", 0.8, "bad")
    with pytest.raises(ValueError, match="out-head"):
        validate_config("out-head", cfg)

def test_packer_conversion_preserves_rvq_stages():
    cfg = {"x": TensorConfig("rvq", 4, 2, "block-max", False, "policy", 1, "ok")}
    assert to_packer_allocation(cfg, group_size=8)["tensors"]["x"]["stages"] == [16, 16]
```

- [ ] **Step 2: Run failing validation tests**

Run: `uv run pytest tests/autoquant/test_validation.py -q`  
Expected: FAIL because the validator and conversion do not exist.

- [ ] **Step 3: Implement hard rails and conversion**

```python
def validate_config(role: str, cfg: TensorConfig) -> None:
    if role == "out-head" and cfg.method == "rvq":
        raise ValueError("out-head must not use rvq")
    if role in {"norm", "bias"} and not cfg.keep_fp16:
        raise ValueError(f"{role} must remain fp16")
    if cfg.method == "rvq" and (cfg.bits not in {2, 3, 4, 6, 8} or cfg.stages < 1):
        raise ValueError("unsupported rvq configuration")
```

`to_packer_allocation` must map RVQ stages to `[1 << bits] * stages`, mark int8/fp16 tensors as dense exceptions, and preserve decision rationale in report metadata.

- [ ] **Step 4: Run validation tests**

Run: `uv run pytest tests/autoquant/test_validation.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/schema.py orka/autoquant/validation.py tests/autoquant/test_validation.py
git commit -m "feat(autoquant): validate and convert allocation decisions"
```

### Task 4: Build registered tools and bounded runner

**Files:**
- Create: `orka/autoquant/tools.py`
- Create: `orka/autoquant/runner.py`
- Test: `tests/autoquant/test_runner.py`

- [ ] **Step 1: Write failing autonomous-loop tests**

```python
def test_runner_emits_verified_success(fake_gratis, small_checkpoint, tmp_path):
    result = run_autoquant(RunConfig(...), fake_gratis, small_checkpoint)
    assert result.status.value == "succeeded"
    assert (tmp_path / "autoquant-report.json").exists()

def test_runner_fails_when_step_limit_is_reached(fake_gratis, small_checkpoint, tmp_path):
    result = run_autoquant(RunConfig(..., max_steps=1), fake_gratis, small_checkpoint)
    assert result.status.value == "failed"
    assert result.failure_reason == "step_limit_exceeded"
```

- [ ] **Step 2: Run failing runner tests**

Run: `uv run pytest tests/autoquant/test_runner.py -q`  
Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement allowlisted tool protocol**

```python
TOOLS = {
    "inspect": inspect_tool,
    "derive": derive_tool,
    "allocate": allocate_tool,
    "pack": pack_tool,
    "pulse_check": pulse_check_tool,
    "refine": refine_tool,
    "verify": verify_tool,
}

def run_autoquant(config, client, source):
    state = HarnessState.running(config)
    while state.step < config.max_steps:
        action = client.next_action(state.messages(), tool_schemas())
        result = execute_validated(action, state, source)
        state = state.record(action, result)
        if acceptance_gate(state, config):
            return state.succeed()
    return state.fail("step_limit_exceeded")
```

Each tool receives typed JSON-compatible arguments. `execute_validated` rejects unregistered tools, validates arguments before side effects, records results, and records normalized failures. `acceptance_gate` requires artifact verification and a measured objective result.

- [ ] **Step 4: Run runner tests**

Run: `uv run pytest tests/autoquant/test_runner.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/tools.py orka/autoquant/runner.py tests/autoquant/test_runner.py
git commit -m "feat(autoquant): add autonomous tool reasoning loop"
```

### Task 5: Integrate structural roles and CLI execution

**Files:**
- Modify: `orka/autoquant/roles.py`
- Modify: `orka/cli/parser.py`
- Modify: `orka/cli/commands.py`
- Modify: `tests/autoquant/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

```python
def test_cmd_autoquant_writes_artifact_report_and_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("ORKA_GRATIS_BASE_URL", "http://gratis")
    args = argparse.Namespace(model=str(model), autonomous=True, max_usd=0.05,
                              max_steps=8, timeout_seconds=600, ...)
    assert cmd_autoquant(args) == 0
    assert (tmp_path / "out.orka").exists()
    assert (tmp_path / "autoquant-report.json").exists()
```

- [ ] **Step 2: Run failing CLI test**

Run: `uv run pytest tests/autoquant/test_cli.py -q`  
Expected: FAIL because autonomous CLI flags and outputs do not exist.

- [ ] **Step 3: Add explicit autonomous command options**

```python
aq.add_argument("--autonomous", action="store_true")
aq.add_argument("--gratis-base-url", required=False)
aq.add_argument("--model", default="gratis-auto")
aq.add_argument("--max-usd", type=float, required=False)
aq.add_argument("--max-steps", type=int, default=16)
aq.add_argument("--timeout-seconds", type=int, default=1800)
```

Require `--autonomous`, endpoint, and `--max-usd` for harness execution. Pass checkpoint shapes into a structural role resolver that distinguishes input embeddings from true output heads, including tied tensors.

- [ ] **Step 4: Run CLI test**

Run: `uv run pytest tests/autoquant/test_cli.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/roles.py orka/cli/parser.py orka/cli/commands.py tests/autoquant/test_cli.py
git commit -m "feat(autoquant): expose autonomous harness CLI"
```

### Task 6: Run integration and regression gates

**Files:**
- Modify: `tests/autoquant/test_integration_pythia.py`
- Modify: `docs/ORKA.md`

- [ ] **Step 1: Extend model-gated integration test**

```python
assert result.returncode == 0, result.stderr
assert artifact.exists()
report = json.loads(report_path.read_text())
assert report["status"] == "succeeded"
assert report["verification"]["passed"] is True
assert report["quality"]["objective_met"] is True
```

- [ ] **Step 2: Run focused harness suite**

Run: `uv run pytest tests/autoquant -q`  
Expected: PASS, with the cached-model test skipped only when Pythia is unavailable.

- [ ] **Step 3: Run quantization regression suite**

Run: `uv run pytest tests -k 'autoquant or quant or allocation or curvature or arch or format'`  
Expected: PASS.

- [ ] **Step 4: Document the failure semantics and run command**

```markdown
orka autoquant MODEL --autonomous --max-usd 0.05 --objective knee --target 0.02 --out OUTPUT
```

Document that Gratis is mandatory and that API, budget, safety, quality, or verification failure ends the run without publishing a successful artifact.

- [ ] **Step 5: Commit**

```bash
git add tests/autoquant/test_integration_pythia.py docs/ORKA.md
git commit -m "docs(autoquant): document autonomous harness execution"
```

## Self-review

- Spec coverage: Tasks 1-2 cover required configuration, state, Gratis-only transport, and budget failure. Tasks 3-4 cover safety, typed tools, traceable autonomous reasoning, and acceptance. Task 5 covers CLI and structural roles. Task 6 covers reporting and all required verification tiers.
- Placeholder scan: no placeholder actions or undefined component references remain.
- Type consistency: `RunConfig`, `HarnessState`, `GratisClient`, `TensorConfig`, `validate_config`, and `run_autoquant` are introduced before later tasks reference them.
