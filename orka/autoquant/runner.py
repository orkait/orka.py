"""Bounded autonomous reasoning loop."""
from __future__ import annotations

import json
import time
from pathlib import Path

from orka.autoquant.gratis import GratisClient, GratisError
from orka.autoquant.harness_schema import HarnessState, RunConfig, RunStatus
from orka.autoquant.tools import TOOLS, ToolContext, tool_schemas


def _parse_action(action: dict) -> tuple[str, dict]:
    fn = action.get("function") or {}
    name = fn.get("name") or action.get("name")
    raw = fn.get("arguments", action.get("arguments", "{}"))
    if isinstance(raw, str):
        args = json.loads(raw) if raw else {}
    elif isinstance(raw, dict):
        args = raw
    else:
        raise ValueError("tool arguments must be an object")
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be an object")
    return str(name), args


def execute_validated(action: dict, ctx: ToolContext) -> dict:
    name, args = _parse_action(action)
    if name not in TOOLS:
        raise ValueError("unregistered_tool")
    return TOOLS[name](args, ctx)


def acceptance_gate(state: HarnessState, config: RunConfig) -> bool:
    verification = state.verification or {}
    quality = state.quality or {}
    return bool(verification.get("passed") or verification.get("ok")) and bool(
        quality.get("objective_met")
    )


def write_report(state: HarnessState, config: RunConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "autoquant-report.json"
    path.write_text(
        json.dumps(
            {
                "status": state.status.value,
                "failure_reason": state.failure_reason,
                "steps": state.step,
                "spent_usd": state.spent_usd,
                "objective": config.objective,
                "target": config.target,
                "artifact": state.artifact_path,
                "verification": state.verification,
                "quality": state.quality,
                "trace": state.trace,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def run_autoquant(
    config: RunConfig,
    client: GratisClient,
    source,
    *,
    pack_fn=None,
    pulse_fn=None,
    verify_fn=None,
) -> HarnessState:
    source_path = Path(source)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    state = HarnessState.running(config)
    ctx = ToolContext(
        config=config,
        source=source_path,
        state=state,
        pack_fn=pack_fn,
        pulse_fn=pulse_fn,
        verify_fn=verify_fn,
    )
    started = time.monotonic()
    try:
        while state.step < config.max_steps:
            if time.monotonic() - started > config.timeout_seconds:
                return _finish(state, config, "timeout_exceeded")
            action = client.next_action(state.messages(), tool_schemas())
            state.spent_usd = getattr(client, "spent_usd", state.spent_usd)
            try:
                result = execute_validated(action, ctx)
            except ValueError as exc:
                if str(exc) == "unregistered_tool":
                    return _finish(state, config, "unregistered_tool")
                result = {"ok": False, "error": str(exc)}
            state.record(action, result)
            if acceptance_gate(state, config) or result.get("accepted"):
                return _finish(state.succeed(), config)
        return _finish(state, config, "step_limit_exceeded")
    except GratisError as exc:
        reason = "gratis_budget_exhausted" if "budget" in str(exc).lower() else "gratis_unavailable"
        return _finish(state, config, reason)
    except StopIteration:
        return _finish(state, config, "gratis_unavailable")


def _finish(state: HarnessState, config: RunConfig, reason: str | None = None) -> HarnessState:
    if reason is not None:
        state.fail(reason)
    if state.status == RunStatus.RUNNING:
        state.fail("incomplete")
    write_report(state, config)
    return state
