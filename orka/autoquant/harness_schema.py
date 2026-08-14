"""Contracts for bounded autonomous AutoQuant runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


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
    artifact_path: Path | None = None
    prompts: Path | None = None
    source: Path | None = None
    group_size: int = 8

    def __post_init__(self) -> None:
        if not self.gratis_base_url:
            raise ValueError("Gratis endpoint is required")
        if not self.model:
            raise ValueError("model is required")
        if self.max_usd <= 0:
            raise ValueError("max_usd must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if self.objective not in {"min-bits", "max-quality", "knee"}:
            raise ValueError(f"unsupported objective: {self.objective}")


@dataclass
class HarnessState:
    status: RunStatus = RunStatus.RUNNING
    step: int = 0
    spent_usd: float = 0.0
    trace: list[dict] = field(default_factory=list)
    candidate: dict | None = None
    roles: dict[str, str] = field(default_factory=dict)
    allocation: dict | None = None
    quality: dict | None = None
    verification: dict | None = None
    artifact_path: str | None = None
    failure_reason: str | None = None

    @classmethod
    def running(cls, config: RunConfig) -> HarnessState:
        artifact = config.artifact_path or (config.output_dir / "out.orka")
        return cls(artifact_path=str(artifact))

    def messages(self) -> list[dict]:
        out = [{"role": "system", "content": "Select one registered AutoQuant tool."}]
        for item in self.trace:
            out.append({"role": "assistant", "content": str(item.get("action"))})
            out.append({"role": "tool", "content": str(item.get("result"))})
        return out

    def record(self, action: dict, result: dict) -> HarnessState:
        self.trace.append({"action": action, "result": result})
        self.step += 1
        if isinstance(result, dict):
            if result.get("candidate") is not None:
                self.candidate = result["candidate"]
            if result.get("roles") is not None:
                self.roles = result["roles"]
            if result.get("allocation") is not None:
                self.allocation = result["allocation"]
            if result.get("quality") is not None:
                self.quality = result["quality"]
            if result.get("verification") is not None:
                self.verification = result["verification"]
            if result.get("artifact_path"):
                self.artifact_path = result["artifact_path"]
        return self

    def succeed(self) -> HarnessState:
        self.status = RunStatus.SUCCEEDED
        self.failure_reason = None
        return self

    def fail(self, reason: str) -> HarnessState:
        self.status = RunStatus.FAILED
        self.failure_reason = reason
        return self
