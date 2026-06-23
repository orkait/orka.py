"""Post-assignment strategy protocol.

A PostAssignmentStrategy is a composable compression step applied to a candidate AFTER
the RVQ stage loop and before its manifest entry. The pipeline applies the registered
strategies in order; each decides (`applies`) whether it runs for a given candidate +
config, then mutates the candidate in place (`apply`).

Pluggable by design: add a new trick by subclassing this and appending an instance to
POST_ASSIGNMENT_STRATEGIES in __init__.py - the pipeline loop does not change.

Strategies duck-type the context (a pack_pipeline.PackCtx); they intentionally do not
import it, so this package has no dependency back on pack_pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class PostAssignmentStrategy(ABC):
    #: short identifier, matches the STRATEGY_REGISTRY entry
    name: str = ""

    @abstractmethod
    def applies(self, ctx, c: dict) -> bool:
        """Whether this strategy runs for candidate ``c`` under config ``ctx``."""

    @abstractmethod
    def apply(self, ctx, c: dict) -> None:
        """Run the strategy, mutating ``c`` in place."""
