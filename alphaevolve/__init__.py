"""PWB AlphaEvolve package - discover & evolve trading strategies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evolution.controller import Controller
from .store.sqlite import ProgramStore

__all__ = [
    "AlphaEvolve",
    "Strategy",
    "backtest",
    "strategies",
]


@dataclass
class Strategy:
    """Simple container for strategy code and metrics."""

    id: str
    code: str
    metrics: dict[str, Any]


class AlphaEvolve:
    """Convenience wrapper for running the evolution loop."""

    def __init__(
        self,
        initial_program_paths: list[str],
        config_path: str,
        *,
        store: ProgramStore | None = None,
    ) -> None:
        self.initial_program_paths = [Path(p) for p in initial_program_paths]
        self.config_path = Path(config_path)
        self.store = store or ProgramStore()
        self.controller = Controller(self.store)

    async def run(self, iterations: int = 1) -> Strategy:
        """Run the evolution loop for a fixed number of iterations."""
        for _ in range(iterations):
            await self.controller._spawn(None)  # type: ignore[attr-defined]
            await asyncio.sleep(0.01)
        best = self.store.top_k(k=1)
        if not best:
            raise RuntimeError("No strategies generated")
        row = best[0]
        return Strategy(id=row["id"], code=row["code"], metrics=row["metrics"])
