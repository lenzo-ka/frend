"""Declared tiergraph work policy shared by frend graph operations."""

from __future__ import annotations

from tiergraph import BudgetExhausted, WorkBudget, WorkMeter

__all__ = [
    "BudgetExhausted",
    "DEFAULT_TIERGRAPH_WORK_BUDGET",
    "TiergraphWorkBudget",
    "work_meter",
]

# One request may cross several tiergraph aggregate operations.  The limit is in
# tiergraph's deterministic logical steps, not seconds, so identical work has the
# same boundary on every machine.  Operations refuse whole with BudgetExhausted.
DEFAULT_TIERGRAPH_WORK_BUDGET = WorkBudget(steps=1 << 24)

type TiergraphWorkBudget = WorkBudget | WorkMeter


def work_meter(budget: TiergraphWorkBudget | None = None) -> WorkMeter:
    """Return one meter for a complete frend request."""
    if budget is None:
        budget = DEFAULT_TIERGRAPH_WORK_BUDGET
    if isinstance(budget, WorkMeter):
        return budget
    if isinstance(budget, WorkBudget):
        return WorkMeter(budget)
    raise TypeError("work_budget must be a tiergraph WorkBudget or WorkMeter")
