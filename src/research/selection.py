"""Research eligibility checklist. Never auto-selects a winner.

Gates are heuristics for human review. They are not statistically validated
thresholds and do not prove profitability or live readiness.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import VARIANT_ORDER, EligibilityGates, ResearchError
from .metrics import ExperimentMetrics

GATE_DISCLAIMER = (
    "These are research eligibility gates: simple heuristics to avoid treating "
    "a thin or one-trade result as a candidate. They are NOT statistically "
    "validated thresholds, not a proof of robustness, not an optimized "
    "scoring model, and not a claim that a passing variant is profitable or "
    "live-ready."
)


class ResearchSelectionError(ResearchError):
    """Selection inputs are invalid (for example, test-period leakage)."""


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class VariantChecklist:
    experiment_id: str
    gates: tuple[GateResult, ...]
    eligible: bool
    advisory_rank: int | None


@dataclass(frozen=True)
class SelectionReport:
    variants: tuple[VariantChecklist, ...]
    eligible_ids: tuple[str, ...]
    disclaimer: str = GATE_DISCLAIMER


def _gate_sample_size(metrics: ExperimentMetrics, gates: EligibilityGates) -> GateResult:
    passed = metrics.closed_trades >= gates.min_closed_trades
    return GateResult(
        name="sample_size",
        passed=passed,
        detail=(
            f"closed_trades={metrics.closed_trades} "
            f"(need >= {gates.min_closed_trades})"
        ),
    )


def _gate_expectancy(metrics: ExperimentMetrics) -> GateResult:
    net = metrics.average_net_pnl_per_closed_trade
    r_value = metrics.average_R_per_closed_trade
    passed = net is not None and r_value is not None and net > 0 and r_value > 0
    return GateResult(
        name="positive_expectancy",
        passed=passed,
        detail=(
            f"average_net_pnl_per_closed_trade={net} "
            f"average_R_per_closed_trade={r_value}"
        ),
    )


def _gate_gross(metrics: ExperimentMetrics) -> GateResult:
    passed = metrics.total_gross_pnl > 0
    return GateResult(
        name="positive_gross",
        passed=passed,
        detail=f"total_gross_pnl={metrics.total_gross_pnl}",
    )


def _gate_drawdown(metrics: ExperimentMetrics, gates: EligibilityGates) -> GateResult:
    limit = gates.max_drawdown_pct_of_starting_capital * metrics.starting_capital
    passed = metrics.max_drawdown <= limit
    return GateResult(
        name="global_max_drawdown",
        passed=passed,
        detail=(
            f"max_drawdown={metrics.max_drawdown} "
            f"(limit {limit} = {gates.max_drawdown_pct_of_starting_capital} "
            f"* starting_capital)"
        ),
    )


def _gate_concentration(metrics: ExperimentMetrics, gates: EligibilityGates) -> GateResult:
    largest = metrics.largest_win_net_pnl
    win_sum = metrics.sum_of_winning_net_pnl
    if largest is None or win_sum is None or win_sum <= 0:
        return GateResult(
            name="concentration",
            passed=False,
            detail="no winning trades to measure concentration",
        )
    share = largest / win_sum
    passed = share < gates.max_largest_win_share_of_wins
    return GateResult(
        name="concentration",
        passed=passed,
        detail=(
            f"largest_win_share={share} "
            f"(need < {gates.max_largest_win_share_of_wins})"
        ),
    )


def _gate_monthly(metrics: ExperimentMetrics, gates: EligibilityGates) -> GateResult:
    months_ok = metrics.months_with_closed_trades >= gates.min_months_with_trades
    if metrics.total_net_pnl > 0:
        share = metrics.max_month_share_of_net
        month_ok = share is not None and share <= gates.max_single_month_net_share
    else:
        month_ok = True
        share = metrics.max_month_share_of_net
    passed = months_ok and month_ok
    return GateResult(
        name="monthly_robustness",
        passed=passed,
        detail=(
            f"months_with_closed_trades={metrics.months_with_closed_trades} "
            f"(need >= {gates.min_months_with_trades}); "
            f"max_month_share_of_net={share}"
        ),
    )


def evaluate_variant(
    experiment_id: str,
    metrics: ExperimentMetrics,
    gates: EligibilityGates,
) -> VariantChecklist:
    checks = (
        _gate_sample_size(metrics, gates),
        _gate_expectancy(metrics),
        _gate_gross(metrics),
        _gate_drawdown(metrics, gates),
        _gate_concentration(metrics, gates),
        _gate_monthly(metrics, gates),
    )
    return VariantChecklist(
        experiment_id=experiment_id,
        gates=checks,
        eligible=all(item.passed for item in checks),
        advisory_rank=None,
    )


def _sort_key(metrics: ExperimentMetrics) -> tuple[float, float, float]:
    r_value = metrics.average_R_per_closed_trade
    pf = metrics.profit_factor
    return (
        -(r_value if r_value is not None else float("-inf")),
        -(pf if pf is not None else float("-inf")),
        metrics.max_drawdown,
    )


def evaluate_selection(
    metrics_by_id: dict[str, ExperimentMetrics],
    gates: EligibilityGates,
    *,
    period_roles: dict[str, str],
) -> SelectionReport:
    for experiment_id, role in period_roles.items():
        if role != "DEVELOPMENT":
            raise ResearchSelectionError(
                f"Refusing to select using period_role={role!r} "
                f"for experiment {experiment_id}"
            )
    if set(metrics_by_id) != set(VARIANT_ORDER):
        raise ResearchSelectionError(
            f"Selection requires variants {list(VARIANT_ORDER)}, "
            f"got {sorted(metrics_by_id)}"
        )

    checklists = [
        evaluate_variant(experiment_id, metrics_by_id[experiment_id], gates)
        for experiment_id in VARIANT_ORDER
    ]
    eligible = [item for item in checklists if item.eligible]
    ranked = sorted(eligible, key=lambda item: _sort_key(metrics_by_id[item.experiment_id]))
    rank_map = {
        item.experiment_id: index + 1 for index, item in enumerate(ranked)
    }
    finalized = tuple(
        VariantChecklist(
            experiment_id=item.experiment_id,
            gates=item.gates,
            eligible=item.eligible,
            advisory_rank=rank_map.get(item.experiment_id),
        )
        for item in checklists
    )
    return SelectionReport(
        variants=finalized,
        eligible_ids=tuple(item.experiment_id for item in ranked),
    )
