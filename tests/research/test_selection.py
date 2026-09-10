"""Selection checklist is development-only and does not auto-pick a winner."""

from __future__ import annotations

import pytest

from src.research.config import ROLE_DEVELOPMENT, VARIANT_ORDER
from src.research.metrics import ExperimentMetrics
from src.research.selection import ResearchSelectionError, evaluate_selection

from .conftest import default_gates, make_result, make_trade, flat_equity
from src.research.metrics import compute_metrics


def _metrics(*, closed: int, net: float, r: float, gross: float, dd: float, months: int, largest: float, win_sum: float, month_share: float | None) -> ExperimentMetrics:
    trades = [
        make_trade(trade_id=index + 1, net_pnl=net / closed if closed else 0.2, R=r)
        for index in range(max(closed, 1))
    ]
    if closed == 0:
        trades = []
    result = make_result(trades[:closed] if closed else [])
    base = compute_metrics(result, flat_equity(result.cash if closed else 20.0))
    data = base.__dict__.copy()
    data.update(
        {
            "closed_trades": closed,
            "total_net_pnl": net,
            "total_gross_pnl": gross,
            "average_net_pnl_per_closed_trade": net / closed if closed else None,
            "average_R_per_closed_trade": r if closed else None,
            "max_drawdown": dd,
            "months_with_closed_trades": months,
            "largest_win_net_pnl": largest if closed else None,
            "sum_of_winning_net_pnl": win_sum if closed else None,
            "max_month_share_of_net": month_share,
            "profit_factor": 1.2 if closed else None,
        }
    )
    return ExperimentMetrics(**data)


def test_selection_refuses_non_development_role():
    metrics = {experiment_id: _metrics(closed=40, net=1.0, r=0.1, gross=2.0, dd=1.0, months=8, largest=0.1, win_sum=1.0, month_share=0.2) for experiment_id in VARIANT_ORDER}
    with pytest.raises(ResearchSelectionError, match="period_role"):
        evaluate_selection(
            metrics,
            default_gates(),
            period_roles={"A": "OUT-OF-SAMPLE / UNTOUCHED TEST", **{k: ROLE_DEVELOPMENT for k in VARIANT_ORDER if k != "A"}},
        )


def test_eligible_rank_is_advisory_not_a_winner():
    metrics = {
        "A": _metrics(closed=40, net=1.0, r=0.2, gross=2.0, dd=2.0, months=8, largest=0.1, win_sum=1.0, month_share=0.2),
        "B": _metrics(closed=40, net=1.0, r=0.5, gross=2.0, dd=1.0, months=8, largest=0.1, win_sum=1.0, month_share=0.2),
        "C": _metrics(closed=5, net=-1.0, r=-0.2, gross=-0.1, dd=12.0, months=2, largest=0.9, win_sum=1.0, month_share=0.9),
        "D": _metrics(closed=40, net=1.0, r=0.1, gross=2.0, dd=3.0, months=8, largest=0.1, win_sum=1.0, month_share=0.2),
    }
    report = evaluate_selection(
        metrics,
        default_gates(),
        period_roles={experiment_id: ROLE_DEVELOPMENT for experiment_id in VARIANT_ORDER},
    )
    assert "C" not in report.eligible_ids
    assert report.eligible_ids[0] == "B"
    by_id = {item.experiment_id: item for item in report.variants}
    assert by_id["B"].advisory_rank == 1
    assert by_id["C"].eligible is False
    assert "winner" not in report.disclaimer.lower()
    assert "NOT statistically validated" in report.disclaimer
