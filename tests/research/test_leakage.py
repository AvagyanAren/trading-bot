"""Leakage guards: compare/selection never see 2025."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.research.config import ROLE_DEVELOPMENT, ROLE_TEST, VARIANT_ORDER
from src.research.pipeline import run_compare
from src.research.selection import ResearchSelectionError

from .conftest import make_research_config
from .test_selection import _metrics


def test_compare_rejects_test_period_snapshot(tmp_path: Path):
    research = make_research_config(tmp_path)
    root = research.output_dir / "development"
    for experiment_id in VARIANT_ORDER:
        folder = research.variant(experiment_id).folder
        dest = root / folder
        dest.mkdir(parents=True, exist_ok=True)
        role = ROLE_TEST if experiment_id == "A" else ROLE_DEVELOPMENT
        (dest / "config_snapshot.yaml").write_text(
            yaml.safe_dump(
                {
                    "experiment_id": experiment_id,
                    "period_role": role,
                    "label": research.variant(experiment_id).label,
                }
            ),
            encoding="utf-8",
        )
        metrics = _metrics(
            closed=40,
            net=1.0,
            r=0.2,
            gross=2.0,
            dd=1.0,
            months=8,
            largest=0.1,
            win_sum=1.0,
            month_share=0.2,
        )
        rows = [{"metric": key, "value": value} for key, value in metrics.as_pairs()]
        pd.DataFrame(rows).to_csv(dest / "metrics.csv", index=False)

    with pytest.raises(ResearchSelectionError, match="period_role"):
        run_compare(research)


def test_selection_has_no_test_period_argument():
    import inspect
    from src.research.selection import evaluate_selection as fn

    assert "test" not in inspect.signature(fn).parameters
