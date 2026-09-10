"""Pipeline file outputs and CLI argument checks."""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.research.config import ROLE_DEVELOPMENT, VARIANT_ORDER
from src.research.equity import EQUITY_COLUMNS
from src.research.experiments import run_experiment
from src.research.pipeline import run_compare, write_experiment_dir
from tests.strategy.conftest import (
    entry_ready_frame,
    indicator_config,
    strategy_config,
)

from .conftest import make_research_config


def test_cli_test_stage_requires_candidate():
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "run_research.py"), "--stage", "test"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2


def test_pipeline_writes_expected_files(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, breakout_period=3)
    research = make_research_config(tmp_path)
    candles = entry_ready_frame(24, indicator, strategy, start="2024-01-01 00:00:00")
    extra = entry_ready_frame(8, indicator, strategy, start="2024-02-01 00:00:00")
    candles = pd.concat([candles, extra], ignore_index=True)
    for experiment_id in VARIANT_ORDER:
        variant_strategy = research.strategy_for(experiment_id)
        strategy_for_run = strategy_config(
            tmp_path,
            breakout_period=3,
            rsi_filter_enabled=variant_strategy.rsi_filter_enabled,
        )
        artifacts = run_experiment(
            candles,
            variant=research.variant(experiment_id),
            strategy_config=strategy_for_run,
            indicator_config=indicator,
            backtest_config=research.backtest_for(experiment_id),
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            period_role=ROLE_DEVELOPMENT,
            symbol="BTCUSDT",
            interval="5m",
        )
        write_experiment_dir(
            artifacts,
            research.output_dir / "development" / artifacts.folder,
            summary_name="summary.txt",
            symbol="BTCUSDT",
            interval="5m",
        )
    dest = run_compare(research)
    assert (dest / "comparison.txt").exists()
    assert (dest / "comparison.csv").exists()
    assert (dest / "selection_checklist.txt").exists()
    checklist = (dest / "selection_checklist.txt").read_text(encoding="utf-8")
    assert "NOT statistically validated" in checklist
    equity = pd.read_csv(
        research.output_dir / "development" / "A_baseline" / "equity.csv"
    )
    assert list(equity.columns) == list(EQUITY_COLUMNS)
    assert "ending capital" not in checklist.lower()
