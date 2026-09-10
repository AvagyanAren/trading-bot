"""Research YAML loading and A/B/C/D isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.config import (
    REQUIRED_STARTING_CAPITAL,
    VARIANT_ORDER,
    ResearchConfigError,
    load_research_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_repo_research_config_isolates_variants():
    research = load_research_config(REPO_ROOT / "config" / "research.yaml")
    assert research.base_backtest.starting_capital == REQUIRED_STARTING_CAPITAL
    assert tuple(research.variants) == VARIANT_ORDER
    a = research.strategy_for("A")
    b = research.strategy_for("B")
    c_bt = research.backtest_for("C")
    d = research.strategy_for("D")
    assert a.rsi_filter_enabled is True
    assert b.rsi_filter_enabled is False
    assert a.breakout_period == b.breakout_period == 20
    assert d.breakout_period == 50
    assert research.backtest_for("A").take_profit == pytest.approx(0.02)
    assert c_bt.take_profit == pytest.approx(0.03)
    assert research.backtest_for("A").starting_capital == 20
    assert research.development.start.isoformat() == "2024-01-01"
    assert research.test.start.isoformat() == "2025-01-01"


def test_rejects_starting_capital_override(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "strategy.yaml").write_text(
        (REPO_ROOT / "config" / "strategy.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (config_dir / "backtest.yaml").write_text(
        (REPO_ROOT / "config" / "backtest.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (config_dir / "research.yaml").write_text(
        (REPO_ROOT / "config" / "research.yaml").read_text(encoding="utf-8").replace(
            "take_profit: 0.02\n",
            "take_profit: 0.02\n    starting_capital: 21\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ResearchConfigError, match="starting_capital"):
        load_research_config(config_dir / "research.yaml")


def test_rejects_starting_capital_not_20(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "strategy.yaml").write_text(
        (REPO_ROOT / "config" / "strategy.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    backtest = (REPO_ROOT / "config" / "backtest.yaml").read_text(encoding="utf-8")
    (config_dir / "backtest.yaml").write_text(
        backtest.replace("starting_capital: 20", "starting_capital: 21"),
        encoding="utf-8",
    )
    (config_dir / "research.yaml").write_text(
        (REPO_ROOT / "config" / "research.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ResearchConfigError, match="starting_capital"):
        load_research_config(config_dir / "research.yaml")
