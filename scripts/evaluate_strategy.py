"""CLI entry point for the v0.3 strategy engine.

Reads the v0.2 enriched store, evaluates the configured LONG-only entry
hypothesis, writes a validation report and a LONG_ENTRY CSV, and exits 0
on PASS or 1 on FAIL.

Does not download data, does not talk to Binance, does not size positions,
does not apply stop loss or take profit, and does not simulate fills.

Usage:
    python scripts/evaluate_strategy.py
    python scripts/evaluate_strategy.py --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import load_config  # noqa: E402
from src.indicators.engine import (  # noqa: E402
    IndicatorConfigError,
    load_indicator_config,
)
from src.strategy.engine import (  # noqa: E402
    StrategyConfigError,
    StrategyInputError,
    load_strategy_config,
)
from src.strategy.pipeline import (  # noqa: E402
    StrategyEvaluationError,
    render_console_summary,
    run_strategy_evaluation,
)

DEFAULT_DATA_CONFIG = PROJECT_ROOT / "config" / "data.yaml"
DEFAULT_INDICATOR_CONFIG = PROJECT_ROOT / "config" / "indicators.yaml"
DEFAULT_STRATEGY_CONFIG = PROJECT_ROOT / "config" / "strategy.yaml"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_strategy",
        description=(
            "Evaluate the configured LONG-only entry hypothesis on the "
            "enriched kline store and write a validation report."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Path to the data pipeline configuration (default: config/data.yaml)",
    )
    parser.add_argument(
        "--indicators-config",
        type=Path,
        default=DEFAULT_INDICATOR_CONFIG,
        help="Path to the indicator configuration (default: config/indicators.yaml)",
    )
    parser.add_argument(
        "--strategy-config",
        type=Path,
        default=DEFAULT_STRATEGY_CONFIG,
        help="Path to the strategy configuration (default: config/strategy.yaml)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step progress output; print only the final summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        data_config = load_config(args.config)
        indicator_config = load_indicator_config(args.indicators_config)
        strategy_config = load_strategy_config(args.strategy_config)
    except (OSError, KeyError, ValueError, IndicatorConfigError, StrategyConfigError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_ERROR

    progress = None if args.quiet else lambda message: print(message, flush=True)

    try:
        result = run_strategy_evaluation(
            data_config,
            indicator_config,
            strategy_config,
            progress=progress,
        )
    except (StrategyEvaluationError, StrategyInputError) as error:
        print(f"\nStrategy evaluation aborted: {error}", file=sys.stderr)
        return EXIT_ERROR

    print(render_console_summary(result))
    return EXIT_PASS if result.passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
