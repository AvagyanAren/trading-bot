"""CLI entry point for the v0.5 research / analytics layer.

Reuses v0.3 signals and the v0.4 backtest engine. Does not download data,
does not talk to Binance, does not optimize parameters, and does not
overwrite v0.4 reports.

Usage:
    python scripts/run_research.py --stage development
    python scripts/run_research.py --stage compare
    python scripts/run_research.py --stage test --candidate A
    python scripts/run_research.py --stage integrity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.config import BacktestConfigError  # noqa: E402
from src.data.pipeline import load_config  # noqa: E402
from src.indicators.engine import (  # noqa: E402
    IndicatorConfigError,
    load_indicator_config,
)
from src.research.config import ResearchConfigError  # noqa: E402
from src.research.config import load_research_config
from src.research.pipeline import (  # noqa: E402
    ResearchPipelineError,
    run_compare,
    run_development,
    run_integrity,
    run_test,
)
from src.research.selection import ResearchSelectionError  # noqa: E402
from src.strategy.engine import StrategyConfigError  # noqa: E402

DEFAULT_DATA_CONFIG = PROJECT_ROOT / "config" / "data.yaml"
DEFAULT_INDICATOR_CONFIG = PROJECT_ROOT / "config" / "indicators.yaml"
DEFAULT_STRATEGY_CONFIG = PROJECT_ROOT / "config" / "strategy.yaml"
DEFAULT_BACKTEST_CONFIG = PROJECT_ROOT / "config" / "backtest.yaml"
DEFAULT_RESEARCH_CONFIG = PROJECT_ROOT / "config" / "research.yaml"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_research",
        description=(
            "Run v0.5 research experiments on the enriched historical store "
            "without changing the v0.4 simulator."
        ),
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("development", "compare", "test", "integrity"),
        help="Research stage to run",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Variant id A/B/C/D for --stage test (required for test)",
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
        help="Path to the baseline strategy configuration (default: config/strategy.yaml)",
    )
    parser.add_argument(
        "--backtest-config",
        type=Path,
        default=DEFAULT_BACKTEST_CONFIG,
        help="Path to the baseline backtest configuration (default: config/backtest.yaml)",
    )
    parser.add_argument(
        "--research-config",
        type=Path,
        default=DEFAULT_RESEARCH_CONFIG,
        help="Path to the research configuration (default: config/research.yaml)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step progress output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "test" and not args.candidate:
        print("Configuration error: --stage test requires --candidate", file=sys.stderr)
        return EXIT_ERROR

    try:
        data_config = load_config(args.config)
        indicator_config = load_indicator_config(args.indicators_config)
        research = load_research_config(
            args.research_config,
            strategy_config_path=args.strategy_config,
            backtest_config_path=args.backtest_config,
        )
    except (
        OSError,
        KeyError,
        ValueError,
        IndicatorConfigError,
        StrategyConfigError,
        BacktestConfigError,
        ResearchConfigError,
    ) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_ERROR

    progress = None if args.quiet else lambda message: print(message, flush=True)

    try:
        if args.stage == "development":
            results = run_development(
                data_config, indicator_config, research, progress=progress
            )
            print(
                f"Wrote {len(results)} development experiments under "
                f"{research.output_dir / 'development'}"
            )
        elif args.stage == "compare":
            dest = run_compare(research, progress=progress)
            print(f"Wrote development comparison under {dest}")
        elif args.stage == "test":
            artifacts = run_test(
                data_config,
                indicator_config,
                research,
                args.candidate,
                progress=progress,
            )
            print(
                f"Wrote OUT-OF-SAMPLE report for {artifacts.experiment_id} under "
                f"{research.output_dir / 'test' / artifacts.folder}"
            )
        else:
            run_integrity(
                data_config, indicator_config, research, progress=progress
            )
            print(
                "Integrity replay matched frozen v0.4 totals "
                f"({research.output_dir / 'integrity' / 'A_full_period'})"
            )
    except (ResearchPipelineError, ResearchSelectionError, ResearchConfigError) as error:
        print(f"\nResearch aborted: {error}", file=sys.stderr)
        return EXIT_FAIL
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
