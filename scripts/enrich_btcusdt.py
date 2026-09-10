"""CLI entry point for the v0.2 indicator enrichment pipeline.

Reads the validated processed store, computes EMA/RSI/volume SMA on the full
chronological series, writes month-partitioned enriched Parquet, and exits
0 on PASS or 1 on FAIL.

Does not download data, does not talk to Binance, and does not generate
trading signals.

Usage:
    python scripts/enrich_btcusdt.py
    python scripts/enrich_btcusdt.py --quiet
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
    IndicatorInputError,
    load_indicator_config,
)
from src.indicators.pipeline import (  # noqa: E402
    EnrichmentError,
    render_console_summary,
    run_enrichment,
)

DEFAULT_DATA_CONFIG = PROJECT_ROOT / "config" / "data.yaml"
DEFAULT_INDICATOR_CONFIG = PROJECT_ROOT / "config" / "indicators.yaml"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enrich_btcusdt",
        description=(
            "Compute EMA20, EMA50, RSI14 and Volume SMA20 on the validated "
            "processed kline store and write enriched Parquet."
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
    except (OSError, KeyError, ValueError, IndicatorConfigError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_ERROR

    progress = None if args.quiet else lambda message: print(message, flush=True)

    try:
        result = run_enrichment(data_config, indicator_config, progress=progress)
    except (EnrichmentError, IndicatorInputError) as error:
        print(f"\nEnrichment aborted: {error}", file=sys.stderr)
        return EXIT_ERROR

    print(render_console_summary(result))
    return EXIT_PASS if result.passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
