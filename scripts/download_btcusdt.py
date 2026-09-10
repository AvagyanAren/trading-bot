"""CLI entry point for the v0.1 historical data pipeline.

Reads ``config/data.yaml``, ensures the raw archives are present and verified,
rebuilds the processed Parquet store, validates it, writes the validation
report and exits 0 on PASS or 1 on FAIL.

Usage:
    python scripts/download_btcusdt.py
    python scripts/download_btcusdt.py --config config/data.yaml --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader import DownloadError  # noqa: E402
from src.data.normalizer import NormalizerError  # noqa: E402
from src.data.parser import ParserError  # noqa: E402
from src.data.pipeline import (  # noqa: E402
    load_config,
    render_console_summary,
    run_pipeline,
)

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "data.yaml"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_btcusdt",
        description=(
            "Download, normalize and validate Binance Spot historical klines "
            "for the symbol and period defined in config/data.yaml."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the data pipeline configuration (default: config/data.yaml)",
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
        config = load_config(args.config)
    except (OSError, KeyError, ValueError) as error:
        print(f"Configuration error in {args.config}: {error}", file=sys.stderr)
        return EXIT_ERROR

    progress = None if args.quiet else lambda message: print(message, flush=True)

    try:
        result = run_pipeline(config, progress=progress)
    except (DownloadError, ParserError, NormalizerError) as error:
        print(f"\nPipeline aborted: {error}", file=sys.stderr)
        return EXIT_ERROR

    print(render_console_summary(result))
    return EXIT_PASS if result.passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
