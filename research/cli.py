"""Command line entry point for earnings-call research briefs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from research.earnings_research import DEFAULT_OUTPUT_DIR, run_research_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn a free-text research task into a finished earnings-call analysis."
    )
    parser.add_argument("task", help="Research task, e.g. 'find evidence for AI capex cyclicality'")
    parser.add_argument("--max-evidence", type=int, default=12)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument(
        "--no-qdrant",
        action="store_true",
        help="Use only local transcript keyword scoring.",
    )
    parser.add_argument(
        "--no-asknews",
        action="store_true",
        help="Skip the default live AskNews research augmentation.",
    )
    parser.add_argument(
        "--asknews-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for the live AskNews DeepNews step.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON result instead of Markdown analysis.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages on stderr.",
    )
    parser.add_argument("--output", type=Path, help="Write the finished analysis to a file.")
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evidence packages are written.",
    )
    parser.add_argument(
        "--no-package",
        action="store_true",
        help="Do not write the side-effect evidence package.",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    status_callback = None
    if not args.quiet:
        status_callback = lambda message: print(f"[research] {message}", file=sys.stderr)

    result = run_research_analysis(
        args.task,
        max_evidence=args.max_evidence,
        max_queries=args.max_queries,
        use_qdrant=not args.no_qdrant,
        output_dir=args.package_dir,
        write_package=not args.no_package,
        use_asknews=not args.no_asknews,
        asknews_timeout=args.asknews_timeout,
        status_callback=status_callback,
    )
    rendered = (
        json.dumps(result, indent=2)
        if args.json
        else result["analysis_markdown"]
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
