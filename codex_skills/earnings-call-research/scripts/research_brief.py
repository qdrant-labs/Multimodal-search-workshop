#!/usr/bin/env python3
"""Run the repo's earnings-call research brief tool from a Codex skill."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _find_repo(start: Path) -> Path | None:
    for path in [start, *start.parents]:
        if (path / "data" / "transcripts").exists() and (path / "workshop").exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--max-evidence", type=int, default=12)
    parser.add_argument(
        "--use-qdrant",
        action="store_true",
        help="Optionally use the repo's MCP/Qdrant search if it is already implemented.",
    )
    parser.add_argument(
        "--no-qdrant",
        action="store_true",
        help="Compatibility flag; local transcript scoring is already the default.",
    )
    parser.add_argument("--no-asknews", action="store_true")
    parser.add_argument("--asknews-timeout", type=float, default=120.0)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--no-package", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo = _find_repo(args.repo.resolve()) or _find_repo(Path(__file__).resolve())
    if repo is None:
        raise SystemExit("Could not find the Multimodal-search-workshop repo. Pass --repo.")

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    status_callback = None
    if not args.quiet:
        status_callback = lambda message: print(f"[research] {message}", file=sys.stderr)

    os.environ["EARNINGS_RESEARCH_REPO"] = str(repo)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from earnings_research import DEFAULT_OUTPUT_DIR, run_research_analysis

    result = run_research_analysis(
        args.task,
        max_evidence=args.max_evidence,
        use_qdrant=args.use_qdrant and not args.no_qdrant,
        output_dir=args.package_dir or DEFAULT_OUTPUT_DIR,
        write_package=not args.no_package,
        use_asknews=not args.no_asknews,
        asknews_timeout=args.asknews_timeout,
        status_callback=status_callback,
    )
    print(json.dumps(result, indent=2) if args.json else result["analysis_markdown"])


if __name__ == "__main__":
    main()
