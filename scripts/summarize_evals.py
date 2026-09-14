#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from univtac.eval.summarize import (
    apply_filters,
    apply_latest_only,
    discover_eval_dirs,
    load_record,
    record_sort_key,
    render_markdown,
    split_csv,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write markdown/csv summaries for one experiment.")
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "eval_result")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--latest-only", action="store_true", default=True)
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--policies", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--demos", type=str, default=None)
    parser.add_argument("--config-names", type=str, default=None)
    parser.add_argument("--inference-configs", type=str, default=None)
    parser.add_argument("--train-run-ids", type=str, default=None)
    parser.add_argument("--eval-ids", type=str, default=None)
    parser.add_argument("--act-paper-run-patterns", type=str, default="paper")
    parser.add_argument("--print", action="store_true", dest="should_print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_runs:
        args.latest_only = False

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (args.results_root / args.experiment_name / "summary" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_patterns = split_csv(args.act_paper_run_patterns)
    assert paper_patterns is not None

    eval_dirs = discover_eval_dirs(args.results_root)
    records = [
        record
        for eval_dir in eval_dirs
        if (record := load_record(eval_dir, args.results_root, paper_patterns)) is not None
    ]

    filter_args = argparse.Namespace(
        experiment_names=args.experiment_name,
        policies=args.policies,
        methods=args.methods,
        tasks=args.tasks,
        demos=args.demos,
        config_names=args.config_names,
        inference_configs=args.inference_configs,
        task_configs=None,
        train_run_ids=args.train_run_ids,
        eval_ids=args.eval_ids,
    )
    records = apply_filters(records, filter_args)
    if args.latest_only:
        records = apply_latest_only(records)
    else:
        records = sorted(records, key=record_sort_key)

    markdown_args = argparse.Namespace(
        results_root=args.results_root,
        output=output_dir / "tables.md",
        format="markdown",
        experiment_names=args.experiment_name,
        policies=args.policies,
        methods=args.methods,
        tasks=args.tasks,
        demos=args.demos,
        config_names=args.config_names,
        inference_configs=args.inference_configs,
        task_configs=None,
        train_run_ids=args.train_run_ids,
        eval_ids=args.eval_ids,
        latest_only=args.latest_only,
        act_paper_run_patterns=args.act_paper_run_patterns,
        should_print=args.should_print,
    )
    markdown = render_markdown(records, markdown_args)
    (output_dir / "tables.md").write_text(markdown, encoding="utf-8")
    write_csv(records, output_dir / "summary.csv")

    if args.should_print:
        print(markdown)
        print(f"\nCSV: {output_dir / 'summary.csv'}")
    else:
        print(f"Wrote summary to {output_dir}")


if __name__ == "__main__":
    main()
