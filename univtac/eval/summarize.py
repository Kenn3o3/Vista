#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


FINAL_RESULT_RE = re.compile(
    r"Final Result:\s*(?P<succ>\d+)/(?P<total>\d+)\((?P<rate>[0-9.]+)%\)\s+success\."
)


def split_csv(raw_value: str | None, *, cast=str) -> set | None:
    if raw_value is None:
        return None
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        return None
    return {cast(item) for item in values}


def infer_demo(config_name: str) -> int | None:
    match = re.search(r"_demo(\d+)$", config_name)
    if match is None:
        return None
    return int(match.group(1))


def strip_demo_suffix(config_name: str) -> str:
    return re.sub(r"_demo\d+$", "", config_name)


def format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def safe_mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return mean(items)


@dataclass(frozen=True)
class EvalRecord:
    experiment_name: str
    policy_name: str
    method: str
    task_name: str
    config_name: str
    inference_config: str
    demo: int | None
    train_run_id: str
    eval_id: str
    success: int
    total: int
    success_rate: float
    avg_steps: float | None
    avg_time: float | None
    success_avg_steps: float | None
    success_avg_time: float | None
    failed_avg_steps: float | None
    failed_avg_time: float | None
    status: str
    eval_dir: str
    ckpt_path: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def detect_method(
    policy_name: str,
    config_name: str,
    train_run_id: str,
    ckpt_path: str,
    paper_patterns: set[str],
) -> str:
    raw = f"{config_name} {train_run_id} {ckpt_path}".lower()
    base = strip_demo_suffix(config_name)
    method_map = {
        ("DP", "dp"): "DP",
        ("DP", "manifeel"): "ManiFeel",
        ("DP", "equidiff_tact"): "EquiDiff-Tact",
        ("ACT", "univtac"): "UniVTAC",
        ("ISP", "so2"): "ISP-SO2",
        ("ISP", "so3"): "ISP-SO3",
        ("ViTAL", "dp"): "ViTAL-DP",
        ("ViTAL", "act"): "ViTAL-ACT",
        ("VISTA", "so2"): "VISTA-SO2",
        ("VISTA", "so3"): "VISTA-SO3",
        ("VISTA", "vista"): "VISTA-SO3",
    }
    mapped = method_map.get((policy_name, base))
    if mapped is not None:
        return mapped
    if policy_name == "ACT":
        if any(pattern in raw for pattern in paper_patterns):
            return "ACT+paper_encoder"
        return "ACT"
    if policy_name == "ISP":
        return f"ISP_{base}"
    if policy_name == "VISTA":
        return f"VISTA_{base}"
    if policy_name == "ViTAL":
        return f"ViTAL_{base}"
    return policy_name


def resolve_train_dir(root: Path, eval_meta: dict) -> Path | None:
    try:
        return (
            root
            / "checkpoints"
            / eval_meta["experiment_name"]
            / eval_meta["task_name"]
            / eval_meta["policy_name"]
            / eval_meta["config_name"]
            / eval_meta["train_run_id"]
        )
    except KeyError:
        return None


def infer_ckpt_path(train_dir: Path) -> str:
    candidates = [
        train_dir / "policy_best.ckpt",
        train_dir / "best.ckpt",
        train_dir / "checkpoints" / "best.ckpt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def infer_eval_metadata_from_dir(root: Path, results_root: Path, eval_dir: Path) -> dict | None:
    try:
        relative = eval_dir.relative_to(results_root)
    except ValueError:
        return None

    parts = relative.parts
    if len(parts) != 7:
        return None

    experiment_name, task_name, policy_name, config_name, train_run_id, inference_config, eval_id = parts
    eval_meta = {
        "experiment_name": experiment_name,
        "task_name": task_name,
        "policy_name": policy_name,
        "config_name": config_name,
        "train_run_id": train_run_id,
        "inference_config_name": inference_config,
        "eval_id": eval_id,
        "eval_dir": str(eval_dir),
        "ckpt_path": "",
    }

    train_dir = resolve_train_dir(root, eval_meta)
    if train_dir is None:
        return eval_meta

    train_meta = read_json(train_dir / "run_metadata.json")
    if isinstance(train_meta, dict):
        eval_meta["experiment_name"] = str(train_meta.get("experiment_name", eval_meta["experiment_name"]))
        eval_meta["task_name"] = str(train_meta.get("task_name", eval_meta["task_name"]))
        eval_meta["policy_name"] = str(train_meta.get("policy_name", eval_meta["policy_name"]))
        eval_meta["config_name"] = str(
            train_meta.get("config_name", train_meta.get("variant", eval_meta["config_name"]))
        )
        eval_meta["train_run_id"] = str(train_meta.get("run_id", eval_meta["train_run_id"]))

        checkpoint_dir = train_meta.get("checkpoint_dir")
        if isinstance(checkpoint_dir, str) and checkpoint_dir:
            ckpt_path = infer_ckpt_path(Path(checkpoint_dir))
        else:
            ckpt_path = infer_ckpt_path(train_dir)
        if ckpt_path:
            eval_meta["ckpt_path"] = ckpt_path

    return eval_meta


def discover_eval_dirs(results_root: Path) -> list[Path]:
    eval_dirs: set[Path] = set()
    for pattern in ("**/run_metadata.json", "**/metadata.json", "**/log.log"):
        for path in results_root.glob(pattern):
            eval_dirs.add(path.parent)
    return sorted(eval_dirs)


def summarize_episode_metadata(
    metadata_path: Path,
) -> tuple[
    int,
    int,
    float,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    payload = read_json(metadata_path)
    if not isinstance(payload, dict):
        return 0, 0, 0.0, None, None, None, None, None, None

    success = 0
    total = 0
    all_steps: list[float] = []
    all_time: list[float] = []
    success_steps: list[float] = []
    success_time: list[float] = []
    failed_steps: list[float] = []
    failed_time: list[float] = []

    for item in payload.values():
        if not isinstance(item, dict):
            continue
        total += 1
        result = item.get("result")
        cost_step = item.get("cost_step")
        cost_time = item.get("cost_time")
        if isinstance(cost_step, (int, float)):
            all_steps.append(float(cost_step))
        if isinstance(cost_time, (int, float)):
            all_time.append(float(cost_time))
        if result == "success":
            success += 1
            if isinstance(cost_step, (int, float)):
                success_steps.append(float(cost_step))
            if isinstance(cost_time, (int, float)):
                success_time.append(float(cost_time))
        elif result == "failed":
            if isinstance(cost_step, (int, float)):
                failed_steps.append(float(cost_step))
            if isinstance(cost_time, (int, float)):
                failed_time.append(float(cost_time))

    rate = 100.0 * success / total if total else 0.0
    return (
        success,
        total,
        rate,
        safe_mean(all_steps),
        safe_mean(all_time),
        safe_mean(success_steps),
        safe_mean(success_time),
        safe_mean(failed_steps),
        safe_mean(failed_time),
    )


def summarize_from_log(log_path: Path) -> tuple[int, int, float] | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(errors="replace")
    match = FINAL_RESULT_RE.search(text)
    if match is None:
        return None
    return int(match.group("succ")), int(match.group("total")), float(match.group("rate"))


def load_record(eval_dir: Path, results_root: Path, paper_patterns: set[str]) -> EvalRecord | None:
    root = repo_root()
    eval_meta = read_json(eval_dir / "run_metadata.json")
    if not isinstance(eval_meta, dict):
        eval_meta = infer_eval_metadata_from_dir(root, results_root, eval_dir)
    if not isinstance(eval_meta, dict):
        return None

    config_name = str(eval_meta.get("config_name", eval_meta.get("variant", "")))
    inference_config = str(
        eval_meta.get("inference_config_name", eval_meta.get("inference_config", eval_meta.get("task_config", "")))
    )
    train_run_id = str(eval_meta.get("train_run_id", eval_meta.get("run_id", "")))
    policy_name = str(eval_meta.get("policy_name", ""))
    task_name = str(eval_meta.get("task_name", ""))
    experiment_name = str(eval_meta.get("experiment_name", ""))
    ckpt_path = str(eval_meta.get("ckpt_path", ""))

    train_dir = resolve_train_dir(
        root,
        {
            "experiment_name": experiment_name,
            "task_name": task_name,
            "policy_name": policy_name,
            "config_name": config_name,
            "train_run_id": train_run_id,
        },
    )
    if train_dir is not None:
        train_meta = read_json(train_dir / "run_metadata.json")
        if isinstance(train_meta, dict):
            experiment_name = str(train_meta.get("experiment_name", experiment_name))
            task_name = str(train_meta.get("task_name", task_name))
            policy_name = str(train_meta.get("policy_name", policy_name))
            config_name = str(train_meta.get("config_name", train_meta.get("variant", config_name)))
            train_run_id = str(train_meta.get("run_id", train_run_id))
            if not ckpt_path:
                checkpoint_dir = train_meta.get("checkpoint_dir")
                if isinstance(checkpoint_dir, str) and checkpoint_dir:
                    ckpt_path = infer_ckpt_path(Path(checkpoint_dir))
                else:
                    ckpt_path = infer_ckpt_path(train_dir)

    metadata_path = eval_dir / "metadata.json"
    log_path = eval_dir / "log.log"
    success, total, rate, avg_steps, avg_time, success_avg_steps, success_avg_time, failed_avg_steps, failed_avg_time = (
        summarize_episode_metadata(metadata_path)
    )
    status = "complete" if total > 0 else "missing_metadata"

    if total == 0:
        log_summary = summarize_from_log(log_path)
        if log_summary is None:
            return None
        success, total, rate = log_summary
        status = "log_only"
        avg_steps = None
        avg_time = None
        success_avg_steps = None
        success_avg_time = None
        failed_avg_steps = None
        failed_avg_time = None

    method = detect_method(policy_name, config_name, train_run_id, ckpt_path, paper_patterns)

    return EvalRecord(
        experiment_name=experiment_name,
        policy_name=policy_name,
        method=method,
        task_name=task_name,
        config_name=config_name,
        inference_config=inference_config,
        demo=infer_demo(config_name),
        train_run_id=train_run_id,
        eval_id=str(eval_meta.get("eval_id", "")),
        success=success,
        total=total,
        success_rate=rate,
        avg_steps=avg_steps,
        avg_time=avg_time,
        success_avg_steps=success_avg_steps,
        success_avg_time=success_avg_time,
        failed_avg_steps=failed_avg_steps,
        failed_avg_time=failed_avg_time,
        status=status,
        eval_dir=str(eval_dir),
        ckpt_path=ckpt_path,
    )


def apply_filters(records: list[EvalRecord], args: argparse.Namespace) -> list[EvalRecord]:
    experiment_names = split_csv(args.experiment_names)
    policies = split_csv(args.policies)
    methods = split_csv(args.methods)
    tasks = split_csv(args.tasks)
    demos = split_csv(args.demos, cast=int)
    config_names = split_csv(args.config_names)
    inference_configs = split_csv(args.inference_configs) or split_csv(args.task_configs)
    train_run_ids = split_csv(args.train_run_ids)
    eval_ids = split_csv(args.eval_ids)

    filtered: list[EvalRecord] = []
    for record in records:
        if experiment_names is not None and record.experiment_name not in experiment_names:
            continue
        if policies is not None and record.policy_name not in policies:
            continue
        if methods is not None and record.method not in methods:
            continue
        if tasks is not None and record.task_name not in tasks:
            continue
        if demos is not None and record.demo not in demos:
            continue
        if config_names is not None and record.config_name not in config_names:
            continue
        if inference_configs is not None and record.inference_config not in inference_configs:
            continue
        if train_run_ids is not None and record.train_run_id not in train_run_ids:
            continue
        if eval_ids is not None and record.eval_id not in eval_ids:
            continue
        filtered.append(record)
    return filtered


def record_sort_key(record: EvalRecord) -> tuple:
    return (
        record.experiment_name,
        record.task_name,
        record.policy_name,
        record.config_name,
        record.demo if record.demo is not None else -1,
        record.train_run_id,
        record.inference_config,
        record.eval_id,
    )


def apply_latest_only(records: list[EvalRecord]) -> list[EvalRecord]:
    latest: dict[tuple[str, str, str, str, str, str], EvalRecord] = {}
    for record in records:
        key = (
            record.experiment_name,
            record.task_name,
            record.policy_name,
            record.config_name,
            record.train_run_id,
            record.inference_config,
        )
        current = latest.get(key)
        if current is None or record.eval_id > current.eval_id:
            latest[key] = record
    return sorted(latest.values(), key=record_sort_key)


def build_detail_rows(records: list[EvalRecord]) -> list[list[str]]:
    rows: list[list[str]] = []
    for record in sorted(records, key=record_sort_key):
        rows.append(
            [
                record.experiment_name,
                record.task_name,
                record.policy_name,
                record.method,
                record.config_name,
                record.inference_config,
                str(record.demo) if record.demo is not None else "n/a",
                record.train_run_id,
                record.eval_id,
                f"{record.success}/{record.total}",
                format_float(record.success_rate),
                format_float(record.avg_steps),
                format_float(record.avg_time),
                record.status,
            ]
        )
    return rows


def build_method_summary_rows(records: list[EvalRecord]) -> list[list[str]]:
    grouped: dict[tuple[str, str, str], list[EvalRecord]] = {}
    for record in records:
        key = (record.experiment_name, record.method, record.inference_config)
        grouped.setdefault(key, []).append(record)

    rows: list[list[str]] = []
    for key in sorted(grouped):
        experiment_name, method, inference_config = key
        group = grouped[key]
        rows.append(
            [
                experiment_name,
                method,
                inference_config,
                str(len(group)),
                format_float(safe_mean(item.success_rate for item in group)),
                format_float(safe_mean(item.avg_steps for item in group if item.avg_steps is not None)),
                format_float(safe_mean(item.avg_time for item in group if item.avg_time is not None)),
            ]
        )
    return rows


def build_task_summary_rows(records: list[EvalRecord]) -> list[list[str]]:
    grouped: dict[tuple[str, str, str, str], list[EvalRecord]] = {}
    for record in records:
        key = (
            record.policy_name,
            record.method,
            record.config_name,
            record.inference_config,
        )
        grouped.setdefault(key, []).append(record)

    rows: list[list[str]] = []
    for key in sorted(grouped):
        policy_name, method, config_name, inference_config = key
        group = grouped[key]
        rows.append(
            [
                policy_name,
                method,
                config_name,
                inference_config,
                str(len(group)),
                format_float(safe_mean(item.success_rate for item in group)),
                format_float(safe_mean(item.avg_steps for item in group if item.avg_steps is not None)),
                format_float(safe_mean(item.avg_time for item in group if item.avg_time is not None)),
            ]
        )
    return rows


def render_markdown(records: list[EvalRecord], args: argparse.Namespace) -> str:
    lines = [
        "# Eval Summary",
        "",
        f"- Results root: `{args.results_root}`",
        f"- Matching runs: `{len(records)}`",
        f"- Latest only: `{args.latest_only}`",
        "",
    ]

    if not records:
        lines.extend(["_No matching runs._", ""])
        return "\n".join(lines)

    lines.extend(
        [
            "## Method Summary",
            "",
            markdown_table(
                ["Experiment", "Method", "Inference", "Runs", "Mean SR (%)", "Mean Steps", "Mean Time (s)"],
                build_method_summary_rows(records),
            ),
            "",
            "## Per-Task Summary",
            "",
        ]
    )

    sectioned: dict[tuple[str, str], list[EvalRecord]] = {}
    for record in records:
        key = (record.experiment_name, record.task_name)
        sectioned.setdefault(key, []).append(record)

    for experiment_name, task_name in sorted(sectioned):
        section_records = sorted(sectioned[(experiment_name, task_name)], key=record_sort_key)
        lines.extend(
            [
                f"### Task: {task_name}",
                "",
                f"- Experiment: `{experiment_name}`",
                f"- Matching runs: `{len(section_records)}`",
                "",
                markdown_table(
                    ["Policy", "Method", "Config", "Inference", "Runs", "Mean SR (%)", "Mean Steps", "Mean Time (s)"],
                    build_task_summary_rows(section_records),
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Details",
            "",
            markdown_table(
                [
                    "Experiment",
                    "Task",
                    "Policy",
                    "Method",
                    "Config",
                    "Inference",
                    "Demo",
                    "Train Run",
                    "Eval ID",
                    "Success",
                    "SR (%)",
                    "Avg Steps",
                    "Avg Time (s)",
                    "Status",
                ],
                build_detail_rows(records),
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_csv(records: list[EvalRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    empty = EvalRecord(
        experiment_name="",
        policy_name="",
        method="",
        task_name="",
        config_name="",
        inference_config="",
        demo=None,
        train_run_id="",
        eval_id="",
        success=0,
        total=0,
        success_rate=0.0,
        avg_steps=None,
        avg_time=None,
        success_avg_steps=None,
        success_avg_time=None,
        failed_avg_steps=None,
        failed_avg_time=None,
        status="",
        eval_dir="",
        ckpt_path="",
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(records[0]).keys()) if records else list(asdict(empty).keys()),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_results_root = root / "eval_result"
    default_output = root / "eval_result" / "summary" / "eval_summary.md"

    parser = argparse.ArgumentParser(description="Summarize evaluation runs in the clean UniVTAC layout.")
    parser.add_argument("--results-root", type=Path, default=default_results_root)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown")
    parser.add_argument("--experiment-names", type=str, default=os.environ.get("EXPERIMENT_NAME"))
    parser.add_argument("--policies", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--demos", type=str, default=None)
    parser.add_argument("--config-names", type=str, default=None)
    parser.add_argument("--inference-configs", type=str, default=None)
    parser.add_argument("--task-configs", type=str, default=None, help="Deprecated alias for --inference-configs.")
    parser.add_argument("--train-run-ids", type=str, default=None)
    parser.add_argument("--eval-ids", type=str, default=None)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument(
        "--act-paper-run-patterns",
        type=str,
        default="paper",
        help="Comma-separated substrings used to classify ACT runs as ACT+paper_encoder.",
    )
    parser.add_argument("--print", action="store_true", dest="should_print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_patterns = split_csv(args.act_paper_run_patterns)
    assert paper_patterns is not None
    eval_dirs = discover_eval_dirs(args.results_root)
    records = [
        record
        for eval_dir in eval_dirs
        if (record := load_record(eval_dir, args.results_root, paper_patterns)) is not None
    ]
    records = apply_filters(records, args)
    if args.latest_only:
        records = apply_latest_only(records)
    else:
        records = sorted(records, key=record_sort_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "markdown":
        text = render_markdown(records, args)
        args.output.write_text(text, encoding="utf-8")
        if args.should_print:
            print(text)
        return

    if args.format == "json":
        payload = [asdict(record) for record in records]
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        args.output.write_text(text, encoding="utf-8")
        if args.should_print:
            print(text, end="")
        return

    write_csv(records, args.output)
    if args.should_print:
        print(f"Wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
