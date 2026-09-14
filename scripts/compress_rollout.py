#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from univtac.eval.summarize import apply_latest_only, discover_eval_dirs, load_record, record_sort_key, split_csv

VALID_RESULTS = ("success", "failed")
VALID_SELECTORS = ("shortest", "longest")


@dataclass(frozen=True)
class RolloutCandidate:
    experiment_name: str
    task_name: str
    variant: str
    config_name: str
    inference_config: str
    train_run_id: str
    eval_id: str
    seed: str
    result: str
    cost_step: float | None
    cost_time: float | None
    sort_key: str
    sort_value: float
    source_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy shortest/longest success/failed rollout videos for one experiment."
    )
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "eval_result")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sort-key", choices=("cost_step", "cost_time"), default="cost_step")
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Keep every eval run instead of only the latest eval_id per config.",
    )
    parser.add_argument("--act-paper-run-patterns", type=str, default="paper")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    token = re.sub(r"_+", "_", token).strip("._")
    return token or "unknown"


def seed_sort_value(seed: str) -> tuple[int, str]:
    try:
        return 0, f"{int(seed):012d}"
    except ValueError:
        return 1, seed


def find_records(args: argparse.Namespace):
    paper_patterns = split_csv(args.act_paper_run_patterns) or set()
    eval_dirs = discover_eval_dirs(args.results_root)
    records = [
        record
        for eval_dir in eval_dirs
        if (record := load_record(eval_dir, args.results_root, paper_patterns)) is not None
        and record.experiment_name == args.experiment_name
    ]
    if not args.all_runs:
        records = apply_latest_only(records)
    return sorted(records, key=record_sort_key)


def resolve_video_path(video_dir: Path, seed: str, result: str) -> Path | None:
    exact = video_dir / f"{seed}_{result}.mp4"
    if exact.is_file():
        return exact

    matches = sorted(video_dir.glob(f"{seed}_*.mp4"))
    if not matches:
        return None
    for match in matches:
        if result in match.stem:
            return match
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_sort_value(payload: dict, preferred_key: str) -> tuple[str, float] | None:
    fallback_key = "cost_time" if preferred_key == "cost_step" else "cost_step"
    for key in (preferred_key, fallback_key):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return None


def load_candidates(record, sort_key: str) -> tuple[list[RolloutCandidate], list[str]]:
    eval_dir = Path(record.eval_dir)
    metadata_path = eval_dir / "metadata.json"
    payload = json.loads(metadata_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {metadata_path}")

    candidates: list[RolloutCandidate] = []
    warnings: list[str] = []
    video_dir = eval_dir / "video"
    for seed, episode in sorted(payload.items()):
        if not isinstance(episode, dict):
            warnings.append(f"Skip non-dict episode metadata: {metadata_path} seed={seed}")
            continue

        result = str(episode.get("result", ""))
        if result not in VALID_RESULTS:
            warnings.append(f"Skip episode with unknown result: {metadata_path} seed={seed} result={result!r}")
            continue

        sort_info = resolve_sort_value(episode, sort_key)
        if sort_info is None:
            warnings.append(f"Skip episode without cost_step/cost_time: {metadata_path} seed={seed}")
            continue
        actual_sort_key, sort_value = sort_info

        source_path = resolve_video_path(video_dir, str(seed), result)
        if source_path is None:
            warnings.append(f"Skip episode with missing video: {metadata_path} seed={seed} result={result}")
            continue

        cost_step = episode.get("cost_step")
        cost_time = episode.get("cost_time")
        candidates.append(
            RolloutCandidate(
                experiment_name=record.experiment_name,
                task_name=record.task_name,
                variant=record.policy_name,
                config_name=record.config_name,
                inference_config=record.inference_config,
                train_run_id=record.train_run_id,
                eval_id=record.eval_id,
                seed=str(seed),
                result=result,
                cost_step=float(cost_step) if isinstance(cost_step, (int, float)) else None,
                cost_time=float(cost_time) if isinstance(cost_time, (int, float)) else None,
                sort_key=actual_sort_key,
                sort_value=sort_value,
                source_path=source_path,
            )
        )

    return candidates, warnings


def assign_base_names(records) -> dict:
    base_names = {
        record: safe_token(
            f"{record.task_name}_{record.policy_name}_{record.config_name}_{record.inference_config}"
        )
        for record in records
    }

    counts: dict[str, int] = {}
    for base in base_names.values():
        counts[base] = counts.get(base, 0) + 1
    for record, base in list(base_names.items()):
        if counts[base] > 1:
            base_names[record] = safe_token(f"{base}_{record.train_run_id}")

    counts.clear()
    for base in base_names.values():
        counts[base] = counts.get(base, 0) + 1
    for record, base in list(base_names.items()):
        if counts[base] > 1:
            base_names[record] = safe_token(f"{base}_{record.eval_id}")

    counts.clear()
    for base in base_names.values():
        counts[base] = counts.get(base, 0) + 1
    dedupe_index: dict[str, int] = {}
    for record, base in list(base_names.items()):
        if counts[base] <= 1:
            continue
        dedupe_index[base] = dedupe_index.get(base, 0) + 1
        base_names[record] = safe_token(f"{base}_{dedupe_index[base]}")
    return base_names


def choose_candidate(candidates: list[RolloutCandidate], selector: str) -> RolloutCandidate | None:
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda item: (item.sort_value, seed_sort_value(item.seed)))
    if selector == "shortest":
        return ordered[0]
    if selector == "longest":
        return ordered[-1]
    raise ValueError(f"Unknown selector: {selector}")


def cleanup_from_manifest(output_dir: Path) -> int:
    manifest_path = output_dir / "manifest.csv"
    if not manifest_path.is_file():
        return 0

    removed = 0
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            copied_to = row.get("copied_to", "").strip()
            if not copied_to:
                continue
            target = Path(copied_to)
            try:
                target.relative_to(output_dir)
            except ValueError:
                continue
            if target.is_file():
                target.unlink()
                removed += 1
    return removed


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    fieldnames = [
        "experiment_name",
        "task_name",
        "variant",
        "config_name",
        "train_run_id",
        "inference_config",
        "eval_id",
        "result",
        "selector",
        "seed",
        "cost_step",
        "cost_time",
        "sort_key",
        "sort_value",
        "source_path",
        "copied_to",
        "note",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.results_root / args.experiment_name / "rollouts_summary")
    records = find_records(args)
    if not records:
        raise SystemExit(
            f"No eval runs found for experiment {args.experiment_name!r} under {args.results_root}"
        )

    base_names = assign_base_names(records)
    warnings: list[str] = []
    manifest_rows: list[dict[str, str]] = []
    selected_count = 0
    missing_slots = 0

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        removed = cleanup_from_manifest(output_dir)
    else:
        removed = 0

    for record in records:
        candidates, candidate_warnings = load_candidates(record, args.sort_key)
        warnings.extend(candidate_warnings)
        grouped = {
            result: [candidate for candidate in candidates if candidate.result == result]
            for result in VALID_RESULTS
        }

        for result in VALID_RESULTS:
            for selector in VALID_SELECTORS:
                chosen = choose_candidate(grouped[result], selector)
                if chosen is None:
                    missing_slots += 1
                    manifest_rows.append(
                        {
                            "experiment_name": record.experiment_name,
                            "task_name": record.task_name,
                            "variant": record.policy_name,
                            "config_name": record.config_name,
                            "train_run_id": record.train_run_id,
                            "inference_config": record.inference_config,
                            "eval_id": record.eval_id,
                            "result": result,
                            "selector": selector,
                            "seed": "",
                            "cost_step": "",
                            "cost_time": "",
                            "sort_key": args.sort_key,
                            "sort_value": "",
                            "source_path": "",
                            "copied_to": "",
                            "note": "no available rollout for this result",
                        }
                    )
                    continue

                target_name = safe_token(
                    f"{base_names[record]}_{result}_{selector}_seed{chosen.seed}.mp4"
                )
                target_path = output_dir / target_name
                if not args.dry_run:
                    shutil.copy2(chosen.source_path, target_path)

                selected_count += 1
                manifest_rows.append(
                    {
                        "experiment_name": chosen.experiment_name,
                        "task_name": chosen.task_name,
                        "variant": chosen.variant,
                        "config_name": chosen.config_name,
                        "train_run_id": chosen.train_run_id,
                        "inference_config": chosen.inference_config,
                        "eval_id": chosen.eval_id,
                        "result": chosen.result,
                        "selector": selector,
                        "seed": chosen.seed,
                        "cost_step": "" if chosen.cost_step is None else f"{chosen.cost_step:.6f}",
                        "cost_time": "" if chosen.cost_time is None else f"{chosen.cost_time:.6f}",
                        "sort_key": chosen.sort_key,
                        "sort_value": f"{chosen.sort_value:.6f}",
                        "source_path": str(chosen.source_path),
                        "copied_to": str(target_path),
                        "note": "",
                    }
                )

    if not args.dry_run:
        write_manifest(manifest_rows, output_dir / "manifest.csv")

    print(
        f"Processed {len(records)} eval groups, selected {selected_count} rollout files, "
        f"missing {missing_slots} slots, cleaned {removed} previous files."
    )
    if args.dry_run:
        print(f"Dry run only. Nothing copied. Planned output dir: {output_dir}")
    else:
        print(f"Wrote rollout summary to {output_dir}")
        print(f"Manifest: {output_dir / 'manifest.csv'}")

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings[:20]:
            print(f"  - {warning}")
        if len(warnings) > 20:
            print(f"  - ... {len(warnings) - 20} more warnings omitted")


if __name__ == "__main__":
    main()
