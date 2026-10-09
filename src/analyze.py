"""Aggregate per-model record CSVs into numbers.json + summary tables.

Single source of truth: numbers.json feeds BOTH the DEV post generator and
the paper builder (same pattern as the reference implementation), so the two
can never disagree.

    python analyze.py --records-dir results/raw --out numbers.json

Input: paraphrase_records_<model>.csv files (written by the Kaggle task).
Output: numbers.json with per-model metrics + cross-model aggregates,
        plus a markdown summary printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stability_core as sc  # noqa: E402

RECORD_COLUMNS = [
    "item_id", "category", "variant", "repeat", "seed", "response",
    "chosen_letter", "chosen_option", "correct_letter", "correct_option",
    "is_correct", "parse_ok", "model",
]


def load_records(records_dir: Path) -> dict[str, pd.DataFrame]:
    """Group rows of all paraphrase_records_*.csv (searched recursively) by model.

    Real downloads nest files as <task>/<version>/<model>/<run_id>/. If a
    model appears in multiple run directories (retries), the richest CSV wins
    — rows are never concatenated across runs, which would corrupt the
    10-cells-per-item denominators. Empty CSVs (errored runs) are skipped.
    """
    files = sorted(records_dir.rglob("paraphrase_records_*.csv"))
    if not files:
        raise SystemExit(f"no paraphrase_records_*.csv found under {records_dir}")
    best: dict[str, pd.DataFrame] = {}
    for f in files:
        try:
            df = pd.read_csv(f)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            print(f"note: skipping unreadable CSV (errored run): {f.name}")
            continue
        if df.empty:
            print(f"note: skipping empty CSV (errored run): {f.name}")
            continue
        missing = [c for c in RECORD_COLUMNS if c not in df.columns]
        if missing:
            raise SystemExit(f"{f}: missing columns {missing}")
        col_model = df["model"].iloc[0] if "model" in df.columns else None
        file_model = f.stem.replace("paraphrase_records_", "")
        model = str(col_model) if isinstance(col_model, str) and col_model else file_model
        prev = best.get(model)
        if prev is None or len(df) > len(prev):
            if prev is not None:
                print(f"note: {model}: multiple runs found, keeping richest "
                      f"({len(df)} vs {len(prev)} rows)")
            best[model] = df
    if not best:
        raise SystemExit(f"all record CSVs under {records_dir} were empty (errored runs)")
    return best


def coerce_bools(df: pd.DataFrame) -> pd.DataFrame:
    """CSV round-trips booleans as strings sometimes; normalize."""
    for col in ("is_correct", "parse_ok"):
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().isin(("true", "1"))
    return df


def metrics_for_model(df: pd.DataFrame) -> dict:
    return sc.compute_metrics(coerce_bools(df.copy()).to_dict("records"))


def extract_metrics_from_logs(root: Path) -> dict[str, dict]:
    """Fallback: pull METRICS_JSON= payloads out of downloaded run logs.

    The Kaggle task prints one METRICS_JSON= line per completed run. This
    recovers every figure/table number even when record CSVs are missing from
    the download. Richest payload wins per model (retry runs duplicate logs).
    """
    found: dict[str, dict] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() == ".csv":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            idx = line.find("METRICS_JSON=")
            if idx < 0:
                continue
            blob = line[idx + len("METRICS_JSON="):]
            start = blob.find("{")
            if start < 0:
                continue
            try:
                # raw_decode: parse the first complete JSON object and ignore
                # any trailing junk (quotes, ANSI codes, wrapping log formats).
                m, _ = json.JSONDecoder().raw_decode(blob[start:])
            except json.JSONDecodeError:
                continue
            if "saa" not in m:
                continue  # partial/aborted payload — never fabricate zeros
            model = str(m.get("model") or "unknown")
            if model not in found or m.get("n_cells", 0) > found[model].get("n_cells", 0):
                found[model] = m
    return found


def leaderboard_row(model: str, m: dict) -> dict:
    lo, hi = m.get("saa_ci", (float("nan"),) * 2)
    expected = m.get("expected_cells", 320)
    n_cells = m.get("n_cells", 0)
    return {
        "model": model,
        "saa": round(m.get("saa", 0.0), 4),
        "saa_ci_half": round((hi - lo) / 2.0, 4) if hi == hi else None,
        "accuracy": round(m.get("accuracy", 0.0), 4),
        "accuracy_ci": [round(m["accuracy_ci"][0], 4), round(m["accuracy_ci"][1], 4)]
        if "accuracy_ci" in m else None,
        "flip_rate": round(m.get("flip_rate", 0.0), 4),
        "run_instability": round(m.get("run_instability", 0.0), 4),
        "prompt_instability": round(m.get("prompt_instability", 0.0), 4),
        "parse_rate": round(m.get("parse_rate", 0.0), 4),
        "n_cells": n_cells,
        "expected_cells": expected,
        # Partial runs (API cells lost) stay in the numbers but are flagged so
        # the post/paper never quotes them as if they were full grids.
        "complete": bool(n_cells >= 0.9 * expected),
        "n_saa_scored": m.get("n_saa_scored", 0),
        "n_saa_excluded_incomplete": m.get("n_saa_excluded_incomplete", 0),
        "per_variant_accuracy": {
            v: round(d["accuracy"], 4) for v, d in m.get("per_variant", {}).items()
        },
        "per_category_accuracy": {
            c: round(d["accuracy"], 4) for c, d in m.get("per_category", {}).items()
        },
        "shuffled_position_accuracy": {
            k: (round(v, 4) if v is not None else None)
            for k, v in m.get("shuffled_position_accuracy", {}).items()
        },
        "flip_items": m.get("flip_items", []),
    }

# %% PART2
def cross_model_aggregates(rows: list[dict]) -> dict:
    if not rows:
        return {}
    best = max(rows, key=lambda r: r["saa"])
    worst_flip = max(rows, key=lambda r: r["flip_rate"])
    by_acc = sorted(rows, key=lambda r: r["accuracy"], reverse=True)
    return {
        "n_models": len(rows),
        "total_cells": sum(r["n_cells"] for r in rows),
        "best_saa_model": best["model"],
        "best_saa": best["saa"],
        "highest_flip_model": worst_flip["model"],
        "highest_flip_rate": worst_flip["flip_rate"],
        "accuracy_spread": round(by_acc[0]["accuracy"] - by_acc[-1]["accuracy"], 4),
        "mean_run_instability": round(
            sum(r["run_instability"] for r in rows) / len(rows), 4),
        "mean_prompt_instability": round(
            sum(r["prompt_instability"] for r in rows) / len(rows), 4),
        "rank_changes_accuracy_to_saa": _rank_changes(rows),
    }


def _rank_changes(rows: list[dict]) -> list[dict]:
    """Models whose rank differs between accuracy and SAA ordering."""
    by_acc = sorted(rows, key=lambda r: r["accuracy"], reverse=True)
    by_saa = sorted(rows, key=lambda r: r["saa"], reverse=True)
    acc_rank = {r["model"]: i for i, r in enumerate(by_acc)}
    saa_rank = {r["model"]: i for i, r in enumerate(by_saa)}
    moved = []
    for r in rows:
        delta = acc_rank[r["model"]] - saa_rank[r["model"]]
        if delta != 0:
            moved.append({
                "model": r["model"],
                "accuracy_rank": acc_rank[r["model"]] + 1,
                "saa_rank": saa_rank[r["model"]] + 1,
                "delta": delta,
            })
    return sorted(moved, key=lambda x: -abs(x["delta"]))


def md_summary(numbers: dict) -> str:
    lines = ["| Model | SAA | Acc | Flip | Prompt-inst | Parse |",
             "|---|---|---|---|---|---|"]
    for r in numbers["leaderboard"]:
        lines.append(
            f"| {r['model']} | {r['saa']:.3f} | {r['accuracy']:.3f} "
            f"| {r['flip_rate']:.3f} | {r['prompt_instability']:.3f} "
            f"| {r['parse_rate']:.3f} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records-dir", default="results/raw")
    ap.add_argument("--out", default="numbers.json")
    args = ap.parse_args(argv)

    root = Path(args.records_dir)
    rows = []
    full_metrics: dict[str, dict] = {}

    # Primary: per-cell record CSVs (full recomputation path).
    try:
        by_model = load_records(root)
    except SystemExit:
        by_model = {}
    for model, df in sorted(by_model.items()):
        m = metrics_for_model(df)
        full_metrics[model] = m
        rows.append(leaderboard_row(model, m))

    # Fallback: METRICS_JSON= lines in run logs (covers models whose CSV
    # didn't survive the download; both sources may mix safely).
    logged = extract_metrics_from_logs(root)
    for model, m in sorted(logged.items()):
        if model in full_metrics:
            continue
        print(f"note: {model} recovered from run logs (no record CSV)")
        full_metrics[model] = m
        rows.append(leaderboard_row(model, m))

    if not rows:
        raise SystemExit(
            f"no results found in {root}: expected paraphrase_records_*.csv "
            "files or METRICS_JSON= lines in run logs")

    numbers = {
        "benchmark": {
            "n_items": len(sc.ITEMS),
            "variants": list(sc.VARIANTS),
            "n_repeats": sc.N_REPEATS,
            "cells_per_model": len(sc.VARIANTS) * sc.N_REPEATS * len(sc.ITEMS),
        },
        "leaderboard": rows,
        "aggregates": cross_model_aggregates(rows),
        "metrics": full_metrics,
    }
    Path(args.out).write_text(
        json.dumps(numbers, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} models)\n")
    print(md_summary(numbers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

