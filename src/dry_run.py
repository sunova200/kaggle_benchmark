"""Dry-run harness: simulate full benchmark runs with mock models.

Four synthetic model personalities exercise every code path the real run
will take (parsing, flips, unparsable cells, position bias) without any API:

- oracle: always answers correctly with a bare letter.
- letter_banker: always answers "A" (tests position/gold-mapping logic).
- phrasing_weathervane: correct on canonical/terse, flips to a wrong option
  when the stem is reworded or verbose (a "prompt instability" model).
- chatty_refuser: sometimes replies with unparsable prose (parse-rate path).

Run:  python dry_run.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stability_core as sc


def run_mock(model_fn, cells=None) -> dict:
    """Run the full grid through a mock model and aggregate metrics."""
    cells = cells if cells is not None else sc.build_cells()
    records = []
    for row in cells.itertuples(index=False):
        row_dict = row._asdict()
        opts = tuple(row_dict["options_json"])
        response = model_fn(row_dict)
        records.append(sc.make_record(row_dict, response, opts))
    return sc.compute_metrics(records)


# ---------- mock model personalities ----------
def oracle_model(row: dict) -> str:
    idx = tuple(row["options_json"]).index(row["correct_option"])
    return sc.LETTERS[idx]


def letter_banker_model(row: dict) -> str:
    return "A"


def phrasing_weathervane_model(row: dict) -> str:
    idx = tuple(row["options_json"]).index(row["correct_option"])
    if row["variant"] in ("reworded", "verbose"):
        return sc.LETTERS[(idx + 1) % 4]  # wrong answer under rephrasing
    return sc.LETTERS[idx]


def chatty_refuser_model(row: dict) -> str:
    idx = tuple(row["options_json"]).index(row["correct_option"])
    # Deterministic "mood": refuse every 7th cell based on seed+repeat.
    if (row["seed"] + row["repeat"] + hash(row["item_id"]) ) % 7 == 0:
        return "I would need more context to answer reliably."
    return f"The answer is {sc.LETTERS[idx]}"


MOCKS = {
    "oracle": oracle_model,
    "letter_banker": letter_banker_model,
    "phrasing_weathervane": phrasing_weathervane_model,
    "chatty_refuser": chatty_refuser_model,
}


def main() -> None:
    problems = sc.validate_answer_key()
    if problems:
        print("ANSWER KEY PROBLEMS:", problems)
        raise SystemExit(1)
    cells = sc.build_cells()
    print(f"grid: {len(cells)} cells "
          f"({cells['item_id'].nunique()} items x {len(sc.VARIANTS)} variants x {sc.N_REPEATS} repeats)\n")
    for name, fn in MOCKS.items():
        m = run_mock(fn, cells)
        print(f"--- {name} ---")
        print(sc.format_report(m))
        print()
        # Invariants that must hold for every model personality:
        assert m["n_cells"] == len(cells)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["saa"] <= 1.0
        assert 0.0 <= m["flip_rate"] <= 1.0
        assert m["accuracy_ci"][0] <= m["accuracy"] <= m["accuracy_ci"][1]
    # Spot-check the designed behaviors:
    mo = run_mock(MOCKS["oracle"], cells)
    assert mo["accuracy"] == 1.0 and mo["flip_rate"] == 0.0, "oracle must be perfect"
    mv = run_mock(MOCKS["phrasing_weathervane"], cells)
    assert mv["flip_rate"] > 0.5, "weathervane should flip on most items"
    assert mv["prompt_instability"] > 0.5, "weathervane instability is prompt-driven"
    mb = run_mock(MOCKS["letter_banker"], cells)
    assert mb["accuracy"] < 0.5, "always-A should score below chance"
    mr = run_mock(MOCKS["chatty_refuser"], cells)
    assert mr["parse_rate"] < 1.0, "chatty model should have unparsed cells"
    assert mr["saa"] < 1.0, "unparsed cells must hurt SAA"
    print("all dry-run invariants passed")


if __name__ == "__main__":
    main()