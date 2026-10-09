"""Generate synthetic record CSVs for pipeline testing (no API needed).

Writes results/raw/paraphrase_records_<model>.csv for four mock personalities
using the same scoring path as the real benchmark:

    python make_fixture.py --out results/raw

These files exist ONLY to validate analyze.py -> numbers.json -> figures ->
paper. Never publish numbers derived from them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stability_core as sc  # noqa: E402
from dry_run import MOCKS  # noqa: E402


def build_records(model_name: str, model_fn) -> pd.DataFrame:
    cells = sc.build_cells()
    records = []
    for row in cells.itertuples(index=False):
        row_dict = row._asdict()
        opts = tuple(row_dict["options_json"])
        response = model_fn(row_dict)
        rec = sc.make_record(row_dict, response, opts)
        rec["model"] = model_name
        records.append(rec)
    return pd.DataFrame(records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/fixtures")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in MOCKS.items():
        df = build_records(name, fn)
        path = out / f"paraphrase_records_{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
