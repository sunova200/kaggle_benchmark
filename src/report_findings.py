"""Print a compact findings summary from numbers.json (for post drafting)."""

import json
from pathlib import Path

d = json.loads(Path("numbers.json").read_text(encoding="utf-8"))
a = d["aggregates"]
rows = d["leaderboard"]

print("== aggregates ==")
print(json.dumps(a, indent=1))

print("\n== accuracy vs SAA gap (biggest stability losses first) ==")
for r in sorted(rows, key=lambda x: -(x["accuracy"] - x["saa"]))[:10]:
    flag = "" if r["complete"] else "  PARTIAL"
    print(f"  {r['model']:<45} acc {r['accuracy']:.3f}  saa {r['saa']:.3f}  "
          f"gap {r['accuracy'] - r['saa']:+.3f}  cells {r['n_cells']}{flag}")

print("\n== families (regeneration check) ==")
for fam in ("claude-sonnet", "claude-opus", "gpt-5", "gpt-6", "gemini-3.",
            "gemini-2.5", "qwen3", "grok"):
    line = [r for r in rows if fam in r["model"]]
    for r in sorted(line, key=lambda x: x["model"]):
        print(f"  {r['model']:<45} saa {r['saa']:.3f}  acc {r['accuracy']:.3f}  "
              f"flip {r['flip_rate']:.3f}")

print("\n== worst instability ==")
for r in sorted(rows, key=lambda x: -x["prompt_instability"])[:6]:
    print(f"  {r['model']:<45} prompt-inst {r['prompt_instability']:.3f}  "
          f"run-inst {r['run_instability']:.3f}")

print("\n== partial runs (n_cells < 288) ==")
for r in rows:
    if not r["complete"]:
        print(f"  {r['model']:<45} cells {r['n_cells']}  "
              f"saa-scored-items {r['n_saa_scored']}  "
              f"excluded {r['n_saa_excluded_incomplete']}")

print("\n== position bias (shuffled, spread max-min) ==")
for r in sorted(rows, key=lambda x: -(
        max([v for v in x["shuffled_position_accuracy"].values() if v is not None] or [0])
        - min([v for v in x["shuffled_position_accuracy"].values() if v is not None] or [1])))[:5]:
    pos = {k: round(v, 2) for k, v in r["shuffled_position_accuracy"].items()
           if v is not None}
    vals = list(pos.values())
    if vals:
        print(f"  {r['model']:<45} spread {max(vals)-min(vals):.2f}  {pos}")
