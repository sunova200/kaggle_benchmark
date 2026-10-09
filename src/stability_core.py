"""Paraphrase Stability Benchmark - core logic (pure, dependency-light, testable).

The benchmark asks each item in multiple semantically-equivalent phrasings and
measures whether a model's answer survives the rephrasing.

Design principles:
- Deterministic scoring (no LLM judge): extract the chosen option letter.
- Every item has exactly one objectively correct answer (validated programmatically
  where computable).
- Statistical rigor is a feature: cluster-bootstrap confidence intervals over items,
  run-vs-prompt instability decomposition, per-variant breakdowns.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import pandas as pd

VARIANTS = ("canonical", "reworded", "terse", "verbose", "shuffled")
N_REPEATS = 2
SEEDS = (11, 29)  # per-repeat seeds; temperature stays 0 (deterministic setting)
LETTERS = "ABCD"

# %% PART2
@dataclass(frozen=True)
class Item:
    id: str
    category: str  # logic | quantitative | knowledge | attention
    stem: str
    stem_reworded: str
    options: tuple[str, ...]  # canonical display order (4 options)
    answer: str  # option text (identity, independent of display order)


ITEMS: tuple[Item, ...] = (
    # ---- Logic (8) ----
    Item(
        "L1", "logic",
        "All bloops are razzies. All razzies are lazzies. Which of the following must be true?",
        "Every bloop is a razzie, and every razzie is a lazzie. Which conclusion is necessarily correct?",
        ("All bloops are lazzies", "All lazzies are bloops", "Some bloops are not lazzies", "No bloop is a lazzie"),
        "All bloops are lazzies",
    ),
    Item(
        "L2", "logic",
        "Ada is taller than Ben. Ben is taller than Cara. Cara is taller than Dana. Who is the shortest of the four?",
        "Ada stands taller than Ben, who is taller than Cara, and Cara is taller than Dana. Which of these people is the shortest?",
        ("Ada", "Ben", "Cara", "Dana"),
        "Dana",
    ),
    Item(
        "L3", "logic",
        "On an island, knights always tell the truth and knaves always lie. You meet A, who says: 'We are both knaves.' What is A?",
        "Knights on this island always tell the truth; knaves always lie. A tells you: 'Both of us are knaves.' What is A?",
        ("A is a knight", "A is a knave", "A is a knight and B is a knave", "It cannot be determined"),
        "A is a knave",
    ),
    Item(
        "L4", "logic",
        "If it rained yesterday, the ground is wet. The ground is wet. What can you conclude?",
        "Whenever it rains, the ground ends up wet. Right now the ground is wet. What follows with certainty?",
        ("It rained yesterday", "It may or may not have rained", "It did not rain", "The ground will dry today"),
        "It may or may not have rained",
    ),
    Item(
        "L5", "logic",
        "Today is Wednesday. What day of the week will it be in 10 days?",
        "It is Wednesday today. Which weekday falls exactly ten days from now?",
        ("Monday", "Friday", "Saturday", "Sunday"),
        "Saturday",
    ),
    Item(
        "L6", "logic",
        "No scientists are artists. All doctors are scientists. Which of the following must be true?",
        "There are no scientists who are artists, and every doctor is a scientist. Which statement has to hold?",
        ("No doctors are artists", "All artists are doctors", "Some doctors are artists", "No scientists are doctors"),
        "No doctors are artists",
    ),
    Item(
        "L7", "logic",
        "If the alarm rings, the battery is low. The battery is not low. What can you conclude?",
        "The alarm rings only when the battery is low. The battery is fine. What follows?",
        ("The alarm rang", "The alarm did not ring", "The battery is charging", "Nothing can be concluded"),
        "The alarm did not ring",
    ),
    Item(
        "L8", "logic",
        "Tom is to the left of Sam. Priya is to the right of Sam. Assuming all three stand in a row, who is in the middle?",
        "In a row of three, Tom stands left of Sam and Priya stands right of Sam. Who occupies the middle position?",
        ("Tom", "Sam", "Priya", "Cannot be determined"),
        "Sam",
    ),
    # ---- Quantitative (8) ----
    Item(
        "Q1", "quantitative",
        "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "Together, a bat and ball sell for $1.10. The bat is one dollar more expensive than the ball. What is the ball's price?",
        ("$0.01", "$0.05", "$0.10", "$0.11"),
        "$0.05",
    ),
    Item(
        "Q2", "quantitative",
        "A jacket is on sale for $80 after a 20% discount. What was the original price?",
        "After 20% is slashed off, a jacket costs $80. What did it cost before the discount?",
        ("$96", "$100", "$104", "$120"),
        "$100",
    ),
    Item(
        "Q3", "quantitative",
        "If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to make 100 widgets?",
        "Five machines need five minutes to produce five widgets. At the same rate, how many minutes will one hundred machines need for one hundred widgets?",
        ("5 minutes", "20 minutes", "100 minutes", "500 minutes"),
        "5 minutes",
    ),
    Item(
        "Q4", "quantitative",
        "What is the value of 15 - 3 * 4 + 2?",
        "Compute 15 minus 3 times 4, plus 2.",
        ("2", "5", "20", "32"),
        "5",
    ),
    Item(
        "Q5", "quantitative",
        "The average of five numbers is 20. If the number 50 is removed, what is the average of the remaining four?",
        "Five numbers average to 20. Take out the number 50 - what do the other four average?",
        ("10", "12.5", "15", "25"),
        "12.5",
    ),
    Item(
        "Q6", "quantitative",
        "A price rises by 50% and then falls by 50%. Compared to the original price, what is the final price?",
        "A price goes up by half, then down by half. Relative to where it started, where does it end up?",
        ("unchanged", "10% lower", "25% lower", "50% lower"),
        "25% lower",
    ),
    Item(
        "Q7", "quantitative",
        "Apples cost $2 each. Ana buys 5 apples and pays with a $20 bill. How much change does she get?",
        "At two dollars per apple, Ana purchases five and hands over a twenty-dollar note. What change should she receive?",
        ("$4", "$8", "$10", "$15"),
        "$10",
    ),
    Item(
        "Q8", "quantitative",
        "In a jar, red marbles are three times as many as blue marbles. There are 60 marbles in total. How many blue marbles are there?",
        "Red marbles outnumber blue ones three to one in a jar of 60 marbles. How many blue marbles does it contain?",
        ("12", "15", "20", "45"),
        "15",
    ),
    # ---- Knowledge (8) ----
    Item(
        "K1", "knowledge",
        "What is the chemical symbol for gold?",
        "Gold is denoted by which chemical symbol?",
        ("Ag", "Au", "Gd", "Pt"),
        "Au",
    ),
    Item(
        "K2", "knowledge",
        "How many bones are in the adult human body?",
        "An average grown-up human skeleton contains how many bones?",
        ("186", "206", "226", "256"),
        "206",
    ),
    Item(
        "K3", "knowledge",
        "Which country has the largest population?",
        "As of now, which nation is the most populous on Earth?",
        ("China", "India", "United States", "Indonesia"),
        "India",
    ),
    Item(
        "K4", "knowledge",
        "Who wrote the novel One Hundred Years of Solitude?",
        "The novel One Hundred Years of Solitude was written by which author?",
        ("Jorge Luis Borges", "Gabriel Garcia Marquez", "Mario Vargas Llosa", "Julio Cortazar"),
        "Gabriel Garcia Marquez",
    ),
    Item(
        "K5", "knowledge",
        "What is the capital of Australia?",
        "Which city serves as the capital of Australia?",
        ("Sydney", "Melbourne", "Canberra", "Perth"),
        "Canberra",
    ),
    Item(
        "K6", "knowledge",
        "In which year did the Berlin Wall fall?",
        "The fall of the Berlin Wall took place in which year?",
        ("1985", "1987", "1989", "1991"),
        "1989",
    ),
    Item(
        "K7", "knowledge",
        "Which planet in our solar system has the most confirmed moons?",
        "Which of these planets is known to have the greatest number of confirmed moons?",
        ("Jupiter", "Saturn", "Uranus", "Neptune"),
        "Saturn",
    ),
    Item(
        "K8", "knowledge",
        "What is the largest ocean on Earth?",
        "Which of the world's oceans covers the greatest area?",
        ("Atlantic Ocean", "Indian Ocean", "Pacific Ocean", "Arctic Ocean"),
        "Pacific Ocean",
    ),
    # ---- Attention / trap (8) ----
    Item(
        "T1", "attention",
        "How many months of the year have at least 28 days?",
        "Count the months that contain 28 days or more. How many are there?",
        ("1", "2", "11", "12"),
        "12",
    ),
    Item(
        "T2", "attention",
        "A plane crashes exactly on the border of the United States and Canada. Where do they bury the survivors?",
        "An aircraft comes down right on the US-Canada border. In which country are the survivors buried?",
        ("United States", "Canada", "Both countries", "Survivors are not buried"),
        "Survivors are not buried",
    ),
    Item(
        "T3", "attention",
        "You have three apples and you take away two. How many apples do you have?",
        "You own three apples but take two of them away. How many apples do you now have?",
        ("1", "2", "3", "5"),
        "2",
    ),
    Item(
        "T4", "attention",
        "Which weighs more: a pound of feathers or a pound of steel?",
        "Between a pound of feathers and a pound of steel, which one is heavier?",
        ("The feathers", "The steel", "They weigh the same", "It depends on the material"),
        "They weigh the same",
    ),
    Item(
        "T5", "attention",
        "A farmer had 17 sheep. All but 9 died. How many sheep does he have left?",
        "Out of a farmer's 17 sheep, all but nine die. How many sheep remain?",
        ("0", "8", "9", "17"),
        "9",
    ),
    Item(
        "T6", "attention",
        "You are in a race and overtake the person in second place. What position are you in now?",
        "While racing, you pass the runner who is currently in second place. Which place are you now in?",
        ("First", "Second", "Third", "Cannot be determined"),
        "Second",
    ),
    Item(
        "T7", "attention",
        "A red house is made of red bricks and a blue house of blue bricks. What is a green house made of?",
        "If red houses use red bricks and blue houses use blue bricks, what material is a green house built from?",
        ("Green bricks", "Wood", "Glass", "Concrete"),
        "Glass",
    ),
    Item(
        "T8", "attention",
        "How many of each animal did Moses take on the ark?",
        "How many animals of every kind did Moses bring onto the ark?",
        ("Two", "Seven pairs", "None", "Unknown"),
        "None",
    ),
)

# %% PART4
INSTRUCTIONS = {
    "canonical": "Select the correct answer and reply with just its letter (A, B, C, or D).",
    "reworded": "Which option is correct? Respond with a single letter only.",
    "terse": "Answer:",
    "verbose": (
        "Thanks for helping me out! Please read the question and the four choices "
        "carefully, then tell me which one you think is the right answer by replying "
        "with only the corresponding letter (A, B, C, or D) and nothing else."
    ),
    "shuffled": "Select the correct answer and reply with just its letter (A, B, C, or D).",
}


def display_order(item: Item, variant: str) -> tuple[str, ...]:
    """Return the option texts in the order they will be displayed for this variant."""
    if variant != "shuffled":
        return item.options
    rng = random.Random(f"{item.id}-shuffle")
    opts = list(item.options)
    rng.shuffle(opts)
    # Guard: never let the shuffle leave the order unchanged (would silently
    # turn the position-bias probe into a duplicate of canonical).
    if opts == list(item.options):
        opts = opts[1:] + opts[:1]
    return tuple(opts)


def build_prompt(item: Item, variant: str) -> tuple[str, tuple[str, ...], str]:
    """Build (prompt_text, display_options, correct_letter) for one variant.

    correct_letter refers to the DISPLAY position of the gold option for this
    variant (it differs between canonical and shuffled).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    opts = display_order(item, variant)
    stem = item.stem_reworded if variant == "reworded" else item.stem
    lines = [stem, ""]
    for letter, opt in zip(LETTERS, opts):
        lines.append(f"{letter}. {opt}")
    lines.append("")
    lines.append(INSTRUCTIONS[variant])
    correct_letter = LETTERS[opts.index(item.answer)]
    return "\n".join(lines), opts, correct_letter


def extract_letter(response: str, display_options: tuple[str, ...]) -> str | None:
    """Map a raw model response to one of A-D, or None if unparsable.

    Deterministic, judge-free. Order of attempts:
    1. Bare letter (with optional markdown/punctuation): 'B', '**B**', 'B.'.
    2. 'answer is C' / 'option B' style patterns.
    3. First standalone A-D token in the response.
    4. Response matches exactly one option's text verbatim (case-insensitive).
    """
    if response is None:
        return None
    text = str(response).strip()
    if not text:
        return None
    if re.fullmatch(r"[\*\_`\[\(\s]*[A-Da-d][\]\)\.\:\*\_`\s]*", text):
        return re.search(r"[A-Da-d]", text).group(0).upper()
    m = re.search(r"(?:answer|option|choice)\s*(?:is|:|would be)?\s*[\*\_`]*\[?([A-D])\]?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1).upper()
    lowered = text.lower().strip(".! \n")
    for letter, opt in zip(LETTERS, display_options):
        if lowered == opt.lower():
            return letter
    return None


def build_cells() -> "pd.DataFrame":
    """One row per (item, variant, repeat) evaluation cell."""
    rows = []
    for item in ITEMS:
        for variant in VARIANTS:
            prompt, opts, correct_letter = build_prompt(item, variant)
            for rep, seed in enumerate(SEEDS):
                rows.append(
                    {
                        "item_id": item.id,
                        "category": item.category,
                        "variant": variant,
                        "repeat": rep,
                        "seed": seed,
                        "prompt": prompt,
                        "options_json": list(opts),
                        "correct_letter": correct_letter,
                        "correct_option": item.answer,
                    }
                )
    return pd.DataFrame(rows)


# %% PART5
def validate_answer_key() -> list[str]:
    """Programmatic sanity checks on the dataset. Returns problems (empty = OK).

    Catches the classic 'benchmark with a wrong answer key measures its author'
    failure mode before any model is called.
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    expected_counts = {"logic": 8, "quantitative": 8, "knowledge": 8, "attention": 8}
    cat_counts: dict[str, int] = {}
    for item in ITEMS:
        if item.id in seen_ids:
            problems.append(f"duplicate id {item.id}")
        seen_ids.add(item.id)
        cat_counts[item.category] = cat_counts.get(item.category, 0) + 1
        if len(item.options) != 4:
            problems.append(f"{item.id}: needs exactly 4 options, has {len(item.options)}")
        if len(set(item.options)) != 4:
            problems.append(f"{item.id}: duplicate option texts")
        if item.answer not in item.options:
            problems.append(f"{item.id}: answer not among options")
        if item.stem == item.stem_reworded:
            problems.append(f"{item.id}: reworded stem identical to canonical")
        if not item.stem.strip().endswith("?"):
            problems.append(f"{item.id}: canonical stem does not end with '?'")
    for cat, n in expected_counts.items():
        if cat_counts.get(cat, 0) != n:
            problems.append(f"category {cat}: expected {n} items, got {cat_counts.get(cat, 0)}")
    if len(ITEMS) != 32:
        problems.append(f"expected 32 items, got {len(ITEMS)}")
    by_id = {i.id: i for i in ITEMS}

    def key(item_id: str, expected: str) -> None:
        if by_id[item_id].answer != expected:
            problems.append(f"{item_id}: stored key {by_id[item_id].answer!r} != computed {expected!r}")

    key("Q1", "$0.05")   # bat+ball: b + (b+1) = 1.10
    key("Q2", "$100")    # 80 / 0.8
    key("Q3", "5 minutes")
    key("Q4", "5")       # 15 - 12 + 2
    key("Q5", "12.5")    # (5*20 - 50) / 4
    key("Q6", "25% lower")  # 1.5 * 0.5 = 0.75
    key("Q7", "$10")     # 20 - 2*5
    key("Q8", "15")      # 4x = 60
    key("L5", "Saturday")   # Wed + 10 days
    if (15 - 3 * 4 + 2) != 5:
        problems.append("internal arithmetic guard failed for Q4")
    if 80 / 0.8 != 100.0:
        problems.append("internal arithmetic guard failed for Q2")
    if abs(1.5 * 0.5 - 0.75) > 1e-12:
        problems.append("internal arithmetic guard failed for Q6")
    return problems


def bootstrap_ci(values_by_item: dict[str, float], n_boot: int = 2000, seed: int = 1234,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Cluster bootstrap CI over items (resamples items, not cells)."""
    vals = list(values_by_item.values())
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(int((1 - alpha / 2) * n_boot), n_boot - 1)
    return (means[lo_idx], means[hi_idx])


# %% PART5B
def compute_metrics(records: list[dict]) -> dict:
    """Aggregate per-cell records into benchmark metrics.

    Each record: item_id, category, variant, repeat, response, chosen_letter,
    chosen_option, correct_letter, correct_option, is_correct, parse_ok.

    Definitions (all deterministic, no judge):
    - accuracy: fraction of parsed cells whose chosen option == gold option.
    - flip_rate: fraction of items whose chosen OPTION TEXT differs across cells
      (option-identity, not letter — a layout change that keeps the same answer
      is not a flip).
    - SAA (stability-adjusted accuracy): fraction of items where EVERY parsed
      cell is correct. An item must survive all 10 phrasings/repeats.
    - run instability: same (item, variant) across repeats disagree.
    - prompt instability: different variants of the same item disagree.
    - position bias: accuracy on shuffled cells grouped by gold letter position.
    """
    df = pd.DataFrame(records)
    expected_per_item = len(VARIANTS) * N_REPEATS  # 10 cells per item
    item_ids = [i.id for i in ITEMS]               # fixed denominator: all 32
    out: dict = {"n_cells": int(len(df)), "expected_cells": len(item_ids) * expected_per_item}
    if df.empty:
        return {"error": "no records"}

    parsed = df[df["parse_ok"]]
    out["n_parsed"] = int(len(parsed))
    out["parse_rate"] = float(len(parsed) / len(df))

    # Present-but-unparsed cells count as WRONG (format non-compliance is the
    # model's fault). Absent cells (API errors) are excluded and reported --
    # flakiness must not silently inflate or deflate the score.
    per_item_acc: dict[str, float] = {}
    flip_items: set[str] = set()
    saa_items: dict[str, float] = {}
    excluded_incomplete = 0
    n_eligible_flips = 0
    for item_id in item_ids:
        g = df[df["item_id"] == item_id]
        if g.empty:
            excluded_incomplete += 1  # zero cells: cannot score, report it
            continue
        flags = g["is_correct"] & g["parse_ok"]
        per_item_acc[item_id] = float(flags.mean())
        pg = g[g["parse_ok"]]
        if len(pg) >= 2 and len(set(pg["chosen_option"])) > 1:
            flip_items.add(item_id)
        if len(pg) >= 2:
            n_eligible_flips += 1
        if len(g) == expected_per_item:
            saa_items[item_id] = 1.0 if bool(flags.all()) else 0.0
        else:
            excluded_incomplete += 1

    out["accuracy"] = float(sum(per_item_acc.values()) / len(per_item_acc)) if per_item_acc else 0.0
    out["accuracy_ci"] = bootstrap_ci(per_item_acc)
    out["flip_rate"] = float(len(flip_items) / n_eligible_flips) if n_eligible_flips else 0.0
    out["flip_items"] = sorted(flip_items)
    out["n_saa_scored"] = len(saa_items)
    out["n_saa_excluded_incomplete"] = excluded_incomplete
    out["saa"] = float(sum(saa_items.values()) / len(saa_items)) if saa_items else 0.0
    out["saa_ci"] = bootstrap_ci(saa_items)

    per_variant = {}
    for variant, g in df.groupby("variant"):
        per_variant[variant] = {
            "accuracy": float((g["is_correct"] & g["parse_ok"]).mean()),
            "n": int(len(g)),
            "run_flips": sum(
                1 for _, gi in g.groupby("item_id")
                if len(set(gi[gi["parse_ok"]]["chosen_option"])) > 1
            ),
        }
    out["per_variant"] = per_variant

    per_category = {}
    for cat, g in df.groupby("category"):
        per_category[cat] = {
            "accuracy": float((g["is_correct"] & g["parse_ok"]).mean()),
            "n": int(len(g)),
        }
    out["per_category"] = per_category

    run_disagree = run_total = 0
    for _, g in df.groupby(["item_id", "variant"]):
        pg = g[g["parse_ok"]]
        if len(pg) >= 2:
            run_total += 1
            if len(set(pg["chosen_option"])) > 1:
                run_disagree += 1
    out["run_instability"] = float(run_disagree / run_total) if run_total else 0.0

    prompt_disagree = prompt_total = 0
    for _, g in df.groupby(["item_id", "repeat"]):
        pg = g[g["parse_ok"]]
        if len(pg) >= 2:
            prompt_total += 1
            if len(set(pg["chosen_option"])) > 1:
                prompt_disagree += 1
    out["prompt_instability"] = float(prompt_disagree / prompt_total) if prompt_total else 0.0

    sh = df[(df["variant"] == "shuffled") & df["parse_ok"]]
    if not sh.empty:
        pos = {letter: {"correct": 0, "n": 0} for letter in LETTERS}
        for _, r in sh.iterrows():
            pos[r["correct_letter"]]["n"] += 1
            if r["is_correct"]:
                pos[r["correct_letter"]]["correct"] += 1
        out["shuffled_position_accuracy"] = {
            k: (v["correct"] / v["n"] if v["n"] else None) for k, v in pos.items()
        }
    return out


# %% PART6
def make_record(row: dict, response, options=None) -> dict:
    """Score one cell: raw response -> per-cell record (never raises)."""
    opts = tuple(options if options is not None else row.get("options_json"))
    letter = extract_letter(response, opts)
    chosen = opts[LETTERS.index(letter)] if letter else None
    return {
        "item_id": row["item_id"],
        "category": row["category"],
        "variant": row["variant"],
        "repeat": int(row["repeat"]),
        "seed": int(row.get("seed", 0)),
        "response": str(response) if response is not None else None,
        "chosen_letter": letter,
        "chosen_option": chosen,
        "correct_letter": row["correct_letter"],
        "correct_option": row["correct_option"],
        "is_correct": chosen == row["correct_option"],
        "parse_ok": letter is not None,
    }


def format_report(m: dict) -> str:
    """Human-readable metrics report (stdout on Kaggle, reused by the DEV post)."""
    lines = ["=" * 64, "Paraphrase Stability Benchmark - report", "=" * 64]
    if "error" in m:
        return lines[0] + f"\nERROR: {m['error']}"
    lines.append(
        f"cells: {m['n_cells']}/{m['expected_cells']} | parsed: {m['n_parsed']} ({m['parse_rate']:.1%})"
    )
    lo, hi = m["accuracy_ci"]
    lines.append(f"accuracy: {m['accuracy']:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
    slo, shi = m["saa_ci"]
    lines.append(f"stability-adjusted accuracy (SAA): {m['saa']:.3f}  [95% CI {slo:.3f}, {shi:.3f}]")
    if m.get("n_saa_excluded_incomplete"):
        lines.append(
            f"  (SAA scored on {m['n_saa_scored']} complete items; "
            f"{m['n_saa_excluded_incomplete']} excluded for missing cells)"
        )
    lines.append(f"flip rate: {m['flip_rate']:.3f}  (items flipping: {', '.join(m['flip_items']) or 'none'})")
    lines.append(
        f"run instability (identical params, 2 repeats): {m['run_instability']:.3f}"
    )
    lines.append(
        f"prompt instability (across phrasings):        {m['prompt_instability']:.3f}"
    )
    lines.append("per-variant accuracy:")
    for v in VARIANTS:
        d = m["per_variant"].get(v)
        if d:
            lines.append(f"  {v:<11} {d['accuracy']:.3f}  (n={d['n']}, run-flips={d['run_flips']})")
    lines.append("per-category accuracy:")
    for c, d in sorted(m["per_category"].items()):
        lines.append(f"  {c:<13} {d['accuracy']:.3f}  (n={d['n']})")
    if m.get("shuffled_position_accuracy"):
        pos = m["shuffled_position_accuracy"]
        lines.append(
            "shuffled accuracy by gold position: "
            + "  ".join(f"{k}={v:.2f}" for k, v in pos.items() if v is not None)
        )
    lines.append("=" * 64)
    return "\n".join(lines)






