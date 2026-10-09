# Paraphrase Stability Benchmark

**Kaggle Benchmarking Challenge submission** (`#kagglechallenge`, due Oct 11, 2026).

Ask 32 objectively-scored MC items in **5 semantically-equivalent phrasings**
(canonical, reworded, terse, verbose, options-shuffled) x 2 repeats = **320 cells
per model**. Scored deterministically (no LLM judge). Headline metric:
**Stability-Adjusted Accuracy (SAA)** = fraction of items answered correctly
under *every* phrasing, with a cluster-bootstrap 95% CI.

## Why this benchmark (strategy)

- Answers the #1 criticism on the leading submissions ("single draws, no error
  bars"): every score ships with bootstrap CIs, 10 shots per item, and a
  run-vs-prompt instability decomposition.
- No LLM judge anywhere: letter extraction only → no judge-bias critique.
- Distinctive leaderboard number: SAA separates models that are *right* from
  models that are *right by luck of phrasing*.

## Layout

```
paraphrase-stability-bench/
├── src/
│   ├── stability_core.py     # items, prompt builders, extraction, metrics (single source of truth)
│   ├── build_notebook.py     # generates dist/kaggle_notebook.py
│   ├── dry_run.py            # 4 mock model personalities; asserts metric invariants
│   ├── make_fixture.py       # mock record CSVs (pipeline testing only — never publish these)
│   ├── analyze.py            # record CSVs -> numbers.json + markdown summary
│   └── figures.py            # numbers.json -> fig1..fig5 PNG (shared by post + paper)
├── paper/
│   ├── build_paper.py        # numbers.json + figures -> paper/main.pdf (reportlab)
│   └── main.pdf              # built paper
├── figures/                  # generated figures
├── numbers.json              # GENERATED: every number the post & paper cite
├── results/raw/              # paraphrase_records_<model>.csv (real runs or fixtures)
├── tests/
│   ├── test_core.py          # unit tests (answer key, parsing, metrics edge cases)
│   ├── test_notebook_mock.py # executes the built notebook against a mock kbench
│   └── test_publish_pipeline.py  # fixtures -> numbers -> figures -> paper, end to end
├── dist/
│   └── kaggle_notebook.py    # ← paste or `kaggle b t push` THIS (generated)
├── docs/
│   ├── SUBMISSION_GUIDE.md   # Kaggle CLI/UI + figures/paper + DEV publishing checklist
│   └── dev_post_draft.md     # full DEV article draft (fill in real numbers)
└── kaggle-benchmarks-lib/    # clone of the official library (reference only)
```

## Commands

```bash
python -m pytest tests/test_core.py -q            # unit tests
python src/dry_run.py                             # mock-model end-to-end invariants
python src/build_notebook.py                      # regenerate dist/kaggle_notebook.py
python tests/test_notebook_mock.py                # execute built notebook against mock kbench
python -m pytest tests/ -q                        # everything incl. publish pipeline

# after real runs:
python src/analyze.py --records-dir results/raw --out numbers.json
python src/figures.py --numbers numbers.json --out figures
python paper/build_paper.py --numbers numbers.json --figures figures --out paper/main.pdf
```

**After any edit to `stability_core.py` or `build_notebook.py`, re-run the test suite.**
**Post/paper numbers always come from `numbers.json` — never hand-type them.**

## Deploy

See `docs/SUBMISSION_GUIDE.md`. Short version: create task at
<https://www.kaggle.com/benchmarks/tasks/new>, paste `dist/kaggle_notebook.py`,
Save Version, then use **Evaluate More Models** on the task page.

## Metric definitions

| Metric | Definition |
|---|---|
| accuracy | mean over items of (correct cells / present cells); unparsed cells = wrong; 95% cluster bootstrap CI over items |
| **SAA** | items where ALL 10 cells are correct / complete items only (incomplete items excluded *and reported*) |
| flip rate | items whose chosen **option text** differs across parsed cells / items with ≥2 parsed cells (letter-identity, layout-independent) |
| run instability | share of (item, variant) pairs whose 2 repeats disagree (backend nondeterminism at temp 0) |
| prompt instability | share of (item, repeat) groups whose variants disagree (pure phrasing sensitivity) |
| parse rate | cells whose reply mapped to A–D; unparsed counts against accuracy & SAA, never silently dropped |
