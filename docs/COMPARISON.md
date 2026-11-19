# Output comparison with the reference repository

**Reference:** a classmate's repository (`github.com/akriii/AI_Lab`), cloned to
`../AI_Lab_reference`. Only its `results/` CSVs and its README were used, for
**output comparison only**. None of its code was opened. All my code was written independently.

**My run:** all six scripts were re-run on 2026-11-19 (Python 3.12.10, numpy 1.26.4,
scipy 1.17.1, pandas 3.0.1, Tesseract 5.4.0). The comparison script compares CSVs
column by column. Floating-point costs are compared with a tolerance of 1e-6 km.
`runtime_ms` is ignored everywhere, because it depends on the machine.

## Summary

| Lab | Verdict |
|---|---|
| 1 Search foundations | **Identical** (9/9 runs: path, hops, cost, expanded, max frontier) |
| 2 Informed search | **Identical** (12/12 runs, including weighted A*). Heuristic check agrees |
| 3 Benchmark | **Identical on the 40 shared runs.** Summary differs only because the query set differs |
| 4 Fuzzy / ANFIS | Linear model, data split and fuzzy outputs identical. ANFIS differs by ≤ 0.3 %: **not meaningful** |
| 5 Robotics | **Identical** (all 7 shared CSVs + rule text). Mine has an extra `model_complexity.csv` |
| 6 OCR | **Different**, as expected for OCR. Explained below |

---

## Lab 1: search foundations
`results/lab01/metrics.csv`: all 9 rows (3 queries × BFS/DFS/UCS) have the same path, hops,
`path_cost_km` (max difference 0 km), `expanded` and `max_frontier`.

Note: before this run, my old committed file had UCS Q1 cost `…050693` and the new one has
`…050694`. That's a 1e-12 km difference in the last digit, caused by floating-point rounding when
summing the edge lengths. It doesn't matter.

## Lab 2: informed search
`results/lab02/metrics.csv`: all 12 rows match in path, hops, cost, expanded and max frontier.
That covers UCS, GBFS, A* and weighted A* with w = 1.5 on Q1–Q3. `gap_pct` differs by at most 9e-15.
The only difference is the label: the reference calls it `Weighted A* (1.5h)`, mine `Weighted A* (w=1.5)`.

`heuristic_check.csv` has a **different format by design**:
- Reference: 10 sampled airports per goal (30 rows), each with haversine km, true km and an admissible flag.
- Mine: one summary row per goal covering **all 3,282 airports that can reach the goal**
  (violations, mean/median/min/max of h/true, consistency check). Any violation would be
  written to `heuristic_violations.csv`; there were none.
- Cross-check: for the reference's 30 sampled airports, my Dijkstra-on-reversed-graph true cost
  and my heuristic match theirs to within 2e-12 km. Both repos find no violations.

## Lab 3: benchmark
`results/lab03/benchmark.csv`: the reference has 40 runs, mine has 100.
- **Shared runs (10 query-file pairs × BFS/UCS/GBFS/A\*): 40/40 identical** in path, hops,
  cost, expanded, max frontier and `found`.
- **Why mine has more rows:**
  1. I include **DFS**, because the task asks for all 5 algorithms. The reference benchmarks only 4.
  2. I add **10 random pairs** (seed 2023), because the task asks for "the query file plus extra
     random pairs". The reference uses only the 10 query-file pairs.
- **Why the summaries differ:** different query sets. When I restrict my runs to the same
  10 queries and 4 algorithms, my numbers **reproduce the reference summary exactly**:

  | Algorithm | mean cost km | mean hops | mean expanded |
  |---|---|---|---|
  | A* | 14,321.11 | 2.2 | 62.7 |
  | BFS | 16,469.27 | 1.8 | 323.9 |
  | GBFS | 15,087.33 | 2.1 | 2.1 |
  | UCS | 14,321.11 | 2.2 | 2,700.4 |

  My full summary has a success rate of 0.9, because random pair R08 has no route. The
  reference shows 1.0 because every query-file pair is solvable.
- My summary also has extra columns: `runs`, `mean_gap_pct`, `optimal_rate`.

## Lab 4: fuzzy logic and ANFIS
- `split.csv` and `fuzzy_fixed_points.csv`: **identical**.
- Linear regression: **identical** in all 4 metrics.
- ANFIS (difference = mine − reference):

  | Model | val RMSE | val MAE | test RMSE | test MAE |
  |---|---|---|---|---|
  | ANFIS-2MF | −0.00013 (−0.03 %) | −0.00001 | +0.00006 (+0.01 %) | +0.00038 (+0.12 %) |
  | ANFIS-3MF | +0.00135 (+0.30 %) | +0.00050 | +0.00029 (+0.06 %) | +0.00045 (+0.14 %) |

- **Is it meaningful? No.**
  - The data, the split (`random_state=42`) and the linear baseline are identical, so the inputs are the same.
  - On this machine, running the script twice gives *byte-identical* metrics. The ANFIS code has
    no unseeded randomness, so this isn't "training randomness".
  - The cause is numerical. ANFIS is trained with `scipy.optimize.minimize(L-BFGS-B)` using
    finite-difference gradients and stops after only `maxiter=30`. Different numpy/scipy/BLAS
    builds round differently, and an optimiser stopped early lands at a slightly different point.
    My own previously committed results (from an earlier environment) also differed from today's
    run by a similar amount.
  - Both runs reach the **same conclusion**: ANFIS-3MF < ANFIS-2MF < linear on validation
    RMSE, so the same model is selected. A difference of ≤ 0.0014 RMSE is about 0.1 % of the
    target's spread (CO(GT) has a standard deviation of about 1.45 mg/m³). That is far smaller
    than the gap between the models (≈ 0.06 RMSE between linear and ANFIS).

## Lab 5: robotics sensor control
`feature_scores.csv`, `metrics.csv`, `noise_robustness.csv`, `rule_per_class.csv`, `split.csv`,
`tree_depth_validation.csv` and `rule_controller.txt` are all **identical**.
My version also writes `model_complexity.csv`, which has no counterpart in the reference.

## Lab 6: handwritten prescription OCR

| Pipeline | exact | NED | CER | blanks |
|---|---|---|---|---|
| Reference baseline | 0.0 % | 0.860 | 0.881 | 44 |
| Reference `grayscale_border_resize` | 1.1 % (1/89) | 0.767 | 0.788 | 15 |
| My baseline | 0.0 % | 0.910 | 0.922 | 62 |
| My `preprocessed` (2× upscale, required) | 0.0 % | 0.981 | 0.980 | 81 |
| My `height150_psm7` (own pipeline) | 0.0 % | 0.836 | 0.859 | 33 |

These differences come from three separate sources:

1. **Slightly different normalisation.** The reference deletes *all* spaces when normalising
   (`tabazimax`), while I only squeeze them (`tab azimax`). Both divide the edit distance by the
   longer string's length (checked on their CSV). To remove this effect, I re-scored
   the reference's raw predictions with **my** metric code: baseline NED 0.858 / CER 0.884,
   preprocessing NED 0.759 / CER 0.783. That's almost the same as their own numbers, so the
   metric difference is small.
2. **Different OCR output on the same raw images.** Both baselines feed the raw photo to
   Tesseract, but only **39/89 raw outputs are identical**. The reference gets text on 27 images
   where I get a blank, and the reverse happens on 9 images. The most likely cause is a different
   Tesseract version or trained-data model (I use 5.4.0 from UB Mannheim). The reference README
   mentions Google Colab, where the default Tesseract is an older 4.x build. I can't confirm this
   without reading their code, which I deliberately didn't do.
3. **Different preprocessing.** My required pipeline uses a 2× upscale, which my ablation
   (`results/lab06/ablation.csv`) shows *hurts* on these already-large photos (58 → 81 blanks).
   The reference's resize step evidently doesn't enlarge the images this way. My own
   `height150_psm7` pipeline fixes most of that (33 blanks) but is still behind the reference's
   best (15 blanks, NED 0.767).

Both repos agree on the main conclusion: Tesseract reads almost none of these handwritten labels
correctly (≤ 1 exact match out of 89), and "no text detected" is the most common failure.
