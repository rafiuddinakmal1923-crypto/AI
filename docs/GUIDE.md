# Study guide: Labs 2, 3 and 6

This guide explains what each script does, why it does it, and how to read the
results. All numbers come from the files in `results/`.

---

## Lab 2: Informed (heuristic) search

**Script:** `all_experiments/lab02_informed_search.py`

### The idea
Lab 1's Uniform-Cost Search (UCS) always expands the cheapest frontier node, where
"cheapest" means `g(n)`, the distance flown so far. It has no sense of direction, so it
spreads out in every direction like a ripple until it reaches the goal. An informed search
adds a **heuristic** `h(n)`: an *estimate* of the distance still to go.

| Algorithm | Priority (smaller = expand first) | Optimal? |
|---|---|---|
| UCS | `g(n)` | yes |
| Greedy best-first (GBFS) | `h(n)` | no: ignores the cost already paid |
| A* | `g(n) + h(n)` | yes, if `h` never overestimates |
| Weighted A* | `g(n) + 1.5·h(n)` | no: may be up to 1.5× the optimum |

All four already exist in `utils/core.py` (`best_first` with different priorities).
The heuristic comes from `utils/search_runner.make_heuristic`. For an airport it returns the
**great-circle (haversine) distance to the goal**: the straight-line distance over the Earth's surface.

### Why the haversine heuristic is admissible
Every flight leg's cost *is* the great-circle distance between its two airports. Any chain of
flights from `n` to the goal is therefore at least as long as the direct great-circle line
(the triangle inequality on a sphere). So `h(n) ≤ true cost`, which is what **admissible** means.
The script doesn't just trust this argument; it checks it:

1. **True cost to the goal for every airport.** Running Dijkstra *from the goal* on the
   **reversed** graph (every edge flipped) gives the cheapest cost from every airport *to* the
   goal in one pass (`true_cost_to_goal`).
2. Compare `h(n)` with that true cost for all 3,282 airports that can reach each goal.
3. As a bonus it also checks **consistency**: `h(n) ≤ cost(n,m) + h(m)` for every edge. This
   matters because our A* never re-opens a closed node, so it needs a consistent heuristic to
   be guaranteed optimal.

Result (`results/lab02/heuristic_check.csv`): **0 violations** and 0 inconsistent edges for all
three goals. The mean ratio `h/true` is 0.88–0.94, so the heuristic is a good but not perfect
estimate. The ratio is exactly 1 for airports with a direct flight to the goal.

### Results (`results/lab02/metrics.csv`)

| Query | UCS expanded | A* expanded | GBFS expanded (cost gap) | Weighted A* expanded (cost gap) |
|---|---|---|---|---|
| Q1 KUL→KEF | 1851 | 69 | 2 (+9.07 %) | 2 (+7.09 %) |
| Q2 SIN→EZE | 3002 | 151 | 3 (0 %) | 3 (0 %) |
| Q3 NRT→CPT | 3042 | 63 | 3 (+6.74 %) | 4 (+0.78 %) |

How to read this:
- **A* finds exactly the same route and cost as UCS**, but expands 95–98 % fewer airports.
  That is the whole point of an admissible heuristic: same answer, much less work.
- **GBFS is extremely fast but greedy.** On Q1 it flies KUL→LHR→KEF, which is 9 % longer
  than the optimum, because LHR "looks" close to Iceland.
- **Weighted A*** sits in between. Inflating `h` makes it trust the estimate more, so it
  expands almost as few nodes as GBFS while staying closer to the optimum. With weight `w`,
  the cost is guaranteed to be at most `w` times the optimum.
- `max_frontier` shows memory use. UCS's frontier stays large because it explores in every direction.

---

## Lab 3: Search engineering and benchmarking

**Files:** `lab03_search_engineering.py`, `utils/search_runner.py`, `utils/__main__.py`, `utils/test_lab03.py`

### Benchmark design
- **Queries:** all 10 pairs in `search_queries.csv`, plus **10 random pairs**.
- **Reproducibility:** random pairs come from `np.random.default_rng(2023)`. A fixed seed gives
  the same pairs every run, so anyone can reproduce the table. Starts are only drawn from airports
  with an outgoing route, and goals from airports with an incoming route.
- **One row per (query, algorithm)** in `benchmark.csv`, with `found`, path, hops, cost, expanded,
  max frontier, runtime and `gap_pct` relative to UCS.

### Sanity checks (the "engineering" part)
Before trusting the numbers, the script asserts:
1. every returned route only uses real flights (`route_is_valid`);
2. all algorithms agree on whether a route exists (they are all *complete* on a finite graph);
3. A* cost == UCS cost on every solvable query (A* is optimal);
4. BFS has the fewest hops on every solvable query (BFS optimises the *number of edges*).

### Results (`results/lab03/benchmark_summary.csv`, 20 queries)

| Algorithm | success | mean cost km | mean hops | mean expanded | optimal |
|---|---|---|---|---|---|
| BFS | 0.9 | 11,784 | 2.39 | 581 | 28 % |
| DFS | 0.9 | 35,874 | 22.39 | 1,814 | 22 % |
| UCS | 0.9 | 10,161 | 2.78 | 1,796 | 100 % |
| GBFS | 0.9 | 10,800 | 2.94 | 168 | 28 % |
| A* | 0.9 | 10,161 | 2.78 | 209 | 100 % |

- The success rate is 0.9 because one random pair (R08) has **no route at all**. On that
  pair every algorithm has to exhaust the 3,286 reachable airports before giving up. That one
  pair inflates every algorithm's mean expansions, which is why the chart also shows medians.
- **DFS** finds a route but a terrible one: 22 hops, 3.5× the optimal distance. It follows
  whichever neighbour is listed first (alphabetical order), not anything sensible.
- **A*** gives UCS's answers for about 1/9 of the work. Mean expanded is 209 against 1,796,
  and the median is far lower still.
- Runtimes are in milliseconds and vary from run to run (other programs, CPU caching). Use them
  only for rough comparisons; `expanded` is the reliable, machine-independent cost measure.

### Command-line tool
`python -m utils ...` runs `utils/__main__.py`. Python always looks for that file when you
"run a package". It uses `argparse`:
- `--algorithm` is restricted with `choices=`, so typos are rejected automatically;
- airport codes are upper-cased and checked against the airport list. An unknown code calls
  `parser.error(...)`, which prints the usage and a clear message to stderr and exits with code 2.
  `difflib.get_close_matches` suggests similar codes;
- exit code 0 = route found, 1 = no route exists, 2 = bad input.

### Tests (`utils/test_lab03.py`)
- `test_astar_cost_equals_ucs_cost`: runs for each of the 10 query pairs (parametrised)
- `test_bfs_gives_fewest_hops`: runs for each of the 10 query pairs
- `test_cli_finds_a_route`, `test_cli_rejects_bad_input` (unknown start, unknown goal, not
  3 letters, unknown algorithm). `pytest.raises(SystemExit)` catches argparse's exit,
  and `capsys` captures the error message.

Run `python -m pytest utils -q` inside `all_experiments/`. All 33 tests pass: 25 new and 8
from the existing `test_search.py`.

---

## Lab 6: OCR on handwritten prescriptions

**Script:** `all_experiments/lab06_handwriting_ocr.py`

### Setup
`pytesseract` only *calls* the Tesseract program. `setup_tesseract()` looks for
`tesseract.exe` in the usual Windows places, sets `pytesseract.pytesseract.tesseract_cmd`, and
prints install instructions if the program is missing. Version used: Tesseract 5.4.0 (UB Mannheim build).

### Pipelines

| Pipeline | Image | Tesseract mode |
|---|---|---|
| `baseline` | raw photo | `--psm 3` (automatic page layout, the default) |
| `preprocessed` (required) | grayscale → autocontrast → 40 px white border → **2× upscale** | `--psm 3` |
| `height150_psm7` (my own) | grayscale → autocontrast → border → **resize to 150 px high** | `--psm 7` (single text line) |

### Scoring
Before comparing, both strings are **normalised**: lowercase, punctuation removed, runs of
spaces squeezed (`" Tab. Azimax\n"` → `"tab azimax"`).
- **Exact match:** the normalised strings are identical.
- **Levenshtein edit distance:** the minimum number of single-character insertions,
  deletions or substitutions needed to turn one string into the other. It's computed with the
  classic dynamic-programming table (`levenshtein()`).
- **Normalised edit distance (NED):** edits ÷ length of the longer string. 0 = perfect,
  1 = nothing in common. Averaged over images.
- **Character error rate (CER):** total edits ÷ total characters in the true labels, over the whole dataset.
- **Blank predictions:** Tesseract returned nothing usable.

### Results (`results/lab06/metrics.csv`, 89 images)

| Pipeline | exact | mean NED | CER | blanks |
|---|---|---|---|---|
| baseline | 0 % | 0.910 | 0.922 | 62 |
| preprocessed | 0 % | 0.981 | 0.980 | 81 |
| height150_psm7 | 0 % | 0.836 | 0.859 | 33 |

`error_analysis.csv` counts the failure types. "No text detected" dominates for the first two
pipelines. With my pipeline, Tesseract nearly always outputs *something*, but mostly the wrong letters.

### Why the required preprocessing made things *worse* (the ablation)
Run with `--ablation` (saved as `ablation.csv`). Each step is added one at a time:

| Step | blanks (psm 3) | NED (psm 3) | blanks (psm 7) | NED (psm 7) |
|---|---|---|---|---|
| raw photo | 62 | 0.910 | 42 | 0.861 |
| + grayscale | 63 | 0.909 | 41 | 0.851 |
| + autocontrast | 62 | 0.898 | 40 | 0.833 |
| + white border | 58 | 0.881 | 43 | 0.844 |
| + 2× upscale | **81** | **0.981** | 48 | 0.866 |
| + resize to 150 px high | 49 | 0.885 | **33** | 0.836 |
| + resize to 250 px high | 53 | 0.872 | 44 | 0.843 |

- Grayscale, autocontrast and the border each help a little.
- **The 2× upscale hurts.** Upscaling is meant for *small, low-resolution* text. These
  photos are already 660–1900 px wide with letters 150–300 px tall. Tesseract's models were
  trained on printed text a few dozen pixels high, so doubling the size takes the letters even
  further from what it knows. Its layout analysis then often decides there is no text at all.
- Resizing to a moderate fixed height (upscaling small images *and* shrinking big ones) works better.
- `--psm 7` ("this image is one line of text") helps almost everywhere, because every crop
  really is a single line. It skips the page-layout step, which is where many blanks come from.

Caveat: I chose 150 px from three options tried *on the same 89 images*, so the gain is a
little optimistic. With a bigger dataset you would tune this on a separate validation set.

### Why accuracy is so low (0 exact matches)
1. **Tesseract is built for printed text.** Its LSTM model learned machine fonts with
   regular shapes and spacing. Cursive handwriting has joined letters, slanted strokes and
   letter shapes that change with every writer.
2. **Doctors' handwriting is extreme.** Many labels are hard for a *human* to read
   (see `ocr_failures.png`: "lipiget", "uriguard", "cefiget 40mg").
3. **Vocabulary.** Drug names (azimax, tonoflex, dicloran) aren't English words, so
   Tesseract's English language model can't use word knowledge to fix mistakes.
4. **Photo quality.** Shadows, folds, blur, logos behind the text, black wedges from
   rotation (see `sample_grid.png`) and blue ink on grey paper.
5. **Strict metric.** Exact match needs every character right, so even "covaes" for "covam" counts as wrong.

Handwriting recognition needs a model trained on handwriting: for example TrOCR,
or a CNN+LSTM trained with CTC on handwritten words. Fine-tuning on this kind of data would help
far more than any image preprocessing.
