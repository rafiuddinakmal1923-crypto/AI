# TEB2023 Artificial Intelligence – Lab Work

**Course:** TEB2023 Artificial Intelligence, Universiti Teknologi PETRONAS (UTP)
**Student:** Muhammad Akmal Rafiuddin bin Mohd Rizal (24006013)

Every lab is a plain Python script in `all_experiments/`. Each script finds its
data and output folders relative to its own location, so you can run it from any folder.

| Lab | Topic | Script | Results |
|----|----|----|----|
| 1 | Search foundations: BFS, DFS, UCS | `lab01_search_foundations.py` | `results/lab01/` |
| 2 | Informed search: GBFS, A*, weighted A*, admissibility | `lab02_informed_search.py` | `results/lab02/` |
| 3 | Search engineering: benchmark, CLI, tests | `lab03_search_engineering.py` | `results/lab03/` |
| 4 | Fuzzy logic and ANFIS (air quality) | `lab04_fuzzy_anfis.py` | `results/lab04/` |
| 5 | Robotics: sensor-to-action control | `lab05_robotics_sensor_control.py` | `results/lab05/` |
| 6 | Handwritten prescription OCR (Tesseract, follows the lecturer's handout) | `lab06_handwriting_ocr.py` | `results/lab06/` |
| 7 | Airline tweet sentiment (bag of words + Naive Bayes) | `lab07_airline_sentiment.py` | `results/lab07/` |
| 8 | Genetic algorithm for regression vs OLS | `lab08_genetic_algorithm_regression.py` | `results/lab08/` |


## Setup (Windows)

```powershell
python -m pip install -r requirements.txt
```

Lab 6 also needs the **Tesseract OCR program**, because `pytesseract` is only a wrapper around it:

```powershell
winget install --id UB-Mannheim.TesseractOCR
```

The script looks in `C:\Program Files\Tesseract-OCR\` automatically. If you installed it
elsewhere, set `$env:TESSERACT_CMD = "D:\path\to\tesseract.exe"`.

## How to run each lab

```powershell
python all_experiments/lab01_search_foundations.py
python all_experiments/lab02_informed_search.py
python all_experiments/lab03_search_engineering.py
python all_experiments/lab04_fuzzy_anfis.py
python all_experiments/lab05_robotics_sensor_control.py
python all_experiments/lab06_handwriting_ocr.py              # about 2 minutes
python all_experiments/lab06_handwriting_ocr.py --ablation   # extra: tests each preprocessing step (~3 more minutes)
python all_experiments/lab06_handwriting_ocr.py --limit 5    # quick check on 5 images
python all_experiments/lab07_airline_sentiment.py
python all_experiments/lab08_genetic_algorithm_regression.py
```

Route finder (Lab 3) and tests. Run these from inside `all_experiments/`:

```powershell
cd all_experiments
python -m utils --algorithm astar --start KUL --goal KEF
python -m utils --algorithm bfs --start SIN --goal EZE
python -m pytest utils -q
```

`--algorithm` can be `bfs`, `dfs`, `ucs`, `gbfs` or `astar`. Airport codes are
3-letter IATA codes. An unknown code gives a clear error message, plus suggestions when a close match exists.

## Folder layout

```
all_datasets/          one folder per lab (Lab 1-3, Lab 4, Lab 5, Lab 6, Lab 7, Lab 8)
all_experiments/       lab scripts
  utils/               shared search code (core.py, openflights.py, plotting.py),
                       search_runner.py (heuristic + run-by-name), __main__.py (CLI), tests
results/labNN/         CSVs and figures produced by each script
assignments/           assignment documents
  reference_drafts/    worked reference drafts (study only, not for submission as-is)
docs/                  study guide and output comparison
```
