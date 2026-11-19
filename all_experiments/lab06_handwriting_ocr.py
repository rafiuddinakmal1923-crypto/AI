"""
Lab 6 of 8 - OCR on Handwritten Prescriptions
Objective: measure how well Tesseract reads doctors' handwriting, and whether
simple image preprocessing helps.

Data:   all_datasets/Lab 6/img/img/*.jpg            (89 cropped photos, one line each)
        all_datasets/Lab 6/doctor_handwriting_labels.csv   (filename, label)

Pipelines compared
    baseline           raw photo -> Tesseract (default page segmentation, --psm 3)
    preprocessed       grayscale -> autocontrast -> white border -> 2x upscale -> Tesseract
    height150_psm7     (my own) grayscale -> autocontrast -> white border -> resize to 150 px
                       high -> Tesseract told the image is ONE line of text (--psm 7).
                       Why: (1) these photos are already large, so a 2x upscale makes the
                       letters far bigger than Tesseract is trained on; a fixed height
                       brings every image to a similar, moderate letter size. (2) every crop
                       is a single line, so --psm 7 skips the page-layout step that often
                       decides there is "no text". Run with --ablation to see each step's effect.

Metrics (after normalising: lowercase, no punctuation, single spaces)
    exact-match accuracy, mean normalised edit distance, character error rate (CER),
    number of blank predictions.

Saves to results/lab06/: metrics.csv, predictions.csv, error_analysis.csv,
preprocessed/*.jpg, sample_grid.png, ocr_failures.png  (+ ablation.csv with --ablation)

Needs the Tesseract PROGRAM (not just the pytesseract package). On Windows:
    winget install --id UB-Mannheim.TesseractOCR
Run:  python all_experiments/lab06_handwriting_ocr.py          (add --limit 5 for a quick test)
      python all_experiments/lab06_handwriting_ocr.py --ablation   (also test each step, ~5 min)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")                     # save figures to file, no pop-up window needed
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Paths are built from this file's location, so it works from any terminal folder.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                # .../AI/all_experiments
ROOT = HERE.parent                                    # .../AI
DATA = ROOT / "all_datasets" / "Lab 6"
IMAGE_DIR = DATA / "img" / "img"
LABELS_FILE = DATA / "doctor_handwriting_labels.csv"
RESULTS = ROOT / "results" / "lab06"

BORDER_PX = 40            # white margin added around the text (Tesseract dislikes text touching the edge)
UPSCALE = 2               # enlarge so pen strokes are several pixels thick
TARGET_HEIGHT = 150       # my pipeline: every image is resized to this height (px), up or down
OCR_LANGUAGE = "eng"

# Places tesseract.exe is usually installed on Windows (checked in this order).
TESSERACT_LOCATIONS = [
    os.environ.get("TESSERACT_CMD", ""),
    shutil.which("tesseract") or "",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]

INSTALL_HELP = r"""
Tesseract OCR is not installed (pytesseract is only a Python wrapper around tesseract.exe).

How to install it on Windows:
  1. In PowerShell:   winget install --id UB-Mannheim.TesseractOCR
     (or download the installer from https://github.com/UB-Mannheim/tesseract/wiki
      and keep the default folder C:\Program Files\Tesseract-OCR)
  2. Close and re-open your terminal / VS Code, then run this script again.

If you installed it somewhere else, either set an environment variable
      $env:TESSERACT_CMD = "D:\path\to\tesseract.exe"
or add this line to the script after `import pytesseract`:
      pytesseract.pytesseract.tesseract_cmd = r"D:\path\to\tesseract.exe"
"""


def heading(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 0 - make sure Tesseract can be found
# ---------------------------------------------------------------------------
def setup_tesseract() -> str:
    """Point pytesseract at tesseract.exe and return its version, or exit with help."""
    for location in TESSERACT_LOCATIONS:
        if location and Path(location).is_file():
            pytesseract.pytesseract.tesseract_cmd = location
            try:
                return str(pytesseract.get_tesseract_version())
            except pytesseract.TesseractNotFoundError:
                continue
    print(INSTALL_HELP, file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1 - images and preprocessing
# ---------------------------------------------------------------------------
def load_image(path: Path) -> Image.Image:
    """Open a photo the right way up (phones store rotation in EXIF) as RGB."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def to_gray(image: Image.Image) -> Image.Image:
    return ImageOps.grayscale(image)                              # colour carries no information here


def gray_autocontrast(image: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(to_gray(image), cutoff=2)       # ink -> near black, paper -> near white


def clean(image: Image.Image) -> Image.Image:
    """grayscale -> autocontrast -> white border (shared by both preprocessing pipelines)."""
    return ImageOps.expand(gray_autocontrast(image), border=BORDER_PX, fill=255)


def preprocess(image: Image.Image) -> Image.Image:
    """The required pipeline: grayscale -> autocontrast -> white border -> 2x upscale."""
    cleaned = clean(image)
    return cleaned.resize((cleaned.width * UPSCALE, cleaned.height * UPSCALE), Image.Resampling.LANCZOS)


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    scale = height / image.height
    return image.resize((max(1, round(image.width * scale)), height), Image.Resampling.LANCZOS)


def preprocess_fixed_height(image: Image.Image) -> Image.Image:
    """My pipeline: same cleaning, but resize to a fixed height instead of always doubling."""
    return resize_to_height(clean(image), TARGET_HEIGHT)


def no_preprocessing(image: Image.Image) -> Image.Image:
    return image


# name -> (image function, Tesseract options, short description)
PIPELINES = {
    "baseline": (no_preprocessing, "--psm 3", "raw photo, default page segmentation"),
    "preprocessed": (preprocess, "--psm 3", "gray + autocontrast + border + 2x upscale"),
    "height150_psm7": (preprocess_fixed_height, "--psm 7",
                       f"gray + autocontrast + border + resize to {TARGET_HEIGHT}px high, single text line"),
}


# ---------------------------------------------------------------------------
# Step 2 - text normalisation and scoring
# ---------------------------------------------------------------------------
def normalise(text: str) -> str:
    """lowercase, drop punctuation/symbols, squeeze spaces: ' Tab. Azimax\\n' -> 'tab azimax'."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)        # keep only letters, digits and spaces
    return " ".join(text.split())


def levenshtein(a: str, b: str) -> int:
    """Minimum number of single-character inserts, deletes or substitutions to turn a into b.

    Classic dynamic programming, keeping only one previous row of the table.
    """
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,                        # delete char_a
                current[j - 1] + 1,                     # insert char_b
                previous[j - 1] + (char_a != char_b),   # substitute (free if equal)
            ))
        previous = current
    return previous[-1]


def failure_type(truth: str, raw_prediction: str, prediction: str, distance: int) -> str:
    """Put each prediction into one bucket so we can count the typical mistakes."""
    if prediction == truth:
        return "correct"
    if not raw_prediction.strip():
        return "no text detected"
    if not prediction:
        return "symbols only (no letters/digits)"
    if distance <= 2:
        return "near miss (1-2 character edits)"
    if set(truth.split()) & set(prediction.split()):
        return "partly right (shares a whole word)"
    if distance / max(len(truth), len(prediction)) <= 0.5:
        return "similar letters (edit distance <= 50%)"
    return "mostly wrong"


def score(truth_raw: str, raw_prediction: str) -> dict:
    truth = normalise(truth_raw)
    prediction = normalise(raw_prediction)
    distance = levenshtein(truth, prediction)
    return {
        "truth_norm": truth,
        "prediction_norm": prediction,
        "exact_match": prediction == truth,
        "blank": prediction == "",
        "edit_distance": distance,
        # 0 = identical, 1 = nothing in common (divide by the longer string so it stays in [0, 1])
        "normalised_edit_distance": distance / max(len(truth), len(prediction), 1),
        "truth_chars": len(truth),
        "failure_type": failure_type(truth, raw_prediction, prediction, distance),
    }


# ---------------------------------------------------------------------------
# Step 3 - run OCR
# ---------------------------------------------------------------------------
def run_ocr(labels: pd.DataFrame, save_dir: Path) -> pd.DataFrame:
    """Run every pipeline on every image. Returns one row per (image, pipeline)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(labels.itertuples(index=False)):
        original = load_image(IMAGE_DIR / item.filename)
        cleaned = preprocess(original)
        # Evidence of what Tesseract saw. Saved at 1x size as JPEG: the 2x PNG is ~1 MB per
        # image (~90 MB in total) and upscaling adds no new detail, so a smaller copy is enough.
        evidence = cleaned.resize((cleaned.width // UPSCALE, cleaned.height // UPSCALE))
        evidence.save(save_dir / f"{index:03d}.jpg", quality=85)

        for name, (prepare, options, _) in PIPELINES.items():
            image = cleaned if prepare is preprocess else prepare(original)   # reuse, don't redo
            started = perf_counter()
            raw = pytesseract.image_to_string(image, lang=OCR_LANGUAGE, config=options)
            runtime_ms = (perf_counter() - started) * 1000
            rows.append({
                "image_id": index,
                "filename": item.filename,
                "pipeline": name,
                "truth": item.label,
                "raw_prediction": raw.strip(),
                "runtime_ms": runtime_ms,
                **score(item.label, raw),
            })
        if (index + 1) % 10 == 0 or index + 1 == len(labels):
            print(f"  OCR done for {index + 1}/{len(labels)} images")
    return pd.DataFrame(rows)


def summarise(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in predictions.groupby("pipeline", sort=False):
        rows.append({
            "pipeline": name,
            "description": PIPELINES[name][2],
            "images": len(group),
            "exact_match_accuracy": group["exact_match"].mean(),
            "mean_normalised_edit_distance": group["normalised_edit_distance"].mean(),
            # CER over the whole dataset: total edits / total characters in the true labels
            "character_error_rate": group["edit_distance"].sum() / group["truth_chars"].sum(),
            "blank_predictions": int(group["blank"].sum()),
            "median_runtime_ms": group["runtime_ms"].median(),
        })
    return pd.DataFrame(rows)


def run_ablation(labels: pd.DataFrame) -> pd.DataFrame:
    """Add the preprocessing steps one at a time, with both page-segmentation modes,
    to see which step helps and which hurts. (Optional: python ... --ablation)"""
    steps = {
        "1 raw photo": no_preprocessing,
        "2 + grayscale": to_gray,
        "3 + autocontrast": gray_autocontrast,
        "4 + white border": clean,
        "5a + 2x upscale (required pipeline)": preprocess,
        "5b + resize to 150 px high": preprocess_fixed_height,
        "5c + resize to 250 px high": lambda image: resize_to_height(clean(image), 250),
    }
    originals = [load_image(IMAGE_DIR / name) for name in labels["filename"]]
    rows = []
    for step, prepare in steps.items():
        for options in ("--psm 3", "--psm 7"):
            def read(i):
                raw = pytesseract.image_to_string(prepare(originals[i]), lang=OCR_LANGUAGE, config=options)
                return score(labels["label"].iloc[i], raw)
            # Tesseract runs as a separate program, so 4 threads really do run 4 OCR jobs at once.
            with ThreadPoolExecutor(max_workers=4) as pool:
                scores = pd.DataFrame(list(pool.map(read, range(len(labels)))))
            rows.append({
                "step": step,
                "tesseract_options": options,
                "exact_match_accuracy": scores["exact_match"].mean(),
                "mean_normalised_edit_distance": scores["normalised_edit_distance"].mean(),
                "character_error_rate": scores["edit_distance"].sum() / scores["truth_chars"].sum(),
                "blank_predictions": int(scores["blank"].sum()),
            })
            print(f"  {step:<38} {options}: blanks={rows[-1]['blank_predictions']:>3}  "
                  f"NED={rows[-1]['mean_normalised_edit_distance']:.3f}")
    return pd.DataFrame(rows)


def error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Count each failure type per pipeline, with one example of each."""
    failures = predictions[predictions["failure_type"] != "correct"]
    rows = []
    for (name, kind), group in failures.groupby(["pipeline", "failure_type"], sort=False):
        example = group.iloc[0]
        rows.append({
            "pipeline": name,
            "failure_type": kind,
            "count": len(group),
            "share_of_images": len(group) / predictions["image_id"].nunique(),
            "example_filename": example["filename"],
            "example_truth": example["truth_norm"],
            "example_prediction": example["raw_prediction"],
        })
    table = pd.DataFrame(rows)
    # Keep pipelines in their usual order; within a pipeline, most common failure first.
    table["pipeline_order"] = table["pipeline"].map({name: i for i, name in enumerate(PIPELINES)})
    table = table.sort_values(["pipeline_order", "count"], ascending=[True, False])
    return table.drop(columns="pipeline_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4 - figures
# ---------------------------------------------------------------------------
def short(text: str, limit: int = 28) -> str:
    text = " ".join(str(text).split()) or "(blank)"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def plot_sample_grid(predictions: pd.DataFrame, image_ids: list[int], path: Path) -> None:
    """Left: raw photo, right: preprocessed image, with every pipeline's prediction."""
    figure, axes = plt.subplots(len(image_ids), 2, figsize=(11, 2.1 * len(image_ids)))
    for row, image_id in enumerate(image_ids):
        group = predictions[predictions["image_id"] == image_id].set_index("pipeline")
        original = load_image(IMAGE_DIR / group["filename"].iloc[0])
        axes[row, 0].imshow(original)
        axes[row, 1].imshow(clean(original), cmap="gray")
        axes[row, 0].set_title(f"raw photo - truth: {group['truth_norm'].iloc[0]!r}", fontsize=9, loc="left")
        axes[row, 1].set_title(" | ".join(f"{name}: {short(group.loc[name, 'raw_prediction'], 18)}"
                                          for name in PIPELINES), fontsize=7, loc="left")
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle("Left: raw photo.  Right: preprocessed (gray + autocontrast + border) "
                    "with each pipeline's OCR output", fontsize=10, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_failures(predictions: pd.DataFrame, pipeline: str, path: Path, n: int = 8) -> None:
    """The n worst (highest edit distance) non-blank mistakes of one pipeline."""
    group = predictions[(predictions["pipeline"] == pipeline) & ~predictions["exact_match"]]
    worst = group.sort_values(["blank", "normalised_edit_distance"], ascending=[True, False]).head(n)
    figure, axes = plt.subplots(len(worst), 1, figsize=(7, 1.9 * len(worst)))
    for axis, item in zip(list(getattr(axes, "flat", [axes])), worst.itertuples()):
        axis.imshow(load_image(IMAGE_DIR / item.filename))
        axis.set_title(f"truth: {item.truth_norm!r}   OCR: {short(item.raw_prediction)!r}   "
                       f"({item.failure_type}, NED={item.normalised_edit_distance:.2f})",
                       fontsize=8, loc="left")
        axis.axis("off")
    figure.suptitle(f"Worst OCR mistakes - pipeline '{pipeline}'", fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 6: Tesseract OCR on handwritten prescriptions")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N images (quick test)")
    parser.add_argument("--ablation", action="store_true",
                        help="also measure each preprocessing step on its own (slower)")
    args = parser.parse_args()

    heading("Step 0 - Check Tesseract")
    version = setup_tesseract()
    print(f"Tesseract {version} at {pytesseract.pytesseract.tesseract_cmd}")

    heading("Step 1 - Load labels")
    labels = pd.read_csv(LABELS_FILE)
    missing = [name for name in labels["filename"] if not (IMAGE_DIR / name).is_file()]
    if missing:
        print(f"WARNING: {len(missing)} labelled images are missing and will be skipped: {missing[:5]}")
        labels = labels[~labels["filename"].isin(missing)]
    if args.limit:
        labels = labels.head(args.limit)
    print(f"{len(labels)} labelled images, e.g. {labels['label'].head(3).tolist()}")

    heading("Step 2 - Run the OCR pipelines")
    RESULTS.mkdir(parents=True, exist_ok=True)
    predictions = run_ocr(labels.reset_index(drop=True), RESULTS / "preprocessed")
    predictions.to_csv(RESULTS / "predictions.csv", index=False)

    heading("Step 3 - Metrics")
    metrics = summarise(predictions)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    print(metrics.drop(columns="description").round(3).to_string(index=False))

    heading("Step 4 - Error analysis (most common failure types)")
    errors = error_analysis(predictions)
    errors.to_csv(RESULTS / "error_analysis.csv", index=False)
    print(errors[["pipeline", "failure_type", "count", "share_of_images"]].round(2).to_string(index=False))

    heading("Step 5 - Figures")
    best = metrics.sort_values("mean_normalised_edit_distance").iloc[0]["pipeline"]
    # Sample grid: the best pipeline's 3 best and 3 worst images, so both cases are visible.
    ranked = predictions[predictions["pipeline"] == best].sort_values("normalised_edit_distance")
    sample_ids = list(dict.fromkeys(ranked["image_id"].head(3).tolist() + ranked["image_id"].tail(3).tolist()))
    plot_sample_grid(predictions, sample_ids, RESULTS / "sample_grid.png")
    plot_failures(predictions, best, RESULTS / "ocr_failures.png")
    print(f"Best pipeline by edit distance: {best}")

    if args.ablation:
        heading("Step 6 - Ablation: which preprocessing step helps?")
        ablation = run_ablation(labels.reset_index(drop=True))
        ablation.to_csv(RESULTS / "ablation.csv", index=False)
        print(ablation.round(3).to_string(index=False))
    print(f"All files saved in {RESULTS}")


if __name__ == "__main__":
    main()
