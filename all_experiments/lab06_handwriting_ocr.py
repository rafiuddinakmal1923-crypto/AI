
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # save figures to file, no pop-up window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageFile, ImageOps

# ---------------------------------------------------------------------------
# Paths are built from this file's location, so it works from any terminal folder.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                # .../AI/all_experiments
ROOT = HERE.parent                                    # .../AI
DATA = ROOT / "all_datasets" / "Lab 6"
IMAGE_DIR = DATA / "img" / "img"
LABELS_FILE = DATA / "doctor_handwriting_labels.csv"
RESULTS = ROOT / "results" / "lab06"

# Fixed experiment settings (from the handout). Only the IMAGE changes between pipelines.
OCR_CONFIG = "--oem 1 --psm 7"
BORDER_PX = 20
MAX_WIDTH = 1200
ENLARGE = 1.5

# Some phone photos are slightly truncated; load them anyway instead of crashing.
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
# Setup - make sure Tesseract can be found
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
# Question 1 - validate the data
# ---------------------------------------------------------------------------
def load_labels(limit: int | None = None) -> pd.DataFrame:
    """Read the label file and check every labelled image exists exactly once."""
    labels = pd.read_csv(LABELS_FILE)
    labels["path"] = [IMAGE_DIR / name for name in labels["filename"]]
    missing = [str(p.name) for p in labels["path"] if not p.is_file()]
    assert labels["filename"].is_unique, "a filename appears twice in the label file"
    assert not missing, f"labelled images missing from {IMAGE_DIR}: {missing[:5]}"
    if limit:
        labels = labels.head(limit)
    return labels.reset_index(drop=True)


def load_image(path: Path) -> Image.Image:
    """Read the pixels now (and close the file) - the raw photo, exactly as stored."""
    with Image.open(path) as image:
        return image.copy()


def plot_sample_grid(labels: pd.DataFrame, path: Path) -> None:
    """3 x 3 grid of the first nine photos with their labels."""
    figure, axes = plt.subplots(3, 3, figsize=(10, 6))
    for axis, row in zip(axes.flat, labels.head(9).itertuples()):
        axis.imshow(load_image(row.path))
        axis.set_title(row.label, fontsize=9)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Question 2 - OCR and normalisation
# ---------------------------------------------------------------------------
def normalise(text: str) -> str:
    """Lowercase and keep only letters and digits: 'Tab Azimax 40mg' -> 'tabazimax40mg'.

    The SAME rule is applied to the label and the prediction, so differences in case,
    spacing or punctuation are not counted as OCR errors.
    """
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def read_text(image: Image.Image) -> tuple[str, str, str]:
    """OCR one image. Returns (prediction, status, detail).

    If Tesseract itself fails on an image we RECORD the failure ("engine_error") and keep
    the image in the benchmark, instead of crashing or silently dropping it.
    """
    try:
        return pytesseract.image_to_string(image, config=OCR_CONFIG).strip(), "ok", ""
    except pytesseract.TesseractError as error:
        return "", "engine_error", str(error.message)[:160]


def run_ocr(labels: pd.DataFrame, images: list, pipeline: str) -> pd.DataFrame:
    """One auditable row per image: label, prediction, both normalised, engine status."""
    rows = []
    for item, image in zip(labels.itertuples(), images):
        prediction, status, detail = read_text(image)
        rows.append({
            "filename": item.filename,
            "pipeline": pipeline,
            "truth": item.label,
            "prediction": prediction,
            "truth_norm": normalise(item.label),
            "prediction_norm": normalise(prediction),
            "ocr_status": status,
            "ocr_detail": detail,
        })
        done = len(rows)
        if done % 20 == 0 or done == len(labels):
            print(f"  {pipeline}: {done}/{len(labels)} images")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Question 3 - measure recognition quality
# ---------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    """Minimum number of single-character inserts, deletes or substitutions to turn a into b.

    Dynamic programming, keeping one row of the table at a time:
    previous[j] = distance between the first i-1 characters of a and the first j of b.
    """
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                current[j - 1] + 1,                     # insert char_b
                previous[j] + 1,                        # delete char_a
                previous[j - 1] + (char_a != char_b),   # substitute (free if equal)
            ))
        previous = current
    return previous[-1]


def add_distances(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["exact"] = frame["truth_norm"] == frame["prediction_norm"]
    frame["edits"] = [levenshtein(t, p) for t, p in zip(frame["truth_norm"], frame["prediction_norm"])]
    # divide by the LONGER string so the value stays between 0 (identical) and 1
    longest = np.maximum(frame["truth_norm"].str.len(), frame["prediction_norm"].str.len()).clip(lower=1)
    frame["normalised_edit_distance"] = frame["edits"] / longest
    return frame


def summarise(frame: pd.DataFrame) -> dict:
    return {
        "pipeline": frame["pipeline"].iat[0],
        "exact_match_accuracy": frame["exact"].mean(),
        "mean_normalised_edit_distance": frame["normalised_edit_distance"].mean(),
        # CER = total edits / total reference characters (can be > 1 if the OCR adds junk)
        "character_error_rate": frame["edits"].sum() / frame["truth_norm"].str.len().sum(),
        "blank_predictions": int((frame["prediction_norm"] == "").sum()),
        "engine_errors": int((frame["ocr_status"] == "engine_error").sum()),
    }


# ---------------------------------------------------------------------------
# Question 4 - one preprocessing pipeline
# ---------------------------------------------------------------------------
def to_grayscale(image: Image.Image) -> Image.Image:
    """Turn the photo the right way up (phones store rotation in EXIF) and drop colour."""
    return ImageOps.exif_transpose(image).convert("L")


def add_border(image: Image.Image) -> Image.Image:
    return ImageOps.expand(image, border=BORDER_PX, fill=255)      # text must not touch the edge


def cap_width(image: Image.Image) -> Image.Image:
    if image.width > MAX_WIDTH:                                     # bound the runtime
        new_height = round(image.height * MAX_WIDTH / image.width)
        image = image.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)
    return image


def enlarge(image: Image.Image) -> Image.Image:
    new_size = (round(image.width * ENLARGE), round(image.height * ENLARGE))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def preprocess(image: Image.Image) -> Image.Image:
    """grayscale -> 20 px white border -> cap width at 1,200 px -> enlarge 1.5x."""
    return enlarge(cap_width(add_border(to_grayscale(image))))


# ---------------------------------------------------------------------------
# Question 6 - failure analysis
# ---------------------------------------------------------------------------
def failure_type(row) -> str:
    """engine error = Tesseract crashed; no text detected = segmentation failure (found no
    text line); character confusion = text was found but read wrongly (recognition failure)."""
    if row.ocr_status == "engine_error":
        return "engine error"
    if row.prediction_norm == "":
        return "no text detected"
    return "character confusion"


def analyse_failures(frame: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """The n worst non-exact predictions (highest normalised edit distance), tagged."""
    worst = (frame[~frame["exact"]]
             .sort_values(["normalised_edit_distance", "filename"], ascending=[False, True])
             .head(n).copy())
    worst["failure_type"] = [failure_type(row) for row in worst.itertuples()]
    return worst


def plot_failures(errors: pd.DataFrame, path: Path) -> None:
    """The six hardest images with reference text and OCR output."""
    figure, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    for axis, row in zip(axes.flat, errors.head(6).itertuples()):
        axis.imshow(load_image(IMAGE_DIR / row.filename))
        axis.set_title(f"Reference: {row.truth}\nOCR: {row.prediction[:28] or '(blank)'}", fontsize=8)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def failure_counts(frame: pd.DataFrame) -> pd.Series:
    """Failure type of EVERY non-exact prediction (not only the worst eight)."""
    wrong = frame[~frame["exact"]]
    return pd.Series([failure_type(row) for row in wrong.itertuples()]).value_counts()


# ---------------------------------------------------------------------------
# Extra (my own, optional): ablation - add the preprocessing steps one at a time
# ---------------------------------------------------------------------------
def run_ablation(labels: pd.DataFrame, originals: list) -> pd.DataFrame:
    """Same 89 images, same OCR config, same metrics; only the preprocessing differs."""
    steps = {
        "1 raw photo (baseline)": lambda im: im,
        "2 + grayscale": to_grayscale,
        "3 + 20 px border": lambda im: add_border(to_grayscale(im)),
        "4 + width cap 1200": lambda im: cap_width(add_border(to_grayscale(im))),
        "5 + enlarge 1.5x (= handout pipeline)": preprocess,
        "extra: grayscale + autocontrast, then steps 3-5":
            lambda im: enlarge(cap_width(add_border(ImageOps.autocontrast(to_grayscale(im), cutoff=2)))),
    }
    rows = []
    for step, prepare in steps.items():
        frame = add_distances(run_ocr(labels, [prepare(im) for im in originals], step))
        rows.append(summarise(frame))
    return pd.DataFrame(rows).rename(columns={"pipeline": "step"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 6: Tesseract OCR on handwritten prescriptions")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N images (quick test)")
    parser.add_argument("--ablation", action="store_true",
                        help="extra: also measure each preprocessing step on its own")
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    RESULTS.mkdir(parents=True, exist_ok=True)

    heading("Setup - Tesseract")
    version = setup_tesseract()
    print(f"OCR mode: live | Tesseract {version} at {pytesseract.pytesseract.tesseract_cmd}")
    print(f"OCR configuration (fixed for every image): {OCR_CONFIG}")

    heading("Question 1 - Validate the data")
    labels = load_labels(args.limit)
    print(f"Labelled images: {len(labels)} | unique labels: {labels['label'].nunique()}")
    plot_sample_grid(labels, RESULTS / "sample_grid.png")

    heading("Question 2 - Baseline OCR on the raw photos")
    originals = [load_image(p) for p in labels["path"]]
    baseline = run_ocr(labels, originals, "baseline")
    print(baseline[["truth", "prediction", "prediction_norm", "ocr_status"]].head(8).to_string(index=False))
    print("OCR engine errors:", int((baseline["ocr_status"] == "engine_error").sum()))

    heading("Question 3 - Measure baseline quality")
    baseline = add_distances(baseline)
    print(pd.DataFrame([summarise(baseline)]).round(4).to_string(index=False))

    heading("Question 4 - Preprocess (grayscale, border, width cap, 1.5x)")
    save_dir = RESULTS / "preprocessed"
    save_dir.mkdir(exist_ok=True)
    prepared = []
    for i, image in enumerate(originals):
        cleaned = preprocess(image)
        cleaned.save(save_dir / f"{i:03d}.png")
        prepared.append(cleaned)
    print(f"Prepared images: {len(prepared)} | example size: {prepared[0].size} mode: {prepared[0].mode}")

    heading("Question 5 - Compare the two pipelines fairly")
    processed = add_distances(run_ocr(labels, prepared, "grayscale_border_resize"))
    pd.concat([baseline, processed]).to_csv(RESULTS / "predictions.csv", index=False)
    metrics = pd.DataFrame([summarise(baseline), summarise(processed)]).set_index("pipeline")
    metrics.to_csv(RESULTS / "metrics.csv")
    print(metrics.round(4).to_string())
    change = (metrics.loc["grayscale_border_resize"] - metrics.loc["baseline"]).rename("change (preprocessed - baseline)")
    print("\n" + change.round(3).to_frame().to_string())

    heading("Question 6 - Analyse the failures")
    errors = analyse_failures(processed)
    errors[["filename", "truth", "prediction", "failure_type"]].to_csv(RESULTS / "error_analysis.csv", index=False)
    print(errors[["truth", "prediction", "failure_type"]].to_string(index=False))
    plot_failures(errors, RESULTS / "ocr_failures.png")
    print("\nFailure types over ALL wrong predictions:")
    for name, frame in [("baseline", baseline), ("grayscale_border_resize", processed)]:
        print(f"  {name}: {failure_counts(frame).to_dict()}")

    if args.ablation:
        heading("Extra - Ablation: which preprocessing step helps?")
        ablation = run_ablation(labels, originals)
        ablation.to_csv(RESULTS / "ablation.csv", index=False)
        print(ablation.round(3).to_string(index=False))

    print(f"\nSaved in {RESULTS}: {sorted(p.name for p in RESULTS.iterdir())}")


if __name__ == "__main__":
    main()
