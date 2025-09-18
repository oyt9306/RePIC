from __future__ import annotations
import json
from glob import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Characters to strip from token boundaries
PUNCT_STRIP = '.,!?;:()[]{}"\'-'

# ---------------- Text & concept normalization ----------------

def strip_brackets(token: str) -> str:
    """Remove surrounding angle brackets if present. Example: '<Backpack>' -> 'Backpack'."""
    return token[1:-1] if token.startswith("<") and token.endswith(">") else token

def tokenize_caption(caption: str) -> List[str]:
    """
    Tokenize a caption into lowercased tokens, removing boundary punctuation.
    Also removes model special tokens like '</s>'.
    """
    caption = caption.replace("</s>", "").lower()
    return [w.strip(PUNCT_STRIP) for w in caption.split() if w.strip(PUNCT_STRIP)]

# ---------------- Single-concept matching ----------------

def predict_for_single_gt(
    caption: str,
    all_concepts: Sequence[str],
    gt_concept: str,
) -> Optional[str]:
    """
    Return a prediction aligned with the single GT concept:
      - the correct concept name (unbracketed, lowercased), e.g., 'backpack'
      - "MISS"  (a different concept from the candidate pool is detected)
      - None    (no concept signal in the caption)

    Matching is token-level exact match (case-insensitive) against:
      - the bracketed form (e.g., '<backpack>') and
      - the unbracketed form (e.g., 'backpack').
    """
    tokens = set(tokenize_caption(caption))  # lowercased token set

    # Concepts present in caption (unbracketed, lowercased)
    present_unbr = {
        strip_brackets(c.lower())
        for c in all_concepts
        if (c.lower() in tokens) or (strip_brackets(c.lower()) in tokens)
    }

    gt_unbr = strip_brackets(gt_concept).lower()
    non_gt_unbr = {
        strip_brackets(c.lower())
        for c in all_concepts
        if strip_brackets(c.lower()) != gt_unbr
    }

    if gt_unbr in present_unbr:
        return gt_unbr                  # correct detection
    if present_unbr & non_gt_unbr:
        return "MISS"                   # wrong concept detected
    return None                         # no signal

# ---------------- Metrics ----------------

def compute_metrics(
    predictions: Sequence[Optional[str]],
    targets_with_brackets: Sequence[str],
) -> Tuple[float, float, float]:
    """
    Compute recall, precision, and F1.

      correct:   predictions[i] == strip_brackets(targets[i]).lower()
      precision: correct / (# predictions != None)   ["MISS" counts in denominator]
      recall:    correct / (total GT)
      F1:        harmonic mean of precision and recall
    """
    target_clean = [strip_brackets(t).lower() for t in targets_with_brackets]

    correct = sum(
        1 for pred, tgt in zip(predictions, target_clean)
        if pred is not None and pred != "MISS" and pred == tgt
    )
    predicted_non_none = sum(p is not None for p in predictions)

    precision = (correct / predicted_non_none) if predicted_non_none > 0 else 0.0
    recall = (correct / len(target_clean)) if target_clean else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return recall, precision, f1

# ---------------- I/O helpers ----------------

def load_records_single(json_path: Path) -> List[Tuple[str, str, str]]:
    """
    Load records from a JSON file where each row is:
        [image_path: str, ["<concept>"], model_caption: str, ...]
    Returns: List of (image_path, gt_concept, caption)
    """
    with json_path.open("r") as f:
        data = json.load(f)

    records = []
    for row in data:
        image_path = row[0]
        gt_concept = row[1][0]  # single concept (bracketed)
        caption = row[2]
        records.append((image_path, gt_concept, caption))
    return records

# ---------------- Main evaluation ----------------

def evaluate_glob_single(json_glob: str) -> None:
    """
    Evaluate all JSON files matching the glob pattern for single-concept detection.
    Prints per-file Precision / Recall / F1 (percent).
    """
    for path in glob(json_glob):
        jp = Path(path)
        records = load_records_single(jp)

        # Candidate pool from this file (original bracketed forms)
        all_concepts = sorted({c for _, c, _ in records})

        # Build aligned predictions/targets
        preds: List[Optional[str]] = []
        tgts:  List[str] = []
        for _, gt_c, caption in records:
            preds.append(predict_for_single_gt(caption, all_concepts, gt_c))
            tgts.append(gt_c)

        recall, precision, f1 = compute_metrics(preds, tgts)

        print(
            f"file: {jp.name:<60}  "
            f"metric (P/R/F1): {precision*100:5.1f} / {recall*100:5.1f} / {f1*100:5.1f}"
        )

if __name__ == "__main__":
    evaluate_glob_single("../save_script/single*.json")