from __future__ import annotations
import json
import re
from glob import glob
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

PUNCT_STRIP = '.,!?;:()[]{}"\'-'

def strip_brackets(token: str) -> str:
    """Remove angle brackets from a token. 
    Example: '<Anya>' -> 'Anya'."""
    return token[1:-1] if token.startswith("<") and token.endswith(">") else token

def tokenize_caption(caption: str) -> List[str]:
    """
    Split the caption into tokens and strip simple punctuation marks from both ends.
    Also removes special tokens like '</s>' from the model output.
    """
    caption = caption.replace("</s>", "")
    return [w.strip(PUNCT_STRIP) for w in caption.split() if w.strip(PUNCT_STRIP)]

def extract_matches_for_pair(
    caption: str,
    all_concepts: Sequence[str],
    gt_pair: Sequence[str],
) -> List[Optional[str]]:
    """
    For a single sample (image), return prediction results for its GT concept pair.
    The output length is always 2, with each position being one of:
      - The correct concept name (brackets removed, e.g., 'Anya')
      - "MISS" (another incorrect concept was detected)
      - None (no relevant concept detected)

    Logic:
      1) Detect concepts present in the caption from the set of all_concepts (token-level exact match).
      2) For each GT position:
         - If the GT appears → return GT name
         - If GT not found but another (non-GT) concept appears → return "MISS"
         - Otherwise → return None
    """
    tokens = set(tokenize_caption(caption))

    # Concepts present in caption (bracket-stripped form)
    present = {strip_brackets(c) for c in all_concepts if c in tokens or strip_brackets(c) in tokens}

    # Cleaned GT and non-GT sets
    gt_clean = [strip_brackets(g) for g in gt_pair]
    non_gt_clean = {strip_brackets(c) for c in all_concepts if c not in gt_pair}

    results: List[Optional[str]] = []
    for gt in gt_clean:
        if gt in present:
            results.append(gt)          # Correct detection
        elif present & non_gt_clean:
            results.append("MISS")      # Wrong concept present
        else:
            results.append(None)        # No detection
    return results

def compute_metrics(
    predictions: Sequence[Optional[str]],
    targets_with_brackets: Sequence[str],
) -> Tuple[float, float, float]:
    """
    Compute recall, precision, and F1 score.

    Args:
        predictions: flattened list like ['Anya', 'MISS', None, ...]
        targets_with_brackets: flattened list like ['<Anya>', '<Bond>', '<Bluey>', ...]

    Definitions:
        - correct: predictions[i] == strip_brackets(targets[i])
        - precision: correct / (# predictions != None) 
                     (includes "MISS" as non-None → penalizes precision)
        - recall: correct / total number of GT targets
        - f1: harmonic mean of precision and recall
    """
    target_clean = [strip_brackets(t) for t in targets_with_brackets]

    correct = sum(
        1 for pred, tgt in zip(predictions, target_clean)
        if pred is not None and pred != "MISS" and pred == tgt
    )
    predicted_non_none = sum(1 for p in predictions if p is not None)

    precision = correct / predicted_non_none if predicted_non_none > 0 else 0.0
    recall = correct / len(target_clean) if target_clean else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return recall, precision, f1

def load_records(json_path: Path) -> List[Tuple[str, List[str], str]]:
    """
    Load records from a JSON file.
    Each record is expected to have the format (based on the uploaded file):
        [
            image_path: str,
            concepts: ["<A>", "<B>"],    # 2 GT concepts
            model_caption: str,          # model-generated caption
            prompt_or_meta: str          # (unused)
        ]
    Returns:
        List of tuples: (image_path, concepts, model_caption)
    """
    with json_path.open("r") as f:
        data = json.load(f)

    records = []
    for row in data:
        image_path, concepts, caption = row[0], row[1], row[2]
        records.append((image_path, concepts, caption))
    return records

def evaluate_dir(json_glob: str) -> None:
    """
    Evaluate precision, recall, and F1 score for all JSON files matching the glob.
    Prints results per file.
    """
    for path in glob(json_glob):
        json_path = Path(path)
        records = load_records(json_path)

        # 1) Collect all concept candidates from the file
        all_concepts = sorted({c for _, cs, _ in records for c in cs})

        # 2) Flatten predictions and targets
        preds_flat: List[Optional[str]] = []
        tgts_flat: List[str] = []

        for _, gt_pair, caption in records:
            preds_flat.extend(extract_matches_for_pair(caption, all_concepts, gt_pair))
            tgts_flat.extend(gt_pair)  # keep original bracket form

        # 3) Compute metrics
        recall, precision, f1 = compute_metrics(preds_flat, tgts_flat)

        # 4) Print results
        print(
            f"file: {json_path.name:<60}  "
            f"metric (P/R/F1): {precision*100:5.1f} / {recall*100:5.1f} / {f1*100:5.1f}"
        )

if __name__ == "__main__":
    evaluate_dir("../save_script/*-2_concepts.json")