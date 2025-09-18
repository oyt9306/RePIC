from __future__ import annotations
import json
from glob import glob
from pathlib import Path
from typing import List, Sequence, Tuple, Optional, Iterable

# Characters to strip from token boundaries
PUNCT_STRIP = '.,!?;:()[]{}"\'-'

# --------- Text & concept normalization ---------

def strip_brackets(token: str) -> str:
    """Remove surrounding angle brackets if present. '<Thor>' -> 'Thor'."""
    return token[1:-1] if token.startswith("<") and token.endswith(">") else token

def tokenize_caption(caption: str) -> List[str]:
    """
    Tokenize a caption into lowercased tokens, removing a small set of boundary punctuations.
    Also removes '</s>' if present in model outputs.
    """
    caption = caption.replace("</s>", "").lower()
    return [w.strip(PUNCT_STRIP) for w in caption.split() if w.strip(PUNCT_STRIP)]

def normalize_concepts(concepts: Iterable[str]) -> List[str]:
    """Lowercase all concepts as strings (including angle brackets)."""
    return [c.lower() for c in concepts]

# --------- Matching logic (N concepts) ---------

def predict_for_gt_list(
    caption: str,
    all_concepts: Sequence[str],
    gt_list: Sequence[str],
) -> List[Optional[str]]:
    """
    For a single sample, return predictions aligned with its ground-truth concepts (N-long).
    For each GT, output one of:
      - correct concept name (brackets removed, e.g., 'thor')
      - "MISS" (a different concept was detected in the caption)
      - None (no concept signal)
    Matching is token-level exact match (case-insensitive) against:
      - the bracketed form (e.g., '<thor>') and
      - the unbracketed form (e.g., 'thor').
    """
    tokens = set(tokenize_caption(caption))  # lowercased token set

    # Build the set of concept strings present in the caption (unbracketed form)
    # We check both '<name>' and 'name' against tokens.
    all_concepts_norm = normalize_concepts(all_concepts)
    present_unbracketed = {
        strip_brackets(c) for c in all_concepts_norm
        if (c in tokens) or (strip_brackets(c) in tokens)
    }

    # Prepare GT (lowercased; keep unbracketed string for comparison)
    gt_unbr = [strip_brackets(g.lower()) for g in gt_list]
    non_gt_unbr = {strip_brackets(c.lower()) for c in all_concepts if c not in gt_list}

    preds: List[Optional[str]] = []
    for gt in gt_unbr:
        if gt in present_unbracketed:
            preds.append(gt)          # correct detection
        elif present_unbracketed & non_gt_unbr:
            preds.append("MISS")      # wrong concept detected
        else:
            preds.append(None)        # no signal
    return preds

# --------- Metrics ---------

def compute_metrics(
    predictions: Sequence[Optional[str]],
    targets_with_brackets: Sequence[str],
) -> Tuple[float, float, float]:
    """
    Compute recall, precision, and F1.
      correct: predictions[i] == strip_brackets(targets[i])  (case-insensitive)
      precision: correct / (# predictions != None)    ["MISS" counts in denominator]
      recall:    correct / (total GT)
    """
    target_clean = [strip_brackets(t).lower() for t in targets_with_brackets]

    correct = 0
    for pred, tgt in zip(predictions, target_clean):
        if pred is not None and pred != "MISS" and pred.lower() == tgt:
            correct += 1

    predicted_non_none = sum(p is not None for p in predictions)
    precision = (correct / predicted_non_none) if predicted_non_none > 0 else 0.0
    recall = (correct / len(target_clean)) if target_clean else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return recall, precision, f1

# --------- I/O helpers ---------

def load_records(json_path: Path) -> List[Tuple[str, List[str], str]]:
    """
    Load records from a JSON file.
    Each row is expected as:
      [
        image_path: str,
        concepts: List[str],        # N GT concepts, e.g., ["<thor>", "<ironman>", ...]
        model_caption: str,         # model-generated caption
        ...                         # optional extra fields are ignored
      ]
    Returns: List of (image_path, concepts, caption)
    """
    with json_path.open("r") as f:
        data = json.load(f)

    records = []
    for row in data:
        image_path = row[0]
        concepts = row[1]
        caption = row[2]
        records.append((image_path, concepts, caption))
    return records

# --------- Main evaluation ---------

def evaluate_glob(json_glob: str) -> None:
    """
    Evaluate all JSON files matching a glob pattern.
    Prints per-file Precision / Recall / F1 (in percent).
    """
    for path in glob(json_glob):
        jp = Path(path)
        records = load_records(jp)

        # Pool of candidate concepts in this file (as provided; keep original forms)
        all_concepts: List[str] = sorted({c for _, cs, _ in records for c in cs})

        # Flatten predictions and targets
        preds_flat: List[Optional[str]] = []
        tgts_flat: List[str] = []

        for _, gt_list, caption in records:
            preds_flat.extend(predict_for_gt_list(caption, all_concepts, gt_list))
            tgts_flat.extend(gt_list)  # keep bracketed strings

        recall, precision, f1 = compute_metrics(preds_flat, tgts_flat)

        print(
            f"file: {jp.name:<60}  "
            f"metric (P/R/F1): {precision*100:5.1f} / {recall*100:5.1f} / {f1*100:5.1f}"
        )

if __name__ == "__main__":
    evaluate_glob("../save_script/*-4_concepts.json")