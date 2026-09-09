"""Per-error-type P/R/F0.5 breakdown for a run.

Usage:
  python -m en_evaluation.analyze_per_type <run_dir> [--m2 PATH]

Reads <run_dir>/results.json (must have source, reference, corrected) and the gold
M2 file (default: data/en_bea/train/bea-train.m2). Computes per-error-category
TP/FP/FN using ERRANT, then writes per_type_breakdown.txt and .json to the run dir.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from gec_metrics import get_metric

import sys as _sys
REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(REPO_ROOT))
from src.en_utils.tokenizer import tokenize as _en_tokenize  # noqa: E402


def _normalize(text: str) -> str:
    """Match en_main._normalize_for_en_metric: re-tokenize to BEA style."""
    try:
        text = _en_tokenize(str(text))
    except Exception:
        text = str(text)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def parse_m2_blocks(m2_path: Path) -> list[tuple[str, list[tuple[int, int, str, str]]]]:
    """Return list of (source, [(start, end, type, corr_str), ...]) per sentence."""
    blocks: list[tuple[str, list[tuple[int, int, str, str]]]] = []
    cur_source = None
    cur_edits: list[tuple[int, int, str, str]] = []
    with open(m2_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_source is not None:
                    blocks.append((cur_source, cur_edits))
                    cur_source = None
                    cur_edits = []
                continue
            if line.startswith("S "):
                if cur_source is not None:
                    blocks.append((cur_source, cur_edits))
                cur_source = line[2:]
                cur_edits = []
            elif line.startswith("A "):
                payload = line[2:]
                # format: start end|||type|||correction|||REQUIRED|||-NONE-|||annotator
                parts = payload.split("|||")
                span = parts[0]
                etype = parts[1]
                corr = parts[2]
                annot = int(parts[5]) if len(parts) > 5 else 0
                if annot != 0:
                    continue
                if etype == "noop":
                    continue
                s, e = span.split()
                cur_edits.append((int(s), int(e), etype, corr))
        if cur_source is not None:
            blocks.append((cur_source, cur_edits))
    return blocks


def normalize_type(t: str) -> str:
    """Strip the M/U/R operator prefix to get a coarse category like NOUN:NUM."""
    if t.startswith(("M:", "U:", "R:")):
        return t[2:]
    return t


def edit_key(s: int, e: int, corr: str) -> tuple[int, int, str]:
    """Match edits by span + correction string (whitespace-normalized)."""
    return (s, e, " ".join(corr.split()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--m2", default=str(REPO_ROOT / "data/en_bea/train/bea-train.m2"))
    parser.add_argument("--start_index", type=int, default=0,
                        help="0-based offset into the M2 file (must match the run's start_index).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    m2_path = Path(args.m2)

    with open(run_dir / "results.json", encoding="utf-8") as f:
        results = json.load(f)["results"]

    gold_blocks = parse_m2_blocks(m2_path)
    # Subset to the run's slice.
    gold_blocks = gold_blocks[args.start_index:args.start_index + len(results)]
    if len(gold_blocks) != len(results):
        raise ValueError(
            f"Length mismatch: {len(gold_blocks)} gold blocks vs {len(results)} predictions"
        )

    cls = get_metric("errant")
    metric = cls(cls.Config(beta=0.5, language="en"))

    # Aggregate per coarse type
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for (gold_src, gold_edits), row in zip(gold_blocks, results):
        # Tokenize source & hypothesis to align with gold M2 token positions.
        src = _normalize(row["source"])
        hyp = _normalize(row["corrected"])

        # System edits via ERRANT
        sys_edits = metric.edit_extraction(src, hyp)
        sys_set = {}
        for e in sys_edits:
            k = edit_key(e.o_start, e.o_end, e.c_str)
            sys_set[k] = e.type  # tagged type, e.g. "R:VERB:SVA"

        gold_set = {}
        for s, en, etype, corr in gold_edits:
            k = edit_key(s, en, corr)
            gold_set[k] = etype

        # TP / FN by gold type
        for k, gtype in gold_set.items():
            cat = normalize_type(gtype)
            if k in sys_set:
                tp[cat] += 1
            else:
                fn[cat] += 1

        # FP by system type
        for k, stype in sys_set.items():
            if k not in gold_set:
                cat = normalize_type(stype)
                fp[cat] += 1

    rows = []
    cats = sorted(set(tp) | set(fp) | set(fn))
    total_tp = total_fp = total_fn = 0
    for cat in cats:
        a, b, c = tp[cat], fp[cat], fn[cat]
        total_tp += a
        total_fp += b
        total_fn += c
        precision = a / (a + b) if (a + b) else 0.0
        recall = a / (a + c) if (a + c) else 0.0
        beta2 = 0.25
        f05 = (
            (1 + beta2) * precision * recall / (beta2 * precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append({
            "type": cat,
            "tp": a,
            "fp": b,
            "fn": c,
            "gold": a + c,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f0_5": round(f05, 4),
        })

    rows.sort(key=lambda r: r["gold"], reverse=True)

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f = (
        1.25 * overall_p * overall_r / (0.25 * overall_p + overall_r)
        if (overall_p + overall_r) > 0
        else 0.0
    )

    out_json = {
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "f0_5": round(overall_f, 4),
        },
        "by_type": rows,
    }
    with open(run_dir / "per_type_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    lines = [
        f"=== Per-Error-Type Breakdown ({run_dir.name}) ===",
        f"Overall (span+correction match): TP={total_tp}  FP={total_fp}  FN={total_fn}  "
        f"P={overall_p:.4f}  R={overall_r:.4f}  F0.5={overall_f:.4f}",
        "",
        f"{'TYPE':<14}{'GOLD':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>9}{'R':>9}{'F0.5':>9}",
    ]
    for r in rows:
        lines.append(
            f"{r['type']:<14}{r['gold']:>6}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}"
            f"{r['precision']:>9.4f}{r['recall']:>9.4f}{r['f0_5']:>9.4f}"
        )
    text = "\n".join(lines)
    with open(run_dir / "per_type_breakdown.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
