"""
Universal Dynamic Grader
========================
Scores an agent's cleaned dataset against the ground truth embedded
in a dynamically-generated task. Works for both LLM-generated tasks
and static fallback tasks.

The grader is schema-agnostic: it inspects the ground truth structure
to determine which checks to apply, making it compatible with any
dataset domain the generator produces.
"""

from typing import Any, Dict, List, Tuple
import re
import datetime


# ── Helpers ───────────────────────────────────────────────────────────────

def _pk_column(rows: List[Dict]) -> str:
    """Guess the primary key column name."""
    if not rows:
        return "id"
    candidates = [k for k in rows[0].keys() if k in ("id", "order_id", "tx_id", "employee_id", "record_id")]
    return candidates[0] if candidates else list(rows[0].keys())[0]


def _rows_by_pk(rows: List[Dict], pk: str) -> Dict[Any, Dict]:
    return {r.get(pk): r for r in rows if pk in r}


def _normalise_val(v: Any) -> Any:
    """Light normalisation for comparison tolerance."""
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, str):
        return v.strip()
    return v


# ── Main grader ───────────────────────────────────────────────────────────

def grade_dynamic_task(task: Dict[str, Any], action_dict: Dict[str, Any]) -> Tuple[float, str]:
    """
    Grade a cleaned dataset against the task's ground truth.
    Returns (score 0.0-1.0, feedback string).
    """
    submitted: List[Dict] = action_dict.get("cleaned_rows", [])
    gt: Dict = task.get("_ground_truth", {})
    clean_rows: List[Dict] = gt.get("clean_rows", [])
    feedback_parts = []
    score = 0.0

    if not clean_rows:
        return 0.5, "No ground truth available — awarded partial credit."

    pk = _pk_column(clean_rows)
    expected_by_pk = _rows_by_pk(clean_rows, pk)
    submitted_by_pk = _rows_by_pk(submitted, pk)

    # ── Component 1: Row count / dropped rows (25%) ───────────────────────
    expected_count = len(clean_rows)
    submitted_count = len(submitted)
    if submitted_count == expected_count:
        score += 0.25
        feedback_parts.append(f"Correct row count: {submitted_count} (+0.25)")
    elif abs(submitted_count - expected_count) == 1:
        score += 0.12
        feedback_parts.append(f"Row count off by 1 — expected {expected_count}, got {submitted_count} (+0.12)")
    else:
        feedback_parts.append(f"Wrong row count — expected {expected_count}, got {submitted_count}")

    # ── Component 2: Correct PKs present (no extra, none missing) (15%) ──
    expected_pks  = set(expected_by_pk.keys())
    submitted_pks = set(submitted_by_pk.keys())
    if expected_pks == submitted_pks:
        score += 0.15
        feedback_parts.append(f"All {len(expected_pks)} expected rows present (+0.15)")
    else:
        missing = expected_pks - submitted_pks
        extra   = submitted_pks - expected_pks
        if missing:
            feedback_parts.append(f"Missing rows with {pk} in: {missing}")
        if extra:
            feedback_parts.append(f"Unexpected rows with {pk} in: {extra}")

    # ── Component 3: Column-level correctness (60% split across columns) ──
    if not clean_rows:
        return round(score, 2), "\n".join(feedback_parts)

    columns = [c for c in clean_rows[0].keys() if c != pk]
    if not columns:
        score += 0.60
        feedback_parts.append("No non-PK columns to check (+0.60)")
    else:
        per_col = 0.60 / len(columns)
        for col in columns:
            col_correct = 0
            col_total   = len([pk_val for pk_val in expected_pks if pk_val in submitted_by_pk])

            for pk_val, expected_row in expected_by_pk.items():
                sub_row = submitted_by_pk.get(pk_val)
                if sub_row is None:
                    continue
                expected_val  = _normalise_val(expected_row.get(col))
                submitted_val = _normalise_val(sub_row.get(col))

                # Numeric tolerance
                if isinstance(expected_val, float) and submitted_val is not None:
                    try:
                        if abs(float(submitted_val) - expected_val) < 0.02:
                            col_correct += 1
                            continue
                    except (TypeError, ValueError):
                        pass
                # None check
                if expected_val is None and submitted_val is None:
                    col_correct += 1
                    continue
                # Exact match
                if submitted_val == expected_val:
                    col_correct += 1
                elif str(submitted_val).lower() == str(expected_val).lower():
                    col_correct += 1  # case-insensitive pass

            if col_total > 0:
                col_score = per_col * (col_correct / col_total)
                score += col_score
                if col_correct == col_total:
                    feedback_parts.append(f"Column '{col}': all {col_total} values correct (+{col_score:.2f})")
                else:
                    feedback_parts.append(
                        f"Column '{col}': {col_correct}/{col_total} correct "
                        f"(+{col_score:.2f}/{per_col:.2f})"
                    )

    final = round(min(score, 1.0), 4)
    # Clamp to strictly (0, 1) — validators reject exact 0.0 and 1.0
    if final <= 0.0:
        final = 0.01
    elif final >= 1.0:
        final = 0.99
    return final, "\n".join(feedback_parts)
