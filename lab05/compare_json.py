from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def compare_json_nodes(
    expected: Any,
    actual: Any,
    path: str = "$",
    float_tol: float = 1e-5,
    mismatches: list[str] | None = None,
) -> list[str]:
    """Recursively compare two JSON structures with float tolerance."""
    if mismatches is None:
        mismatches = []

    # Type check (allow int-to-float cross comparison for numbers)
    is_exp_num = isinstance(expected, (int, float)) and not isinstance(expected, bool)
    is_act_num = isinstance(actual, (int, float)) and not isinstance(actual, bool)

    if is_exp_num and is_act_num:
        if not math.isclose(expected, actual, rel_tol=float_tol, abs_tol=float_tol):
            mismatches.append(
                f"{path}: expected numeric {expected!r}, got {actual!r} "
                f"(diff: {abs(expected - actual):.2e})"
            )
        return mismatches

    if type(expected) is not type(actual):
        mismatches.append(
            f"{path}: type mismatch (expected {type(expected).__name__}, "
            f"got {type(actual).__name__})"
        )
        return mismatches

    # Dict comparison
    if isinstance(expected, dict):
        exp_keys = set(expected.keys())
        act_keys = set(actual.keys())

        missing = exp_keys - act_keys
        extra = act_keys - exp_keys

        for k in sorted(missing):
            mismatches.append(f"{path}.{k}: missing key in student JSON")
        for k in sorted(extra):
            mismatches.append(f"{path}.{k}: unexpected extra key in student JSON")

        for k in sorted(exp_keys & act_keys):
            compare_json_nodes(
                expected[k], actual[k], f"{path}.{k}", float_tol, mismatches
            )

    # List/Array comparison
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            mismatches.append(
                f"{path}: list length mismatch (expected {len(expected)}, got {len(actual)})"
            )
        # Compare element-by-element up to the shortest length
        for i, (e_elem, a_elem) in enumerate(zip(expected, actual)):
            compare_json_nodes(
                e_elem, a_elem, f"{path}[{i}]", float_tol, mismatches
            )

    # Exact equality for strings, bools, None
    else:
        if expected != actual:
            mismatches.append(
                f"{path}: expected {expected!r}, got {actual!r}"
            )

    return mismatches


def compare_prune_files(
    expected_path: str | Path,
    student_path: str | Path,
    float_tol: float = 1e-5,
    max_errors_display: int = 15,
) -> bool:
    """Load both files, run the comparison, and print an evaluation summary."""
    exp_p = Path(expected_path)
    stu_p = Path(student_path)

    if not exp_p.is_file():
        raise FileNotFoundError(f"Reference file not found: {exp_p}")
    if not stu_p.is_file():
        raise FileNotFoundError(f"Student file not found: {stu_p}")

    with open(exp_p, "r", encoding="utf-8") as f:
        reference_data = json.load(f)

    with open(stu_p, "r", encoding="utf-8") as f:
        student_data = json.load(f)

    mismatches = compare_json_nodes(
        reference_data, student_data, float_tol=float_tol
    )

    print("=" * 70)
    print("PRUNING JSON VERIFICATION REPORT")
    print(f"Reference: {exp_p.name}")
    print(f"Student  : {stu_p.name}")
    print("=" * 70)

    if not mismatches:
        print("RESULT: ALL CHECKS PASSED (100% Match)")
        print("All criteria, masks, indices, bytes, and sweep rows match reference.")
        print("=" * 70)
        return True

    print(f"RESULT: FAILED with {len(mismatches)} mismatch(es):\n")
    for error in mismatches[:max_errors_display]:
        print(f"  ✗ {error}")

    if len(mismatches) > max_errors_display:
        print(f"\n  ... and {len(mismatches) - max_errors_display} more errors omitted.")

    print("=" * 70)
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare student's prune_outputs.json with reference output."
    )
    parser.add_argument("reference", help="Path to your ground-truth JSON file")
    parser.add_argument("student", help="Path to the student's JSON file")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Relative/absolute float tolerance (default: 1e-5)",
    )
    args = parser.parse_args()

    passed = compare_prune_files(args.reference, args.student, float_tol=args.tol)
    raise SystemExit(0 if passed else 1)