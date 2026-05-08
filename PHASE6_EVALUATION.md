# Phase 6 - Evaluation and Regression Safety

Status: Completed

## Phase Goal
Create measurable quality checks so improvements are validated and regressions are visible.

## Tasks
1. Add benchmark case storage API.
2. Add evaluation runner API.
3. Compute pass/fail and aggregate scores.
4. Log evaluation runs for observability.
5. Keep evaluation mode configurable (quick/analytical).

---

## Delivered
- Added `GET /api/eval/cases`.
- Added `POST /api/eval/cases`.
- Added `POST /api/eval/run`.
- Evaluation computes:
  - keyword match score
  - confidence score
  - pass/fail per case
  - aggregated pass rate and averages
- Evaluation run summary is logged through audit event `eval.run`.
