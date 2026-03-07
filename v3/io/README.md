# io/

This folder handles output formatting and submission writing.

## Files

- `submission.py`
  - `build_submission_df()`

## Responsibilities

- Convert prediction frames into challenge format.
- Keep column contract stable: `buyer_id,predicted_id`.
- Merge warm and cold predictions consistently.

## Why This Is Important

Formatting issues can invalidate an otherwise strong model.
By isolating submission logic, you reduce risk of accidental column changes during modeling experiments.
