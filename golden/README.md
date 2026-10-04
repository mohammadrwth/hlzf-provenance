# Golden set

Hand labels for real DSO publications, used by `hlzf eval`. Only files with `verified: true`
count, and only a person who read the PDF should set it.

Three templates are prepared, chosen to cover the cases that matter:

| File | Why this document |
|---|---|
| `wunsiedel-2024.yaml` | simple: three levels, one window per cell |
| `bayreuth-2026-korr.yaml` | the corrected re-publication; label what this file prints |
| `n-ergie-2026.yaml` | several windows per cell, very short windows, two-state holiday rule |

How to label, about 15 minutes per document:

1. `uv run hlzf fetch --only <id>` and open `data/raw/<id>.pdf`. Do not open the review UI for
   this document first, so its answers cannot anchor yours.
2. For every cell write the printed windows as `"HH:MM-HH:MM"`, `[]` if the cell is printed
   empty. Delete cells the table does not have.
3. Set `convention`, list the rules the document states, put your name in `labeled_by` and set
   `verified: true`.
4. `uv run hlzf eval --write-readme` updates the results table in the README.

For other documents, `uv run hlzf golden-template <id>` writes a blank grid after the
document has been processed.
