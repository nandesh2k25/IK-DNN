---
name: notebook_map
description: Keeps notebook_map.md (project root) in sync with the live Colab notebook's cell IDs. Consult/apply this whenever a Colab notebook cell is added, updated, deleted, or moved via colab-proxy-mcp tools, or when the user reports they made a manual edit in the browser and wants it logged. Only available in this project directory.
---

# notebook_map — keeping the cell-ID cache honest

`notebook_map.md` exists so future edits can target a Colab cell by ID directly (`update_cell`/`run_code_cell`) instead of fetching the whole notebook first — full fetches with outputs have repeatedly exceeded the tool's token limit in this project. The map is only safe to rely on if it's kept accurate. That's this skill's job.

## When this applies

- Any time you call `mcp__colab-proxy-mcp__update_cell`, `add_code_cell`, `add_text_cell`, `delete_cell`, or `move_cell` on the `august_thesis_claude` notebook.
- Any time the user says they edited the notebook themselves (in the browser) and wants it logged — they'll tell you explicitly; don't assume a manual edit happened otherwise.

## What to do, immediately after the edit (same turn, don't defer)

**`update_cell`** (content changed, ID unchanged): find that cell's row in `notebook_map.md` by ID, update its `Summary` column if the change affects what the cell does (new function signature, new purpose, section renumbered, etc.). Update the "Current key variables" section too if the edit changes what a downstream cell can assume exists.

**`add_code_cell` / `add_text_cell`** (returns `newCellId`): insert a new row at the correct position in the table (renumber the `#` index column for every row after the insertion point), using the returned ID directly — no re-fetch needed for this.

**`delete_cell`**: remove that row, renumber every row after it.

**`move_cell`**: reorder the row to match, renumber affected `#` indices.

**Manual browser edit reported by the user**: this map can't be trusted incrementally here since you didn't make the edit — do a full `get_cells` (no outputs) and rewrite the table from scratch, rather than guessing what changed.

After any of the above, update the "Last full sync" line's date/commit reference only if you just did a full rewrite (manual-edit case) or a batch of related edits — not required for every single small `update_cell`.

## Before trusting a cached ID for an edit

If it's been a while (new session, long conversation, context compaction) or you have any doubt the map might be stale, don't blindly trust it. Do a narrow `get_cells` (small `cellIndexStart`/`cellIndexEnd` range around the cell you're targeting, `includeOutputs: false`) to confirm the ID still points at what the map says before writing to it. This is cheap; a full-notebook fetch is not.

## What NOT to do

- Don't let `notebook_map.md` drift for "just one small edit" — the whole point is it stays trustworthy, and small unlogged edits are exactly how it goes stale.
- Don't skip the verification step above just because the map exists — it's a cache, not a guarantee.
- Don't fold this file's content into `thesis_doc.md` — that file is for thesis-writing content, this is engineering bookkeeping (same separation of concerns as `handoff.md`/`QUESTIONS.md`/`CLAUDE.md`, see `CLAUDE.md`'s "Where to look for more").
