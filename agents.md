## Agents Working Rules

**Critical Safety Rules Must be followed every turn and regressions must be proactively reported**
- Zeroth Rule: If an edit causes a file to be deleted, corrupted, or severely damaged, immediately stop work, report what happened and what you were doing, and propose restoration options (restore from .bak, rebuild selectively, or request user direction).
- Backup Before Edit: Before modifying any existing file, create a .bak copy (increment suffix as needed). Never skip this step.

## Editing & Checks
- Check planning docs and suggest next likely steps after each turn.
- After edits, run format, lint, and build checks as appropriate. Never run full applications unless explicitly instructed.
- If checks fail, fix the errors and rerun until clean.
- Git usage: read-only only (status, diff, show). Do not pull, push, reset, or restore from HEAD.
- End-of-turn backups: remove temporary .bak files using Remove-ItemSafely (Recycle module). If unavailable, move backups into a root .bak folder instead of deleting.
- Update planning docs / Checklist files after each turn. **Add entries for ad-hoc edits.**
- "_Plan" file is the **source of truth** / Checklist is the current stepwise approach.
- Windows note: file locks sometimes block builds/tests; pausing briefly and rerunning (especially tests) often clears the lock. Avoid repeated rapid retries if the OS might still be holding the file.

## Coding Conventions & Design
- Design for modularity and extensibility; keep a clear separation of concerns.
- Include stub placeholders with TODOs for planned future features.
- Comment only where necessary; avoid unnecessary clutter.
- Favor small, focused modules/functions; refactor if a file approaches “god class” size (~3000+ lines).
- Follow project-appropriate formatting/linting; prefer deterministic, reproducible outputs.
