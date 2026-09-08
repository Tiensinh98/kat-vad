# Lessons constitution — governance

## Purpose

The catalog exists to stop the same mistake twice. It is not a style guide, not a
tutorial, and not a changelog. Every entry costs context on every task, so the
bar is deliberately high.

## Quality standards

- **One lesson, one failure mode.** If the rule needs "and", it is two lessons.
- **Cite the code.** Every lesson carries `path:line`. A lesson whose citation
  no longer exists is a retirement candidate.
- **Rule starts with a verb** and is checkable from a diff.
- **Bad/Good must be real code** from this repo, not pseudocode.
- **No lesson without evidence.** See `GATES.md` gate 1.

## Size discipline

- `meta-index.md` stays under ~500 tokens. It holds CRITICAL/HIGH inline plus the
  trigger map — nothing else.
- If `index.md` passes ~40 entries, run a merge/retire pass before adding more.

## Merge rules

Two lessons merge when they share a root cause, even if the symptoms differ.
Keep the more general statement, union the triggers, keep both file citations.

## Retirement criteria

Retire (move to `archive.md`, never delete) when any holds:

1. The cited code no longer exists and the pattern is gone from the repo.
2. A linter, type checker, or test now catches it automatically.
3. It has been superseded by a more general lesson.
4. It was disproven by measurement.

Retired entries keep their original text plus a one-line reason and date.

## Review cadence

- **After every bug fix or feature:** propose a candidate (CLAUDE.md §7). This is
  mandatory, not optional.
- **At each phase boundary:** review `pending.md`, promote or drop.
- **Quarterly:** re-check citations, run a retirement pass.

## Relationship to the memory bank

Lessons are engineering rules. Project state, decisions, and measurements belong
in `activeContext.md` / `progress.md`. A measured experimental result is **not**
a lesson — the generalizable rule extracted from it might be.
