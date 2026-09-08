# The 5 gates — lesson validation

A candidate becomes a lesson only if **all five** are true. Evaluate as booleans,
in order; the first `false` sends it to `pending.md`.

## Gate 1 — REAL
The problem actually happened in this repo, with evidence: a stack trace, a
failing test, a wrong metric, or a commit that fixed it.
`false` if the only support is "this is a known anti-pattern".

## Gate 2 — RECURRABLE
It can plausibly happen again — the same class of code exists elsewhere, or the
next dataset/backend/phase will re-expose it.
`false` for one-off accidents in code that no longer exists.

## Gate 3 — NON-OBVIOUS
Someone competent, reading the surrounding code carefully, would still get it
wrong. Type checkers, linters, and the existing test suite do not already catch it.
`false` if `ruff`/`mypy`/`pyright` already flags it.

## Gate 4 — ENFORCEABLE
The rule can be stated as one imperative sentence starting with a verb, and a
reviewer can check compliance by reading a diff.
`false` for "be careful with X" or "consider Y".

## Gate 5 — NOT DUPLICATE
`grep -i "<keyword>" index.md` returns nothing that already covers it. If an
existing lesson is close, **extend that one** instead of adding a new entry.

## Check

```bash
grep -i "<keyword>" .project/memory-bank/lessons-learned/index.md
```

## Severity

- `[CRITICAL]` — outage, data loss, or silent corruption (including silently
  invalid experimental results)
- `[HIGH]` — runtime exception on a normal path
- `[MEDIUM]` — edge-case bug or quality degradation
- `[LOW]` — style or maintainability only

`[CRITICAL]` and `[HIGH]` additionally require an entry in the `meta-index.md`
trigger map.
