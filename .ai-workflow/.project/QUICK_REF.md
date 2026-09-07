# Quick Reference Card — CLAUDE.md at a Glance

**Print this. Keep it handy. 1 page.**

---

## Session Start (First 5 minutes)

```
1. Read .project/preference.md          → Sets language, tone, style
2. Read .project/memory-bank/activeContext.md
3. Read .project/memory-bank/lessons-learned/meta-index.md
   ↓
   Ready to work!
```

---

## MCP Tools Quick Ref

| When | Tool | Command |
|------|------|---------|
| **Editing code** | `gitnexus_impact` | Check who calls this → breaks? |
| **Need full context** | `gitnexus_context` | All callers, callees, processes |
| **Renaming symbol** | `gitnexus_rename` | Safe refactor, preview first |
| **Before commit** | `gitnexus_detect_changes` | Verify scope matches expected |
| **Finding code** | `gitnexus_query` | "Find code that does X" |
| **Debugging error** | `gitnexus_query` + context | "Find code related to error" |
| **Using library** | `context7_resolve-library-id` + `query-docs` | Get current API docs |
| **Complex planning** | `sequential-thinking` | "Reason through this task" |

---

## Decision Tree Shorthand

```
CODE CHANGE?
  → gitnexus_impact (blast radius?)
     → gitnexus_context (360° view)
        → Any HIGH/CRITICAL risk?
           YES → WARN user, ask approval
           NO → Implement + commit

COMMIT?
  → gitnexus_detect_changes (right scope?)
  → Max 20 files per commit?
  → git add FILE (never git add -A)

BUG FIX or FEATURE DONE?
  → Add lesson candidate
  → Pass 5 gates? (see GATES.md)
  → YES → Add to index.md
  → NO → Add to pending.md

RENAMING?
  → gitnexus_rename (dry_run: true) first
  → Review preview
  → Approve? → dry_run: false

DEBUGGING?
  → gitnexus_query (find related flows)
  → gitnexus_context (full symbol context)
  → Check: read gitnexus://repo/diaflow-backend/process/{name}

USING LIBRARY?
  → context7_resolve-library-id
  → context7_query-docs
  → NEVER write from memory
```

---

## 5 Gates for Lessons (Boolean)

| Gate | PASS if | FAIL action |
|------|---------|------------|
| **Recurrence** | Seen 2+ times OR CRITICAL+preventive | pending.md |
| **Actionability** | Has BAD, GOOD, rule with verb | Needs analysis |
| **Novelty** | Not duplicate of existing | Update existing |
| **Quality** | Format complete (all required fields) | Complete it |
| **Severity** | Can assign & justify CRITICAL/HIGH/MEDIUM/LOW | Not ready |

---

## Lesson Format (Copy-Paste)

```markdown
## N. [SEVERITY] Category - Title

**Triggers:** keyword1, keyword2, keyword3

**Problem:** One sentence describing the issue.

**Bad:**
```python
# Antipattern code
code_that_breaks()
```

**Good:**
```python
# Correct approach
code_that_works()
```

**Rule:** [Verb] [action]. Avoid/Never/Always/Use [pattern].

**Files:** modules/path/file.py:L42-50
```

---

## Memory Bank Map

| File | What | When |
|------|------|------|
| `activeContext.md` | Current work | **Every session** |
| `lessons-learned/meta-index.md` | 9 critical lessons | **Every task** |
| `lessons-learned/index.md` | All 64 lessons | Code review, refactoring |
| `lessons-learned/GATES.md` | 5 gates as boolean | Adding new lesson |
| `systemPatterns.md` | Architecture decisions | Before code change |
| `debugging-guide.md` | Debug tips | Fixing bugs |
| `common-patterns.md` | How to add features | Implementing |
| `development-commands.md` | Dev setup, testing | Setup or running |

---

## Rules (Don't Break These)

✗ NEVER:
  - `git add -A` or `git add .` — add files by name
  - `git commit --no-verify` — let hooks run
  - Edit without `gitnexus_impact` first
  - Rename with find-and-replace — use `gitnexus_rename`
  - Commit without `gitnexus_detect_changes`
  - Commit > 20 files
  - Commit `.project/` files (except memory bank whitelisted)

✓ ALWAYS:
  - Load lessons-learned before code tasks
  - Run MCP tools before risky actions
  - Update activeContext.md at session end
  - Add lesson candidate after bug fix/feature
  - Use gitnexus_impact before editing code
  - Copy before mutating shared state
  - Type hints on public functions

---

## Lesson Checklist (Adding New)

Before adding to production (index.md):

```
Gate 1: Recurrence
  ☐ Seen 2+ times? OR CRITICAL+preventive?

Gate 2: Actionability
  ☐ BAD example?
  ☐ GOOD example?
  ☐ Rule starts with verb?

Gate 3: Novelty
  ☐ grep "[keywords]" = no match?

Gate 4: Quality
  ☐ **Problem:** present?
  ☐ **Rule:** complete?
  ☐ **Files:** valid path?

Gate 5: Severity
  ☐ CRITICAL/HIGH/MEDIUM/LOW assigned?
  ☐ Justified?

ALL YES? → production (index.md + detailed.md)
ANY NO? → pending.md
```

---

## Blast Radius Severity

When you see `gitnexus_impact` results:

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK direct callers | UPDATE these |
| d=2 | LIKELY AFFECTED indirect | Should TEST |
| d=3 | MAY NEED TESTING transitive | Test if critical |

---

## Quick Checks Before Finishing

```
☐ Code edited?
   → gitnexus_impact run?
   → No HIGH/CRITICAL ignored?
   → All d=1 updated?

☐ Code changed?
   → gitnexus_detect_changes confirms scope?
   → ≤20 files?
   → All files added individually?

☐ Bug fix or feature?
   → Lesson candidate added?
   → All 5 gates passed?
   → Added to index/detailed/pending?

☐ Session done?
   → activeContext.md updated?
```

---

## Fact-Check Sources (2026)

When updating this card or CLAUDE.md:

Research best practices:
- Technical Documentation: Clarity, consistency, enforcement mechanisms
- Policy-as-Code: CI/CD automation for compliance
- ADR Pattern: Architecture decisions with explicit status
- Knowledge Management: Centralized repos, consistent taxonomy

See: `.project/policy.json` for machine-readable rules
See: `.project/DECISION_FLOW.md` for ASCII decision trees

---

## Troubleshooting

**"I'm not sure what to do"**
  → Read `.project/DECISION_FLOW.md` ASCII diagrams
  → Find your situation, follow the tree

**"Should I add this as a lesson?"**
  → Check `.project/memory-bank/lessons-learned/GATES.md`
  → Answer the 5 gate questions
  → If unsure: pending.md (not index.md)

**"Is it safe to edit this function?"**
  → Run: `gitnexus_impact({target: "funcName", direction: "upstream"})`
  → If HIGH/CRITICAL: WARN user, ask approval
  → If LOW/MEDIUM: Continue with caution

**"gitnexus tool errored"**
  → STOP immediately
  → Report to user
  → Don't proceed without fix

**"I forgot which files to load"**
  → Check: Session Start section (above)
  → Or: Memory Bank Map table

---

## Update This Card

When CLAUDE.md changes:
1. Update this QUICK_REF.md
2. Keep it ≤1 page (print-friendly)
3. Extract only essential info
4. Link to full docs for details
5. Commit with CLAUDE.md update

**Last Updated:** 2026-03-17
**Next Review:** Quarterly (or when CLAUDE.md major revision)
