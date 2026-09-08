# Decision Flow Diagram

**Purpose:** Replace vague triggers with explicit decision trees. Every action has a clear path.

**Document source:** `.project/policy.json` + CLAUDE.md triggers

---

## Workflow: Task Assignment → Completion

```
┌─────────────────────────────────────────────────────────────────────┐
│ START: Task/Request Assigned                                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │ What type of task? │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
    CODE TASK?          DOCUMENTATION?         EXPLORATION?
        │                      │                      │
    [YES]                  [YES]                  [YES]
        │                      │                      │
        ▼                      ▼                      ▼
    Load lessons-learned   Read relevant        Load lessons-learned
    meta-index.md          docs/                meta-index.md
                                                   ↓
                                            gitnexus_query({query})
                                                   ↓
                                            gitnexus_context({name})
                                                   ↓
                                            Present findings
                                                   ↓
                                                 DONE
        │
        ▼
    ┌─────────────────────────┐
    │ Code Task Subcategory?  │
    └──────────┬──────────────┘
               │
    ┌──────────┼──────────┬─────────────┐
    │          │          │             │
    ▼          ▼          ▼             ▼
  EDIT?    RENAME?   DEBUG?          CREATE?
    │        │         │               │
 [YES]   [YES]      [YES]            [YES]
    │        │         │               │
    ├─────────────────────────────────┤
    │                                 │
    └─────────────────┬───────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │ Load Lessons-Learned            │
        │ - meta-index.md                 │
        │ - Scan trigger map              │
        │ - Load index.md (if refactoring)│
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │ Run Required MCP Tools          │
        │ (Based on trigger)              │
        └──────────────┬──────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │ EDIT/RENAME:     │ DEBUG/EXPLORE:   │
    │ - gitnexus_impact│ - gitnexus_query │
    │ - gitnexus_context
    │ - Check: impact? │ - gitnexus_context
    │   HIGH/CRITICAL? │                  │
    │   YES → WARN     │ - Report results │
    │   NO → Continue  │ - DONE           │
    └────────┬─────────┴──────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    │ WARNING: HIGH/  │
    │ CRITICAL RISK!  │
    │                 │
    │ [STOP]          │
    │ Request approval│
    │ from user       │
    │                 │
    │ User: Continue? │
    │ [YES] ──┐       │
    │ [NO]  ──┼──→ BLOCKED
    │         │       │
    │         ▼       │
    └────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │ Implement Changes               │
    │ (Follow lessons-learned rules)  │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │ BEFORE COMMIT:                  │
    │ - gitnexus_detect_changes()     │
    │ - Verify scope matches expected │
    │ - Check: ≤20 files? YES         │
    │                                 │
    │ If scope wrong:                 │
    │   - STOP, report to user        │
    │   - Ask: stage different set?   │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │ Commit (one commit)             │
    │ (if multi-step: multiple commits)
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │ Was this a:                     │
    │ - Bug fix?                      │
    │ - Feature completion?           │
    │ - Architecture decision?        │
    │                                 │
    │ ANY YES → Add lesson candidate  │
    │ NO → Skip lessons-learned update│
    └──────────────┬──────────────────┘
                   │
    ┌──────────────┴──────────────────┐
    │ (if YES to bug fix/feature)     │
    │                                 │
    │ Run lesson process:             │
    │ 1. grep -i '[keyword]' check    │
    │ 2. Pass 5 gates?                │
    │    - Recurrence?                │
    │    - Actionability?             │
    │    - Novelty?                   │
    │    - Quality?                   │
    │    - Severity?                  │
    │                                 │
    │ if PASS:                        │
    │ - Add to index.md               │
    │ - Add to detailed.md            │
    │ - Update meta-index.md          │
    │                                 │
    │ if FAIL:                        │
    │ - Add to pending.md             │
    │ - Revisit when recurs           │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │ Update activeContext.md         │
    │ (at session end)                │
    │                                 │
    │ Record: what changed, decisions │
    └──────────────┬──────────────────┘
                   │
                   ▼
              ┌─────────┐
              │ DONE    │
              └─────────┘
```

---

## Decision Tree: When to Load What

```
START: New Session
    │
    ▼
ALWAYS read FIRST (first 10 minutes):
    1. .project/preference.md
       └─ Sets language, tone, working style

    2. .project/memory-bank/activeContext.md
       └─ Understand current branch, recent decisions

    3. .project/memory-bank/lessons-learned/meta-index.md
       └─ 500 tokens, 9 critical lessons + trigger map

    ▼
Then, based on task type:
    │
    ├─ CODE CHANGE?
    │   └─ Load: lessons-learned/index.md
    │       └─ Full 64-lesson catalog
    │
    ├─ REFACTORING?
    │   └─ Load: lessons-learned/index.md (same as above)
    │
    ├─ DEBUGGING?
    │   └─ Load: debugging-guide.md + lessons-learned/meta-index.md
    │
    ├─ IMPLEMENTING FEATURE?
    │   └─ Load: common-patterns.md + systemPatterns.md
    │       └─ How to add agents, tasks, nodes, endpoints
    │
    ├─ PERFORMANCE/OPTIMIZATION?
    │   └─ Load: progress.md (current bottlenecks)
    │       └─ Development-commands.md (profiling tools)
    │
    └─ ARCHITECTURE DECISION?
        └─ Load: systemPatterns.md
            └─ Existing decisions, patterns
```

---

## MCP Tool Decision Tree

```
START: About to take action on code
    │
    ▼
What am I doing?
    │
    ├─ EDITING function/class/method?
    │   └─ Run: gitnexus_impact({target: "symbolName", direction: "upstream"})
    │       └─ Get blast radius (d=1, d=2, d=3)
    │       └─ Check: HIGH/CRITICAL risk?
    │           ├─ YES → WARN user, request approval
    │           └─ NO → Continue to: gitnexus_context()
    │               └─ 360° view of callers, callees, processes
    │
    ├─ RENAMING anything?
    │   └─ Run: gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})
    │       └─ Preview edits
    │       └─ Review graph edits (safe) vs text_search edits (manual review)
    │       └─ Approve? YES → dry_run: false
    │
    ├─ COMMITTING?
    │   └─ Run: gitnexus_detect_changes({scope: "staged"})
    │       └─ Verify symbols changed match expected
    │       └─ Check affected processes
    │       └─ Match scope OK? YES → git commit
    │
    ├─ DEBUGGING bug/error?
    │   └─ Run: gitnexus_query({query: "error message or symptom"})
    │       └─ Get execution flows related to issue
    │       └─ Run: gitnexus_context({name: "suspect function"})
    │           └─ See all callers, callees
    │           └─ Read full trace: gitnexus://repo/diaflow-backend/process/{name}
    │
    ├─ EXPLORING unfamiliar code?
    │   └─ Run: gitnexus_query({query: "concept or keyword"})
    │       └─ Get process-grouped results
    │       └─ Run: gitnexus_context({name: "symbol"})
    │           └─ Full context view
    │
    ├─ USING external library?
    │   └─ Run: context7_resolve-library-id({libraryName: "lib name"})
    │       └─ Get library ID
    │       └─ Run: context7_query-docs({libraryId: "/org/project", query: "topic"})
    │           └─ Get current docs before writing code
    │
    ├─ PLANNING 3+ step task?
    │   └─ Run: sequential-thinking (reason through dependencies)
    │       └─ Propose → verify against codebase → decide
    │
    └─ ROOT CAUSE with 2+ theories?
        └─ Run: sequential-thinking
            └─ Hypothesis loop: propose → verify → decide
```

---

## Lesson-Learned Addition Process (5 Gates)

```
Completed: Bug fix or feature → Add lesson candidate?
    │
    ▼
STEP 1: Check for duplicates
    └─ Run: grep -i '[keyword]' .project/memory-bank/lessons-learned/index.md
    └─ If found: Update existing lesson instead (don't add new one)

    ▼
STEP 2: Pass all 5 gates
    │
    ├─ Gate 1: Recurrence Test
    │   └─ Question: Seen this pattern 2+ times OR high-risk?
    │   └─ PASS: continue
    │   └─ FAIL: Add to pending.md, watch for recurrence
    │
    ├─ Gate 2: Actionability Test
    │   └─ Question: Can someone fix it in < 5 min after reading?
    │   └─ PASS: continue
    │   └─ FAIL: Needs more analysis
    │
    ├─ Gate 3: Novelty Test
    │   └─ Question: Similar to existing lesson?
    │   └─ PASS (no similar): continue
    │   └─ FAIL (similar found): Update existing instead
    │
    ├─ Gate 4: Quality Test
    │   └─ Question: Has BAD code? GOOD code? Rule sentence? File reference?
    │   └─ PASS: continue
    │   └─ FAIL: Complete the format first
    │
    └─ Gate 5: Severity Classification
        └─ Question: What's production impact if ignored?
        └─ CRITICAL: Outage/data loss
        └─ HIGH: Runtime exception
        └─ MEDIUM: Edge case bug
        └─ LOW: Style/maintainability
        └─ PASS: Assigned one of above
        └─ FAIL: Can't classify → needs more context

    ▼
Result:
    │
    ├─ ALL 5 GATES PASS → Add to production
    │   ├─ Add entry to: lessons-learned/index.md (compact form)
    │   ├─ Add entry to: lessons-learned/detailed.md (full explanation)
    │   ├─ If CRITICAL/HIGH: Update meta-index.md trigger map
    │   └─ Commit to git
    │
    └─ ANY GATE FAILS → Add to pending.md
        ├─ Note which gate failed
        ├─ Watch for recurrence
        ├─ Promote to production when gates pass later
        └─ Review quarterly (CONSTITUTION.md Part 7)
```

---

## External Library Decision Tree

```
About to write code involving external library (FastAPI, SQLAlchemy, etc.)
    │
    ▼
NEVER write from memory alone!
    │
    ▼
STEP 1: Resolve library ID
    └─ Run: context7_resolve-library-id({
        libraryName: "fastapi",
        query: "What you need to do"
    })
    └─ Output: Context7-compatible library ID (e.g., "/vercel/fastapi")

    ▼
STEP 2: Query docs
    └─ Run: context7_query-docs({
        libraryId: "/vercel/fastapi",
        query: "How to create endpoint with async"
    })
    └─ Output: Current API docs, examples, version info

    ▼
STEP 3: Write code
    └─ Only after seeing:
       - Current API signatures
       - Configuration options
       - Version-specific behavior
       - Real examples from docs

    ▼
RESULT:
    └─ Code correct for current library version
    └─ No API misuse (deprecated methods, wrong signatures)
```

---

## Quality Check Checklist Before Finishing

```
Before marking task COMPLETE, verify:

☐ If code edited:
  ├─ gitnexus_impact was run
  ├─ No HIGH/CRITICAL risk ignored
  └─ All d=1 (WILL BREAK) dependents updated

☐ If code changed:
  ├─ gitnexus_detect_changes() confirms expected scope
  ├─ ≤20 files per commit
  └─ All files added individually (not git add -A)

☐ If bug fix or feature:
  ├─ Lesson candidate added (passed 5 gates)
  ├─ Added to index/detailed/pending.md
  └─ Meta-index.md updated (if CRITICAL/HIGH)

☐ Session end:
  ├─ activeContext.md updated with what changed
  ├─ Decisions documented
  └─ For user next time: clear handoff
```

---

## Translation: Decision Flow → Action

| Decision | Action | MCP Tool |
|----------|--------|----------|
| "About to edit function" | Load lessons-learned/meta-index.md | None (read) |
| Editing, need safety check | Check blast radius | gitnexus_impact |
| Need 360° view of symbol | All callers/callees/processes | gitnexus_context |
| Renaming across files | Safe refactor with graph awareness | gitnexus_rename |
| Committing changes | Verify scope matches | gitnexus_detect_changes |
| Error or bug | Find related execution flows | gitnexus_query + gitnexus_context |
| Exploring code | Understand architecture | gitnexus_query + gitnexus_context |
| Using new library | Verify API before writing | context7_resolve-library-id + query-docs |
| Planning complex task | Reason through dependencies | sequential-thinking |
| Multiple bug theories | Hypothesis verification | sequential-thinking |
