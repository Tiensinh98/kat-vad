# CLAUDE.md

Execution system for Claude Code.
This is **not documentation** — it is **behavioral logic** and must be followed strictly.

---

# 1. SESSION CHECK (MANDATORY — MUST OUTPUT)

Before responding, confirm:

[ ] Project Initialization available (from `.project/INIT.md` OR provided in prompt)
[ ] Preference loaded from `.project/preference.md`
[ ] Active context loaded from `.project/memory-bank/activeContext.md`

**If any missing → STOP and ask user.**

Apply preference (language, tone, working style) before answering.

**Memory Bank paths:**

- Core: `.project/memory-bank/activeContext.md`, `.project/preference.md`
- Lessons: `.project/memory-bank/lessons-learned/meta-index.md`
- Policy: `.project/policy.json`
- User persistent: `~/.claude/projects/.../memory/MEMORY.md` (auto-loaded by Claude Code)

---

## 1.1 MEMORY BANK — QUICK BOOTSTRAP (EFFICIENCY)

**Minimum session load (low token):**

1. `.project/preference.md`
2. `.project/memory-bank/activeContext.md`
3. Before IMPLEMENTATION/DEBUG: `.project/memory-bank/lessons-learned/meta-index.md`

**Full memory bank update (`update memory bank` or milestone):**

- Reconcile all six core files: `projectbrief.md`, `productContext.md`, `systemPatterns.md`, `techContext.md`, `activeContext.md`, `progress.md`.
- Refresh quantitative claims (LOC, file counts, routes) from the repo; set **Last Memory Bank Update** in `activeContext.md`.

**Path note:** `preference.md` is `.project/preference.md` (not inside `memory-bank/`).

---

# 2. TASK MODE DETECTION (REQUIRED)

Classify the request before doing anything:

- **SIMPLE** → explanation, small question
- **INVESTIGATE** → explore code, trace flow, understand system
- **PLANNING** → define approach, steps, or architecture
- **RESEARCH** → gather external knowledge, APIs, best practices
- **IMPLEMENTATION** → write or modify code
- **DEBUG** → find and fix bugs (with hypotheses)
- **ARCHITECTURE** → system-level design decisions

### Rules

- If unclear → default to **INVESTIGATE**
- NEVER jump directly to IMPLEMENTATION without INVESTIGATE or PLANNING

---

# 3. MEMORY LOADING RULES (LAYERED — ENFORCED)

Memory is loaded in 3 layers:

---

## 3.1 CORE MEMORY (ALWAYS REQUIRED)

Must always be available before any action:

- Preference (language, tone, working style)
- Active context (current task, recent decisions)

If missing → STOP

---

## 3.2 MODE-BASED MEMORY (CONDITIONAL)

Load based on TASK MODE:

### IMPLEMENTATION

- `lessons-learned/meta-index.md`
- `policy.json` (MCP triggers)

### DEBUG

- `lessons-learned/meta-index.md`
- `lessons-learned/index.md`
- `debugging-guide.md` (if available)
- `policy.json`

### INVESTIGATE

- `activeContext.md`
- `systemPatterns.md` (if exploring architecture)
- `techContext.md` (if exploring stack)

### PLANNING

- `meta-index.md`
- `systemPatterns.md`
- `projectbrief.md` (if scope unclear)

### RESEARCH

- External docs via `context7`
- Do NOT rely on memory bank unless needed

### ARCHITECTURE

- `systemPatterns.md`
- `techContext.md`
- `meta-index.md`

### SIMPLE

- No additional memory required

---

## 3.3 MCP MEMORY (CROSS-STEP — CONDITIONAL)

Use `@modelcontextprotocol/server-memory` when:

- Task spans multiple steps
- Entities/relationships must persist across steps
- Debugging requires tracking evolving hypotheses
- Planning involves dependencies across components

### Rules

- Store key entities (modules, bugs, decisions)
- Link relationships (A affects B, bug tied to module)
- Update memory after each major step
- Do NOT use for trivial tasks

---

## 3.4 LOADING PRIORITY

1. CORE MEMORY (mandatory)
2. MODE-BASED MEMORY (based on task)
3. MCP MEMORY (if complexity requires)

---

## 3.5 HARD RULES

- Never load everything by default
- Never skip CORE MEMORY
- Never implement without loading lessons (IMPLEMENTATION mode)
- Never debug without lessons + context
- Prefer minimal sufficient context over full loading

---

# 4. MCP TRIGGERS (ENFORCED)

Use tools automatically when conditions match:

| Condition                 | Tool                                                       |
| ------------------------- | ---------------------------------------------------------- |
| Editing code              | `trace_call_path` → `get_code_snippet`                     |
| Exploring unfamiliar code | `search_graph` → `get_code_snippet`                        |
| Debugging                 | `search_graph` → `trace_call_path` → `get_code_snippet`    |
| Renaming symbol           | `search_graph` → `trace_call_path` (find all usages first) |
| Before commit             | `detect_changes`                                           |
| External library          | `context7_resolve-library-id` → `context7_query-docs`      |
| Complex reasoning         | `sequential-thinking`                                      |
| Implementing code/bug fix | Load `lessons-learned/meta-index.md`                       |

**If triggered → MUST use before proceeding.**

---

# 5. EXECUTION FLOW (DEBATE-DRIVEN — MANDATORY)

Every non-trivial action MUST follow this:

## 5.1 INVESTIGATE (FIRST STEP)

- Identify current state
- Gather context (code, constraints, system)
- Surface assumptions and unknowns

## 5.2 INTERNAL DEBATE (REQUIRED)

Before acting:

### Options

- List possible approaches (≥2 if non-trivial)

### Tradeoffs

- Pros / Cons for each

### Risks

- Failure modes
- Edge cases
- Impact scope

### Decision

- Choose with justification

## 5.3 EXECUTION

- Execute only after debate
- Follow MCP + lessons constraints

## 5.4 VALIDATION

- Check correctness
- Evaluate impact
- Ensure consistency with lessons

## 5.5 STOP

- Present result
- Wait for user confirmation before continuing

---

# 6. HARD STOP CONDITIONS

Stop immediately if:

- Preference not provided
- Required context missing
- MCP tool fails
- `trace_call_path` returns HIGH or CRITICAL impact (without explicit user approval)
- Requirements unclear

---

# 7. LESSONS SYSTEM (MANDATORY)

## Before Implementation

- MUST load `lessons-learned/meta-index.md`
- Apply relevant lessons

## After Fix / Feature

- MUST create a lesson candidate

### Process

1. Check duplicate: `grep -i "[keyword]" .project/memory-bank/lessons-learned/index.md`
2. Validate 5 gates (`GATES.md`)
3. If pass → add to `index.md` + `detailed.md` + update `meta-index.md` trigger map if [CRITICAL] or [HIGH]
4. If fail → add to `pending.md`

### Lesson format (index.md)

```
## N. [SEVERITY] Category - Title
**Triggers:** keyword1, keyword2, ...
**Problem:** one sentence
**Bad:** antipattern code
**Good:** correct code
**Rule:** enforceable rule starting with a verb
**Files:** actual/file/path.py:line
```

### Severity definitions

- `[CRITICAL]` = outage/data loss/silent corruption
- `[HIGH]` = runtime exception in normal paths
- `[MEDIUM]` = edge case bugs / quality degradation
- `[LOW]` = style/maintainability only

**CI check:** Run `.project/scripts/lesson_checks.sh` to detect violations automatically.

---

# 8. COMMUNICATION STYLE (STRICT)

**At the start of every session, read `.project/preference.md` and strictly apply all settings found there.**

Settings in `preference.md` control: Language, Tone, Working Style, Code Name / Email.

Never:

- Use filler ("Certainly", "Of course", "Sure!")
- Write long intros or preambles before doing the actual work
- Be overly formal or polished
- Ask unnecessary clarifying questions if intent is obvious

Always:

- Be direct
- Match user tone
- Focus on execution

---

# 9. RISK & DECISION REPORTING

For every non-trivial task:

- Show risks and impacts
- Show pros and cons for each proposed solution
- Highlight impact scope

---

# 10. CODE STANDARDS

## Python

- PEP 8: `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- Type hints required on all public functions and methods
- No `print()` → use `logging` module
- No bare `except:` — catch specific exception types
- No `global` keyword
- Global variables must be constants (ALL_CAPS) and defined in a separate `constants.py` file, without underscore prefixes
- All imports at module top-level

## Principles (KISS + DRY)

- KISS: Keep it simple
- DRY: No code duplication
- No hard-coded values — use environment variables and constants
- No magic numbers — define as named variables
- Search for existing implementations before creating new code

## Git

- Never `git add .` or `git add -A` — add files individually
- Never `git commit --no-verify`
- Never commit `.project/` files (except whitelisted memory bank files)
- Max 20 files per commit

## Other Rules

- Any change must include documentation and tests — check if they already exist before creating new ones
- All documentation goes in `core/docs/`. Planning documents go in `.project/plans/`.

---

# 11. QUALITY GATES

Run before commit:

```bash
source .venv/bin/activate && ruff check * && mypy * && bandit * && pycycle * && pyright *
```

**Ruff configuration:** See `ruff.toml` for all linting rules, per-file exceptions, and lesson enforcement.

**Python scripts:** Always prefix with `source .venv/bin/activate`. Example: `source .venv/bin/activate && python script.py`

---

# 12. MEMORY BANK

All detailed project knowledge lives in `.project/memory-bank/`. These files are committed and shared with the team.

### Start Every Session By Reading

1. `.project/memory-bank/activeContext.md` — current branch, in-progress work, recent decisions
2. `.project/memory-bank/lessons-learned/meta-index.md` — **always load first** (~400 tokens): all lessons inline + category map + trigger map
3. `.project/memory-bank/lessons-learned/index.md` — full catalog with Bad/Good code; load for code review, refactoring, or complex tasks

### Memory Bank & Quick Reference Index

**Quick Start (1 page):** `.project/QUICK_REF.md`

**Decision Trees (ASCII diagrams):** `.project/DECISION_FLOW.md` — Follow these before every action!

**Machine-Readable Policy (JSON):** `.project/policy.json` — For automation and CI/CD checks

| File                                       | Contents                                                    | When to Read                     |
| ------------------------------------------ | ----------------------------------------------------------- | -------------------------------- |
| `.project/preference.md`                               | Language, tone, working style (per-user, local)            | **Every session**                |
| `.project/memory-bank/activeContext.md`                | Current work, recent decisions, open questions             | **Every session**                |
| `.project/memory-bank/lessons-learned/meta-index.md`   | **Tier 0** — 7 lessons inline + trigger map (~400 tokens)  | **Every task**                   |
| `.project/memory-bank/projectbrief.md`                 | Scope, goals, source of truth, reproduction gates          | New to project                   |
| `.project/memory-bank/productContext.md`               | Why it exists, what KIP buys                               | Understanding context            |
| `.project/memory-bank/systemPatterns.md`               | Forward pass, key decisions, loss wiring, deviations       | Before any code change           |
| `.project/memory-bank/techContext.md`                  | Stack, pinned externals, constraints, commands             | Setup or debugging               |
| `.project/memory-bank/progress.md`                     | What's done, what's left, technical debt                   | Planning                         |
| `.project/memory-bank/lessons-learned/index.md`        | **Tier 1** — full catalog, severity tags + triggers        | Code review, refactoring         |
| `.project/memory-bank/lessons-learned/detailed.md`     | **Tier 2** — root cause + code examples per lesson         | Deep dive on specific lesson     |
| `.project/memory-bank/lessons-learned/GATES.md`        | 5 gates as boolean logic for lesson validation             | Before adding new lesson         |
| `.project/memory-bank/lessons-learned/CONSTITUTION.md` | Quality standards, retirement criteria, merge rules        | Reference on lesson governance   |
| `.project/memory-bank/lessons-learned/pending.md`      | Candidates awaiting 5-gate validation                      | After seeing a new recurring bug |
| `.project/memory-bank/lessons-learned/archive.md`      | Retired lessons (preserved, never deleted)                 | Quarterly review                 |

Operational docs live in `core/docs/`: `COLAB.md` (run sequence), `TRAINING.md`
(objective wiring, deviations, resume contract), `DATA_LAYOUT.md` (on-disk contract).

### Updating the Memory Bank

- **Every session end:** Update `activeContext.md` with what changed.
- **After an architecture decision:** Update `systemPatterns.md`.
- **MANDATORY — After fixing a bug or completing a feature:** Add or update a lesson in `.project/memory-bank/lessons-learned/` following the lesson process in Section 7. This is NOT optional.

---

# 13. CODEBASE MEMORY MCP — Code Intelligence (MANDATORY)

This project is indexed by **codebase-memory-mcp**. Use these tools to understand code, assess impact, and navigate safely.

> If the index is stale, re-index: `codebase-memory-mcp cli index_repository '{"repo_path": "/path/to/project"}'`

## Always Do

- **MUST run `trace_call_path` before editing any symbol.** Before modifying a function, class, or method, run `trace_call_path` and report the blast radius (direct callers, affected files, risk level) to the user.
- **MUST run `detect_changes` before committing** to verify changes only affect expected symbols.
- **MUST warn the user** if `trace_call_path` returns HIGH or CRITICAL impact before proceeding with edits.
- When exploring unfamiliar code, use `search_graph` instead of grepping. It returns ranked, structured results.
- When you need full context on a symbol — callers, callees, execution flows — use `trace_call_path` + `get_code_snippet`.

## When Debugging

1. `search_graph({project: "...", name_pattern: "<suspect>"})` — find functions/classes related to the issue
2. `trace_call_path({project: "...", qualified_name: "<suspect>", direction: "both"})` — see all callers and callees
3. `get_code_snippet({project: "...", qualified_name: "module.ClassName.method"})` — read actual source
4. `query_graph({project: "...", cypher: "MATCH ..."})` — custom analysis if needed

## When Exploring Architecture

1. `get_architecture({project: "...", aspects: ["languages","packages","routes","hotspots"]})` — high-level overview
2. `search_graph({project: "...", name_pattern: "...", label: "Class"})` — find relevant components
3. `trace_call_path` — trace execution flow through components

## When Refactoring

- **Before renaming**: use `search_graph` + `trace_call_path` to find all usages. Review all callers before renaming.
- **Extracting/Splitting**: MUST run `trace_call_path({direction: "both"})` to see all incoming/outgoing refs before moving code.
- After any refactor: run `detect_changes` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `trace_call_path` on it.
- NEVER ignore HIGH or CRITICAL impact warnings.
- NEVER rename symbols with find-and-replace — search all usages first with `search_graph` + `trace_call_path`.
- NEVER commit changes without running `detect_changes` to check affected scope.

## Tools Quick Reference

| Tool               | When to use                      | Command                                                                                 |
| ------------------ | -------------------------------- | --------------------------------------------------------------------------------------- |
| `search_graph`     | Find code by name pattern        | `search_graph({project: "...", name_pattern: "foo", label: "Function"})`                |
| `get_code_snippet` | Read source of a specific symbol | `get_code_snippet({project: "...", qualified_name: "module.Class.method"})`             |
| `trace_call_path`  | Callers/callees blast radius     | `trace_call_path({project: "...", qualified_name: "...", direction: "both", depth: 3})` |
| `detect_changes`   | Pre-commit scope check           | `detect_changes({project: "...", diff: "<git diff output>"})`                           |
| `query_graph`      | Custom Cypher queries            | `query_graph({project: "...", cypher: "MATCH ..."})`                                    |
| `get_architecture` | High-level architecture view     | `get_architecture({project: "...", aspects: ["packages","routes","hotspots"]})`         |
| `search_code`      | Literal text search in source    | `search_code({project: "...", query: "literal text", file_pattern: "*.py"})`            |

## GitNexus → Codebase Memory Tool Mapping

| Old (GitNexus)              | New (Codebase Memory MCP)                                  |
| --------------------------- | ---------------------------------------------------------- |
| `gitnexus_context`          | `get_code_snippet` + `trace_call_path`                     |
| `gitnexus_impact`           | `trace_call_path` (upstream direction)                     |
| `gitnexus_query`            | `search_graph` or `query_graph`                            |
| `gitnexus_detect_changes`   | `detect_changes`                                           |
| `gitnexus_rename` (dry_run) | `search_graph` + `trace_call_path` (find all usages first) |
| `gitnexus_cypher`           | `query_graph`                                              |

## Cypher Query Examples

```cypher
-- Find all functions that call a specific method
MATCH (f:Function)-[:CALLS]->(g) WHERE g.name = 'create_builder' RETURN f.name, f.file_path

-- Find all routes in a module
MATCH (r:Route) WHERE r.file_path CONTAINS 'builder' RETURN r.name, r.file_path

-- Find class and all its methods
MATCH (c:Class)-[:DEFINES_METHOD]->(m:Method) WHERE c.name = 'CRUDBuilder' RETURN m.name

-- Find all HTTP calls from a service
MATCH (f:Function)-[:HTTP_CALLS]->(r:Route) WHERE f.file_path CONTAINS 'services' RETURN f.name, r.name

-- Dead code candidates (in_degree = 0)
MATCH (f:Function) WHERE f.in_degree = 0 AND f.file_path CONTAINS 'modules' RETURN f.name, f.file_path LIMIT 20
```

## Self-Check Before Finishing

Before completing any code modification task, verify:

1. `trace_call_path` was run for all modified symbols
2. No HIGH/CRITICAL impact warnings were ignored
3. `detect_changes` confirms changes match expected scope
4. All direct callers (depth=1) were updated if needed

## Keeping the Index Fresh

After committing code changes, re-index to keep the graph current:

```bash
codebase-memory-mcp cli index_repository '{"repo_path": "/path/to/your-project"}'
```

---

## context7 MCP — Library Documentation (MANDATORY)

**MUST use whenever:**

- Writing or modifying code that imports or calls an external library
- Checking API signatures, config options, or setup steps for any library

**Workflow:**

1. `mcp__context7__resolve-library-id({libraryName: "pytorch"})` — get the library ID
2. `mcp__context7__query-docs({context7CompatibleLibraryID: "/...", query: "specific topic"})` — pull targeted docs

**Never** write library-specific code from memory alone — always verify against current docs.

---

## sequential-thinking MCP — Complex Reasoning (MANDATORY)

**MUST use when:**

- Root cause analysis has 2+ competing hypotheses
- Architecture or design decision requires hypothesis → verify → decide loops
- Planning any task with inter-step dependencies

**Workflow:** Call `sequential-thinking` first → reason through findings → then call `trace_call_path` to validate blast radius before committing to a strategy.

---

# 14. PROJECT OVERVIEW — KAT-VAD

## 14.0 In one sentence

**KAT-VAD** (Kinematics-Aware, definition-conditioned Traffic VAD) = the **LaGoVAD** weakly-supervised, language-definition-conditioned baseline **+** a novel **Kinematic Induction Pathway (KIP)** that induces optical-flow motion evidence at train time (RGB-only at inference) to close LaGoVAD's one weakness: no motion modeling (its lowest score is on the motion-dominated DoTA benchmark).

### 14.0.1 WHICH BRANCH AM I ON? (READ FIRST — the tree differs)

The repository carries **two live branches with different KIP code**. They share
history up to `fac71a3`; the memory bank under `.project/memory-bank/` is tracked
**per branch**, so the one you just loaded describes *this* branch only.

| Branch | What the code is | KIP gate | Tests |
| --- | --- | --- | --- |
| **`main` (tip `ae4fded`)** | **KAT-VAD v1** — the original KIP: `PMGFlowHead` + `KinematicShift` (frozen 321-param MLP gate) + `MotionScoreHead`. Plus the DoTA / PreVAD / TAD / DADA adapters, `core/eda/`, and `core/data/windows.py`. | **v1 only.** No `kip.gate_type`, no `core/kip/ecmr.py`, no gate diagnostics, no train-only inference graph. | **582 collected, 0 fail** (2026-09-24) |
| **`v3` (tip `bb1516c`)** | v1 **+** the 2026-08-30 gate rebuild: four selectable `gate_type`s (`rank`/`mlp_frozen`/`mlp_ste`/`constant`), ECMR, `train_only_modules`, `--dump-kip-diag`. | v3, default `rank`. | 537 green |

**Run `git branch --show-current` before acting on anything in §14.5.** Writing
v3 code on `main` or v1 code on `v3` is the single easiest way to waste a day.
**The MSAD attribution and the 7-arm DADA campaign were produced by the `v3`
branch's code** and are kept here because they are the project's durable record —
`outputs/` is gitignored. **DADA Phase 1 (2026-09-12) and the whole TAD campaign
(2026-09-14/15) were measured by `main`'s own code.** Each results row in
[[progress]]'s ledger carries a *Code* column saying which.

## 14.1 Source-of-Truth Docs (follow strictly — read before any code)

| Doc | What it gives you | Read when |
| --- | --- | --- |
| `core/docs/KAT_VAD_PROPOSAL.md` | The "why": full proposal, intuition, gap analysis, baseline & design justification. | Understanding intent / design decisions |
| `core/docs/KAT_VAD_IMPLEMENTATION_SPEC.md` | The "what/how": exact modules, per-step tensor shapes, losses, training/inference/eval, config flags, build order. | Writing or modifying any code |
| `core/docs/REPORT_KIP_MSAD_DOTA_PREVAD.md` | The measured campaign write-up (renamed 2026-08-29). §12 = PreVAD trunk campaign. | Before citing any number |
| `core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` | **The DADA root cause (2026-09-08).** The clip-length leak (C28), the whole-clip receptive field, the DVS whole-anchor label (C29), the phased fix, and the video-backbone decision with its pre-registered rule. | Before any DADA work, or before arguing about the backbone |
| `core/docs/RESULTS_PREVAD.md` | Gate P0, the PreVAD trunk, and why its KIP A/B is blocked. | Before touching PreVAD |
| **`core/docs/TAD_SETUP.md` §8.1 + §15.1** | **The TAD campaign's only tracked record** (`RESULTS_TAD.md` does not exist yet). §8.1 = Gate T0 = 0.7912, diagnosed. §15.1 = the in-domain clip-classifier collapse, the falsified loss ladder, the warm-start arm. | Before quoting any TAD number |
| **`.project/plans/katvad-tad-loss-ladder.md`** | Why the loss arms, why NOT a DADA-style windowed rebuild (simulated against real `gt` and rejected), and the pre-registration. | Before proposing anything on TAD |
| **`core/docs/DADA_ORIGIN_PHASE0.md`** | **The DADA-2000 *original* release runbook** (2026-09-15). Spanned-zip mechanics, the measured archive layout, the probe, and Gate P1. | Before touching the original release |
| **`.project/plans/katvad-kip-loss-scale-diagnosis.md`** | **The KIP loss-scale diagnosis — pre-registered bars, the D1/D2 results (Appendix A) and the four repair options with the decision table that gates them.** | **Before touching `lambda_rec`, `e_O`, or any KIP arm** |
| **`.project/plans/katvad-flow-zscore-option-a.md`** + **`colab/DADA2000Origin/phase_5_zscore.ipynb`** | Option A: `flow/v2_zscore`, the pre-registered gates and read-out table. **CLOSED 2026-09-26** (App. A): cost removed, KIP-v1 neutral on T2. | Before touching `lambda_rec` or the flow cache |
| **`core/docs/RESULTS_DADA_ORIG_T2.md`** | **The T2 KIP record (2026-09-26):** phase-4 A/B, the D1/D2 diagnosis, the Option A read-out. KIP-on v2 vs off = **+0.0076 [−0.0002, +0.0154]**, a bounded null. | **Before quoting any T2 / KIP number** |
| **`core/docs/D2CITY_EDA.md`** + `.project/plans/katvad-d2city-normal-bag-eda.md` (App. A) | **D2City as an imported negative-bag pool — RUN 2026-09-25: NO-GO.** G-X FAIL (Δ(X−R0) −0.090), G-M fail (−0.012), shortcut AUC 1.000. Lesson C38. | Before any cross-dataset normal pool, or before touching `data/D2City/` |
| **`core/docs/BDDA_EDA.md`** + `.project/plans/katvad-bdda-normal-bag-eda.md` (App. A) | **BDD-A (Berkeley DeepDrive Attention, US, braking-enriched) as a T2 negative pool — RUN 2026-09-26: NO-GO.** G-X FAIL on the calm arm (X 0.5492, Δ −0.127), G-M ≈ 0, shortcut 1.000. Fourth separate-pool failure (C38). | Before any cross-dataset normal pool, or before touching `data/BDDA/` |
| **`.project/plans/katvad-dada-original-corpus.md`** | **The live corpus plan.** Why the original release, the five measured constructions, T2 windowing, Gates P/D0/W, and the DoTA comparison rule. | **Start here for "what next" on data** |
| ~~`.project/plans/katvad-dada-phase2-corpus-rebuild.md`~~ | **Superseded 2026-09-15** by the plan above — it re-sharded the *trimmed* archive, which measures worse on every column. Read for the C33 derivation only. | — |
| `core/docs/v3/KAT-VAD-ARCHITECTURE.md` **(v3 branch only — absent on `main`)** | **The architecture, phase by phase**: input, output and intuition for P1–P7. Start here for "what is the model". | Understanding or changing the model |
| `core/docs/v3/KAT-VAD_spec_v3.md` **(v3 branch only — absent on `main`)** | **Current spec.** Changes A–K, each tied to a measured defect. Supersedes v2. | Before any architectural change |
| `core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md` **(v3 branch only — on `main` the numbers survive ONLY in `.project/memory-bank/`)** | **The attribution result** (2026-09-01, MSAD-trained): the plain-TSM control reproduces the whole +0.09. | Before claiming KIP does anything |
| `core/docs/v3/RESULTS_DADA.md` | **The DADA-2000 campaign** (2026-09-06): the ordering inverts, and the in-domain 0.86 is a clip-level number. | Before quoting any DADA number |
| `core/docs/v3/KAT-VAD_audit_addendum_PreVAD.md` **(v3 branch only)** | Audit of the PreVAD campaign: what is valid, the attribution queue. | Before designing an experiment |
| `core/docs/{TAD,DADA,PREVAD}_SETUP.md` + `core/docs/EDA.md` + `core/docs/v3/setup/*.md` | Per-corpus data and arm runbooks. On `main` only `v3/setup/{DADA,TAD}_V3_SETUP.md` exist, and **their arm matrices prescribe `kip.gate_type`, which `main` cannot parse** — run those arms on `v3`. | Before running a campaign |
| ~~`core/docs/v2/*`~~ | **Does not exist in this tree.** v3 was written to supersede it; the v2 files were never committed here. Do not cite them. | — |
| `.project/plans/katvad-v3-kip-gate-rebuild.md` | **The v3 branch's plan** (gate rebuild; on `main` it is history — the gate work belongs on `v3`). v3 Phase-3 rebuild, Phases 0–5 with measured appendices A–E. | Before any gate work (on `v3`) |
| ~~`.project/plans/katvad-v2-next-steps.md`~~ | **Does not exist in this tree** (same as the `v2/` docs). The v2 attribution tiers survive only as prose in `activeContext.md`. Do not cite it. | — |

> On conflict, the **read the proposal spec and decide what to do next**; the implementation has some code in it but don't need to follow strictly those code implementation but rationally implement as you decided.
> On conflict *between spec versions*: the newest spec wins where it cites a measurement, and the live plan wins over the spec where it cites code — the plan read the tree, the spec did not. Where the plan's appendices A–E report a measurement, they outrank both.

## 14.2 Data flow (at a glance)

```
video ─► frozen CLIP ViT-B/16 ─► F (L×512)
                                  │
                                  ▼
                Temporal encoder (2-layer Transformer, RoPE) ─► v^t (L×512)
                                  │
                    ┌─────── KIP (NOVEL — build from scratch) ───────┐
                    │  PMG-flow head:  v^t → ê_O  (L×256)            │  train-time
                    │  Kinematic gate + adaptive temporal shift       │  target: RAFT
                    │  Motion-score head: ê_O → ŷ_O                   │  flow e_O (cached)
                    └──────────────► v^k (L×512) ◄────────────────────┘
                                  │
   definition Z ─► frozen CLIP text encoder ─► z^t (C×512)
                                  │
                    Co-attention fusion U(v^k, z^t) ─► v^u, z^u
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
              H_bin ─► y^bin (L)          H_mul ─► y^mul (L×C)
              anomaly curve (main)        per-frame category (triage)
                                  │
                                  ▼  (off critical path, optional)
                    ATS sampler ─► MLLM ─► incident report
```

Notation: `L` = sampled frames (variable), `D = 512` hidden size, `C` = #categories in definition `Z`.

## 14.3 Code Boundaries (HARD RULE)

- **`core/`** — the KAT-VAD implementation. **The ONLY place to write or modify code.** Must strictly follow `core/docs/`.
- **`LaGoVAD-PreVAD/`** — baseline reference. **READ-ONLY — never modify.** Re-implement/organize its ideas in `core/`; do **not** copy code blindly.

## 14.4 What to reuse vs. what to build

| | Component | Action |
| --- | --- | --- |
| **Reuse from LaGoVAD as-is** (re-implement in `core/`) | Frozen CLIP image/text encoders, temporal encoder, co-attention fusion `U`, detection head `H_bin`, classification head `H_mul`, losses `L_MIL`, `L_MIL-align`, `L_dvs` (dynamic video synthesis), `L_neg` (hard-negative mining). | Organize, don't copy |
| **Build from scratch** (no source exists — spec §2–5) | **KIP module**: `PMGFlowHead` (§2), `KinematicShift` gate+shift (§3), `MotionScoreHead` (§4), `KIP` wiring (§5.1), and losses `L_KIP_rec`, `L_KIP_align`, `L_kin` (§5.2). | New code |
| **Train-time only** | RAFT flow-target extraction → cached `e_O` (spec §1). Never on the inference/scoring path. | Offline, amortized |
| **Off critical path (optional)** | ATS sampler → MLLM incident reports (reasoning layer, spec §8 Stage 3). | Build last |

The **only splice** into LaGoVAD's forward pass: insert KIP between the temporal encoder and fusion, then route `v^k` (not `v^t`) into fusion `U` and `H_bin` (spec §6).

## 14.5 Where the project stands (2026-09-24 — branch `main` = KAT-VAD **v1**)

**Baseline of truth = commit `b9978ff`** ("feat: Include training/testing for MSAD full"). Treat it as the known-good state; change it deliberately, never incidentally. **This branch is `main`, tip `ae4fded` — the v1 line.** The v3 gate rebuild lives on branch `v3` (tip `bb1516c`); the two diverged at `fac71a3`. See §14.0.1.

- **Code on `main` (v1):** Phases 0–6 complete + DoTA zero-shot adapter, the offline metrics/rescore layer, the PreVAD preprocessor, the **TAD** (2026-09-02) and **DADA-2000** (2026-09-03/04) adapters, and the **`core/eda/` pre-flight profiler** (2026-09-06). KIP is the **v1** module: `PMGFlowHead` → `KinematicShift` (frozen 321-param MLP gate) → `MotionScoreHead`. Plus the **DADA-2000 original** adapter (`core/data/dada_origin.py`, 2026-09-15), the **`core/tools/grad_probe.py`** gradient-attribution CLI (2026-09-18) and the **`core/flow/zscore{,_cache}.py`** z-scored flow-target builder (2026-09-23). **89 Python files (60 source + 29 test), 13,833 source LOC, 25 docs** (measured 2026-09-24; +`D2CITY_EDA.md` 2026-09-25, +`BDDA_EDA.md` 2026-09-26), CPU, data-free.
- **NOT on `main` (v3 branch only):** `kip.gate_type` (`rank`/`mlp_frozen`/`mlp_ste`/`constant`), `core/kip/ecmr.py`, `train_only_modules`, `--dump-kip-diag`, the gate-type checkpoint guard, and the docs `v3/{KAT-VAD-ARCHITECTURE,KAT-VAD_spec_v3,RESULTS_V3_GATE_ATTRIBUTION,KAT-VAD_audit_addendum_PreVAD}.md` + `v3/setup/MSAD_DOTA_V3_SETUP.md`.
- **`main`'s suite is green: 582 collected, 0 fail** (2026-09-24; history in `progress.md`). `pyproject` sets `addopts="-q"`, so count with `pytest --co -q | awk -F': ' '/^core/{s+=$2} END{print s}'`. The v3-written gate matrix in `test_{dada,tad}.py` was collapsed to the one v1 gate (`TestDadaTrains` / `TestTadTrains`). **Never fix a `gate_type` failure by adding `gate_type` to `core/config.py` alone** — the v3 tests also need `ecmr.py`, the STE shift and the diagnostics, and a partial port turns loud failures into a silently wrong gate.
- **Result — the +0.09 is attributed, and it is not motion.** *(Measured with the `v3` branch's code; kept here because it is the project's durable record. `main` cannot re-run the A1/A2/A3/A4 arms — no `gate_type`.)* MSAD-trained, zero-shot DoTA: **A2**, a fixed 50 % channel shift with no flow, no PMG head and no KIP losses, is statistically **indistinguishable from full v1 KIP** (Δ = +0.0109, t95 [−0.0588, +0.0805], n=3) while A2 − A0 = **+0.1025 ± 0.0350**. **KIP's measured contribution is temporal smoothing** (`RESULTS_V3_GATE_ATTRIBUTION.md`, 2026-09-01). The v3 `rank` gate is the *worst* KIP arm there (A1 − A2 = −0.0683) — and it does not exist on `main`. **MSAD stays a bounded null** — "any in-domain effect is < ≈1 AUC point at n=3", not "costs nothing".
- **The smoother's sign depends on the training corpus.** On DADA-2000 (2026-09-06) the pre-registered ordering **inverts**: A2 − A0 = **−0.0918** on zero-shot DoTA and A1 − A2 = **+0.0300** — both signs flipped versus MSAD, confirmed by a second A2 seed. Mechanism: DADA's median clip is **9 stride-8 frames** under a `Conv1d(kernel=9)` score head, so the shift collapses the curve to a per-clip constant. A component whose sign flips with training clip length is a smoothing hyperparameter, not a motion mechanism. Ego-kinematics stays refuted (10/10 arm-seeds); H4′ CONFIRMED.
- **NEW 2026-09-15 — the TAD campaign, and the project's second negative result.** It is **not** about KIP; it is about the supervision. Gate T0 = **0.7912** micro / 0.7578 macro vs a published 89.56 — **failed and diagnosed, not a defect**: labels match LaGoVAD's own `tad_test_anno.json` 100/100, stride and pooling match, and `auc_macro = 0.7578` exonerates frame ordering (shuffled → ≈0.50); the residual is length weighting (equal-clip-weighted micro drops to 0.6574) plus `_ncc`. **Per C8b, 0.7912 is the reference; 89.56 is retired** — it sits between this corpus's length ruler (0.8968) and its clip oracle (0.9226). **In-domain training collapses the model into a clip classifier:** `m0` reaches clip-level AUC **0.9975** and micro 0.9237 ≈ the oracle 0.9226, while macro **falls** 0.7578 → 0.6174 and DoTA transfer 0.6158 → 0.5496. **Three objective-side repairs all failed** (`dvs_anchor_mode=ignore`, `bottomk_weight=1.0`, both, plus a `bottomk_topk_pct=8` dose): `bottomk` moved every column the wrong way *while working mechanically* — range +37 %, gap +78 %, macro down — i.e. **variance with no direction**. **`t2_warm`** (warm-started from the PreVAD trunk) is the only arm to move all four columns right (macro 0.6540, d +0.5482, DoTA macro 0.5887; 26/27/59 % of the gap) and still misses the bar — which buys the strong claim: **TAD's training signal destroys 0.104 of macro it was handed.** Next: the destruction curve (`--stop-after-epochs 3 7 18 36`). **The TAD KIP A/B is BLOCKED by C14.**
- **NEW 2026-09-15 (later) — the training corpus becomes the DADA-2000 ORIGINAL release.** `.project/plans/katvad-dada-original-corpus.md` + `core/docs/DADA_ORIGIN_PHASE0.md`. `data/DADA/dada标注.xlsx` is the **original annotation**; `Cleaned_Metadata.csv` is a strict subset (975 ⊂ 1,962, **0/975** mismatches) adding only `Fault_Label`, which the original release **does not have**. The archive this project trains on trims **both** classes unequally (abnormal raw median 56, normal 152, **original 322**), so dropping full-length abnormal clips beside the archive's normals *inverts* the leak (0.8105 → **0.8539**) rather than closing it. Chosen construction **T2**: negatives cut from **inside the accident videos** → leak **0.5000** by construction, `score_head_kernel` 9 → 3, `mil_topk_pct` 16 → 5. ~~W=16 hop 8, oracle 0.6631, 98.1 % kept~~ — that was a **simulation over the annotation alone**; measured, W=16 FAILS Gate W (oracle 0.7529) and the adopted geometry is **W=20 hop 8** (oracle 0.7037, retention 0.9568, 798 two-class windows). The 0.6631 prediction belongs to W≈24, which fails retention instead (**C35**). **The code already exists on `main`** (`plan_record_windows`, `b48508a`) — only flags change. **Phase 0 P2 PASSES:** layout `DADA2000/{type}/{video:03d}/images/{frame:04d}.png`, 52/52 types match the xlsx, aggregate frame delta **+0.30 %** (vs −79 % for the trimmed archive). **P1/P3 PASS** (397/400 exact, mean signed delta −0.28) and **Gate D0 PASSES at 0.6518** (bar ≥ 0.60; DoTA 0.6708 · DADA-archive 0.5228 on the same frozen features).
- **NEW 2026-09-18/21 — T2 trains without collapsing, and the KIP A/B on it is NEGATIVE and DIAGNOSED.** *(First campaign of KIP measured by `main`'s own code, and the project's only **within-corpus paired** on/off A/B — the two configs differ in exactly one line, `kip.enabled`.)* **In-domain T2, KIP-off, 3 seeds:** `auc_macro` **0.6248** (above the 0.5983 per-frame linear probe) with micro **0.6182** *below* the 0.7037 clip oracle — macro ≥ micro, so **the C14/TAD collapse did NOT reproduce**; T2 is the first corpus here where in-domain training does not destroy localization. **KIP-on costs 0.0129 T2 micro** (t95 [−0.0249, −0.0008], 3/3 seeds — the only interval excluding zero; the DoTA rows are ~20× wider than their own Δ, **do not quote them**). **Diagnosed 2026-09-20 (D1/D2, lesson C37, `outputs/v1/DADA2000_orig_diag_kip_loss_scale/`):** `e_O` is **unnormalized** — a constant global-mean predictor scores MSE **31.64** and `mag_max` (raw pixel units) carries **83.0 %** of `E[s²]` — so `lambda_rec = 1.0` weights `L_KIP_rec` **~32×** an otherwise-O(1) objective, and at the shared temporal encoder `|g_KIP|/|g_task|` = **3.1** (11.6 at the stage-1 end) with `cos` = **−0.001**: KIP **captures the trunk orthogonally**, spending capacity rather than fighting. **The premise is not refuted** — `R²_item` = **0.283**, i.e. the PMG head does fit 28 % of *within-clip* flow variance from `v^t`. Repair = **Option A** — see the next bullet. C24 is untouched by it: every KIP-on arm here is still a fixed ~50 % channel shift.
- **2026-09-26 — Option A RUN: cost removed, KIP-v1 NEUTRAL on T2** (`RESULTS_DADA_ORIG_T2.md`). `V_v2` 1.0047 → `lambda_rec` 0.9953. `rho` **3.105 → 0.179**. T2 micro KIP-on(v2) − off **+0.0076, t95 [−0.0002, +0.0154]** (3/3 +, includes 0 → bounded null). v2 − v1 **+0.0204 [+0.0010, +0.0398]**, so the phase-4 cost was the loss scale. Caveats: `K_v2` 0.72 > the item-mean oracle 0.654 (D1-b fails on v2), and the histogram block `R²` is 0.07. **No fourth seed; stop spending seeds on v1 KIP.** Next = the user's choice of B (T2 learning curve), C (v3 port for C24) or D (write-up).
- **2026-09-23 — Option A AUTHORIZED (construction A1), BUILT (`ae4fded`).** `python -m core.flow.zscore_cache` rebuilds `e_O` from `flow/v1`'s raw `.stats.npy` alone (no frames, no RAFT): the **23 raw stats** are z-scored with **T2-train-window** moments, then the **same** projection `M` → `cache/flow/v2_zscore/DADA2000_orig/`. HARD gates G0 (every v1 `e_O` == its `stats @ M`), G1, G2-b, G2-c; it writes `lambda_rec = round(1/V_v2, 4)` (predicted **≈ 1.0**) into `zscore_manifest.json`. Runbook `colab/DADA2000Origin/phase_5_zscore.ipynb`: KIP-on stage 1+2 × 3 seeds on v2; **KIP-off arms reused from phase 4** (inert to both changed knobs, `TestKipOffIsInert`; §4 asserts the config diff). Read-out bars pre-registered in the plan §6.3: `rho` < 1.0 (capture removed) × paired Δ T2 micro. Known price: the angle histogram becomes **69.6 %** of the target (R-1b reads it out; block re-weighting is not authorized).
- **2026-09-25 — D2City EDA RUN: NO-GO** (`core/docs/D2CITY_EDA.md`, lesson **C38**). Mechanics pass (length AUC 0.5009, R0 0.6763); D2City as the only negatives drops within-DADA `auc_macro` to **0.5864** (Δ **−0.090**, t95 [−0.101, −0.079], 5/5 folds), as extra negatives −0.012; **shortcut AUC 1.000**; DADA-vs-DoTA S-ref 0.9999, so every foreign dashcam corpus is separable on frozen CLIP. V1 crop did not help. **T2 stays; track 0b closed.** *Original plan, kept for the record:* Proposed: positive bags = full-length DADA-original videos, negative bags = D2City dashcam clips (`data/D2City/`: 7 zips × 100, **25 fps**, ~30 s, 63 % 1080p / 37 % 720p, 16:9 vs DADA's 2.40:1). Stride **7** derived (0.280 s vs DADA's 0.267 s; DADA = 30 fps is literature-only, assumption A1). Admission is decided by **G-X**, not by how alike the sources look: a probe with D2City as its *only* negatives must keep within-DADA `auc_macro` ≥ 0.60 and within 0.03 of the in-video-negatives reference R0 (≈ Gate D0). T2 stays the corpus until then. No `core/` change.
- **2026-09-26 — BDD-A EDA RUN: NO-GO** (`core/docs/BDDA_EDA.md`, C38 extended). Same probes/folds (R0 reproduces D2City to 3.1e-5). Calm arm: X **0.5492**, Δ(X−R0) **−0.127** [−0.153, −0.101], 5/5 folds; M Δ ≈ −0.003 (no gain); shortcut **1.000**. R0 started with BDD-A *above* DADA normals (0.374) and it still ended separable. **No more BDD downloads; T2 + Option A stay.**
- **Tracks.** (0) **DADA-2000 original / T2 + Option A** — the live one (bullets above). (0b) ~~**D2City negative pool**~~ — **closed 2026-09-25, NO-GO** (C38); ~~**BDD-A**~~ — **closed 2026-09-26, NO-GO** (C38). (1) **TAD** — campaign run 2026-09-14/15, collapsed (C14); the KIP A/B is blocked; next would be the destruction curve. (2) **DADA-2000 archive** — root-caused 2026-09-08 (length leak C28, whole-anchor DVS label C29) and **superseded as a training corpus** by the original release. (3) **PreVAD trunk transfer** — Gate P0 still unrun. **PreVAD can never host a KIP-on arm.**
- **Evidence, in measurement order:** `RESULTS_MSAD` → `RESULTS_DOTA` → `RESULTS_NCC` → `RESULTS_PHASE_A` → `RESULTS_ARM4_PROBE` → `RESULTS_PREVAD` → **`v3/RESULTS_V3_GATE_ATTRIBUTION`** → **`v3/RESULTS_DADA`**, plus `REPORT_KIP_MSAD_DOTA_PREVAD.md` (and, on branch `v3`, the spec trio). **`outputs/` is gitignored**, so those documents are the only durable record — and on `main`, where `RESULTS_V3_GATE_ATTRIBUTION.md` is absent, `.project/memory-bank/{progress,activeContext}.md` is that record; the **62,254** per-clip `.npz` score files live on the user's disk/Drive.

Detail lives in `.project/memory-bank/{activeContext,progress}.md` — read those, not this section, before acting.

### 14.5.1 Things not to get wrong

- **The v1 gate MLP is never trained, and H4′ is now CONFIRMED.** `(ratio * max_shift).floor().long()` kills the gradient — all 6 tensors get `grad is None`. Worse, the gate is *measurably* near-constant: its input is min-max normalized, so `[0,1]` is the whole reachable domain, and sweeping it moves `s_t` by **0–4 channels out of 128** across 8 seeds (seed 0: exactly 0). **v1's "motion-gated adaptive temporal shift" was, in operation, a fixed ~50 % shift at `s ≈ 58–69`.** Never write "motion-gated" of `gate_type="mlp_frozen"`. Measured 2026-08-30, plan Appendix C. The v3 **rank gate** (`gate_type="rank"`, the default *on branch `v3`*) spans the full `[0, 128]` on every clip and is parameter-free. **On `main` there is no `gate_type` at all: the frozen-MLP gate is the only gate, so every KIP-on run on this branch is a fixed ~50 % shift by construction.**
- **A feature cache is bound to the transform and stride that built it** (lesson 2/13). The pipeline is on `no_center_crop`; all current artifacts are `*_ncc`. Changing `preprocess_frames` or `FRAME_STRIDE` invalidates every cached feature *and every metric measured on it*.
- **Pooling is decided by the label distribution, not the dataset name** (lesson 12). DoTA is ~all-abnormal → per-clip min-max; MSAD/PreVAD → raw. Mixing the two protocols in one table is how a released checkpoint reads as chance.
- **A score head whose kernel spans the clip is a clip classifier.** `ConvScoreHead` is one `Conv1d(512→1, kernel_size=9)`; DADA-2000's median clip is **9** stride-8 frames (MSAD 86, DoTA 13), so every output timestep sees the whole clip. Result: flat curves, `auc_macro` at chance, and a micro AUC that is really video classification. **Check `score_head_kernel` against a new corpus's median sequence length before training on it.** (`RESULTS_DADA.md` §5.)
- **Micro AUC over a test set full of all-normal clips measures clip classification** — the mirror of lesson C12. On DADA-2000, 74 % of test frames come from all-normal clips and a *constant-score-per-clip* oracle scores **0.9069**; our best arm scored 0.8739 with `auc_macro` at chance. Compute that oracle and print it beside any micro number. **Never put a DADA micro AUC next to a published frame-level number.**
- **A benchmark can be unusable in-domain and still be a usable training corpus — and TAD is neither, yet.** TAD's constant-score-per-clip oracle is **0.9226** and its frame-count-only ruler **0.8968**, so the published **89.56 sits between them**: micro AUC on TAD measures clip classification, and 99.69 % of its frame pairs span two clips. Report `auc_macro` and clip-level AUC; **never** put a TAD micro number beside a published frame-level one. Gate T0 reproduces at **0.7912** and that, not 89.56, is the reference (C8b).
- **In-domain WS-MIL training on TAD destroys localization — measured, not inferred.** `m0` reaches clip-level AUC **0.9975** while macro falls 0.7578 → 0.6174; warm-starting from the PreVAD trunk still ends at 0.6540, i.e. **0.104 of macro handed to the model is destroyed in 504 steps**. Every objective-side repair failed, including `bottomk` *while working mechanically* (range +37 %, gap +78 %, macro down — variance with no direction). **The TAD KIP A/B is blocked by C14 until this is resolved.**
- **`descriptions` is not what conditions the model, and `L_neg` has essentially never run.** Two separate text paths: class **definitions** are hardcoded in `core/data/definitions.py` and always feed `z^t`; the `descriptions` field feeds captions → `L_neg` and is shipped **only by PreVAD**, reaches `meta.json` only, and is read by nothing (gap **G4**). 54 of 55 `config.yaml` have `captions_from_definitions: false`, so `core/train.py:375` skips `L_neg` outright — `cap_contrastive_weight: 1.0` is decoration. A `descriptions: null` in a TAD/DoTA/DADA/MSAD annotation is expected and harmless.
- **Do not tune KIP on any delta** (lesson 14). The MSAD +0.09 is attributed to smoothing and does not replicate on DADA; tuning against either number is fitting the benchmark.
- **A raw `kip_rec` is not a fit quality — divide it by what a constant predictor scores.** `e_O` is 23 **unnormalized** flow scalars, so `L_KIP_rec`'s magnitude is a property of **pixel units**. Measured on T2: zero-predictor MSE **71.15**, global-mean **31.64**, item-mean oracle **16.21**, actual `kip_rec` **11.62** → `R²` **0.633** (a *good* fit that reads as a catastrophic loss). At `lambda_rec = 1.0` the term is **~32×** the rest of the objective and reaches the shared trunk at `rho` = **3.1**, `cos` = **−0.001**. Report `R² = 1 − kip_rec / V` **of the cache the run trained on**, and if you change the weight, **derive** it as `1/V` — never sweep it against a Δ (lesson 14). Lesson **C37**.
- **`lambda_rec` belongs to the flow cache.** `flow/v1` → 1.0 (V = 31.64 on T2); `flow/v2_zscore` → the `lambda_rec` in its `zscore_manifest.json` (≈ 1). **0.0316 is v1's `1/V`; on v2 it switches `L_KIP_rec` off.** `--flow-dir` is a CLI path and is **not** in `config.yaml` — record it in the run manifest. A v2 cache is bound to one train split (`train_ids_sha1`); `.stats.npy` stays raw in it (KNN motion key).
- **The flow target has no spatial content** — 23 frame-global scalars lifted to 256-d by a fixed seeded projection (`core/flow/raft_extract.py:52-81`). Say "23 effective dimensions", and never claim KIP localizes anything spatially.
- **PreVAD can never host a KIP-on arm.** It ships CLIP features, no video, so the
  RAFT targets `L_KIP_rec` needs are unbuildable; `require_flow=False` is not a
  workaround (it trains the PMG head to predict zeros). Settled 2026-08-29 —
  do not reopen. PreVAD = KIP-off trunk + Gate P0 only (`PREVAD_SETUP.md` §7.4).
- **Closing the length leak and lowering the clip oracle are two DIFFERENT jobs.** Fixed-length windows close the leak perfectly — `DADA2000_w32s2` and `_w24s1` both measure exactly **0.5000**. The oracle follows the **frame share** (C33's `(F_norm + 0.5X)/(F_norm + X)`), so it moves only when abnormal bags carry more frames: 7.5 % → 0.9766, 35 % → 0.8965, **67 % → 0.6631**. "The clips were too short" explains `w32s2` only — **`w24s1` kept 406 abnormal test clips and its oracle is still 0.8965.** The root cause of both failures is that the negatives came from `0_Normal_Driving`, a separate pool ~3× longer, under a fixed hop. **Cut negatives from inside the abnormal videos.**
- **Every corpus with negative bags is degenerate; every clean corpus has none.** MSAD (negatives, clean, but CCTV) · TAD (negatives, collapses) · DADA-archive (negatives, leaks) · **DoTA (clean — oracle 0.5017, 1,392/1,397 mixed — but `train_clips: 0` and 3 normal clips)** · DADA-original (clean, zero normals). **DoTA cannot be trained on and must stay held out**: every comparison this project owns is defined by DoTA being unseen. The bar a new training corpus must beat is **0.6408**.
- **The "frozen CLIP cannot localize" verdict is about the DADA archive, not about CLIP.** Its frame linear probe reads 0.5228 there and **0.6708 on DoTA**, same features, same transform. Do not cite the 0.5228 as a representation ceiling.
- **`dada标注.xlsx`'s sheets are named the opposite of their contents:** `name="text"` → the type 1–38 taxonomy; **`name="Sheet1"` → the 1,962-row per-clip table**. Detect the sheet by its columns; a hardcoded name parses zero rows and reads like a corrupt file.
- **`preprocess_frames(frames: np.ndarray, crop_size=224, center_crop=True)` takes a decoded ARRAY, not paths, and centre-crops by default.** Pair it with `read_images(paths)` and pass **`center_crop=False`** — the whole project is `_ncc`, and a transform mismatch does not raise (C2, C13).
- **`L_neg` cannot be activated from DADA's `texts` column.** It is a category label — **83 normalized unique strings over 1,962 videos**, top string 12.2 % — so same-caption rows in a batch become false negatives under `asymmetric_infonce_loss`. Plus gap **G4** (no code path reads a caption field) and `N3_MIN_SCORE_RANGE = 0.2` gating the one branch that would matter.
- **An imported normal pool is admitted by a transfer probe, not by similarity.** "Same country, dashcam, similar fps" held for `0_Normal_Driving` too, and it leaked (C28/C32). CLIP separates almost any two corpora, so the source probe (≥ 0.90 ⇒ no in-domain claim) is a *claim* gate; admission = G-X. **Measured on D2City (2026-09-25) and BDD-A (2026-09-26): G-X FAIL both, shortcut 1.000 — third and fourth separate-pool failures (C38).** Match the foreign pool's bag lengths by **drawing all lengths first, then packing** — per-clip drawing is biased short (pending (v)).
- **`LaGoVAD-PreVAD/` is read-only**, and gate against the released checkpoint, not the paper's printed number (lesson 8b).
- **A runbook on `main` is not runnable on `main` just because the file is there.** `core/docs/v3/setup/{DADA,TAD}_V3_SETUP.md` prescribe six arms keyed on `kip.gate_type`; this branch parses none of them. Check the flag exists in `core/config.py` before copy-pasting a runbook command.
- **The memory bank is tracked per branch and the two copies have deliberately diverged** (`main` = v1, `v3` = v3). A `git merge` between them **will** conflict in `{CLAUDE.md,.project/memory-bank/*}` — resolve by branch identity, never by "take theirs".

## 14.6 Delivery Requirements

- Every entry-point script ships with an `argparse`-based CLI (training, inference, flow/feature extraction, evaluation, visualization).
- Follow §10 code standards (PEP 8, type hints on public surface, no `print` → `logging`, no bare `except`, no `global`, constants in `constants.py`).
- Quality gate before commit (§11): `source .venv/bin/activate && ruff check * && mypy * && bandit * && pycycle * && pyright *`.
- Documentation in `core/docs/`; planning docs in `.project/plans/`.

For full architecture details → `.project/memory-bank/systemPatterns.md` | For stack details → `.project/memory-bank/techContext.md`
