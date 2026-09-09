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
| **`main` (tip `6a5f648`)** | **KAT-VAD v1** — the original KIP: `PMGFlowHead` + `KinematicShift` (frozen 321-param MLP gate) + `MotionScoreHead`. Plus the DoTA / PreVAD / TAD / DADA adapters and `core/eda/`. | **v1 only.** No `kip.gate_type`, no `core/kip/ecmr.py`, no gate diagnostics, no train-only inference graph. | **439 collected, 439 pass** (2026-09-09, after Phase 1) |
| **`v3` (tip `bb1516c`)** | v1 **+** the 2026-08-30 gate rebuild: four selectable `gate_type`s (`rank`/`mlp_frozen`/`mlp_ste`/`constant`), ECMR, `train_only_modules`, `--dump-kip-diag`. | v3, default `rank`. | 537 green |

**Run `git branch --show-current` before acting on anything in §14.5.** Writing
v3 code on `main` or v1 code on `v3` is the single easiest way to waste a day.
**All measured results below were produced by the v3 branch's code** and are kept
here because they are the project's durable record — `outputs/` is gitignored.

## 14.1 Source-of-Truth Docs (follow strictly — read before any code)

| Doc | What it gives you | Read when |
| --- | --- | --- |
| `core/docs/KAT_VAD_PROPOSAL.md` | The "why": full proposal, intuition, gap analysis, baseline & design justification. | Understanding intent / design decisions |
| `core/docs/KAT_VAD_IMPLEMENTATION_SPEC.md` | The "what/how": exact modules, per-step tensor shapes, losses, training/inference/eval, config flags, build order. | Writing or modifying any code |
| `core/docs/REPORT_KIP_MSAD_DOTA_PREVAD.md` | The measured campaign write-up (renamed 2026-08-29). §12 = PreVAD trunk campaign. | Before citing any number |
| `core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md` | **The DADA root cause (2026-09-08).** The clip-length leak (C28), the whole-clip receptive field, the DVS whole-anchor label (C29), the phased fix, and the video-backbone decision with its pre-registered rule. | Before any DADA work, or before arguing about the backbone |
| `core/docs/RESULTS_PREVAD.md` | Gate P0, the PreVAD trunk, and why its KIP A/B is blocked. | Before touching PreVAD |
| `core/docs/v3/KAT-VAD-ARCHITECTURE.md` **(v3 branch only — absent on `main`)** | **The architecture, phase by phase**: input, output and intuition for P1–P7. Start here for "what is the model". | Understanding or changing the model |
| `core/docs/v3/KAT-VAD_spec_v3.md` **(v3 branch only — absent on `main`)** | **Current spec.** Changes A–K, each tied to a measured defect. Supersedes v2. | Before any architectural change |
| `core/docs/v3/RESULTS_V3_GATE_ATTRIBUTION.md` **(v3 branch only — on `main` the numbers survive ONLY in `.project/memory-bank/`)** | **The attribution result** (2026-09-01, MSAD-trained): the plain-TSM control reproduces the whole +0.09. | Before claiming KIP does anything |
| `core/docs/v3/RESULTS_DADA.md` | **The DADA-2000 campaign** (2026-09-06): the ordering inverts, and the in-domain 0.86 is a clip-level number. | Before quoting any DADA number |
| `core/docs/v3/KAT-VAD_audit_addendum_PreVAD.md` **(v3 branch only)** | Audit of the PreVAD campaign: what is valid, the attribution queue. | Before designing an experiment |
| `core/docs/{TAD,DADA,PREVAD}_SETUP.md` + `core/docs/EDA.md` + `core/docs/v3/setup/*.md` | Per-corpus data and arm runbooks. On `main` only `v3/setup/{DADA,TAD}_V3_SETUP.md` exist, and **their arm matrices prescribe `kip.gate_type`, which `main` cannot parse** — run those arms on `v3`. | Before running a campaign |
| ~~`core/docs/v2/*`~~ | **Does not exist in this tree.** v3 was written to supersede it; the v2 files were never committed here. Do not cite them. | — |
| `.project/plans/katvad-v3-kip-gate-rebuild.md` | **The live plan.** v3 Phase-3 rebuild, Phases 0–5 with measured appendices A–E. | **Start here for "what next"** |
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

## 14.5 Where the project stands (2026-09-08 — branch `main` = KAT-VAD **v1**)

**Baseline of truth = commit `b9978ff`** ("feat: Include training/testing for MSAD full"). Treat it as the known-good state; change it deliberately, never incidentally. **This branch is `main`, tip `6a5f648` — the v1 line.** The v3 gate rebuild lives on branch `v3` (tip `bb1516c`); the two diverged at `fac71a3`. See §14.0.1.

- **Code on `main` (v1):** Phases 0–6 complete + DoTA zero-shot adapter, the offline metrics/rescore layer, the PreVAD preprocessor, the **TAD** (2026-09-02) and **DADA-2000** (2026-09-03/04) adapters, and the **`core/eda/` pre-flight profiler** (2026-09-06). KIP is the **v1** module: `PMGFlowHead` → `KinematicShift` (frozen 321-param MLP gate) → `MotionScoreHead`. **79 Python files (54 source + 25 test), 10,828 source LOC** (measured 2026-09-09, post-Phase-1), CPU, data-free.
- **NOT on `main` (v3 branch only):** `kip.gate_type` (`rank`/`mlp_frozen`/`mlp_ste`/`constant`), `core/kip/ecmr.py`, `train_only_modules`, `--dump-kip-diag`, the gate-type checkpoint guard, and the docs `v3/{KAT-VAD-ARCHITECTURE,KAT-VAD_spec_v3,RESULTS_V3_GATE_ATTRIBUTION,KAT-VAD_audit_addendum_PreVAD}.md` + `v3/setup/MSAD_DOTA_V3_SETUP.md`.
- **`main`'s suite is green: 439 collected, 439 pass** (2026-09-09, after Phase 1 added 16 tests; Phase 0 had taken it to 423). Earlier the same day it was 418, and before the repair 425 → 413 pass / 12 fail: `core/tests/test_{dada,tad}.py::TestXTrainsUnderEveryGate` were written on the v3 branch and parametrized over `kip.gate_type`, which `core/config.py:207` rejects because v1 has no such field. Repaired by collapsing the matrix to the one gate v1 ships (`kip.gate_signal=flow_norm`) and renaming the classes `TestDadaTrains` / `TestTadTrains`; A0, stage-1 warm-up and the `config.yaml`-recording coverage all survive. **Never fix this by adding `gate_type` to `core/config.py` alone** — the v3 tests also need `ecmr.py`, the STE shift and the diagnostics, and a partial port turns loud failures into a silently wrong gate.
- **Result — the +0.09 is attributed, and it is not motion.** *(Measured with the `v3` branch's code; kept here because it is the project's durable record. `main` cannot re-run the A1/A2/A3/A4 arms — no `gate_type`.)* MSAD-trained, zero-shot DoTA: **A2**, a fixed 50 % channel shift with no flow, no PMG head and no KIP losses, is statistically **indistinguishable from full v1 KIP** (Δ = +0.0109, t95 [−0.0588, +0.0805], n=3) while A2 − A0 = **+0.1025 ± 0.0350**. **KIP's measured contribution is temporal smoothing** (`RESULTS_V3_GATE_ATTRIBUTION.md`, 2026-09-01). The v3 `rank` gate is the *worst* KIP arm there (A1 − A2 = −0.0683) — and it does not exist on `main`. **MSAD stays a bounded null** — "any in-domain effect is < ≈1 AUC point at n=3", not "costs nothing".
- **The smoother's sign depends on the training corpus.** On DADA-2000 (2026-09-06) the pre-registered ordering **inverts**: A2 − A0 = **−0.0918** on zero-shot DoTA and A1 − A2 = **+0.0300** — both signs flipped versus MSAD, confirmed by a second A2 seed. Mechanism: DADA's median clip is **9 stride-8 frames** under a `Conv1d(kernel=9)` score head, so the shift collapses the curve to a per-clip constant. A component whose sign flips with training clip length is a smoothing hyperparameter, not a motion mechanism. Ego-kinematics stays refuted (10/10 arm-seeds); H4′ CONFIRMED.
- **Three tracks are live.** (1) **TAD replication** — code + runbooks shipped 2026-09-02, **nothing trained yet**; next is Gate T0 against `TAD_ZERO_SHOT_AUC = 89.56`. (2) **DADA-2000** — campaign run, analysed, and **root-caused 2026-09-08** (`core/docs/DIAGNOSIS_DADA_FRAME_LEVEL_COLLAPSE.md`): the arms are clip classifiers, the corpus **leaks its label through clip length** (a ruler scores micro **0.8654**; lesson **C28**) and DVS labels the whole anchor clip positive (**C29**). **Phases 0 (reporting) and 1 (three default-off arms: `loss.dvs_anchor_mode=ignore`, `loss.bottomk_weight`, `--equalize-length`) are complete — none is trained yet**; Phase 2 (corpus rebuild, fires C2) and Phase 3 (probe → the backbone decision) are queued. The frame-level probe is **run** (DADA 0.5228 / DoTA 0.6708 macro). (3) **PreVAD trunk transfer** — `PREVAD_SETUP.md`, code prerequisites landed 2026-08-24, next step is Gate P0. **PreVAD can never host a KIP-on arm.**
- **Evidence, in measurement order:** `RESULTS_MSAD` → `RESULTS_DOTA` → `RESULTS_NCC` → `RESULTS_PHASE_A` → `RESULTS_ARM4_PROBE` → `RESULTS_PREVAD` → **`v3/RESULTS_V3_GATE_ATTRIBUTION`** → **`v3/RESULTS_DADA`**, plus `REPORT_KIP_MSAD_DOTA_PREVAD.md` (and, on branch `v3`, the spec trio). **`outputs/` is gitignored**, so those documents are the only durable record — and on `main`, where `RESULTS_V3_GATE_ATTRIBUTION.md` is absent, `.project/memory-bank/{progress,activeContext}.md` is that record; the **62,254** per-clip `.npz` score files live on the user's disk/Drive.

Detail lives in `.project/memory-bank/{activeContext,progress}.md` — read those, not this section, before acting.

### 14.5.1 Things not to get wrong

- **The v1 gate MLP is never trained, and H4′ is now CONFIRMED.** `(ratio * max_shift).floor().long()` kills the gradient — all 6 tensors get `grad is None`. Worse, the gate is *measurably* near-constant: its input is min-max normalized, so `[0,1]` is the whole reachable domain, and sweeping it moves `s_t` by **0–4 channels out of 128** across 8 seeds (seed 0: exactly 0). **v1's "motion-gated adaptive temporal shift" was, in operation, a fixed ~50 % shift at `s ≈ 58–69`.** Never write "motion-gated" of `gate_type="mlp_frozen"`. Measured 2026-08-30, plan Appendix C. The v3 **rank gate** (`gate_type="rank"`, the default *on branch `v3`*) spans the full `[0, 128]` on every clip and is parameter-free. **On `main` there is no `gate_type` at all: the frozen-MLP gate is the only gate, so every KIP-on run on this branch is a fixed ~50 % shift by construction.**
- **A feature cache is bound to the transform and stride that built it** (lesson 2/13). The pipeline is on `no_center_crop`; all current artifacts are `*_ncc`. Changing `preprocess_frames` or `FRAME_STRIDE` invalidates every cached feature *and every metric measured on it*.
- **Pooling is decided by the label distribution, not the dataset name** (lesson 12). DoTA is ~all-abnormal → per-clip min-max; MSAD/PreVAD → raw. Mixing the two protocols in one table is how a released checkpoint reads as chance.
- **A score head whose kernel spans the clip is a clip classifier.** `ConvScoreHead` is one `Conv1d(512→1, kernel_size=9)`; DADA-2000's median clip is **9** stride-8 frames (MSAD 86, DoTA 13), so every output timestep sees the whole clip. Result: flat curves, `auc_macro` at chance, and a micro AUC that is really video classification. **Check `score_head_kernel` against a new corpus's median sequence length before training on it.** (`RESULTS_DADA.md` §5.)
- **Micro AUC over a test set full of all-normal clips measures clip classification** — the mirror of lesson C12. On DADA-2000, 74 % of test frames come from all-normal clips and a *constant-score-per-clip* oracle scores **0.9069**; our best arm scored 0.8739 with `auc_macro` at chance. Compute that oracle and print it beside any micro number. **Never put a DADA micro AUC next to a published frame-level number.**
- **Do not tune KIP on any delta** (lesson 14). The MSAD +0.09 is attributed to smoothing and does not replicate on DADA; tuning against either number is fitting the benchmark.
- **The flow target has no spatial content** — 23 frame-global scalars lifted to 256-d by a fixed seeded projection (`core/flow/raft_extract.py:52-81`). Say "23 effective dimensions", and never claim KIP localizes anything spatially.
- **PreVAD can never host a KIP-on arm.** It ships CLIP features, no video, so the
  RAFT targets `L_KIP_rec` needs are unbuildable; `require_flow=False` is not a
  workaround (it trains the PMG head to predict zeros). Settled 2026-08-29 —
  do not reopen. PreVAD = KIP-off trunk + Gate P0 only (`PREVAD_SETUP.md` §7.4).
- **`LaGoVAD-PreVAD/` is read-only**, and gate against the released checkpoint, not the paper's printed number (lesson 8b).
- **A runbook on `main` is not runnable on `main` just because the file is there.** `core/docs/v3/setup/{DADA,TAD}_V3_SETUP.md` prescribe six arms keyed on `kip.gate_type`; this branch parses none of them. Check the flag exists in `core/config.py` before copy-pasting a runbook command.
- **The memory bank is tracked per branch and the two copies have deliberately diverged** (`main` = v1, `v3` = v3). A `git merge` between them **will** conflict in `{CLAUDE.md,.project/memory-bank/*}` — resolve by branch identity, never by "take theirs".

## 14.6 Delivery Requirements

- Every entry-point script ships with an `argparse`-based CLI (training, inference, flow/feature extraction, evaluation, visualization).
- Follow §10 code standards (PEP 8, type hints on public surface, no `print` → `logging`, no bare `except`, no `global`, constants in `constants.py`).
- Quality gate before commit (§11): `source .venv/bin/activate && ruff check * && mypy * && bandit * && pycycle * && pyright *`.
- Documentation in `core/docs/`; planning docs in `.project/plans/`.

For full architecture details → `.project/memory-bank/systemPatterns.md` | For stack details → `.project/memory-bank/techContext.md`
