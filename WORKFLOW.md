# Claude Code Workflow v2

The Reddit workflow ("v1") is a set of manual habits layered on top of Claude Code. v2 is about leaning on what the tool actually provides — plan mode, hooks, subagents, memory, rewind, slash commands — so the discipline lives in your setup, not your willpower.

This doc is a menu, not a prescription. Pick the level that matches the project.

---

## The v2 core loop

```
plan → constrain → execute → verify → checkpoint
   ↑                                       │
   └──── memory / rules drift forward ─────┘
```

Each step below has a "lightweight / medium / strict" option. Mix as you like.

---

## 1. Planning

**v1 said:** write a plan prompt before coding.
**v2 says:** use the harness's plan mode, and pick the planning depth that matches the change.

- Toggle plan mode with **Shift+Tab** (cycles auto-accept → plan → normal). In plan mode Claude can read but not write — it produces a plan, then `ExitPlanMode` asks you to approve.
- Three planning depths:

  | Level       | When                              | What to ask for                                                                |
  | ----------- | --------------------------------- | ------------------------------------------------------------------------------ |
  | Lightweight | bug fix, single-file change       | one paragraph: root cause + the change                                         |
  | Medium      | new feature in known territory    | per-file change list, edge cases, what's intentionally **not** changing        |
  | Heavy       | architectural / cross-cutting     | spawn the **Plan** subagent — separate context, returns a structured plan      |

- Always include "what's intentionally not in scope" in the plan. Scope creep is the #1 way Claude wanders.

---

## 2. Context discipline

The biggest invisible failure mode is a polluted context window. v1 ignores this. v2 manages it.

- **CLAUDE.md** — project rules: stack, conventions, "never do" list, where to find things. Loaded automatically. Keep under ~200 lines or it dilutes attention.
- **Memory system** — persistent notes across sessions (user preferences, project facts, feedback). Different from CLAUDE.md: memory is yours and personal-ish; CLAUDE.md is the project's.
- **`/compact`** when context gets heavy but the thread is still useful — summarizes and continues.
- **`/clear`** when switching tasks — start fresh. Cheaper and cleaner than `/compact`.
- **Subagents protect the main context** (see §4). Use them for any read-heavy exploration so the raw output never lands in your main thread.

---

## 3. Constraints in prompts

v1 had this right. v2 adds: be specific about *where* and *what to verify*.

Good constraint phrases:

- "Touch only `src/foo/*` and `tests/foo/*`."
- "Use the existing pattern at `src/bar.ts:42` — don't introduce a new one."
- "No new dependencies. If you think you need one, stop and ask."
- "Don't refactor adjacent code, even if it looks wrong. Note it instead."
- "Verify by running `pnpm test foo` — don't summarize, paste the result."

The last one matters most. Agent summaries are aspirational; tool output is real.

---

## 4. Subagents (the biggest v1 → v2 jump)

The `Agent` tool spawns a fresh Claude with its own context. Use it for:

| Use case                                | Subagent type     |
| --------------------------------------- | ----------------- |
| Find where X is defined / used          | `Explore`         |
| Open-ended research across the repo     | `general-purpose` |
| Design an implementation strategy       | `Plan`            |
| Independent second opinion on a diff    | `code-reviewer` (if you have it) |

**Run in parallel** when tasks are independent — one message, multiple `Agent` calls. A 4-way parallel exploration finishes in the time of one.

**Brief them like a new colleague**: they have no memory of your conversation. Include the goal, what you've ruled out, what form of answer you want, and a length cap.

**Trust but verify**: an agent's summary is what it *intended* to do. Read the diff yourself before declaring done.

---

## 5. Hooks — automate the discipline

v1 relies on you remembering to lint, format, test. v2 puts those in `.claude/settings.json` so the harness enforces them.

Three tiers:

**Lightweight** — format on save:
```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "Edit|Write", "hooks": [{ "type": "command", "command": "prettier --write $CLAUDE_FILE" }] }
    ]
  }
}
```

**Medium** — typecheck/lint before Claude says "done":
```json
{
  "hooks": {
    "Stop": [
      { "hooks": [{ "type": "command", "command": "pnpm typecheck && pnpm lint" }] }
    ]
  }
}
```

**Strict** — block destructive bash, require tests pass before commit:
```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "scripts/guard-bash.sh" }] }
    ]
  }
}
```

Hooks fail loudly, which is what you want. The `update-config` skill helps if you don't want to hand-edit JSON.

---

## 6. Permissions — stop the prompt fatigue

Every "Allow this command?" interruption breaks your flow. Configure once:

- `/permissions` to view and edit.
- The **`fewer-permission-prompts`** skill scans your transcript and proposes a sensible allowlist for read-only commands.
- Allow read-only liberally (`ls`, `cat`, `git status`, `pnpm run typecheck`). Keep mutating commands gated.
- Per-project `.claude/settings.json` for project-specific allowances; global `~/.claude/settings.json` for things you trust everywhere.

---

## 7. TDD vs. patch-loop

v1 describes a "build, run tests, fix, repeat" cycle. That's a patch-loop and it's prone to whack-a-mole — Claude papers over symptoms instead of fixing root causes.

Two better options:

- **Real TDD** — "write a failing test for X, then implement until it passes." Claude is unusually good at this because the test pins down the spec. Best for pure logic, parsers, data transforms.
- **Spec-then-build** — write a one-paragraph spec in the plan, build, then run tests. Best for UI / integration work where TDD is awkward.

Either way, when something fails, your next prompt should be **"why did this fail?"** not **"fix this."** The first gets a diagnosis; the second gets a guess.

---

## 8. Verification — don't trust the summary

The single biggest accuracy boost: stop reading Claude's "I implemented X and it works" summaries and start reading the diff.

- After any non-trivial change, run `git diff` yourself or ask: "show me the full diff and call out anything risky."
- For UI changes, open the browser. Type checks ≠ feature works.
- For backend changes, hit the actual endpoint, not just the test.
- If Claude says a test passes, ask it to paste the test runner output.

This is the habit that separates "I shipped a working feature" from "I shipped what looked like a working feature."

---

## 9. Rewind — use it instead of git resets

v1 calls this "checkpointing" and treats it as a workflow you maintain. It's actually built in:

- **`Esc Esc`** — rewind the conversation to an earlier turn. Use when you want to retry from a known-good point.
- **`/rewind`** — restore file state to a previous turn. Faster than `git checkout` for transient mistakes that aren't committed.

When to commit instead: any state you'd be sad to lose. Rewind is for transient experiments; git is for durable progress.

---

## 10. Slash commands and skills — codify your workflows

If you find yourself typing the same prompt twice, make it a slash command.

- **Project commands**: `.claude/commands/<name>.md` — available in this repo.
- **Personal commands**: `~/.claude/commands/<name>.md` — available everywhere.
- **Skills** are richer: a folder with a `SKILL.md` plus optional scripts/examples. Better for multi-step workflows with logic.

Useful starter commands:
- `/review-diff` — "Read the staged diff, list every change, flag risky ones."
- `/scope-check` — "Compare the diff to the plan in this thread. List anything out of scope."
- `/why-failed` — "The last command failed. Diagnose root cause before suggesting a fix."

---

## 11. MCP servers — only what you use

MCP exposes external tools (DBs, APIs, browsers, search) to Claude. Tempting to install many; better to install few.

- Start with zero. Add one when a workflow genuinely needs it (e.g. a Postgres MCP if you're constantly running queries).
- Each MCP adds tools to Claude's context budget. Five lightly-used MCPs is worse than two heavily-used ones.
- Audit periodically: `claude mcp list`, remove unused.

---

## 12. Scope discipline (the v1 refactor rule, generalized)

v1 says "refactor after the feature works." v2 generalizes: **one task, one concern.**

- "While you're in there, also fix Y" is how PRs grow legs.
- If Claude notices something wrong adjacent to its task, the right behavior is to **note it**, not fix it. Add to your CLAUDE.md: "Don't fix unrelated issues — surface them in the response so I can decide."
- Feature work and refactors should be different commits, ideally different sessions.

---

## Anti-patterns to retire from v1

- **"Let me read the file to verify"** when the harness already tracks file state. Wastes context.
- **One mega-prompt** with the whole feature. Plan + small chunks beats it every time.
- **Reading Claude's summary instead of the diff.** Already covered, repeated because it matters.
- **Letting tool output pile up in main context.** Delegate to subagents.
- **Skipping plan mode for "simple" changes.** They're rarely as simple as they look.
- **Manual checkpointing rituals.** Use `Esc Esc` / `/rewind` / git.

---

## Starter setup checklist

For a new project, in order:

1. `CLAUDE.md` — stack, conventions, "never do" list, scope discipline rule.
2. `.claude/settings.json` — at minimum, a Stop hook for typecheck + lint.
3. `/permissions` — allow your common read-only commands.
4. One or two slash commands you'll actually use (`/review-diff`, `/scope-check`).
5. Skip MCP and custom subagents until you hit a real need.

That's the whole upgrade. v1 is "be disciplined." v2 is "make the harness enforce the discipline so you don't have to be."
