---
name: Loop blocker — main is red or task can't proceed
about: The autonomous loop hit pre-existing red on main, or otherwise can't make progress. Filed by the loop or by a human.
title: "main is red: <one-line symptom>"
labels: ["loop-blocker"]
---

## Symptom

<!-- One sentence. Example: "uv run mypy --strict src fails on competitor.py:19" -->

## Affected task(s)

<!-- TASKS.md slug(s) blocked by this issue. -->

-

## Reproduce

```bash
# Exact commands that surface the problem on a fresh main:
git checkout main && git pull --ff-only origin main && uv sync
# <the command that fails>
```

## Suggested fix

<!--
Concrete. One paragraph max. If you know the line and the change, include it.
Example:
  Add `# type: ignore[misc]` on src/vgc_ai/competitor.py:19, OR extend the
  mypy override in pyproject.toml to disable misc errors for the vgc2.* path.
-->

## Filed by

- [ ] Human
- [ ] Autonomous loop (iteration log: `~/vgc-ai-logs/claude/iteration-<TS>.log`)
