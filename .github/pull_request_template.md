<!--
PR template for vgc-ai. Both human and autonomous-loop PRs should follow this.
The autonomous loop fills this in from its bench results JSON automatically;
humans should fill it in by hand.
-->

## Task

<!-- Slug from TASKS.md (e.g. `policy-heuristic-eval`), or "ad-hoc" for human PRs. -->

## What changed

<!-- 1–3 bullets. WHAT, not WHY. Concrete: file paths, new classes, removed code. -->

-

## Why

<!-- 1–3 sentences. The constraint / motivation. Skip if it's identical to the linked task. -->

## Bench evidence

<!--
For policy changes: paste the contents of bench/results/<slug>.json verbatim.
For non-policy changes (refactor, infra, docs): write "N/A — <reason>".
-->

```
N/A
```

## Checks

- [ ] `uv run ruff format .`
- [ ] `uv run ruff check src tests bench`
- [ ] `uv run mypy --strict src`
- [ ] `uv run pytest -q`
- [ ] Touched files only — no unrelated drive-by edits

## Risks / open questions

<!-- What could break? Anything the reviewer should look at twice? -->

-
