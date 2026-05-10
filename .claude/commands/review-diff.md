---
description: Review the current uncommitted changes — summarize every change and flag anything risky.
---

Read the current git state (`git status`, then `git diff` for unstaged and `git diff --cached` for staged) and produce a structured review.

For each modified file:
1. Summarize what changed in one sentence.
2. Flag anything risky: scope creep beyond the stated task, missing test coverage, ignored edge cases, new dependencies, hardcoded values, debug code, secrets, or anything that looks like it was added "while I was in there."
3. Note anything that should be in a separate change.

End with a one-line verdict: **ship it**, **needs work** (list what), or **do not ship** (explain).

Do not run formatters or modify code — this is read-only. If you spot bugs, surface them; don't fix them.
