---
description: Compare current uncommitted changes against the originally-stated task and flag scope creep.
---

The user asked for a specific thing earlier in this conversation (or recently). Find that ask and compare it to what was actually changed.

Steps:
1. Look back through the conversation for the most recent task statement from the user (what they asked you to do).
2. Run `git status` and `git diff` to see what changed.
3. For each change, classify it as:
   - **In scope** — directly serves the stated task.
   - **Adjacent** — touches the same area but wasn't asked for (e.g. a rename, a small refactor, a docstring).
   - **Out of scope** — unrelated to the task entirely.
4. Report the classification as a table or bulleted list.

End with a recommendation: **proceed as-is**, **split into separate changes** (and how), or **revert the out-of-scope parts**.

If the user never stated a clear task, say so and ask them to clarify before judging scope.
