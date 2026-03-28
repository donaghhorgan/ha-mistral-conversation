Get API Docs via chub
======================

When you need documentation for a library or API, fetch it with the `chub` CLI
rather than guessing from training data. This gives you the current, correct
API.

Step 1 — Find the right doc ID
--------------------------------

```bash
chub search "<library name>" --json
```

Pick the best-matching `id` from the results (e.g. `openai/chat`,
`anthropic/sdk`, `stripe/api`). If nothing matches, try a broader term.

Step 2 — Fetch the docs
------------------------

```bash
chub get <id> --lang py    # or --lang js, --lang ts
```

Omit `--lang` if the doc has only one language variant — it will be
auto-selected.

Step 3 — Use the docs
----------------------

Read the fetched content and use it to write accurate code or answer the
question. Do not rely on memorized API shapes — use what the docs say.

Step 4 — Annotate what you learned
------------------------------------

After completing the task, if you discovered something not in the doc — a
gotcha, workaround, version quirk, or project-specific detail — save it so
future sessions start smarter:

```bash
chub annotate <id> "Webhook verification requires raw body"
```

Annotations are local, persist across sessions, and appear automatically on
future `chub get` calls. Keep notes concise and actionable. Don't repeat
what's already in the doc.

Step 5 — Give feedback
-----------------------

Always rate the doc after using it. This helps authors fix outdated or
incorrect docs and prioritize improvements. Include a label and a brief
comment explaining what was good or what needs fixing — specific feedback is
the most valuable.

```bash
chub feedback <id> up --label accurate "Clear examples, models are current"
chub feedback <id> down --label outdated "Lists gpt-4o as latest but gpt-5.4 is out"
```

Available labels: `outdated`, `inaccurate`, `incomplete`, `wrong-examples`,
`wrong-version`, `poorly-structured`, `accurate`, `well-structured`, `helpful`,
`good-examples`.

If you notice the doc has wrong model names, deprecated APIs, missing
features, or incorrect code patterns, always leave a downvote with details so
it can be fixed.

Quick reference
---------------

Goal
  List everything
  Find a doc
  Exact id detail
  Fetch Python docs
  Fetch JS docs
  Save to file
  Fetch multiple
  Save a note
  List notes
  Rate a doc

Command
  chub search
  chub search "stripe"
  chub search stripe/api
  chub get stripe/api --lang py
  chub get openai/chat --lang js
  chub get anthropic/sdk --lang py -o docs.md
  chub get openai/chat stripe/api --lang py
  chub annotate stripe/api "needs raw body"
  chub annotate --list
  chub feedback stripe/api up

Notes
-----

- `chub search` with no query lists everything available
- IDs are `<author>/<name>` — confirm the ID from search before fetching
- If multiple languages exist and you don't pass `--lang`, chub will tell you
  which are available
