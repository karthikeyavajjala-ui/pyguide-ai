# PyGuide AI

**Your Personal Python Programming Assistant** — a local, retrieval-based Python learning app with curated explanations, runnable examples, saved conversations, topic search, progress tracking, dark mode, and practice challenges.

## Run it

This app has no npm dependencies. Python 3.9+ is enough:

```bash
python3 server.py
```

Then open `http://localhost:8000`. The app uses `localStorage` for saved chats, theme, and learning progress. The built-in code runner executes a restricted Python subset in a short-lived subprocess; imports are limited to `math` and `statistics`, and filesystem/network access is disabled. Third-party library examples are shown for learning but marked as local-only.

## Included

- Knowledge-base search and natural-language concept matching
- 40+ curated Python topics with explanations, examples, expected output, and related concepts
- Chat history, feedback, learned-topic tracking, and responsive light/dark UI
- Syntax-highlighted snippets and copy buttons
- Safe Python execution endpoint at `POST /api/run`
- Four editable practice challenges with run and reveal-solution actions
