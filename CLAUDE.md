# CLAUDE.md

## Pre-commit checklist

Every commit must pass all three checks before being committed:

```bash
.venv/bin/ruff check src/ tests/       # lint
.venv/bin/ruff format --check src/ tests/  # formatting
.venv/bin/python -m pytest tests/ -q   # unit tests
```

To auto-fix formatting before committing:

```bash
.venv/bin/ruff format src/ tests/
```
