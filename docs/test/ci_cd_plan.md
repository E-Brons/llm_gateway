# LLM Gateway — CI/CD Plan

> **Status**: Not yet implemented. This doc describes the intended pipeline.

---

## CI Pipeline (GitHub Actions)

Trigger: every push and PR to `main`.

```
lint → test → (future: publish)
```

### Jobs

```yaml
jobs:
  lint:
    # ruff / pyflakes static analysis
    runs-on: ubuntu-latest

  test:
    # pytest unit tests (no services required)
    needs: lint
    strategy:
      matrix:
        python-version: ["3.14"]
    runs-on: ubuntu-latest
```

### Job detail

#### lint

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with: { python-version: "3.14" }
  - run: pip install ruff
  - run: ruff check src/ tests/
```

#### test

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with: { python-version: "3.14" }
  - run: pip install -e ".[test]"
  - run: pytest tests/ --cov=src --cov-fail-under=85
```

No Ollama server in CI — all tests mock network and subprocess calls.

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, always green |
| `feature/*` | Feature branches, PR to main |

---

## Quality Gates

- All 89 unit tests pass
- No ruff lint errors
- Coverage ≥ 85% on `src/`

---

## CD (future)

- Tag `vX.Y.Z` → publish to PyPI as `llm_gateway`
- Consumers install via `pip install llm_gateway` or `git+https://...@vX.Y.Z`
