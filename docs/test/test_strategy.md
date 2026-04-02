# LLM Gateway — Test Strategy

## Architecture

llm_gateway is a pure Python library and local HTTP gateway. There is no frontend. Tests run entirely offline using mocked subprocess and HTTP calls.

---

## Test Levels

### 1. Unit Tests (`tests/`, pytest)

All tests live in `tests/` and run with:

```sh
.venv/bin/python -m pytest tests/
```

No running services required — all network and subprocess calls are mocked.

| File | Covers |
|------|--------|
| `test_core_llm_factory.py` | Config loading from YAML, factory method return types, invalid impl raises, CLI-for-tools raises |
| `test_core_llm_ollama.py` | All 6 Ollama implementations: happy paths, retry behaviour, payload structure, tool call parsing |
| `test_core_llm_litellm.py` | All 6 LiteLLM implementations: happy paths, api_base forwarding, thinking budget, tool call parsing |
| `test_core_llm_cli.py` | All 4 CLI implementations: subprocess format, effort flag, image base64 embedding |
| `test_core_llm_responses.py` | `TextResponse`, `ImageResponse`, `ToolCall`, `ToolCallResponse`: construction, immutability, optional fields |
| `test_core_llm_construction_params.py` | temperature / max_tokens / response_schema forwarded correctly per backend |
| `test_core_llm_retry.py` | `retry_text_completion` and `retry_image_generation`: empty response, exception, exhaustion, correction hint, Transfer-Encoding callback |
| `test_core_llm_workaround.py` | `reset_litellm_client()`: replaces module-level client, updates cache, graceful when unavailable |

**Current count**: 89 tests, all passing.

### What is validated per type

| Type | Validated |
|------|-----------|
| **General** | Happy path, model prefix stripping (Ollama), api_base forwarding (LiteLLM), system prompt via `--system-prompt` (CLI) |
| **Text-gen** | Happy path, empty-response retry until success, exhaustion raises `ValueError`, stream-json format used (CLI) |
| **Reasoning** | Happy path, `thinking_budget` forwarded (LiteLLM), silently ignored (Ollama), `--effort high` used (CLI) |
| **Image-gen** | Happy path, missing image retries then raises, `width`/`height`/`seed` in options payload |
| **Image Inspector** | Happy path, image bytes base64-encoded in payload, multimodal content format (LiteLLM), stream-json embedding (CLI) |
| **Tools** | Happy path, tool definitions forwarded in payload, tool call parsed from response, no-tool-calls returns empty list, ID generated when Ollama omits it |

### 2. Integration Tests (local, manual)

These require a running Ollama server and are not run in CI. Use `pytest -m integration` when Ollama is available locally.

Fixtures in `conftest.py` skip automatically if Ollama is unreachable:

| Fixture | Purpose |
|---------|---------|
| `ollama_url` | Base URL (env `OLLAMA_URL`, default `http://localhost:11434`) |
| `ollama_text_model` | Auto-detects a suitable text model from pulled models |
| `ollama_vision_model` | Auto-detects a vision-capable model |
| `ollama_image_model` | Auto-detects a diffusion/image-gen model |

---

## Running Tests

```sh
# All unit tests (no services required)
.venv/bin/python -m pytest tests/ -v

# With coverage
.venv/bin/python -m pytest tests/ --cov=src --cov-report=term-missing

# Single file
.venv/bin/python -m pytest tests/test_core_llm_ollama.py -v

# Skip slow/integration
.venv/bin/python -m pytest tests/ -m "not integration"
```

---

## Test Naming Convention

- Files: `test_core_llm_<scope>.py` — one file per concern (implementation type, retry logic, etc.)
- Functions: `test_<class>_<scenario>` — e.g. `test_ollama_tools_generates_id_when_missing`
- Mocking: all external calls patched at the module where they are used (e.g. `src.impl.impl_ollama.requests.post`)

---

## Mocking Strategy

| External dependency | How mocked |
|--------------------|------------|
| Ollama HTTP (`requests.post`) | `unittest.mock.patch` on `src.impl.impl_ollama.requests.post` |
| LiteLLM (`litellm.completion`) | `patch` on `src.impl.impl_litellm.litellm.completion` |
| LiteLLM image generation | `patch` on `src.impl.impl_litellm.litellm.image_generation` |
| Claude CLI (`subprocess.run`) | `patch` on `src.impl.impl_cli.subprocess.run` |
| `reset_litellm_client` | `patch` on `src.impl.impl_litellm.reset_litellm_client` to isolate init side-effects |

---

## CI

CI is not configured yet (no `.github/workflows/` exists). See `docs/test/ci_cd_plan.md` for the planned pipeline.
