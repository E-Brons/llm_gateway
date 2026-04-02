# LLM Gateway

General-purpose LLM Gateway — Python library and local HTTP server that unifies access to Ollama, LiteLLM (cloud APIs), and the Claude CLI under a single typed interface.

## Prerequisites

Runs on macOS (Apple arm64).

- **Python** ≥ 3.14
- **Ollama** ≥ 0.19 (includes MLX support)

## Install

```sh
./scripts/install.sh
```

Checks prerequisites and installs Python dependencies into `.venv/`.

## Configure

Copy the example config and edit it:

```sh
cp llm_route.yml.example local/llm_route.yml
```

Override server settings (port, logging, concurrency) via `local/settings.json` — any key in `settings.json` can be overridden there without editing the tracked file.

## Run

```sh
./scripts/run.sh                        # start on default port 4096
./scripts/run.sh --config port:9191     # override port
./scripts/run.sh --config host:0.0.0.0  # override host (requires allow_external_port: true)
./scripts/run.sh --config_file <PATH>   # use a different LLM config file
./scripts/stop.sh                       # stop the gateway
```

The server starts a FastAPI backend on `http://127.0.0.1:8000` by default and waits until `/health` responds before returning.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/models` | List configured models and their capabilities; annotates Ollama models with availability |
| `POST` | `/general` | Open-ended text completion |
| `POST` | `/text_gen` | Structured text generation with retry |
| `POST` | `/reasoning` | Extended-thinking / reasoning-mode completion |
| `POST` | `/image_gen` | Image generation (prompt → base64 PNG) |
| `POST` | `/image_inspector` | Vision: analyse an image (base64 in, text out) |
| `POST` | `/tools` | Tool / function calling |

## Test

```sh
.venv/bin/python -m pytest tests/
```

Unit tests run fully offline with mocked subprocess and HTTP calls.

## Library usage

```python
from src import create_factory

factory = create_factory("local/llm_route.yml")

# Text
result = factory.general().complete([{"role": "user", "content": "Hello"}])
print(result.content)

# Tool use
result = factory.tools().complete(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }],
)
for tc in result.tool_calls:
    print(tc.name, tc.arguments)
```

## Docs

Design specs and architecture docs live in [`docs/`](docs/).
