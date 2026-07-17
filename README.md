# Fintech Agent API

A FastAPI service exposing a **streaming, tool-calling fintech agent** backed by the
OpenAI API, plus a hybrid (vector + keyword) RAG retriever. The codebase follows a clean,
layered architecture with strict type hints and full dependency injection.

## Architecture

```
app/
├── config.py              # Typed settings (pydantic-settings), loaded from env/.env
├── dependencies.py        # FastAPI DI providers (read service singletons off app.state)
├── main.py                # App factory + lifespan that wires the shared OpenAI client
├── models/                # Pydantic request/response models (the contracts)
│   ├── agent.py           # UserRequest, AgentRequest/Response, FintechTransactionQuery, ...
│   ├── llm.py             # ChatMessage, LLMRequest/Response
│   └── rag.py             # RAGConfig, ScoredDocument, RetrieveContextRequest
├── routers/v1/agent.py    # POST /v1/chat — streaming endpoint
└── services/              # Decoupled business logic (no framework imports)
    ├── agent_core.py      # AgentOrchestrator: the tool-calling loop
    ├── llm_service.py     # LLMService: chat completion + streaming
    ├── vector_db.py       # RAGService + Qdrant/in-memory backends
    └── retry.py           # Shared OpenAI rate-limit backoff policy
```

**Request flow for `POST /v1/chat`:** the `AgentOrchestrator` runs the tool-calling loop
(fetching transaction data when needed) and returns a resolved message context; the
`LLMService` then streams the final answer to the client token-by-token via
`StreamingResponse`. Both services are injected with `Depends` and share a single
`AsyncOpenAI` client created during startup.

### Safety & performance note

- **Prompt caching** — the static system instructions are a module-level constant sent as
  the *first* message on every request, so identical prefixes can be cached upstream.
- **Prompt-injection mitigation** — user text is never interpolated into the system prompt;
  it only ever travels in a separate `user` message, and the system prompt instructs the
  model to treat user content as untrusted data.
- **Bounded token spend** — the agent loop has a hard `AGENT_MAX_ITERATIONS` cap and raises
  `MaxIterationsExceededError` (surfaced as HTTP `504`) instead of looping indefinitely.
- **Explicit error handling** — only specific exceptions are caught (`openai.APIError`,
  `pydantic.ValidationError`, Qdrant connection errors), each with structured logging.

## Configuration

Copy `.env.example` to `.env` and set at least `OPENAI_API_KEY`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — (required) | OpenAI-compatible API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for the agent |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for RAG |
| `QDRANT_URL` | _(empty)_ | Qdrant endpoint; empty → in-memory store |
| `QDRANT_API_KEY` | _(empty)_ | Qdrant API key |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |
| `AGENT_MAX_ITERATIONS` | `5` | Hard cap on agent tool-loop steps |
| `REQUEST_TIMEOUT_SECONDS` | `30` | OpenAI client timeout |

## Local development (Poetry)

```bash
poetry install
cp .env.example .env            # then edit OPENAI_API_KEY
poetry run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API, or call it directly:

```bash
curl -N -X POST http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Summarize the transfers on account ACC123 in July 2026."}'
```

`-N` disables curl buffering so you see tokens stream in.

## Testing & linting

```bash
poetry run ruff check app tests     # lint
poetry run ruff format app tests    # format
poetry run pytest                   # unit tests (all network calls are mocked)
```

The suite covers every router, Pydantic model, and service with `unittest.mock` — no real
OpenAI or Qdrant calls are made.

## Deployment (Docker)

The `Dockerfile` uses a multi-stage build (Poetry-based dependency layer → slim runtime)
and runs as a non-root user with a `/health` healthcheck.

**Single container:**

```bash
docker build -t fintech-agent-api .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... fintech-agent-api
```

**Full stack with Qdrant (recommended):**

```bash
export OPENAI_API_KEY=sk-...       # or put it in .env (docker compose reads it automatically)
docker compose up --build
```

`docker compose` starts the API alongside a Qdrant instance, wires `QDRANT_URL` to the
`qdrant` service, waits for Qdrant to become healthy before starting the API, and persists
Qdrant data in the `qdrant_storage` volume. The API is served on
http://localhost:8000 and Qdrant on http://localhost:6333.
