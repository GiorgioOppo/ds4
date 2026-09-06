**English** | [Italiano](HTTP-API.it.md)

# Local HTTP API

The in-process server exposes the model already loaded by the app; it never
creates a second inference service.

## Endpoints

| Endpoint | Compatibility layer |
|---|---|
| `GET /v1/models` | Loaded-model discovery |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/completions` | Legacy OpenAI Completions |
| `POST /v1/messages` | Anthropic Messages |

Streaming and non-streaming requests use the same engine event source. The
server reports the actual GGUF basename even when a request supplies another
model identifier.

Model discovery (`/v1/models` and `/v1/models/{id}`) includes
`context_length`, the total token capacity configured for the loaded backend.
It is separate from `max_completion_tokens`, the server's default response
budget. Clients should reserve space for both the prompt and generated tokens
within that context; the GGUF's theoretical maximum is not advertised.

## Request path

1. `Network.framework` accepts and bounds the request.
2. `HTTPRequest` parsing validates headers and body framing.
3. API-key and endpoint checks reject invalid requests before generation.
4. The endpoint adapter converts JSON into engine conversation types.
5. `RequestGate` serializes inference against Chat and other HTTP clients.
6. Response helpers emit JSON or streaming frames and close the connection.

The default listener is loopback-only and plaintext. Binding to another
interface requires an external TLS reverse proxy; the optional API key does not
provide transport encryption.
