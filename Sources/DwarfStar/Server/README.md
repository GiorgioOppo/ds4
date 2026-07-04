# DwarfStar/Server

Native in-process HTTP server built on `Network.framework`, compatible with
OpenAI and Anthropic-style APIs. It does not launch a subprocess and does not
load a second model. The server wraps the **single shared `InferenceService`**
loaded in Settings, so Chat and HTTP requests use the same engine actor and are
serialized one at a time. This avoids duplicating resident Q4, `mlock`ed buffers,
KV scratch, and expert-cache memory on 16 GB systems.

There is no model choice over HTTP: `/v1/models` advertises exactly the loaded
model (the GGUF basename); a different `model` field in a request is logged and
overridden, and every response reports the real loaded model.

- **`ServerController.swift`** starts/stops the listener and binds it to the
  already-loaded shared engine.
- **`LocalServer.swift`** implements endpoint routing for
  `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/messages`, and
  `/v1/models`, including streaming and non-streaming responses.
- **`ChatRequestParser.swift`** parses request bodies into engine-level types.
- **`ServerView.swift`** renders the server panel.

Request handling limits: bodies are capped at 32 MB (413 beyond that) and a
client must deliver its full request within 60 s or the connection is dropped —
neither can stall the serialized engine, since the gate is only acquired after
the request has been fully read and parsed.

An optional API key can be set in the server panel: when present, every `/v1`
request must send `Authorization: Bearer <key>` (OpenAI style) or
`x-api-key: <key>` (Anthropic style); anything else gets a 401.

HTTP traffic is plaintext. The intended default is `127.0.0.1`; if you expose it
beyond loopback, put it behind TLS — the API key guards against other local
processes, not against network eavesdropping.
