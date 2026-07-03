# DwarfStar/Server

Native in-process HTTP server built on `Network.framework`, compatible with
OpenAI and Anthropic-style APIs. It does not launch a subprocess; GGUF weights
are shared through mmap with the chat engine. Requests are processed one at a
time.

- **`ServerController.swift`** starts/stops the server, applies configuration,
  and wires disk KV.
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
