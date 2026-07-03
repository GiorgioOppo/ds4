# DwarfStar/Server

Native in-process HTTP server built on `Network.framework`, compatible with
OpenAI and Anthropic-style APIs. It does not launch a subprocess and does not
load a second model. The server wraps the **single shared `InferenceService`**
loaded in Settings, so Chat and HTTP requests use the same engine actor and are
serialized one at a time. This avoids duplicating resident Q4, `mlock`ed buffers,
KV scratch, and expert-cache memory on 16 GB systems.

- **`ServerController.swift`** starts/stops the listener and binds it to the
  already-loaded shared engine.
- **`LocalServer.swift`** implements endpoint routing for
  `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/messages`, and
  `/v1/models`, including streaming and non-streaming responses.
- **`ChatRequestParser.swift`** parses request bodies into engine-level types.
- **`ServerView.swift`** renders the server panel.

HTTP traffic is plaintext. The intended default is `127.0.0.1`; if you expose it
beyond loopback, put it behind TLS.
