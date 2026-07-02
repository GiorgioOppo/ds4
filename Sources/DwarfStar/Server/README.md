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

HTTP traffic is plaintext. The intended default is `127.0.0.1`; if you expose it
beyond loopback, put it behind TLS.
