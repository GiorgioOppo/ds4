# DwarfStar/Server

Server HTTP **nativo in-process** (su `Network.framework`), OpenAI/Anthropic-compatible. Niente sottoprocesso; i pesi GGUF sono condivisi (mmap) con l'engine della chat. Una richiesta alla volta.

- **`ServerController.swift`** — avvio/stop, configurazione, wiring del KV su disco.
- **`LocalServer.swift`** — il server: routing degli endpoint (`/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/messages`, `/v1/models`), streaming SSE e non.
- **`ChatRequestParser.swift`** — parsing dei body di richiesta nei tipi dell'engine.
- **`ServerView.swift`** — UI del pannello.

⚠️ HTTP in chiaro: pensato per `127.0.0.1`; oltre il loopback mettilo dietro TLS.
