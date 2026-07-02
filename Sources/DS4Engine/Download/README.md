# DS4Engine/Download

Download dei modelli.

- **`ModelDownloader.swift`** — download GGUF **resumibile** (HTTP Range) dall'endpoint `resolve` di Hugging Face, direttamente in `<ggufDir>/<file>.part`. Nessun script/curl esterno. Verifica l'**integrità SHA-256** del file scaricato contro un digest noto (`ModelTarget.sha256`) quando configurato; altrimenti riporta il digest calcolato. La verifica del contenuto è la difesa robusta (immune alla rotazione delle chiavi della CDN), non il pinning TLS.
