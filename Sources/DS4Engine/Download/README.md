# DS4Engine/Download

Native model download support.

- **`ModelDownloader.swift`** performs **resumable** GGUF downloads with HTTP
  Range requests from Hugging Face `resolve` endpoints, writing directly to
  `<ggufDir>/<file>.part`. It does not shell out to scripts or `curl`.

When `ModelTarget.sha256` is configured, the downloader verifies the downloaded
file against that known SHA-256 digest. Otherwise it reports the calculated
digest. Content verification is the robust defense here because it survives CDN
key rotation; this code does not rely on TLS pinning.
