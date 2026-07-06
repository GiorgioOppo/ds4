# DS4Engine/Download

Native model download support.

- **`ModelDownloader.swift`** performs **resumable** GGUF downloads with HTTP
  Range requests from Hugging Face `resolve` endpoints, writing directly to
  `<ggufDir>/<file>.part`. It does not shell out to scripts or `curl`. The
  downloadable targets (`q2-imatrix`, `q2-q4-imatrix`, `q4-imatrix`, `mtp`,
  `pro-q2-imatrix`) are defined in `ModelDownloader.targets`, all from the
  `antirez/deepseek-v4-gguf` repo. An HF token is resolved as: explicit >
  `HF_TOKEN` env > `~/.cache/huggingface/token`. See the root
  [Configuration Reference](../../../README.md#configuration-reference) for
  the download-related settings.

When `ModelTarget.sha256` is configured, the downloader verifies the downloaded
file against that known SHA-256 digest. Otherwise it reports the calculated
digest. Content verification is the robust defense here because it survives CDN
key rotation; this code does not rely on TLS pinning.
