# Model Management Services

`DownloadRunner.swift` is the main-actor adapter around
`DS4Engine.ModelDownloader`. It maps downloader phases, progress, completion,
and errors into observable UI state and supplies the explicitly selected
Hugging Face token.

Networking, checksums, resume behavior, and token storage belong in
`DS4Engine`. This layer must remain safe to cancel and must not expose secret
tokens in logs or observable status strings.

