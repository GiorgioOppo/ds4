# DwarfStar/Models

GGUF model selection, scanning, and downloads.

- **`ModelPicker.swift`** provides sandbox-friendly selection through
  `NSOpenPanel` plus security-scoped bookmarks, so the same file can be reopened
  across launches.
- **`ModelCatalog.swift`** scans configured folders for available `.gguf` files.
- **`DownloadView.swift` / `DownloadRunner.swift`** provide the UI and driver for
  the native downloader (`DS4Engine.ModelDownloader`), including progress phases
  and integrity verification.
