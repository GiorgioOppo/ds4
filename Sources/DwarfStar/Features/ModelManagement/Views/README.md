# Model Management Views

- `ModelPicker.swift` uses `NSOpenPanel` and security-scoped bookmarks for GGUF
  files outside the app sandbox.
- `DownloadView.swift` renders target selection and `DownloadRunner` progress.

Keep sandbox access balanced: start and stop security-scoped access around the
operation that needs it. Views should delegate downloads to the service and
catalog scans to the model layer.

