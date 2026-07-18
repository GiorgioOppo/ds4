# Model Management Views

- `ModelPicker.swift` uses `NSOpenPanel` and security-scoped bookmarks for
  external GGUFs. Before accepting them it calls the Engine inspection and
  `BackendSelector`, so unknown profiles, GLM 5.2, MTP, shards and Qwen are never saved as a valid local runtime.
- `DownloadView.swift` renders `ModelCatalogRegistry` exclusively: artifact
  type, runtime availability, local state, disk space, phase and progress. It
  offers Download/Resume, Cancel/Retry and Select for the runnable Flash and
  Pro Q2; the three GLM 5.2 variants show download and resume but no
  `Select`.

The sheet is reachable both from the Chat preload and from Settings. It does
not close while a download is active; selecting a managed file neutralizes an
old external bookmark and persists the path in `AppSettings`.
