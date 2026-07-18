# DwarfStar/Features/ModelManagement

Selection and native download of the GGUFs supported by the GUI.

- **`Views/ModelPicker.swift`** provides sandbox-friendly selection through
  `NSOpenPanel` plus security-scoped bookmarks, so the same file can be reopened
  across launches.
- **`Models/ModelCatalog.swift`** identifies the three DeepSeek V4 Flash and the
  single-file Pro Q2 that the current runtime can load; **Browse** remains the
  explicit path for custom GGUFs and validates them before storing them.
- **`Views/DownloadView.swift` / `Services/DownloadRunner.swift`** provide the UI and driver for
  the native downloader (`DS4Engine.ModelDownloader`). The GUI derives all rows
  from `ModelCatalogRegistry`: it does not maintain a second hardcoded list.
  It shows Installed/Partial/Not downloaded state, disk space, phase, progress,
  cancellation and resume. The runner passes the Keychain token configured in
  Settings → Hugging Face without ever showing its full value.

Catalog records and scans live in `Models/`, UI adapters in `Services/`, and
selection/progress rendering in `Views/`. Remote protocol, integrity, resume,
and credential logic stays in `DS4Engine`; never log or persist a plaintext
token in this feature.

New files end up in `Application Support/DwarfStar/models`, not in the
bundle's read-only Resources. Before a download they are also looked up in
the development directories and next to the selected model: a cataloged file
that is already present and non-empty is reused without network or copy,
provided any exact size also matches. A `.part` in the managed folder is kept
after Cancel and resumed on the next attempt.

## Support boundary

The three Flash quantizations and the single-file Pro IQ2 are downloadable
and selectable. The two-shard Pro Q4 package is visible and downloadable, but
keeps the download-only badge and does not automatically become the active
model. The three GLM 5.2 variants are visible and downloadable from their
dedicated HF repository, but remain download-only. MTP and Qwen do not appear
in the main catalog.
