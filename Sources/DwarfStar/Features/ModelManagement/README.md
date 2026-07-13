# DwarfStar/Features/ModelManagement

GGUF model selection, scanning, and downloads.

- **`Views/ModelPicker.swift`** provides sandbox-friendly selection through
  `NSOpenPanel` plus security-scoped bookmarks, so the same file can be reopened
  across launches.
- **`Models/ModelCatalog.swift`** scans configured folders for available `.gguf` files.
- **`Views/DownloadView.swift` / `Services/DownloadRunner.swift`** provide the UI and driver for
  the native downloader (`DS4Engine.ModelDownloader`), including progress phases
  and integrity verification. The runner passes the Keychain token saved in
  Settings → Hugging Face (`HFTokenStore`) as the explicit token; the sheet
  shows which token source, if any, the download will use.

Catalog records and scans live in `Models/`, UI adapters in `Services/`, and
selection/progress rendering in `Views/`. Remote protocol, integrity, resume,
and credential logic stays in `DS4Engine`; never log or persist a plaintext
token in this feature.

## Confine di supporto

Il catalogo include anche artefatti Pro e MTP che possono essere scaricati o
ispezionati, ma non sono consumati dal backend corrente. L'esecuzione locale e
distribuita accetta solo il profilo Flash; il componente MTP separato non viene
caricato e `DS4_SPEC_K` è un esperimento self-speculative indipendente. La
presenza di un target nella GUI non equivale quindi a supporto d'inferenza.
