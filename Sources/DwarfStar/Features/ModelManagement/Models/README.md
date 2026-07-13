# Model Management Models

`ModelCatalog.swift` defines discovered local models, downloadable targets, and
filesystem scanning helpers used by the picker and download UI.

These are lightweight app catalog records, not GGUF parser types. Keep GGUF
metadata parsing in `DS4Core` and remote-download policy in `DS4Engine`.

Catalog availability is intentionally broader than runtime compatibility. Pro
and MTP records are download-only with the current backend: local/distributed
execution accepts Flash, and no load path consumes the separate MTP component.
Do not describe a catalog row as runnable unless the corresponding loader and
shape validation are implemented and tested.
