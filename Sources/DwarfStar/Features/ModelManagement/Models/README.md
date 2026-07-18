# Model Management Models

`ModelCatalog.swift` defines the lightweight `DiscoveredModel` record and the
scanner used by the loading screen. The remote catalog is not duplicated here:
it comes from `DS4Engine.ModelCatalogRegistry`.

These are lightweight app catalog records, not GGUF parser types. Keep GGUF
metadata parsing in `DS4Core` and remote-download policy in `DS4Engine`.

The automatic scanner admits only the `primaryArtifact` of the entries Engine
declares selectable, including Flash and the single Pro Q2; it does not present
MTP, shards, the Pro Q4 package or future architectures as load-ready models.
**Browse** remains available for custom quantizations, but `ModelPicker`
performs inspection and backend selection before saving the bookmark.

The GLM 5.2 entries exist in the remote registry for download, but being
`downloadOnly` they do not enter `selectableEntries` and therefore do not
appear in this scan.
