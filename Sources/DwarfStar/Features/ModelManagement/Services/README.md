# Model Management Services

`DownloadRunner.swift` is the main-actor adapter around
`DS4Engine.ModelDownloader`. It derives the entries from the Engine catalog,
looks for each artifact in the known local directories, and downloads to
Application Support only the missing ones. For multi-file packages it runs the
artifacts in sequence and considers them installed only when all are present.

The runner iterates `ModelCatalogRegistry`, so it can surface different
repositories. A final file is reused without touching the network; for targets
with a pinned exact size the byte count must match as well. Completed GLM
entries stay installed but are not passed to model selection.

Very frequent callbacks are coalesced through an `AsyncStream` bounded to
about 8 UI updates per second. The continuation is closed with `defer` even on
error/cancellation, avoiding a stuck state. Networking, SHA-256, HTTP Range,
disk preflight and token storage stay in `DS4Engine`; this layer does not log
credentials.

The `AsyncStream` buffer keeps only the four most recent events. Together with
the serial delegate queue and the Engine's `F_NOCACHE` I/O, neither the GUI
nor the page cache grows in proportion to the gigabytes already downloaded.

The progress bar assigns 90% to the transfer and 10% to the SHA verification:
it stays monotonic when a resume finishes the download and the verification
phase starts counting bytes from zero again. Every snapshot carries the
current phase, so coalescing cannot hide "Verifying integrity".

`active` holds a struct inside an `@Observable` property: the consumer never
modifies one of its fields while simultaneously re-reading the property. Every
event is applied to a local copy and published with a single assignment,
avoiding overlapping `_modify` accesses and reducing SwiftUI invalidations.
