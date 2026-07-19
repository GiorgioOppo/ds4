**English** | [Italiano](README.it.md)

# ModelManagement/Download

Implements native, resumable GGUF downloads, without `curl` or external
processes.

## Components

- `ModelDownloader.swift`: `acquire` API, local check, safe path, disk space
  preflight, atomic finalization and typed outcome.
- `HTTPRangeFileTransfer.swift`: cache-free HTTP session, chunked reception,
  validated resume and direct writing of the `.part` file.
- `ModelDownloadTypes.swift`: progress, phases and the
  `alreadyPresent`/`downloaded` outcome consumed by the GUI.
- `HFTokenStore.swift`: reads/writes the Hugging Face token in the Keychain,
  masked form and description of the active source.

## Flow

The downloader writes to `<ggufDir>/<name>.part`, resumes from the existing
offset and renames only once the stream has finished. `URLSessionDataDelegate`
delivers `Data` chunks: there is no Swift iteration over every byte of GGUFs
hundreds of GB in size. The ephemeral session uses no HTTP cache, cookies or
credential storage, and the delegate queue is serial: only one chunk is
processed at a time. Progress notifications are throttled so as not to
overload the GUI.

The descriptors used to write the `.part` and for SHA verification after a
resume are marked `F_NOCACHE`: the sequential bytes of the download must not
fill macOS's unified file cache and evict useful model weights or pages. The
verification reads at most 8 MiB at a time and drains the autorelease pool at
every chunk. The resume JSON sidecar is likewise accepted only up to 64 KiB.
The downloader's peak RAM therefore stays independent of the total GGUF size.

A `206` response is accepted only if `Content-Range` starts at the local size.
If the server answers `200` to a Range request, the `.part` is actually
truncated before restarting; `416` is considered complete only when the remote
size matches exactly. ETag/Last-Modified are kept in a small sidecar and sent
as `If-Range` on resume.

New catalog downloads have pinned SHA-256 hashes and are verified before the
rename. A regular, non-empty final file already present immediately yields
`alreadyPresent`: it is neither re-downloaded nor fully re-read on every open.
When the target has `expectedSizeBytes`, the byte count must match; this guard
makes it possible to reuse the hundreds-of-GB GLM GGUFs without a full hash on
every open and without accepting a truncated final file. An empty final file
is never considered a valid GGUF.

The downloader runs a disk space preflight, includes a filesystem margin and
accounts for the `.part` bytes already present. An actor gate prevents
concurrent acquisitions of the same path. The GUI chooses
`~/Library/Application Support/DwarfStar/models/` because the Resources of an
installed app are not writable.

Token resolution follows: explicit → `HF_TOKEN` →
`~/.cache/huggingface/token`. The Keychain is consulted by the GUI, not by the
generic `resolveToken` method, so CLI and tests do not trigger unexpected
prompts.

## Dependencies and security

Foundation handles URLSession/files, CryptoKit handles integrity and Security
the Keychain. Content verification is independent of CDN certificate rotation.
The bearer token is removed when Hugging Face redirects to a different CDN
host. Do not put tokens in URLs, logs or unprotected state.

## Extension

To add a target, use a stable ID and file name, a Hugging Face source with the
revision preferably pinned, exact size when available and an authoritative
SHA-256. `ModelDownloader` builds the URL from the target's `source`, so
different catalogs no longer share a global repository. Preserve cancellation,
resume and progress callbacks; avoid buffers proportional to the model size.

A downloadable target does not imply decoder compatibility. The main catalog
declares runtime support per entry: the three Flash and single-file PRO Q2
models are selectable, while the PRO Q4 package and the three GLM 5.2 remain
`downloadOnly`. MTP is a separate accessory and does not appear among the GUI
models.

Related settings are in the
[Configuration Reference](../../../../README.md#configuration-reference).
