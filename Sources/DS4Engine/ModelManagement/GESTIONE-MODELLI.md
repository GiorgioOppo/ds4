**English** | [Italiano](GESTIONE-MODELLI.it.md)

# Model lifecycle

## Download

The GUI reads `ModelCatalogRegistry`, picks a `ModelCatalogEntry` and downloads
its `ModelTarget`s with `ModelDownloader.acquire`. The file is written as a
`.part`; a subsequent request uses HTTP Range to resume the bytes already
present. The result explicitly distinguishes a fresh download from a regular,
non-empty final file that is already present.
If the target declares `expectedSizeBytes`, the final file is reused only when
the size matches as well; it is not fully re-read on every startup.

The GUI uses `~/Library/Application Support/DwarfStar/models/` as the writable
destination. Before opening the network it also looks for the exact filename in
the development model directories and in the directory of the active GGUF: if
it finds a regular, non-empty final file it reuses it in place, applying the
exact-size check when applicable. An empty final file is not considered valid.

Resume accepts `206` only with a consistent `Content-Range`. If the server
ignores Range and responds `200`, the `.part` is truncated and rewritten; `416`
counts as completion only when the remote size matches the local one.
Cancellation and transport errors preserve the `.part`; the disk-space
preflight, the remote validator and the per-path gate respectively prevent
saturation, appending to changed objects and a double writer.

Before the atomic rename, new downloads are verified against the SHA-256
digest pinned in the catalog. The three Flash files and the single PRO Q2 are
runnable and selectable; the PRO Q4 package and the three GLM 5.2 files remain
`downloadOnly`. MTP is a distinct accessory and is not an entry in the main
catalog.

| Entry | Artifacts | Download | Selection/execution |
|---|---:|---:|---:|
| Flash Q2 imatrix | 1 | yes | yes |
| Flash mixed Q2/Q4 imatrix | 1 | yes | yes |
| Flash Q4 imatrix | 1 | yes | yes |
| Pro Q2 imatrix | 1 | yes | yes |
| Pro Q4 split | 2 shards | yes | no |
| GLM 5.2 IQ2_XXS | 1 | yes | no |
| GLM 5.2 Q2_K | 1 | yes | no |
| GLM 5.2 Q4_K | 1 | yes | no |

The automatic scan offers the three Flash filenames and the selectable Pro Q2.
Completed GLM GGUFs stay excluded until the backend becomes `runnable`.
**Browse** remains available for external files, but
`InferenceService.inspectModel` and `BackendSelector` validate their
architecture and profile before changing the active model.

## Credentials

The Hugging Face token precedence is: explicit value, `HF_TOKEN`, standard
Hugging Face cache file. In the app, the explicit value comes from the
Keychain via `HFTokenStore`; it must not be shown in full or included in error
messages.

## Derived artifacts

`ExpertBundleTool.ensure` opens the GGUF via mmap, derives the experts'
geometry and quantization and calls `ExpertBundle.openOrBuild`. The sidecar and
the dense-Q4 cache are rebuildable: the GGUF always remains the primary
source.

## Local and distributed use

The local service resolves the sidecars from the configured directory. In
distributed mode the coordinator includes the active artifacts in the manifest
and the worker downloads only those not already verified. The assignment
configuration decides whether the worker must use them.

The downloader is more general than the local loader: the PRO Q4 package can
be acquired without being offered as the active model, as can the three GLM
5.2 files coming from a separate Hugging Face repository. No current path
loads the separate MTP component; the demo's self-speculative `DS4_SPEC_K`
does not use those weights.

See [`Download`](Download/README.md) and
[`Distributed/Files`](../Distributed/Files/README.md).
