**English** | [Italiano](README.it.md)

# Model Management Engine Tests

- `HFTokenStoreTests.swift` checks token storage behavior without exposing a
  real credential.
- `ModelDownloaderTests.swift` covers the typed multi-family catalog, runtime
  selectability, per-target Hugging Face source, safe paths, local
  missing/empty/present state, skip-existing outcome, pinned-size rejection,
  free-space calculations, URL construction, readable errors and checksums.

Catalog tests assert that the three complete Flash entries and single-file Pro
Q2 are selectable, Pro Q4 stays download-only with its two-shard package
boundary, and MTP is absent from the main-model entries. Existing-file
tests use non-empty regular fixtures so no network request is needed.
The three GLM 5.2 entries are checked against their pinned filenames, byte
counts, SHA-256 digests and revision, and must remain non-selectable.

Never read the developer's actual Keychain token or contact Hugging Face from a
unit test. Use isolated services, temporary destinations, and fixed payloads.
