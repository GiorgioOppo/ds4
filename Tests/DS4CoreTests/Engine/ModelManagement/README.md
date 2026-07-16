# Model Management Engine Tests

- `HFTokenStoreTests.swift` checks token storage behavior without exposing a
  real credential.
- `ModelDownloaderTests.swift` covers the typed Flash/PRO catalog, runtime
  selectability, safe paths, local missing/empty/present state, skip-existing
  outcome, free-space calculations, URL construction, readable errors and
  checksums.

Catalog tests assert that the three complete Flash entries and single-file Pro
Q2 are selectable, Pro Q4 stays download-only with its two-shard package
boundary, and MTP is absent from the main-model entries. Existing-file
tests use non-empty regular fixtures so no network request is needed.

Never read the developer's actual Keychain token or contact Hugging Face from a
unit test. Use isolated services, temporary destinations, and fixed payloads.
