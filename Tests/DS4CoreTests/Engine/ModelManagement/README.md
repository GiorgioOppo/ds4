# Model Management Engine Tests

- `HFTokenStoreTests.swift` checks token storage behavior without exposing a
  real credential.
- `ModelDownloaderTests.swift` covers target mapping, checksums, download
  planning, and validation helpers.

Never read the developer's actual Keychain token or contact Hugging Face from a
unit test. Use isolated services, temporary destinations, and fixed payloads.

