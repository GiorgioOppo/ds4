# Encryption and Export Compliance — DwarfStar

This document records how DwarfStar uses cryptography, both for Apple export
compliance (App Store / TestFlight) and for operational security clarity.
It is a technical inventory, not legal advice. Export rules and App Store
questions can change; re-run the App Store Connect questionnaire for every
material change to networking, storage, dependencies, or cryptographic code.

Last reviewed against Apple's public guidance and the U.S. BIS guidance:
2026-07-13.

## Compliance Summary

The app is marked as:

```text
ITSAppUsesNonExemptEncryption = NO
```

In this project the value is set in `project.yml` as
`INFOPLIST_KEY_ITSAppUsesNonExemptEncryption: NO`, and is propagated into the
generated `Info.plist`.

The current project inventory supports `NO`: DwarfStar uses operating-system
HTTPS/TLS and Keychain services plus hashing functions, and does not implement
or embed a proprietary encryption algorithm. Apple documents `NO` for apps that
use no encryption or only forms exempt from App Store export-document upload.

This key does **not** by itself decide every U.S. export obligation. Apple's
guidance explicitly notes that some exempt encryption may still require a BIS
year-end self-classification report, while BIS ties that report to the precise
License Exception ENC classification. Therefore this document does not claim
that a report or CCATS can never be required; the publisher must confirm the
classification and distribution scenario.

## Cryptography by Component

| Component | Cryptography Used | Technical assessment |
|---|---|---|
| **Model downloads** (`ModelDownloader` -> `huggingface.co`) | **HTTPS/TLS** through Foundation `URLSession`; catalog-pinned **SHA-256** (`CryptoKit.SHA256`) for every new main-model download | OS-provided TLS plus hashing; consistent with the current `NO` inventory. |
| **Disk KV cache** (`KVCFile`, `DiskKVStore`) | **SHA-1** (`CryptoKit.Insecure.SHA1`) to name checkpoint files | Hashing only; it does not provide confidentiality. |
| **Distributed file transfer** (`DistFiles`, `DistWorker+Files`) | **SHA-256** for file identity and resumable chained checkpoints | Integrity hashing only; the transport itself is not encrypted. |
| **Hugging Face token storage** (`HFTokenStore`) | macOS **Keychain** through the Security framework | OS-provided secure storage; no custom cryptography in DwarfStar. |
| **Local HTTP server** (`LocalServer`) | None; plain HTTP | No encryption or confidentiality. |
| **Distributed inference transport** (`DistTransport`) | None; plain TCP over LAN | No encryption or confidentiality. |
| **Inference engine** (`DS4Core`, `DS4Metal`, `DS4Engine`) | None | Not applicable. |

Important notes:

- **No custom encryption.** `CryptoKit` is used only for hashes: SHA-1 for
  KV-cache names and SHA-256 for model and distributed-file integrity. The
  Security framework is used for the operating-system Keychain.
- **Hashing is not encryption** for export-compliance purposes. It does not hide
  data; it identifies or verifies data.
- **Why content SHA-256 instead of TLS pinning.** Hugging Face `resolve/main/...`
  URLs redirect to an LFS CDN whose public keys are outside this app's control and
  may rotate. ATS public-key pinning would be fragile. Hashing the final content
  protects against corruption or tampering regardless of CDN key rotation.
- **What is verified.** A newly transferred catalog artifact is checked against
  the response byte count and its SHA-256 fixed in `DeepSeekV4ModelCatalog`
  before the `.part` file is renamed to `.gguf`. Resumed transfers are hashed
  with a bounded-memory pass over the complete file, not only the appended
  suffix.
- **Existing-file policy.** A final regular, non-empty file with the exact
  catalog filename is treated as user-owned installed content and is not read
  again merely to hash hundreds of gigabytes. This is an explicit performance
  policy, not a cryptographic assertion about that pre-existing file. Remove or
  rename it to force a fresh, verified acquisition if its provenance is
  uncertain.
- **Credential redirects.** The optional Hugging Face bearer token is sent to
  `huggingface.co`; when the request redirects to an LFS/CDN host, the
  `Authorization` header is removed. The CDN uses its signed redirect URL.
- **TLS is entirely OS-provided.** DwarfStar does not implement TLS; it uses
  Apple networking APIs.

## Security Warning

The local server and distributed transport are intentionally simple and run in
plain text:

- The HTTP server is intended for `127.0.0.1`. If you bind it to `0.0.0.0` or a
  LAN address, prompts and generated text are not encrypted.
- Distributed inference exchanges hidden states and token data between Macs in
  clear text on the local network.

Use these features only on trusted networks. If you expose the server beyond
loopback, place it behind TLS, for example through Caddy, Nginx, WireGuard, or an
SSH tunnel.

## App Store Connect Answer

With the current technical inventory, `ITSAppUsesNonExemptEncryption = NO` is
consistent with Apple's description and streamlines the submission questions.
The Account Holder or App Manager must still answer App Store Connect using the
actual build and intended countries of distribution. If a future version adds
non-exempt encryption, the declaration must be changed and the requested
documentation supplied.

## Official References

- [Apple: `ITSAppUsesNonExemptEncryption`](https://developer.apple.com/documentation/bundleresources/information-property-list/itsappusesnonexemptencryption)
- [Apple: Overview of export compliance](https://developer.apple.com/help/app-store-connect/manage-app-information/overview-of-export-compliance)
- [Apple: Complying with Encryption Export Regulations](https://developer.apple.com/documentation/security/complying-with-encryption-export-regulations)
- [BIS: Annual self-classification](https://www.bis.gov/learn-support/encryption-controls/annual-self-classification)
