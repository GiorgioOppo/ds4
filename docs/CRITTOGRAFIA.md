# Encryption and Export Compliance — DwarfStar

This document records how DwarfStar uses cryptography, both for Apple export
compliance (App Store / TestFlight) and for operational security clarity.

## Compliance Summary

The app is marked as:

```text
ITSAppUsesNonExemptEncryption = NO
```

In this project the value is set in `project.yml` as
`INFOPLIST_KEY_ITSAppUsesNonExemptEncryption: NO`, and is propagated into the
generated `Info.plist`.

That means DwarfStar uses **only exempt encryption**: standard operating-system
HTTPS/TLS and hashing functions. The app does **not** implement or embed custom,
proprietary, or non-exempt encryption algorithms. In practical App Store terms,
this means no annual self-classification report and no CCATS are required for the
current feature set.

## Cryptography by Component

| Component | Cryptography Used | Exempt? |
|---|---|---|
| **Model downloads** (`ModelDownloader` -> `huggingface.co`) | **HTTPS/TLS** through Foundation `URLSession`; **SHA-256** (`CryptoKit.SHA256`) to verify downloaded GGUF integrity | Yes. TLS is OS-provided standard encryption; SHA-256 is hashing, not encryption. |
| **Disk KV cache** (`KVCFile`, `DiskKVStore`) | **SHA-1** (`CryptoKit.Insecure.SHA1`) to name checkpoint files | Yes. This is hashing, not encryption. |
| **Local HTTP server** (`LocalServer`) | None; plain HTTP | Yes. No encryption. |
| **Distributed inference transport** (`DistTransport`) | None; plain TCP over LAN | Yes. No encryption. |
| **Inference engine** (`DS4Core`, `DS4Metal`, `DS4Engine`) | None | Not applicable. |

Important notes:

- **No custom encryption.** The only crypto framework imported by the project is
  `CryptoKit`, and it is used only for hash functions: SHA-1 for KV-cache file
  names, matching the ported `ds4_kvstore.c` format, and SHA-256 for model-file
  integrity verification.
- **Hashing is not encryption** for export-compliance purposes. It does not hide
  data; it identifies or verifies data.
- **Why content SHA-256 instead of TLS pinning.** Hugging Face `resolve/main/...`
  URLs redirect to an LFS CDN whose public keys are outside this app's control and
  may rotate. ATS public-key pinning would be fragile. Hashing the final content
  protects against corruption or tampering regardless of CDN key rotation.
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

With `ITSAppUsesNonExemptEncryption = NO` in `Info.plist`, App Store Connect
should not require encryption documentation for every build. If future versions
add non-exempt encryption, such as custom end-to-end encryption of user data, the
declaration must be changed to `YES` and the required export documentation must
be supplied.
