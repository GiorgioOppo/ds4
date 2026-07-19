**English** | [Italiano](README.it.md)

# Distributed/Files

Manages the identity and local retention of large distributed artifacts.

## Components

- `DistFileHash`: full SHA-256, checkpoint chain and persistent cache
  validated with size and modification date.
- `DistFileStore`: managed directory, name sanitization, manifest and fast
  verification of files already received.

## Flow and dependencies

The coordinator computes the hash and chain once; the worker uses the manifest
to avoid transfers already verified. The corresponding messages are in
[`Protocol/Files`](../Protocol/Files/README.md), reception in
[`Worker/Files`](../Worker/Files/README.md). Depends on Foundation and
CryptoKit.

## Extension

Do not trust names sent over the network, do not promote `.part` files
without a final hash and invalidate the cache when size or mtime change. A new
sidecar requires a new `Kind` in the protocol and an explicit resolution
policy.
