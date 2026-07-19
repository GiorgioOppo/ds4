**English** | [Italiano](README.it.md)

# Project Engine Tests

`ProjectCacheTests.swift` covers indexing, queries, edits, file boundaries, and
the refusal of unsafe symlinks.

Fixtures must live under a temporary project root. Every path-handling change
needs traversal, final-link and linked-parent tests, plus a positive case for
creating genuinely missing nested directories. Tests may use a separate
temporary directory as an escape target, but must assert that its contents were
not read or modified.
