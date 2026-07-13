# Project Engine Tests

`ProjectCacheTests.swift` covers indexing, queries, edits, file boundaries, and
the refusal of unsafe symlinks.

Fixtures must live under a temporary project root. Every path-handling change
needs traversal and symlink escape tests; never let a test read or modify files
outside its fixture directory.

