**English** | [Italiano](README.it.md)

# Projects

`ProjectCache` indexes an imported project separately from the chat memory.
The model explores only the requested parts through the `project_*` and
`file_*` tools.

## Files

- `ProjectCache.swift`: thread-safe singleton, limits, index state and
  central component-by-component path validation.
- `+Indexing`: import, filters, traversal and reload.
- `+Queries`: listing, tree, search and bounded reads.
- `+Editing`: write/edit with a re-read of the current content.
- `+Files`: raw access, line ranges and confined operations.

The security invariants are in
[`SICUREZZA-PERCORSI.md`](SICUREZZA-PERCORSI.md).

## Flow and dependencies

Import records textual relative paths within count and size limits. Contents
are loaded lazily under an LRU-like budget; a search over cold files must not
needlessly evict the cache. The built-ins in
[`Tools/Builtins/Projects`](../Tools/Builtins/Projects/README.md) are the
main consumer.

## Extension

Always bound output and memory, preserve thread safety and route every tool
I/O through `confinedProjectURL`, which rejects traversal and symlinks in any
existing component. Revalidate right before the I/O; destructive operations
require an explicit tool contract and must not operate on directories.
