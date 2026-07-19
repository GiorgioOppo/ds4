**English** | [Italiano](README.it.md)

# Builtins/Files

Exposes raw access and targeted edits to files under the imported root.

## Tools

- `file_read` and `file_lines`: content or line ranges.
- `file_write` and `file_add`: controlled creation/writing.
- `file_modify`: targeted replacement.
- `file_delete`: deletion of a single file, never directories.

## Flow and dependencies

The specs delegate to [`ProjectCache`](../../../Projects/README.md), which
normalizes and re-checks paths. They are all `projectScoped` and read outputs
are bounded.

## Extension

Apply the invariants in
[`Projects/SICUREZZA-PERCORSI.md`](../../../Projects/SICUREZZA-PERCORSI.md),
avoid destructive globs and require unambiguous search text for edits. A new
directory operation needs an explicit review of the authorization model.
