**English** | [Italiano](README.it.md)

# Builtins/Projects

Lets the model explore and modify the project index without
putting the entire repository into the prompt.

## Tools

- `project_inspect`: preferred read-only batch surface. One bounded request can
  combine Git change scope, tree/list orientation, path/content searches, and
  multiple source ranges (12 operations, 48,000-character hard cap).
- `project_tree`, `project_list`, `project_find`: orientation and paths.
- `project_read`, `project_search`: bounded reads and search optionally
  restricted to a subfolder. Retained as compatibility primitives; default
  project agents use `project_inspect` to avoid one inference round per file.
- `project_write`, `project_edit`: writing and exact replacement.
- `project_reload`: index rebuild after external changes.

## Flow and dependencies

All tools use [`ProjectCache`](../../../Projects/README.md) and are
`projectScoped`. Git re-indexes automatically after operations that mutate
the working tree; external editors and scripts require `project_reload`.

## Extension

Prefer aggregated, bounded responses to reduce inference round-trips without
reading a repository wholesale. Separate name search from content search and
do not load cold files into the cache when a streaming scan is sufficient.
