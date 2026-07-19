**English** | [Italiano](README.it.md)

# Project Views

`ProjectView.swift` contains the project-library UI and the app-facing
`ProjectLibrary` bookmark adapter. Imported folders remain in place; managed
GitHub clones live under Application Support and may be deleted when removed.

Indexing and safe file operations belong to `DS4Engine.ProjectCache`. Preserve
the distinction between forgetting an imported bookmark and deleting an
app-managed clone, and never follow untrusted symlinks into files outside the
selected project root.

