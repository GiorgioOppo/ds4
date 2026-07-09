# DwarfStar/Project

- **`ProjectView.swift`** manages the project library. It imports folders through
  sandbox bookmarks, indexes them in `ProjectCache`, and selects the active
  project used by agent `project_*` and `file_*` tools. Importing a project does
  not add its files to chat memory; content enters the model only when a tool or
  attachment provides it.

Two kinds of library entries coexist:

- **User-imported folders** — picked with `NSOpenPanel`, persisted as
  security-scoped bookmarks; removing one only forgets it (the folder on disk
  is untouched).
- **GitHub clones** — created by the chat `github_clone` tool under
  `Application Support/DwarfStar/github-projects`, tracked by plain path (the
  app container needs no bookmark). `ProjectLibrary.syncClonedRepos()` scans
  that folder and keeps the list in sync (it runs when the Project tab or a
  project menu appears), so a repo cloned in chat shows up automatically and is
  marked active. Removing a clone DELETES the copy from disk — otherwise the
  next sync would re-list it; re-clone it anytime with `github_clone`.
