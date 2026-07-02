# DwarfStar/Project

- **`ProjectView.swift`** manages the project library. It imports folders through
  sandbox bookmarks, indexes them in `ProjectCache`, and selects the active
  project used by agent `project_*` and `file_*` tools. Importing a project does
  not add its files to chat memory; content enters the model only when a tool or
  attachment provides it.
