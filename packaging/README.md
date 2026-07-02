# Packaging

Files and scripts for assembling and signing the macOS `.app` bundle.

- **`make_app.sh`** builds `build/DwarfStar.app` from the SwiftPM release binary.
  It copies the executable, applies `Info.plist`, includes the `metal/` kernels
  required at runtime, and performs ad-hoc signing. For distribution, re-sign
  with Developer ID and notarize the bundle. Invoked by `make app`.
- **`Info.plist`** contains bundle metadata. `make_app.sh` resets the essential
  fields with PlistBuddy during packaging.
- **`DwarfStar.entitlements`** declares the sandbox capabilities used by the app:
  user-selected read/write files, app-scoped bookmarks, and network client/server
  access. The entitlements file is referenced by `project.yml`.
