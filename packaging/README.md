# Packaging

Files and scripts for assembling and signing the macOS `.app` bundle.

- **`make_app.sh`** builds `build/DwarfStar.app` from the SwiftPM release binary.
  It copies the executable, applies `Info.plist`, and performs ad-hoc signing
  (identity overridable via `DS4_SIGN_IDENTITY`). The Metal kernels are embedded
  in the binary; the `metal/` sources are copied into `Resources/` only as an
  optional fallback. The ad-hoc app is deliberately signed WITHOUT the App
  Sandbox (Powerbox file access does not work with ad-hoc + sandbox). For
  distribution, re-sign with Developer ID and notarize the bundle. Invoked by
  `make app`.
- **`Info.plist`** contains bundle metadata. `make_app.sh` resets the essential
  fields with PlistBuddy during packaging.
- **`DwarfStar.entitlements`** declares the sandbox capabilities used by the app:
  app sandbox, user-selected read/write files, app-scoped bookmarks, and network
  client/server access. The entitlements file is referenced by `project.yml`
  (`CODE_SIGN_ENTITLEMENTS`) and therefore applies to the Xcode build, not to
  the ad-hoc `make app` bundle.
