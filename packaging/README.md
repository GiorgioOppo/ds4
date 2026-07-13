# Packaging

Files and scripts for assembling and signing the macOS `.app` bundle.

See [`docs/GUIDA-SVILUPPO.md`](../docs/GUIDA-SVILUPPO.md) for the complete
release workflow and [`docs/CRITTOGRAFIA.md`](../docs/CRITTOGRAFIA.md) for the
technical encryption/export-compliance inventory.

- **`make_app.sh`** builds `build/DwarfStar.app` from the SwiftPM release binary.
  It copies the executable, applies `Info.plist`, and performs ad-hoc signing
  (identity overridable via `DS4_SIGN_IDENTITY`). The Metal kernels are embedded
  in the binary. The `metal/` directory copied into `Resources/` is only a
  diagnostic/source snapshot: the production app never reads or compiles that
  copy at runtime. The `make app` bundle is deliberately ad-hoc signed
  **without App Sandbox** (Powerbox file access does not work with ad-hoc +
  sandbox). For distribution, use the signed Xcode workflow, then Developer ID
  signing and notarization as appropriate. Invoked by `make app`.
- **`Info.plist`** contains bundle metadata. `make_app.sh` resets the essential
  fields with PlistBuddy during packaging.
- **`DwarfStar.entitlements`** declares the sandbox capabilities used by the app:
  app sandbox, user-selected read/write files, app-scoped bookmarks, and network
  client/server access. The entitlements file is referenced by `project.yml`
  (`CODE_SIGN_ENTITLEMENTS`) and therefore applies only to the corresponding
  signed Xcode build, not to the ad-hoc `make app` bundle.
