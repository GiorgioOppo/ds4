# packaging

Assemblaggio e firma della `.app`.

- **`make_app.sh`** — costruisce `build/DwarfStar.app` dalla release SwiftPM: copia l'eseguibile, l'`Info.plist`, i kernel `metal/` (richiesti a runtime), firma ad-hoc. Per la distribuzione: ri-firmare con Developer ID e notarizzare. Invocato da `make app`.
- **`Info.plist`** — metadati del bundle (il `make_app.sh` ne reimposta gli essenziali via PlistBuddy).
- **`DwarfStar.entitlements`** — entitlement del sandbox: file user-selected (read-write) + bookmark, network client/server. Referenziato da `project.yml`.
