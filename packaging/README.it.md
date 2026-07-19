[English](README.md) | **Italiano**

# Packaging

File e script per assemblare e firmare il bundle macOS `.app`.

Vedi [`docs/GUIDA-SVILUPPO.md`](../docs/GUIDA-SVILUPPO.it.md) per il flusso di
release completo e [`docs/CRITTOGRAFIA.md`](../docs/CRITTOGRAFIA.it.md) per
l'inventario tecnico di crittografia/conformità all'esportazione.

- **`make_app.sh`** costruisce `build/DwarfStar.app` dal binario release di
  SwiftPM. Copia l'eseguibile, applica `Info.plist` ed esegue la firma ad-hoc
  (identità sovrascrivibile tramite `DS4_SIGN_IDENTITY`). I kernel Metal sono
  incorporati nel binario. La directory `metal/` copiata in `Resources/` è solo
  uno snapshot diagnostico/dei sorgenti: l'app di produzione non legge né
  compila mai quella copia a runtime. Il bundle di `make app` è deliberatamente
  firmato ad-hoc **senza App Sandbox** (l'accesso ai file tramite Powerbox non
  funziona con ad-hoc + sandbox). Per la distribuzione usa il flusso Xcode
  firmato, poi la firma Developer ID e la notarizzazione secondo necessità.
  Invocato da `make app`.
- **`Info.plist`** contiene i metadati del bundle. `make_app.sh` reimposta i
  campi essenziali con PlistBuddy durante il packaging.
- **`DwarfStar.entitlements`** dichiara le capacità sandbox usate dall'app: app
  sandbox, lettura/scrittura di file selezionati dall'utente, bookmark con
  scope dell'app e accesso di rete client/server. Il file di entitlements è
  referenziato da `project.yml` (`CODE_SIGN_ENTITLEMENTS`) e si applica quindi
  solo alla corrispondente build Xcode firmata, non al bundle ad-hoc di
  `make app`.
