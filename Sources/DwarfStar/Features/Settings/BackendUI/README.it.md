[English](README.md) | **Italiano**

# Settings/BackendUI

Livello UI per-backend: una classe base "astratta" più un'implementazione
concreta per backend, scelta dal modello caricato (o ispezionato).

## File principali

- [`BackendSettingsUI.swift`](BackendSettingsUI.swift): classe base e
  factory. Implementa le parti COMUNI — lo scaffold del benchmark
  (rapido/completo/stop/progresso/referto), le righe Disk KV condivise e la
  sezione informativa di fallback — e dichiara i punti di override (nome
  del backend, sezione memoria, extra del benchmark, pannello tuning).
  Istanziabile di proposito: la base è il fallback neutro per
  Qwen/sconosciuto/nessun modello.
- [`DeepSeekSettingsUI.swift`](DeepSeekSettingsUI.swift): implementazione
  DeepSeek V4 — sezione memoria completa (cache esperti, streaming denso,
  Q4 lossy, bundle, Disk KV con budget in token, Raw-KV ring), extra
  auto-tune e pannello tuning slot-cache + usage imatrix. Ospita anche il
  `BundleBuildButton` condiviso.
- [`GLM52SettingsUI.swift`](GLM52SettingsUI.swift): implementazione
  GLM 5.2 — stepper residenza/esperti/arena/streaming, toggle MetalIO e
  staging speculativo, sidecar Q4, Disk KV senza budget (checkpoint
  singolo per modello) e build del sidecar unificato.

## Flusso e dipendenze

`SettingsView` e `TuningView` chiamano `BackendSettingsUI.make(store:dist:)`
a ogni valutazione del body e renderizzano le sezioni restituite. Il
contenuto delle sezioni si lega allo stato di `ChatStore` tramite piccole
struct `@Bindable`, così l'osservazione SwiftUI continua a funzionare
dietro l'indirezione della classe. `store.runSettingsBenchmark` e
`store.buildExpertBundleNow` smistano già da soli sul backend vivo, quindi
gli scaffold comuni non hanno rami condizionali.

## Regole di modifica

I controlli specifici di un backend vivono nella sua sottoclasse, mai
nella sezione di un fratello né dietro la capability di un altro backend.
Un nuovo backend aggiunge una sottoclasse e un caso alla factory. Le
sezioni della base restano generiche: tutto ciò che nomina un motore, un
knob o un preset misurato sta nella sottoclasse.
