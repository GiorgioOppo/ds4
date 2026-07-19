[English](README.md) | **Italiano**

# DwarfStar/App

Shell dell'applicazione e stato condiviso.

- **`DwarfStarApp.swift`** è il punto di ingresso `@main`. Installa la cattura
  dello stderr di `EngineLog`, crea `AppSettings`, `ChatStore` e `MCPStore`, e
  apre la finestra principale.
- **`RootView.swift`** possiede la sidebar `NavigationSplitView` e istanzia i
  controller condivisi per Distributed, Server, Bench e Diagnostics (il
  `ChatStore` è creato in `DwarfStarApp` e passato dall'esterno). I controller
  Server e Bench ricevono il `ChatStore`/engine condiviso invece di caricare
  engine completi indipendenti.
- **`AppSettings.swift`** memorizza le impostazioni persistenti come il
  percorso del modello (`DS4ModelPath`), la lunghezza del contesto
  (`DS4ContextSize`) e la modalità di esecuzione locale/distribuita
  (`DS4EngineMode`). Le Impostazioni possiedono questi valori; gli altri
  controller li leggono attraverso lo stato condiviso dell'app. Consulta la
  [Guida di riferimento alla configurazione](../../../README.it.md#riferimento-di-configurazione)
  alla radice per tutte le chiavi e i valori predefiniti.
- **`AppEnvironment.swift`** risolve i percorsi per le esecuzioni in sviluppo
  rispetto all'app impacchettata, calcola i preset hardware basati sulla RAM
  (contesto predefinito 4096 sotto i 24 GB, 8192 sotto gli 80 GB, 32768 sopra),
  espone gli helper di memoria e possiede la directory scrivibile di download
  dei modelli `~/Library/Application Support/DwarfStar/models/`.

I modelli del catalogo dentro quella directory sono gestiti dall'app e non
necessitano di un bookmark security-scoped. Quando ne viene selezionato uno,
il vecchio bookmark del modello esterno non deve prevalere su di esso al lancio
successivo. Le identità dei modelli remoti e la politica di supporto restano in
`DS4Engine.ModelCatalogRegistry` e nei suoi cataloghi di famiglia, non in
`AppEnvironment`.

## Regole di modifica

- Crea qui lo stato osservabile a livello di processo e iniettalo nelle
  feature.
- Mantieni lo stato specifico di una feature nel controller o nel view model
  di quella feature.
- Preserva il singolo engine locale condiviso e la proprietà main-actor dello
  stato della UI.
- Mantieni i download scrivibili in Application Support; le Resources del
  bundle sono di sola lettura dopo l'installazione.
- Aggiungi nuove destinazioni della sidebar tramite `AppSection` e `RootView`,
  con la loro implementazione sotto `Features/`.
