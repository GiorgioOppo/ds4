# DwarfStar/Features/Chat

View model e UI della chat.

- **`ViewModels/ChatStore.swift`** è il view model `@MainActor @Observable`.
  Possiede `InferenceService`, ne rispecchia lo stream di eventi, gestisce il
  loop dei tool incluso il reinstradamento di `subagent_run` verso l'engine
  (le chiamate malformate — JSON errato, domanda mancante, id agente
  sconosciuto, tool non concedibili — vengono rifiutate con un errore
  esplicativo restituito al modello e mostrato nella trascrizione; le
  esecuzioni valide appaiono subito come una card in corso che mostra in
  streaming i passi interni del sub-agent e li conserva in caso di
  errore/stop), gestisce gli allegati di testo, mostra avvisi di contesto
  quasi pieno, applica le impostazioni di memoria come expert cache, expert
  pread, expert bundle, dense streaming, `mlock`, cache densa Q4, disk KV e
  ring raw-KV, e gestisce **più chat persistenti**. La chat attiva vive in
  `messages`; le chat inattive sono memorizzate su disco. Possiede inoltre le
  manopole di sampling e prefill (`DS4Temperature`, `DS4RepPenalty`,
  `DS4PrefillUnion`, `DS4PrefillChunk`); tutte le chiavi e i valori
  predefiniti sono elencati nel
  [Riferimento di Configurazione](../../../../README.md#configuration-reference)
  alla radice.
- **`Persistence/ChatSession.swift`** definisce il modello di chat `Codable`,
  inclusi i metadati e le voci di trascrizione come `StoredMessage`.
  `ChatSessionStore` persiste un file JSON per chat sotto
  `Application Support/DwarfStar/chats`.
- **`Views/ChatView.swift`** possiede il layout della trascrizione e del
  composer. File di vista focalizzati renderizzano Markdown, messaggi, tool,
  sheet e allegati.
- **`Views/ChatListView.swift`** è il popover delle chat salvate per cambiare,
  rinominare, eliminare e creare chat.
- **`Views/ChatTabView.swift`** commuta la tab Chat in base a modalità e fase
  dell'engine: la `ChatView` locale quando è pronta, un placeholder di
  caricamento/onboarding altrimenti, e `CoordinatorChatView` in modalità
  Distribuita. L'header con i menu progetto/agente/tool vive in `ChatView`.
- **`Views/ContentView.swift`** contiene `ModelLoadView`, il form di
  configurazione pre-caricamento (selezione del modello, impostazioni di
  memoria, contesto, agente, prompt di sistema). Nessuna delle due viste è
  referenziata dalla UI corrente — `RootView`/`ChatTabView` e la tab Settings
  hanno assorbito questo flusso, quindi il file è un candidato alla rimozione.

## Ciclo di vita della selezione del modello

`ChatStore+ModelLifecycle` scansiona la directory di download gestita dall'app
e le cartelle dei modelli di sviluppo alla ricerca di voci di catalogo
dichiarate selezionabili da `DS4Engine`: le tre varianti Flash e il Pro Q2 a
file singolo. Il Pro Q4 split, gli artefatti GLM 5.2 solo download, MTP e i
GGUF sconosciuti non appaiono come candidati al caricamento con un clic. I
file manuali restano raggiungibili tramite il picker **Browse** validato nelle
Impostazioni.

Selezionare un file di catalogo eseguibile scaricato in Application Support
persiste il suo semplice percorso gestito e rimuove un eventuale bookmark
esterno più vecchio. Al riavvio, `restoreModelBookmark` preserva quella scelta
gestita; un bookmark obsoleto non deve sostituirla. Il progresso di download
del modello e la policy di rete non appartengono a `ChatStore`: restano nella
feature ModelManagement e in `DS4Engine`.

## Riaprire una chat

Dopo il riavvio dell'app o dopo il cambio di chat, l'engine non possiede più
la KV cache di quella conversazione. Al primo nuovo invio, la cronologia
visibile viene renderizzata di nuovo tramite `InferenceService.sendWithHistory`.
Il disk KV può ripristinare il prefisso, poi i turni successivi tornano
all'esecuzione incrementale append-only.

## Engine condiviso

Quando il modello è caricato, `ChatStore.sharedEngine` espone l'unico
`InferenceService` locale usato da Chat, Server e Benchmark locale. L'avvio
del Server fallisce finché questo engine non è pronto. Il Benchmark è
consentito solo mentre la chat è inattiva, perché le esecuzioni di benchmark
riscrivono lo stato KV.

## Proprietà dei file

I DTO delle conversazioni sono in `Models`, la conversione di storage e i file
di sessione in `Persistence`, lo stato UI mutabile in `ViewModels` e tutto il
rendering in `Views`. I tipi condivisi dell'engine restano in `DS4Engine` o
`DS4Core` invece di spostarsi nella feature GUI.

Vedi [`FLOW.md`](FLOW.md) per il ciclo di vita end-to-end di messaggi, tool,
sessioni ed engine condiviso. Ogni directory figlia ha le proprie regole di
proprietà e modifica in un README locale.
