[English](README.md) | **Italiano**

# DwarfStar/Features/Settings

- **`Views/SettingsView.swift`** effettua il rendering delle Impostazioni
  globali. Il percorso del modello e la lunghezza del contesto vengono
  configurati una sola volta tramite `AppSettings` ed ereditati ovunque: Chat,
  Server, Benchmark, Diagnostica e Worker. La sezione **Hugging Face**
  memorizza il token di download nel Keychain di macOS tramite
  `DS4Engine.HFTokenStore` (campo di sola scrittura, riga di stato oscurata; il
  downloader lo riceve esplicitamente, con precedenza sulla variabile
  d'ambiente `HF_TOKEN` e su `~/.cache/huggingface/token`).
  La sezione Modello apre inoltre **Scarica…**, il cui sheet appartiene a
  `Features/ModelManagement`: le tre varianti Flash del catalogo e la Pro Q2 a
  file singolo possono essere selezionate dopo il download, mentre la Pro Q4
  split resta visibilmente solo scaricabile. Le tre quantizzazioni GLM 5.2 sono
  scaricabili dalla revisione bloccata del loro repository e diventano
  selezionabili quando `GLM52RuntimeGate.enabled` è attivo.
  Le sezioni benchmark e memoria arrivano da [`BackendUI/`](BackendUI/README.it.md):
  una classe base astratta più un'implementazione DeepSeek e una GLM, scelte
  dal modello selezionato, così ogni backend vede sempre i propri controlli.
- **`Views/MCPServersView.swift`** effettua il rendering del pannello MCP:
  `MCPStore` rende persistenti i server MCP configurati (UserDefaults,
  import/export JSON in formato `mcpServers`) e li invia a `MCPManager.shared`
  (in `DS4Engine/Tools/MCP/`), che possiede le connessioni attive; la vista
  mostra lo stato per server, gli strumenti esposti e uno sheet di aggiunta
  server (comando stdio o URL Streamable-HTTP).

Il pannello seleziona anche la modalità di esecuzione:

- **Locale** carica il singolo engine in-process usato da Chat, Server e dal
  Benchmark locale.
- **Distribuita** configura la rotta del coordinator e le opzioni di trasporto
  usate dal cluster di Worker.

La sezione Memoria controlla il profilo di runtime usato al successivo
caricamento del modello: slot della cache degli esperti, pool di esperti
mixed-quant con budget in byte, `pread` diretto degli esperti, sidecar dei
bundle di esperti, streaming dei pesi densi, `mlock` best-effort, cache Q4
delle proiezioni di attenzione, KV su disco con relativo budget e ring raw-KV.
La maggior parte dei valori predefiniti tiene conto della RAM; l'attuale
percorso per RAM ridotta preferisce lo streaming e i buffer caldi bloccati in
memoria al mantenere residente ogni peso denso. Il preset veloce abilita
`DS4MultiQuantCache` per impostazione predefinita dopo l'A/B esatto del
2026-07-16; il suo interruttore può ripristinare il bypass off-class legacy.
La chiave UserDefaults e il valore predefinito di ogni impostazione sono
documentati nella [Guida di riferimento alla configurazione](../../../../README.it.md#riferimento-di-configurazione)
alla radice.

L'ispezione del modello viene eseguita senza caricare Metal. I controlli
Benchmark e Memoria sono mostrati solo quando `ModelInfo`/`RuntimeModelDescriptor`
dichiara `deepSeekPerformanceTuning`; un modello Qwen riconosciuto non riceve
quindi mai impostazioni solo-DeepSeek per esperti, NSA, bundle o
riquantizzazione.

L'azione **Auto-tune record-holder** usa il livello decisionale puro in
`DS4Engine/Inference/Autotuning`. Misura una sola volta la radice calda
caricata e ogni candidato unico al massimo una volta; una configurazione
ripetuta è un hit di cache senza ricaricamento. L'intera osservazione valida
con la mediana di decode più alta rimane il record. Gli slot della cache degli
esperti vengono percorsi verso l'alto un passo di manifest alla volta; dopo una
promozione la prima sconfitta chiude la manopola, mentre il fallback al vicino
inferiore è usato solo quando il passo iniziale verso l'alto perde. La
promozione richiede un risultato di decode strettamente più alto, qualità
full-logit bit-exact immutabile, prefill entro il −8%, stabilità tail/head
≥0.75, il pavimento di RAM della run e al massimo 128 MiB di swapout a regime.
Una radice caricata sotto il 10% entra in modalità vincolata con un pavimento
immutabile relativo alla radice, una riserva di 512 MiB e nessun delta
residente positivo. Saltare le ripetizioni rinuncia alla stima del rumore ABBA,
quindi un record fortunato può causare falsi negativi conservativi ma non può
aggirare alcuna protezione.
Il profilo d'uso è congelato e il ring Raw-KV resta attivo.
I valori candidati restano locali al processo; le Impostazioni rendono
persistente solo il finalista dopo un warmup riuscito dell'agente attivo e una
sonda finale di swap a regime. Una transazione di adozione durevole ripristina
lo snapshot completo delle preferenze iniziali dopo un crash o un commit
interrotto. Il pannello possiede l'avanzamento e lo Stop, ed espone nel Finder
il report Markdown/JSON generato.

**Browse** è il percorso avanzato per un GGUF esterno. Convalida il
descrittore di runtime prima di aggiornare `DS4ModelPath` e rende persistente
un bookmark security-scoped solo per i file esterni. Un modello del catalogo
sotto Application Support viene salvato come percorso gestito dall'app e
cancella un eventuale bookmark esterno obsoleto.

`Views/` possiede la presentazione e lo store MCP rivolto all'app. Le chiavi
persistenti e i valori predefiniti devono restare retrocompatibili; le
impostazioni di layout del modello si applicano al caricamento successivo. Le
credenziali passano sempre attraverso gli helper dell'engine basati su Keychain
e non devono comparire nei log, nelle impostazioni esportate o in UserDefaults.
