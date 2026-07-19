[English](DS4CORE-INFERENCE.md) | **Italiano**

# Componenti di DS4Core rivolti all'inferenza

Componenti puramente rivolti all'inferenza che non toccano la GPU. I contratti
portabili sono tenuti separati dal frontend di ciascun backend di modello.

- **`Tokenization/Backends/DeepSeekV4/DeepSeekV4Tokenizer.swift`** implementa
  il BPE con i token di controllo di DeepSeek-V4 come BOS/EOS, `<｜User｜>`,
  `<think>` e `｜DSML｜`. Tra i punti di ingresso importanti ci sono
  `tokenizeRenderedChat` e `tokenText`; `Tokenizer` resta un alias di
  compatibilità.
- **`Tokenization/Backends/GLM52/GLM52Tokenizer.swift`** implementa il BPE
  GPT-2, il pretokenizer `glm4`, i token di controllo GLM e la politica di
  stop. La factory dell'API lo seleziona senza implicare che il runtime
  numerico GLM sia disponibile.
- **`Conversation/Models/ConversationModels.swift`** definisce `ToolSpec`,
  `ToolCall` e `ChatTurn`; i file sotto
  `Conversation/Backends/DeepSeekV4/DSML` renderizzano i prompt di chat/tool ed
  estraggono le chiamate dal DSML generato.
- **`Conversation/Backends/GLM52`** possiede il framing dei ruoli GLM, il
  reasoning effort, l'XML nativo dei tool e il relativo parser
  strict/incrementale.
- **`Generation/Sampler.swift`** implementa il sampling con temperatura,
  top-k/top-p, min-p e penalità di ripetizione.
- **`Model/Common/ModelArchitecture.swift`** rileva e descrive la famiglia del
  modello senza costruire un decoder.
- **`Model/Backends/DeepSeekV4/DeepSeekV4Configuration.swift`** valida i
  profili DeepSeek consumati dal runtime attuale.
- **`Model/Backends/GLM52/GLM52Configuration.swift`** valida l'esatta geometria
  `glm-dsa` mantenendo esplicita la disponibilità del runtime.
- **`Diagnostics/LoadProgress.swift`** è un reporter di avanzamento singleton
  thread-safe: il percorso di caricamento del modello scrive milestone e
  avanzamenti per unità, e la UI interroga `snapshot` per renderizzare una
  barra determinata.

I parametri di sampling esposti all'utente (temperatura, penalità di
ripetizione, il top-k fisso) sono documentati nel
[Riferimento di configurazione](../README.it.md#riferimento-di-configurazione) alla
radice del repository.

Per il ciclo di vita end-to-end vedere
[PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.it.md); per l'ownership delle cartelle
leggere [`Sources/DS4Core/README.md`](../Sources/DS4Core/README.it.md).
