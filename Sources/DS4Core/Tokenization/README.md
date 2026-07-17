# Tokenization

Contratto comune e implementazioni di tokenizzazione specifiche dei backend.

## Struttura

- [`API/`](API/README.md): superficie minima `TokenizerProtocol`.
- [`Common/`](Common/README.md): primitive byte-level riutilizzabili.
- [`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.md): BPE JoyAI/DeepSeek,
  token speciali e reasoning; conserva l'alias storico `Tokenizer`.
- [`Backends/GLM52/`](Backends/GLM52/README.md): GPT-2 byte-level BPE,
  pretokenizer `glm4`, controlli GLM e stop policy.
- [`Backends/Qwen/`](Backends/Qwen/README.md): punto di estensione, senza
  implementazione fittizia.

## Flusso

I tokenizer concreti leggono le tabelle tramite
[`GGUFModel`](../Formats/GGUF/README.md).
Il testo semplice attraversa pre-tokenizzazione e merge BPE; una chat già resa
riconosce prima i token speciali indicizzati per byte iniziale. Gli id generati
alimentano prefill e decode; gli id in uscita vengono ricomposti in byte/testo.

`neutralizeSpecialTokenLiterals(in:)` contiene i confini del prompt: spezza le
sequenze letterali che il modello classifica come token di controllo quando
provengono da dati non fidati. Va applicato ai singoli campi prima del rendering
(system, user, storico, risultati e schemi tool), mai alla chat già renderizzata:
BOS, ruoli e delimitatori aggiunti dal renderer devono restare token atomici.
La variante `inJSON:` decodifica e visita ricorsivamente chiavi e valori prima
di riserializzare: anche un delimitatore occultato con escape `\uXXXX` viene
neutralizzato prima che il renderer degli schemi o degli argomenti lo espanda.

## Regole di modifica

La corrispondenza byte-per-byte col tokenizer selezionato è obbligatoria.
Preservare longest-match dei token speciali e fallback sui singoli byte; testare
Unicode, CJK, delimitatori DSML e input malformati. Evitare allocazioni nel ciclo
caldo senza misurarne l'impatto sul prefill.
