# Tokenization

Tokenizer byte-level BPE compatibile con il vocabolario e il pre-tokenizer
JoyAI/DeepSeek contenuti nel GGUF.

## File principali

- [`Tokenizer.swift`](Tokenizer.swift): carica vocaboli e merge, riconosce token
  di controllo, tokenizza testo o chat renderizzata e detokenizza gli id.
- [`ByteLevel.swift`](ByteLevel.swift): codifica byte-level e decodifica UTF-8
  usate internamente dal BPE.
- [`ThinkMode.swift`](ThinkMode.swift): scelta tipizzata della modalità reasoning.

## Flusso

`Tokenizer` legge le tabelle tramite [`GGUFModel`](../Formats/GGUF/README.md).
Il testo semplice attraversa pre-tokenizzazione e merge BPE; una chat già resa
riconosce prima i token speciali indicizzati per byte iniziale. Gli id generati
alimentano prefill e decode; gli id in uscita vengono ricomposti in byte/testo.

## Regole di modifica

La corrispondenza byte-per-byte col tokenizer del modello è obbligatoria.
Preservare longest-match dei token speciali e fallback sui singoli byte; testare
Unicode, CJK, delimitatori DSML e input malformati. Evitare allocazioni nel ciclo
caldo senza misurarne l'impatto sul prefill.
