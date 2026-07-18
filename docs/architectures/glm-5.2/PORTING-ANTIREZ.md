# Port dal branch `antirez/ds4:glm5.2`

Il branch [`glm5.2`](https://github.com/antirez/ds4/tree/glm5.2) è il riferimento
funzionale, non una sorgente da copiare monoliticamente. Il suo runtime GLM è
integrato nel grande grafo C/Objective-C/Metal originale; DwarfStar Swift lo
separa invece per architettura, con oracle CPU e test prima di collegare ogni
kernel alla generazione.

## Cosa introduce upstream

- shape `glm-dsa` e loader dei 79 blocchi, con 78 layer autoregressivi;
- template ChatGLM/GLM, reasoning High/Max e tool grammar XML piatta;
- MLA/DSA con cache KV-LoRA compatta, RoPE tail e indexer 32×128;
- IndexShare: full-indexer nei layer 0, 1, 2, 6, 10, …, 74;
- attenzione causale completa per le righe corte e selezione top-2.048 per il
  contesto lungo;
- routing sigmoid top-8, bias usato solo per la selezione, pesi non-biased
  normalizzati e scala 2,5;
- tre layer FFN densi, poi MoE routed più esperto condiviso;
- caricamento resident o streaming SSD degli esperti e kernel Metal dedicati;
- percorsi separati per token singolo, batch prefill e indexed prefill.

Il branch non fornisce un backend GLM CPU/CUDA/ROCm di produzione equivalente,
non abilita MTP nella normale generazione e non contiene benchmark affidabili
utilizzabili come promessa prestazionale per questa app.

## Cosa è già portato qui

| Area | DwarfStar Swift |
|---|---|
| Detector e blocco backend | registrato `glm-dsa`, rifiuto esplicito finché non runnable |
| Metadata/shape | validazione stretta dei 31 campi usati dal grafo |
| Tensor directory | schema completo, mappa payload-free e planner letture top-8 |
| Lettura pesi | `pread` bounded su descrittori e piani top-8 e slot-cache LRU con pinning del batch e hit byte-identici; MetalIO da collegare |
| Tokenizer/chat/tool | implementati con golden test sugli ID reali |
| Router | oracle CPU e kernel Metal dedicato |
| DSA/IndexShare | layout, policy, scorer/top-k CPU, primitive Metal e top-k GPU multi-blocco (argsort+merge) |
| Attenzione compatta | oracle CPU doppio (espanso vs assorbito) e kernel Metal staged qk_lowrank/indexed/value_project (F32 e Q8_0) confrontati con l'oracle |
| FFN/MoE/output head | oracle CPU F32-ref, dequant K-quant di riferimento e kernel Metal di validazione per tutte le fasi (routed K-quant, denso/shared/output head Q8_0); famiglie ottimizzate per-quant mancanti |
| Cache | planner F16 lazy; KV-LoRA norm/store e indexer-K norm/RoPE/store isolati |
| Decoder | non ancora presente |

## Scelte deliberate diverse da upstream

1. **Backend separato.** Nessun ramo GLM nei tipi DeepSeek e nessun alias che
   possa inviare un GGUF al decoder sbagliato.
2. **Capacità lazy.** Upstream dimensiona normalmente la cache compatta alla
   finestra logica; qui il planner usa slab append-only e budget, evitando che
   una finestra vuota da 100k prenoti subito circa 8,87 GiB.
3. **Frontend prima del decoder.** Tokenizer e protocollo tool sono testabili e
   selezionabili per diagnostica, ma non implicano capability di inferenza.
4. **Correttezza prima delle scorciatoie batch.** I percorsi prefill ottimizzati
   verranno abilitati solo dopo il confronto dei logits; la variante token-major
   corta di upstream non viene assunta numericamente equivalente.
5. **Streaming attuale DwarfStar.** Il port dovrà riusare circuit breaker,
   MetalIO/pread e limiti di memoria già presenti, invece di duplicare il
   sottosistema C upstream.

## Gate di qualità

Un kernel isolato verde non basta. Per rendere GLM eseguibile servono almeno:

- test GPU realmente eseguiti su Metal, non saltati per assenza del device;
- confronto tensor-by-tensor su uno strato denso e uno MoE;
- hash/tolleranza documentata di tutti i 154.880 logits;
- equivalenza fra decode token singolo e prefill per lo stesso prefisso;
- qualità invariata fra resident e streaming;
- stop, reasoning e tool call verificati nel loop di chat completo;
- cancellazione, cambio schermata e rilascio memoria verificati in GUI.
