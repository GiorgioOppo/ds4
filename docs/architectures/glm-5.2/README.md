# GLM 5.2

Questa cartella descrive il port nativo di GLM 5.2 (`glm-dsa`) in DwarfStar.
Lo stato corrente non è più soltanto “download”: detector, contratto GGUF,
tokenizer e protocollo chat sono implementati e verificati. Il decoder resta
però intenzionalmente non eseguibile; GUI, demo, server e benchmark non possono
ancora selezionare GLM.

## Stato corrente

| Capacità | Stato |
|---|---|
| Catalogo e download GUI | sì, tre GGUF monolitici con revisione/size/SHA fissati |
| Riconoscimento `general.architecture = glm-dsa` | sì |
| Selezione frontend/tokenizer | sì |
| Configurazione e schema GGUF | sì, geometria stretta e 1.809 tensori |
| Mappa pesi e piano letture top-8 | sì, payload-free e quant-block-aware |
| Lettura payload dal GGUF | sì, `pread` bounded su descrittori e piani top-8 (record gate\|up\|down) |
| Tokenizer GPT-2 + pretokenizer `glm4` | sì |
| Template chat, reasoning e tool XML nativi | sì |
| Oracle CPU di router, DSA/IndexShare e cache compatta | sì |
| Kernel Metal GLM | parziali e non collegati a un decoder |
| Prefill, decode e output logits end-to-end | no |
| Selezione GUI o `DS4Demo` | no |
| Server, benchmark, KV checkpoint e distribuzione | no |

Le entry del catalogo restano `downloadOnly`. Un file completato può essere
ispezionato e tokenizzato, ma `BackendSelector` lo rifiuta con un errore
“backend non ancora implementato”; non esiste alcun fallback al decoder
DeepSeek V4.

## Contratto verificato sul GGUF reale

Il 17 luglio 2026 è stato letto l'header reale della variante IQ2_XXS dello
snapshot Antirez `2638b3b878f5c6cc3ae7334b8dbea1275025f52e`:

- 66 metadata KV e 1.809 descrittori tensoriali;
- architettura `glm-dsa`, 79 blocchi memorizzati e 78 blocchi autoregressivi;
- hidden 6.144, vocabolario 154.880, 64 teste, KV-LoRA 512 e RoPE tail 64;
- 256 esperti routed, top-8, un esperto condiviso e tre layer densi;
- indexer 32×128, top-k 2.048 e full-indexer su 21 layer;
- contesto dichiarato 1.048.576 token;
- `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = glm4`;
- BOS `154822 = [gMASK]`, `<sop> = 154824`, EOS
  `154820 = <|endoftext|>`.

Il test opzionale `GLM52RealHeaderIntegrationTests` usa una copia sparse della
dimensione originale: valida configurazione, vocabolario e tutti i descrittori
senza leggere il payload dei pesi. I test ordinari usano fixture sintetiche e
non richiedono il GGUF.

## Cache DSA e memoria

Il layout compatto F16 conserva per token:

- 78 × 512 valori KV-LoRA;
- 78 × 64 valori RoPE;
- 21 × 128 chiavi indexer.

Il totale è 95.232 byte/token, circa 372 MiB a 4.096 token e 8,87 GiB a
100.000 token. Il planner DwarfStar cresce a slab append-only e può imporre un
budget residente: una finestra logica grande non provoca quindi, da sola,
l'allocazione immediata dell'intera cache. Il runtime dovrà comunque decidere
come gestire contesti che superano RAM e budget SSD prima di dichiarare il
milione di token supportato.

## Manifest download

| Variante | Filename | Dimensione esatta | SHA-256 |
|---|---|---:|---|
| IQ2_XXS RoutedIQ | `GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf` | 211.075.856.448 byte | `a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0` |
| Q2_K RoutedQ2K | `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` | 262.036.650.048 byte | `b9fa49d010dad35b96418c45831c212a746715b0646c1121ccfc414455bd6fe5` |
| Q4_K RoutedQ4K | `GLM-5.2-UD-Q4_K_RoutedQ4K.gguf` | 434.170.886.208 byte | `7160879c87756236eea16ec6bfeb19288d16fa94dcfcef3a5ed5f38b1383d3a5` |

Sono alternative monolitiche, non shard. Il downloader scrive su `.part`, usa
buffer limitati, supporta resume e verifica SHA a blocchi senza caricare il
GGUF in RAM.

## Avanzamento del runtime

La sequenza di abilitazione è vincolante:

1. collegare la mappa pesi validata alle letture SSD/MetalIO top-8 e alle cache
   — avviato: `GLM52PayloadReader` esegue descrittori e piani top-8 con `pread`
   bounded (doppia prova dei limiti, rifiuto dei GGUF troncati all'apertura) e
   `GLM52ExpertSlotCache` fornisce la cache LRU per-esperto con hit
   byte-identici e pinning del batch; restano MetalIO e residency;
2. completare Q/KV-LoRA, RoPE, indexer, attenzione DSA e IndexShare;
3. completare layer densi, MoE routed/shared, RMS residuale e output head;
4. confrontare embedding, ogni layer e logits con un oracle indipendente;
5. verificare prefill e decode su prompt reali, incluse chat e tool call;
6. misurare RAM, pressure, SSD e cancellazione della sessione;
7. solo allora cambiare il catalogo da `downloadOnly` e abilitare GUI/demo;
8. server, benchmark, checkpoint KV e distribuzione vengono dopo il runtime
   locale.

La mappa dettagliata rispetto al branch upstream è in
[`PORTING-ANTIREZ.md`](PORTING-ANTIREZ.md); nomi, forme e tipi GGUF sono in
[`CONTRATTO-GGUF.md`](CONTRATTO-GGUF.md).
