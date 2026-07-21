[English](README.md) | **Italiano**

# GLM 5.2

Questa cartella descrive il port nativo di GLM 5.2 (`glm-dsa`) in DwarfStar.
Il port è **operativo end to end**: il motore streaming
(`GLM52ResidentModel`) esegue prefill e decode dal GGUF reale su una
macchina da 16 GB, e GUI, demo, server locale, benchmark, auto-tune e
checkpoint disk-KV selezionano GLM nativamente. Il pannello Benchmark esegue
sia Speed (sweep di throughput, decode p99) sia Correctness (Top-1/2/3
teacher-forced su `forwardBatch` layer-major, piano di campionamento e tipi
di risultato condivisi con DeepSeek), e il pannello Diagnostica tokenizza e
mostra template chat/tool col tokenizer e renderer GLM. Resta solo la
distribuzione, per ora esclusiva DeepSeek.

## Stato corrente

| Capacità | Stato |
|---|---|
| Catalogo e download GUI | sì, tre GGUF monolitici con revisione/dimensione/SHA fissati |
| Riconoscimento di `general.architecture = glm-dsa` | sì |
| Configurazione, schema GGUF, tokenizer, protocollo chat/tool | sì |
| Prefill, decode e logits end-to-end | **sì** — motore streaming, prefill layer-major, decode greedy/campionato |
| Selezione GUI e `DS4Demo` | **sì** (chat, settings per-backend, tuning) |
| Server locale (semantica stateless OpenAI/Anthropic) | **sì**, con riuso incrementale del prefisso KV |
| Benchmark e auto-tune | **sì** (bottone GUI e `DS4_GLM_AUTOTUNE=1`) |
| Checkpoint disk-KV | sì, store per prefisso come DeepSeek (`GLM52DiskKVStore`): più conversazioni affiancate, restore del prefisso più lungo, budget in token con eviction, `state.glmkv` legacy adottato automaticamente |
| Profiling per fase | sì — report in stile DeepSeek più split GPU/host |
| Distribuzione | no (solo DeepSeek) |

## Motore e ottimizzazioni misurate (M1 Pro 16 GB, IQ2_XXS)

Il motore streamma: i 3 layer dense iniziali residenti (floor adattivo sotto
pressione RAM — residenti in più vengono paginati dal sistema e costano
~750 ms/token di residency al driver, misurato), i 75 layer sparse
streammano i tensori grossi per token con prefetch double-buffered, e gli
esperti instradati arrivano come record contigui dall'arena di staging. Le
ottimizzazioni sono default del motore, ognuna con verdetto misurato:

- **fusione dei commit** (`DS4_GLM_FUSE`): FFN del layer N e trunk del layer
  N+1 in un command buffer — metà delle attese sincrone (~154 → ~77/token);
- **kernel vettorizzati**: dot IQ2_XXS/Q8_0/Q4_K con FMA float4 e letture
  larghe (−19% GPU);
- **fase B del prefill multi-token** (`DS4_GLM_PREFILL_MOE`, sulla tecnica
  GEMM fusa di ds4): il loop sui token sta dentro i kernel, così i pesi di
  ogni esperto staged attraversano la DRAM una volta per tile di 4 token
  invece che una volta per token, e una wave è 3 dispatch invece di 3 per
  applicazione — bit-esatto rispetto al percorso per-applicazione (fissato
  dai test; il guadagno end-to-end sul prefill va ancora misurato, fase
  "experts" baseline 253 ms/token);
- **MoE batched** (`DS4_GLM_MOE_BATCH`): tutti gli esperti instradati in due
  dispatch; l'esperto condiviso (sempre attivo) resta separato;
- **router fuso su GPU** (`DS4_GLM_GPU_ROUTER`): matvec + sigmoid + top-8
  nel commit del trunk, readback di 64 byte (−18% di prefill);
- **argmax su device** per il decode greedy (readback 4 byte), **coppia
  qA+kvA** in un dispatch, **top-k dell'indexer su device** (contesti oltre
  2.048);
- **mlock dei pesi residenti** (`DS4_GLM_MLOCK`): head da 433 a 39 ms/token;
- **letture parallele nel prefill** (`DS4_GLM_READ_SPLIT`, solo prefill —
  misurato controproducente in decode, dove il ritmo seriale del fill è ciò
  che lascia banda SSD alle letture demand degli esperti);
- **sidecar Q4 dei layer** come pack unico (`<gguf>.q4dense`, sezioni per
  layer, ripristinabile, qualsiasi sottoinsieme utile) più il bundle esperti
  (`<gguf>.expbundle`) nello stesso formato contenitore.

Misurato sull'IQ2_XXS pubblicato con sidecar a 58/75 layer: **prefill
~1,05 s/token, decode ~3,0 s/token (0,33 tok/s)** sostenuti su 64 token —
contro i 6,0 s/token di baseline a inizio campagna. Il profilo residuo del
decode è ~69% I/O SSD: le prossime leve sono il completamento del sidecar
(vincolato dal disco) e il decode self-speculative.

Bocciate con misura (restano come knob opt-in, verdetti nei commenti del
codice): speculazione esperti su SSD saturo, letture F_NOCACHE, MetalIO per
lo stream layer in decode, cache esperti usage-driven (il routing GLM è
quasi uniforme: i top esperti coprono ~15% delle route con 2 GB di budget,
contro il 69% di hit DeepSeek).

## Contratto verificato sul GGUF reale

Il 17 luglio 2026 è stato letto l'header reale della variante IQ2_XXS dello
snapshot Antirez `2638b3b878f5c6cc3ae7334b8dbea1275025f52e`:

- 66 KV di metadati e 1.809 descrittori di tensori;
- architettura `glm-dsa`, 79 blocchi memorizzati e 78 autoregressivi;
- hidden 6.144, vocabolario 154.880, 64 teste, KV-LoRA 512 e coda RoPE 64;
- 256 esperti instradati, top-8, un esperto condiviso e tre layer densi;
- indexer 32×128, top-k 2.048 e full indexer su 21 layer;
- contesto dichiarato di 1.048.576 token;
- `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = glm4`;
- BOS `154822 = [gMASK]`, `<sop> = 154824`, EOS
  `154820 = <|endoftext|>`.

Il test opzionale `GLM52RealHeaderIntegrationTests` usa una copia sparsa
della dimensione originale: valida configurazione, vocabolario e tutti i
descrittori senza leggere il payload dei pesi. I test ordinari usano fixture
sintetiche e non richiedono il GGUF.

## Cache DSA e memoria

Il layout compatto F16 tiene per token:

- 78 × 512 valori KV-LoRA;
- 78 × 64 valori RoPE;
- 21 × 128 chiavi indexer.

Il totale è 95.232 byte/token, circa 372 MiB a 4.096 token e 8,87 GiB a
100.000 token. Il planner di DwarfStar cresce in slab append-only e può
imporre un budget residente: una finestra logica ampia non provoca quindi,
da sola, l'allocazione immediata dell'intera cache.

## Manifest di download

| Variante | Nome file | Dimensione esatta | SHA-256 |
|---|---|---:|---|
| IQ2_XXS RoutedIQ | `GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf` | 211.075.856.448 byte | `a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0` |
| Q2_K RoutedQ2K | `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` | 262.036.650.048 byte | `b9fa49d010dad35b96418c45831c212a746715b0646c1121ccfc414455bd6fe5` |
| Q4_K RoutedQ4K | `GLM-5.2-UD-Q4_K_RoutedQ4K.gguf` | 434.170.886.208 byte | `7160879c87756236eea16ec6bfeb19288d16fa94dcfcef3a5ed5f38b1383d3a5` |

Sono alternative monolitiche, non shard. Il downloader scrive su `.part`,
usa buffer limitati, supporta la ripresa e verifica lo SHA a blocchi senza
caricare il GGUF in RAM.

## Cosa resta

1. decode self-speculative (bozza economica + verifica `forwardBatch` —
   l'unico modo oltre il tetto di I/O dello stream layer per token) e, più
   avanti, MTP sul blocco nextn mai eseguito (blk78);
2. progresso/cancellazione del prefill e checkpoint a metà prefill (parità
   DeepSeek sui prompt lunghi);
3. copertura completa del sidecar (vincolo disco: ~2,4 GB per layer
   mancante);
4. distribuzione.

La mappa dettagliata rispetto al branch upstream è in
[`PORTING-ANTIREZ.it.md`](PORTING-ANTIREZ.it.md); nomi, shape e tipi GGUF
sono in [`CONTRATTO-GGUF.it.md`](CONTRATTO-GGUF.it.md).
