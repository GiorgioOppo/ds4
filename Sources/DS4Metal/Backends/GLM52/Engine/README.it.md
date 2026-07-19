[English](README.md) | **Italiano**

# Motore GLM 5.2

`GLM52ResidentModel` è il motore reale sul GGUF reale: valida lo schema,
costruisce la mappa dei pesi e carica il grafo di decode residente —
attention, FFN densa/condivisa e output head caricati una volta per layer,
esperti routed serviti in streaming per token tramite
`GLM52StreamedExpertProvider` (planner + slot cache LRU, slicing dei record
esatto al byte) — poi guida il prefill token per token e il decode greedy
sulle cache residenti in crescita. Le righe di embedding vengono lette per
token direttamente dal tensore Q8_0 `token_embd` tramite la lettura ranged
con limiti del reader.

Limiti di scopo deliberati:

- `layerCount` può troncare lo stack dalla testa. I tre layer densi iniziali
  sono Q8_0 da un capo all'altro e girano già oggi sul file pubblicato; gli
  esperti routed richiedono un tipo con un kernel validato — sono supportati
  Q8_0, i quattro K-quant e IQ2_XXS (il formato routed pubblicato); qualsiasi
  altro tipo viene rifiutato al caricamento
  (`GLM52StreamedExpertProviderError.unsupportedExpertType`), perché un
  output silenziosamente sbagliato è peggio di un errore.
- Questo è cablaggio di roadmap, non abilitazione: `BackendSelector` rifiuta
  ancora `glm-dsa`, e il catalogo resta `downloadOnly`, finché non passa il
  gate di parità dei logits sul GGUF reale a modello completo.

`GLM52GreedyDecoding` mantiene l'argmax (pareggi risolti all'indice più
basso) e il loop di generazione liberi da Metal, così entrambi si testano in
unit test senza un device.

# Streaming da SSD

`GLM52LayerStreamer` è l'analogo GLM dello StreamingDecoder di DeepSeek: i
layer sparsi oltre `residentLayerCount` tengono residente solo il loro
piccolo stato (norm, righe del router, proiezione dell'indexer, cache di
decode — ~12 MiB/layer) mentre i grandi tensori Q8_0 vengono letti con pread
DIRETTAMENTE in due set riutilizzabili di staging buffer, con il riempimento
del layer successivo sovrapposto al calcolo GPU del layer corrente. Lo
streaming è una strategia di memoria, mai numerica: il percorso in streaming
esegue le stesse identiche funzioni del grafo residente sui buffer in
staging.

Manopole (demo: DS4_GLM_RESIDENT_LAYERS / DS4_GLM_ACTIVE_EXPERTS /
DS4_GLM_EXPERT_SLOTS): budget dei layer residenti, tetto degli esperti routed
(troncamento in ordine di rank — meno I/O, qualità inferiore), dimensione
della slot cache degli esperti. Dopo ogni token il motore riscalda la slot
cache di ciascun layer sparso con gli esperti selezionati da quel token
(`GLM52StreamedExpertProvider.prefetch`, serializzato rispetto al thread di
decode). Aritmetica onesta: un passaggio completamente in streaming su 78
layer legge ~36 GiB/token — il budget residente, l'accorpamento dei record
degli esperti già svolto da `read(plan:)` e i futuri percorsi MTLIO/bundle
sono ciò che rende praticabili le macchine da 16-32 GiB.

MetalIO (`DS4_GLM_MTLIO=1`): lo streamer riempie i propri slot di staging
tramite una `MTLIOCommandQueue` (SSD → MTLBuffer, nessuna copia pread via
CPU), con una probe di warm-up al caricamento e fallback permanente per
l'esecuzione verso pread su qualsiasi anomalia — la stessa disciplina del
backend ExpertBundle di DeepSeek.

I BUNDLE di esperti sono per ora un non-obiettivo deliberato: reimpacchettare
in modo contiguo i record gate|up|down duplicherebbe ~190 GiB su disco per
trasformare tre pread adiacenti in uno, e `read(plan:)` accorpa già le
letture di ciascun esperto in un'unica destinazione packed. Da riconsiderare
solo se il profiling mostra che il percorso degli esperti è limitato dai seek
anziché dalla banda.
