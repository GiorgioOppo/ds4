[English](README.md) | **Italiano**

# Motore Laguna

`LagunaResidentModel` è il motore residente di primo taglio: ogni tensore
validato della ricetta ufficiale Q8_0-signal / esperti Q4_K viene caricato
una volta in `MTLBuffer` shared (upstream Laguna richiede residenza completa
— niente streaming SSD, sidecar o cache esperti), ogni layer possiede una KV
cache ad anello F16 (512 righe sui blocchi sliding-window, il contesto
configurato sui blocchi full-attention) e il grafo per-token rispecchia
`laguna_graph_forward_token`: RMSNorm → matvec Q8_0 accoppiati (Q/K, V/gate)
→ norm/RoPE per-testa → store sull'anello → attention GQA gated → proiezione
di output e residuo → FFN denso o instradato. Dispaccia le primitive GLM
condivise dove l'upstream le condivide (`kernel_glm52_rms_norm_f32`,
`kernel_glm52_matvec_pair_sg`, `kernel_glm52_router_select` con top-N
configurabile, i matvec K-quant di `glm52_moe`) più i kernel Laguna accanto. La
selezione del router viene riletta sull'host per indirizzare gli slab degli
esperti, come il chained decode GLM.

Gli esperti instradati possono essere Q2_K, Q3_K o Q4_K per layer (coerenti,
come garantisce lo schema), quindi girano sia il file ufficiale Q4_K_M sia il
misto RoutedQ2_K-Last27Q3_K; gli helper di dot Q3_K vivono accanto agli altri
K-quant in `metal/glm5.2/glm52_quant.metal`. Limiti di scope deliberati di
questo taglio, rifiutati con errori distinti al caricamento: la ricetta
legacy F16/Q4_K (i suoi percorsi matvec non sono cablati).
`LagunaResidentModelOptions.layerCount` tronca lo stack dal fronte per le
prove di bring-up.

I prompt multi-token usano un prefill layer-major. Norm+RoPE di Q/K lavora
sull'intero chunk, le nuove righe K/V vengono predisposte in F16, la GQA
causale gated valuta tutte le righe in parallelo (condividendo ogni lettura
K/V tra le sei teste query di produzione) e infine le righe vengono
committate. Lo staging conserva la semantica sequenziale quando un chunk
sliding-window attraversa il wrap dell'anello. I piani delle attivazioni
restano allocati e crescono solo quando un prompt successivo richiede un
chunk più largo. `DS4_PREFILL_BATCH=0` ripristina i dispatch
dell'attention per-token per confronti diagnostici di logits o prestazioni.
`DS4_PREFILL_DENSE_MM` è attivo per
default e batchea anche RMSNorm, Q/K/V/gate, proiezione di output, router e
FFN denso/condiviso. Usa le matmul Q8_0 multi-token condivise con DeepSeek;
lo staging delle attivazioni in F16 rende il risultato numericamente vicino
ma non bit-identico. `=0` mantiene disponibile il percorso matvec esatto.
Sul GGUF misto, M1 Pro, top-6 e chunk 256, l'A/B su 414 token ha ridotto il
prefill da 73,8 a 32,6 secondi (5,61 → 12,72 token/s).

Durante il decode della chat, il sampler legge direttamente il buffer Metal
shared dei logits invece di copiare tutti i 100.352 float in un nuovo array
Swift per ogni token. Una repetition penalty esplicita resta corretta e crea
una copia privata, perché deve modificare i logits selezionati.

Tre varianti decode lossless restano disponibili per A/B, ma sono
disabilitate per default perché sul target M1 Pro hanno peggiorato il
wall-clock: baseline 1,94–1,97 tok/s, `DS4_DECODE_CHAINED=1` 1,17 tok/s,
`DS4_DECODE_SPLIT_K=1` 0,59 tok/s e
`DS4_DISCARD_UPLOAD_PAGES=1` 1,74 tok/s (prompt ~1k, top-6, cache 2 GiB).
Il primo accoda il tail esperti N e il trunk attention/router N+1; il secondo
usa sui 12 layer globali GQA split-K oltre
`DS4_DECODE_SPLIT_K_MIN` (default 384); il terzo scarta dal mmap le pagine
interne già copiate. I percorsi restano utili su GPU/pressioni memoria
diverse, ma vanno abilitati esplicitamente.

`DS4_SHARED_EXPERT_OVERLAP` avvia lo shared expert
residente prima del primo wait sugli slab routed. Lo shared scrive in un
accumulatore separato mentre le `pread` avanzano; la chiusura conserva
esattamente l'associazione `(after_attn + routed) + shared`. Il percorso vale
sia per decode sia per prefill layer-major e `=0` mantiene il precedente
gather-then-shared. Resta off nel preset M1 Pro: l'A/B top-10 è risultato
instabile e le misure stabilizzate hanno favorito il percorso non sovrapposto.

Sui 12 layer globali `DS4_INDEXED_ATTN` (default on) costruisce durante il prefill un indice
F16 compresso a centroidi di blocco. Oltre 4096 token, ogni testa calcola gli
score e seleziona top-32 blocchi interamente sulla GPU, poi l'attention legge
le K/V F16 originali dei blocchi scelti più gli ultimi 512 token densi. Il
costo decode dell'attention globale è quindi limitato a circa
`topBlocks × blockSize + recent`, invece di crescere con tutto il contesto.
Laguna non contiene i pesi di compressore/indexer appresi di DeepSeek:
l'indice a centroidi è perciò una selezione sparsa approssimata, mentre le K/V
dei token selezionati restano esatte. I controlli sono
`DS4_LONG_ATTN_BLOCK=16`, `DS4_LONG_ATTN_TOP_BLOCKS=32`,
`DS4_LONG_ATTN_RECENT=512` e `DS4_LONG_ATTN_THRESHOLD=4096`.
`DS4_INDEXED_ATTN=0` ripristina
l'attention densa originale.

La KV attiva cresce lazy: i 12 layer globali partono da
`DS4_KV_INITIAL=512`, mentre i 36 layer sliding
mantengono il ring da 512 righe. Un contesto massimo 32k non riserva quindi
subito ~1,5 GiB: sul test da 999 token la KV iniziale è scesa da 1.608 a
96 MiB e il decode da 0,36 a 1,67 tok/s. Quando si cambia conversazione, la
capacità globale in eccesso viene rilasciata.

Il servizio GUI supporta inoltre checkpoint Disk KV nativi `LKV1`, indicizzati
per prefisso come DeepSeek/GLM. La KV live resta in Metal durante il decode;
l'SSD serve a sospendere/ripristinare sessioni senza rifare il prefill. La
scrittura è sequenziale `F_NOCACHE`, senza costruire un secondo `Data` grande
quanto la cache. Sotto 512 token il file costa ~192 KiB/token; oltre la
finestra costa ~48 KiB/token più ~72 MiB fissi per i ring sliding.

`LagunaResidentModelOptions.expertStreaming` è una divergenza opt-in e
dichiarata dall'upstream (che per Laguna impone la residenza completa): il
percorso "segnale" Q8_0 resta residente e gli slab degli esperti instradati
vengono letti con `pread`/`F_NOCACHE` direttamente in una cache LRU di buffer
Metal shared. Gli hit saltano completamente I/O e copia. Demo e GUI usano
2.048 MiB per default durante lo streaming (529 slot sul GGUF misto provato);

Il gather top-10 parallelo di Laguna raggiunge il tetto SSD direttamente dal GGUF.
`DS4_EXPERT_CACHE_MB=3072` è disponibile per gli A/B e `=0` disabilita
la cache. Sul target M1 Pro 16 GB, 3.072 MiB alza gli hit dal 46% al 53% e
riduce il gather, ma la pressione memoria peggiora sia il prefill sia il
decode: il miglior wall-clock osservato resta quello da 2.048 MiB.
`DecodeProfile`
(`profileReport()`) riporta il costo per fase.

`DS4_ACTIVE_EXPERTS` riduce il top-k
effettivamente eseguito da 10 a `1...10`. Il router seleziona direttamente
top-N e rinormalizza i pesi sugli esperti rimasti, come DeepSeek; buffer,
staging, I/O e compute si riducono insieme. Il default del motore 10 conserva
la numerica upstream; il preset GUI M1 Pro/10 GiB usa top-6 per il profilo
veloce misurato. `DS4_EXPERT_CACHE_SLOTS` può fissare il numero di slot
globali e ha precedenza sul budget MiB; `DS4_RESIDENT_LAYERS` mantiene
residenti i primi N layer routed anche quando lo streaming è attivo.

Gli altri controlli lossless condivisi con DeepSeek sono
`DS4_PREFILL_CHUNK` (default 256, ulteriormente limitato dalla cache),
`DS4_EXPERT_PREAD` (on), `DS4_PREAD_SPLIT` (`1...8`),
`DS4_WILLNEED_EXPERTS` (per il fallback mmap), `DS4_MTLIO` (opt-in con
fallback automatico), `DS4_MLOCK` (opt-in su head e pool esperti) e
`DS4_NSG` (`1...8`, default 4). Gli alias `DS4_LAGUNA_*` sono accettati
soltanto per compatibilità con vecchi script.

`DS4_PREFILL_MOE_BATCH=1` abilita una variante sperimentale
expert-major del MoE prefill. Resta disabilitata per default: sullo stesso
GGUF ha aumentato il prefill da 12,8 a 13,2 s.

`LagunaRuntimeGate.enabled` resta `false` finché questo motore non passa la
parità end-to-end dei logits contro il motore C di riferimento su pesi reali;
selezione, disponibilità a catalogo e dispatch della demo dipendono tutti da
quella sola costante.
