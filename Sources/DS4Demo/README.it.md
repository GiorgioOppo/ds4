[English](README.md) | **Italiano**

# DS4Demo

`DS4Demo` è un eseguibile CLI di demo e diagnostica per l'engine in puro Swift
(`DS4Core` + `DS4Metal`). Funziona senza l'app SwiftUI e senza collegamenti a
engine esterni: nessun engine C, nessuna libreria statica, nessun percorso di
inferenza in sottoprocesso.

È utile per:

- avviare il runtime Metal e compilare i kernel incorporati;
- eseguire un self-test della GPU;
- fare l'audit dei dtype dei tensori di un GGUF e dei token speciali del
  tokenizer;
- eseguire uno smoke test forward di un solo token;
- generare token reali misurando prefill, decode e I/O degli esperti.

[`Command/main.swift`](Command/main.swift) apre un GGUF con mmap senza copie,
rileva la quantizzazione MoE, esegue un forward pass e, quando richiesto,
effettua il prefill del prompt layer-major seguito dal decode in streaming.

I profili DeepSeek eseguibili sono Flash e Pro Q2 in un unico GGUF. La demo
deriva numero di layer, dimensioni, compressione, RoPE e geometria del router
dal profilo validato. Non assembla il pacchetto Pro Q4 a due file.

La demo usa `maxKeys = 4096` per default; `DS4_DEMO_CONTEXT` può selezionare
una diversa capacità configurata per misurazioni A/B controllate a contesto
vuoto. Il campionamento è greedy per default (`temperature = 0`), mentre le
variabili `DS4_DEMO_*` qui sotto abilitano gli stessi filtri del sampler usati
dall'engine. È pensata per la misurazione dell'engine, non per la UX di chat;
per la conversazione normale usare l'app DwarfStar.

## Argomenti posizionali

```sh
swift run DS4Demo [gguf-path] [maxNew] [prompt]
```

| # | Argomento | Default | Significato |
|---|---|---|---|
| nessuno | *(nessuno)* | nessuno | Solo bring-up di Metal + self-test GPU. Non è richiesto alcun modello. |
| 1 | `gguf-path` | nessuno | Percorso di un file `.gguf`. Apre il modello, stampa la quantizzazione MoE rilevata ed esegue uno smoke test forward. |
| 2 | `maxNew` | `4` | Numero di token da generare. `0` significa solo smoke test forward, nessuna generazione in streaming. |
| 3 | `prompt` | `"ciao come stai? rispondi in 1 parola"` | Testo utente reso attraverso il chat template del modello. `@/path/file` usa il CONTENUTO del file come prompt (testi lunghi / benchmark di prefill, senza quoting di shell) — troncato a `DS4_PROMPT_MAX_CHARS` caratteri (default `12000`, ≈3k token); prompt più output devono stare in `DS4_DEMO_CONTEXT`. |

Gli argomenti sono posizionali. Per passare un prompt bisogna passare anche
`maxNew`:

```sh
swift run DS4Demo model.gguf 32 "Explain RoPE"
```

Il prompt viene letto da `CommandLine.arguments[3]`; racchiuderlo tra
virgolette se contiene spazi. Non esistono flag `--prompt` né flag di
campionamento.

## Variabili d'ambiente

Le opzioni avanzate sono variabili d'ambiente perché l'engine legge le stesse
manopole a runtime nella demo, nell'app e nei test.
Il confronto completo fra nomi, default e semantica di DeepSeek, GLM e Laguna
è in [Parametri runtime dei backend](../../docs/BACKEND-RUNTIME-PARAMETERS.it.md).

Come leggere le manopole:

- Le manopole di **diagnostica** aggiungono misurazioni e logging senza
  cambiare il modello.
- Le **manopole di memoria / I/O** cambiano come i pesi vengono letti o tenuti
  residenti; contano soprattutto quando il modello non entra in RAM e l'I/O su
  SSD domina il decode.
- Le **manopole dell'expert cache** mantengono residenti su GPU gli esperti MoE
  caldi. Costano RAM wired ma possono ridurre le letture da SSD quando il
  routing è concentrato.
- Le **manopole numeriche / di qualità** cambiano intenzionalmente il calcolo o
  il numero di esperti attivi; usarle per test A/B, non per run di parità.

Regola pratica: prima eseguire con `DS4_DIAG=1`, poi cambiare una manopola alla
volta e confrontare token/s, `gather`, `route/attn`, hit-rate dell'expert cache
e throughput a regime.

### Variabili specifiche della demo

| Variabile | Valori | Default | Effetto |
|---|---|---|---|
| `DS4_DEMO_CONTEXT` | `1024...1000000` | `4096` | Capacità `maxKeys` configurata usata per costruire il decoder. Cambia solo la capacità: un prompt breve identico mantiene il KV vivo ugualmente vuoto, il che ne fa la manopola controllata per i test A/B sulla capacità di contesto. |
| `DS4_DEMO_LIVE_CONTEXT` | `8...(DS4_DEMO_CONTEXT-maxNew-1)` | non impostata | Sostituisce il normale stream di token del prompt breve con un contesto chat sintetico di esattamente N token, ottenuto piastrellando il testo ordinario del prompt fornito preservando l'inquadratura BOS/User/Assistant. Pensata solo per frontiere di prestazioni a contesto vivo; la qualità del testo generato non è un benchmark semantico. |
| `DS4_PROMPT_FILE` | percorso | non impostata | Legge il prompt da un file UTF-8. Equivale alla forma posizionale `@/path/file` ed evita lo splitting degli argomenti di shell per i prompt lunghi. |
| `DS4_PROMPT_MAX_CHARS` | intero positivo | `12000` | Numero massimo di caratteri letti da un file di prompt prima del troncamento. Il prompt tokenizzato finale più l'output richiesto devono comunque stare in `DS4_DEMO_CONTEXT`. |
| `DS4_TYPES_ONLY` | presente, di solito `=1` | off | Modalità audit GGUF. Stampa i dtype dei tensori critici, gli id speciali del tokenizer e la tokenizzazione del prompt, poi esce prima di costruire il decoder. Usarla per prima quando un modello produce testo senza senso o quando si valida una nuova quantizzazione. |
| `DS4_DIAG` | `=1` | off | Run diagnostico completo in streaming. Prima della generazione stampa le manopole attive, misura la banda del disco con `F_NOCACHE` e verifica se i pesi MTP esistono nel GGUF. Dopo la generazione stampa il routing per layer, la concentrazione degli esperti, l'allocazione degli slot di cache e la banda effettiva di gather rispetto al tetto SSD misurato. |
| `DS4_ACTIVE_EXPERTS` | `1...6` | `6` | Riduce quanti esperti MoE instradati vengono effettivamente usati per token. Abbassa I/O e tempo di gather ma cambia la qualità perché entrambi i profili DeepSeek supportati usano top-6. Utile come modalità degradata a bassa RAM o per stimare il costo dell'I/O degli esperti. |
| `DS4_USAGE_FILE` | percorso o `off` | `<gguf-path>.usage.json` | File JSON per la imatrix d'uso, cioè le scelte storiche di esperti fatte dal router. Tenerla abilitata consente al run successivo di pre-riscaldare la cache con gli esperti storicamente caldi. Usare un percorso dedicato per benchmark ripetibili; usare `off` per run a freddo. |
| `DS4_WARMUP` | intero `>=0` | `0`; con `DS4_DIAG`, `min(4, maxNew-1)` | Esclude i primi N token generati dal profilo di decode. I primi token pagano spesso costi una tantum come cache fredda, wiring dei buffer e assestamento della memoria. |
| `DS4_AB_TRACE` | prefisso di output | non impostata | Scrive i metadati `<prefix>.json` e i logit `<prefix>.f32` sull'intero vocabolario per l'harness A/B. I logit sono trattenuti copy-on-write e serializzati solo dopo la regione cronometrata di prefill/decode. Pensata per `scripts/metal_ab.sh`, non per la generazione normale. |
| `DS4_AB_TRACE_FRAMES` | `1...64` | `9` | Numero massimo di frame di logit trattenuti: un risultato di prefill seguito dai risultati dei forward di decode. Il tetto rigido limita la RAM extra a circa 32 MiB sul vocabolario Flash anche se `maxNew` è accidentalmente molto grande. |

### Manopole dell'engine

`DS4Demo` costruisce uno `StreamingDecoder`, quindi eredita le stesse manopole
di runtime usate dall'app. Ogni riga dichiara se l'opzione cambia solo
storage/scheduling, può alterare l'ordine di accumulazione oppure è
deliberatamente lossy. Non dedurre l'equivalenza numerica dal solo fatto che
una manopola è orientata alle prestazioni. Le stesse manopole sono documentate
dal punto di vista dell'app nella sezione
[Configuration Reference](../../README.it.md#riferimento-di-configurazione) del README
principale.

| Variabile | Valori | Default | Effetto |
|---|---|---|---|
| `DS4_RAW_RING` | `=1` | off | Memorizza il KV raw in un ring `nSWA` invece dell'intero contesto. È un buffer Metal in memoria condivisa, non Disk KV su SSD. La sliding-window attention legge solo le ultime 128 righe, quindi la memoria del KV raw è costante; una finestra avvolta viene riordinata e convertita F32→F16 da un solo dispatch GPU. Non elimina la KV cache compressa. |
| `DS4_PREFILL_UNION` | intero | `192` | Numero massimo di esperti raggruppati insieme nell'I/O di prefill layer-major. Ogni gruppo rilegge da disco l'intera unione (con `DS4_EXPERT_PREAD` la page cache viene bypassata), quindi i byte/token di prefill scalano con unione ÷ token-per-gruppo: al vecchio default `64` il gather leggeva ~1.7 GB/token (~257 esperti!) e saturava l'SSD; `192` copre ~3× i token per lettura — misurato come migliore su M1 Pro. Costa ~1.3 GB impacchettati × 2 (pipeline) di memoria transitoria durante il prefill; abbassarlo sulle macchine con RAM stretta. Mai sotto `k` (6). |
| `DS4_PREFILL_FFN_BATCH` | `=0` disabilita | on | FFN di prefill in batch: tutte le FFN dei token di un gruppo vengono codificate in UN solo command buffer Metal (un commit+wait per gruppo) invece di uno per token. Su Flash il vecchio percorso per token produceva 43 layer × 512 token ≈ 22k sincronizzazioni GPU per chunk; Pro ha 61 layer. Encoder seriale ⇒ ordine di dispatch e numerica identici. Usare `=0` solo per verifiche di parità A/B contro il vecchio percorso. |
| `DS4_PREFILL_CHUNK` | intero | `512` | Token per chunk di prefill. Ogni chunk ricarica i pesi densi di TUTTI i layer (~6 GB con `DENSE_STREAM`), quindi un chunk più grande ammortizza quel costo fisso su più token (`1024` lo dimezza) al prezzo di ~160 KB/token di attivazioni transitorie extra. Solo per prompt lunghi — un chunk non supera mai la lunghezza del prompt. |
| `DS4_PREFILL_MM` | `=1` | off (OPT-IN) | FFN instradata E condivisa del prefill in batch attraverso kernel matrice-matrice: i pesi degli esperti vengono letti una volta per tile 64×32 per TUTTI i token del gruppo invece di una volta per token (matvec), e l'esperto condiviso esegue 3 matmul per GRUPPO invece di 3 matvec per token (pesi condivisi Q8_0; i residenti `DS4_SHARED_Q4` mantengono il percorso per token). Stessa matematica, diverso ordine di accumulazione (MMA simdgroup, f16 intermedio) — gli output sono vicini ma NON bit-identici al percorso matvec, perciò resta opt-in finché non è validata in A/B. Questo percorso quant ottimizzato è solo per la shape Flash; tenerlo spento per Pro Q2, che usa il percorso matvec supportato. Anche i gruppi con `DS4_ACTIVE_EXPERTS` < 6 o con meno di 8 token ricadono su matvec. Costa ~80 MB di staging transitorio extra per chunk su Flash. |
| `DS4_PREFILL_ROUTE_BATCH` | intero (`0`/`1` = off) | `32` | Fase route del prefill in batch: route/attention di fino a N token consecutivi vengono codificate in UN solo command buffer — lo snapshot scratch di ogni token (input FFN + selezione del router) viene copiato via blit lato GPU prima che il token successivo lo sovrascriva, e la CPU legge tutte le selezioni dopo una singola attesa. Riduce di N× le sincronizzazioni della fase A. Encoder seriale ⇒ numerica identica. Il benchmark completo dell'app testa 16/32/64/128 dal vivo. |
| `DS4_GPU_INDEXER_TOPK` | `0` disabilita | on | Il decode a contesto lungo tiene i punteggi dell'indexer, la maschera top-K esatta equivalente allo heap e l'attention in un solo command buffer. `0` ripristina la storica selezione su CPU per verifiche di parità. |
| `DS4_ADAPTIVE_SPLITK` | `0` disabilita | on | Sceglie esattamente `min(32, max(1, ceil((rawRows + compressedRows)/32)))` workgroup di FlashAttention. Non c'è arrotondamento a potenze di due: 128 righe usano 4 workgroup e 129 ne usano 5, evitando il vecchio gradino 4→8. `0` ripristina la storica profondità fissa di 32 per A/B. |
| `DS4_VECTOR_COPY` | `=1` abilita | off | Copie contigue F32/F16 sperimentali impacchettate a quattro. Esatte su M1 Pro, code scalari e trasporto dei bit F16 inclusi, ma l'A/B full-model bilanciato per ordine è risultato neutro/leggermente negativo in decode; la copia generica resta il default. |
| `DS4_FLASH_KV_STAGE` | `=1` abilita | off | Fonde il gather del raw ring, lo staging F32→F16 della cache compressa e il padding parziale di K/V/maschera in un solo dispatch. Esatto nei test wrapped/non-wrapped e su 2,197,760 logit full-model. Su M1 Pro è risultato circa neutro in decode e circa +2% in prefill, quindi resta opt-in per altri chip/frontiere di contesto. |
| `DS4_ROPE_PAIR` | `=1` abilita | off | Specializzazione RoPE in-place solo a coppie; il decode ricostruisce anche le posizioni affini su GPU salvo `DS4_ROPE_AFFINE=0`. Baseline/pair/affine sono bit-identici sulle shape testate normal/YaRN/inverse. L'A/B end-to-end su M1 Pro non ha mostrato guadagni in decode, perciò il kernel generico resta il default misurato. |
| `DS4_ROPE_AFFINE` | `=0` disabilita le posizioni affini | on quando `DS4_ROPE_PAIR=1` | Mantiene il kernel a coppie ma ripristina l'array di posizioni fornito dall'host. Non ha effetto mentre `DS4_ROPE_PAIR` è spento. |
| `DS4_DENSE_Q4_KERNEL` | `0` disabilita | on | Matvec denso-Q4 residente dedicato. Riusa l'esatta implementazione delle righe Q4_K ma rimuove il wrapper sintetico con ID di esperto singolo. Impostare `0` per A/B di bit-parità/prestazioni. |
| `DS4_FUSED_ROUTER_PROBS` | `0` disabilita | on | Calcola lo `sqrt(softplus(logit))` del router in un solo dispatch vettoriale invece di due passaggi scalari. Impostare `0` per A/B di parità. |
| `DS4_FUSED_ROUTER_FINALIZE` | `0` disabilita | on | Combina la selezione top-6 e la normalizzazione bit-identica dei pesi di route, risparmiando un dispatch su ogni layer instradato. |
| `DS4_FUSED_COMP_PROJ` | `0` disabilita | on | Calcola insieme le proiezioni KV+gate del compressore per F16 e Q8, condividendo il traffico di attivazioni e un dispatch preservando l'ordine di riduzione di ciascuna. |
| `DS4_DEMO_TEMPERATURE` | float | `0` | Temperatura di campionamento della demo. `0` mantiene lo storico decoding greedy; `0.3` è una scelta focalizzata per il modello a 2 bit. |
| `DS4_DEMO_TOP_K` | intero | `0` | Tetto sui candidati; usare `40` con temperatura diversa da zero per evitare la coda rumorosa del vocabolario da 129k. |
| `DS4_DEMO_TOP_P` / `DS4_DEMO_MIN_P` | float | `1` / `0` | Filtri nucleus e di probabilità minima per il sampler della demo. |
| `DS4_DEMO_REPEAT_PENALTY` | float | `1` | Valori sopra 1 scoraggiano i loop; `1.1` corrisponde al default della GUI. |
| `DS4_DEMO_REPEAT_LAST_N` | intero | `64` | Finestra dei token recenti usata dalla penalità di ripetizione. |
| `DS4_EXPERT_CACHE_SLOTS` | intero | `0` (off) | Abilita una cache GPU LRU per layer per gli esperti MoE. Ogni slot costa circa 6.9 MB wired per layer sul modello a 2 bit. `8` è il minimo effettivo quando abilitata. Se lo hit-rate sale, il gather da SSD scende; se sale la pressione sulla RAM, le prestazioni possono peggiorare. |
| `DS4_EXPERT_CACHE_UNIFORM` | `=1` | off | Disabilita la ridistribuzione degli slot guidata dall'uso. Per default i layer con routing concentrato ricevono più slot a parità di budget totale. Usarla per confronti A/B. |
| `DS4_MULTI_QUANT_CACHE` | `=1` | off | Dà ai layer di esperti misti IQ2/Q4 pool per layer dimensionati correttamente invece di bypassare la cache legacy a classe di dimensione singola. I conteggi di slot sono allocati contro il budget effettivo in byte della cache legacy, quindi abilitarla non aumenta silenziosamente la RAM di cache pianificata. Numerica esatta; tenere off/on come coppia A/B finché non è validata sul Mac target. |
| `DS4_EXPERT_BUNDLE` | `=1` | off | Sidecar `<gguf>.expbundle` con gli slab gate/up/down di ogni esperto instradato CONTIGUI: un miss della slot cache diventa un singolo burst sequenziale da ~7 MB invece di tre letture sparse da ~2 MB. Stessi byte, stessa numerica. Il PRIMO run lo costruisce accanto al modello (duplica la regione degli esperti su disco — decine di GB, saltato se lo spazio scarseggia); i run successivi lo aprono in millisecondi. Invalidato automaticamente quando il modello cambia. |
| `DS4_MTLIO` | `=1` | off | Caricamento rapido di risorse Apple Metal per i miss della slot cache degli esperti durante il decode. Un esperto interleaved viene caricato con un solo comando contiguo direttamente nel suo `MTLBuffer` di destinazione; il prefill resta su `pread` paralleli. Qualsiasi errore fa ricadere permanentemente su `pread`. Stessi byte e stessa numerica; fare A/B sul Mac target perché MetalIO non è costantemente più veloce su M1 Pro. |
| `DS4_MTLIO_MIN_GBS` | float | `1.5` | Soglia automatica del circuit-breaker. I tempi sono aggregati in finestre da 64 MiB; due finestre lente consecutive spostano il caricamento del modello su `pread`. Piccoli batch e stalli isolati vengono ignorati. |
| `DS4_BUNDLE_DIR` | percorso di directory | non impostata (sidecar accanto al GGUF) | Dove il sidecar `.expbundle` viene COSTRUITO quando impostata (`<dir>/<gguf-name>.expbundle`); la lettura prova sempre prima il fratello `<gguf>.expbundle`, poi la directory. L'app sandboxed la imposta al proprio Application Support (non può scrivere accanto a un modello scelto dal picker); per condividere UNA sola copia tra demo e app, puntare la demo alla stessa dir: `DS4_BUNDLE_DIR="$HOME/Library/Application Support/DwarfStar/expert-bundle"`. |
| `DS4_POOL_INTERLEAVE` | `=0` disabilita | on | Layout del pool della slot cache: ogni slot contiene gli slab gate\|up\|down dell'esperto CONTIGUI (stesso layout di record del sidecar bundle), come tre viste di un unico buffer con la dimensione del record passata come stride dell'esperto ai kernel MoE. Un cache miss con il bundle diventa UNA sola `pread` da ~7 MB direttamente nello slot (una syscall invece di tre, I/O più grandi a parità di queue depth). Stessi byte, stessi kernel ⇒ numerica identica. `=0` ripristina il layout storico a 3 buffer (verifiche di parità). |
| `DS4_EXPERT_PREAD` | `=1` | off | Legge gli slab degli esperti con `pread` + `F_NOCACHE` direttamente nei buffer di destinazione, bypassando la page cache di sistema. Evita che il churn degli esperti sfratti i pesi densi ed è spesso utile sui sistemi da 16 GB. La numerica è invariata. |
| `DS4_PREAD_SPLIT` | `1...8` | `1` | Con `DS4_EXPERT_PREAD=1`: numero di pread CONCORRENTI per slab di esperto nel riempimento della slot cache. I miss in decode sono pochi per layer (~2-3 × 3 slab ⇒ queue depth NVMe ~6-9), ma il disco raggiunge il suo tetto solo con ~24 richieste in volo (la sonda DIAG "random parallelo"): dividere ogni slab in N intervalli disgiunti allineati a 16 KB letti in parallelo alza la queue depth a parità di byte. Stessi byte, stessa numerica. Fare uno sweep di `2/3/4` e leggere la banda di gather rispetto al tetto nel verdetto DIAG. |
| `DS4_PREFETCH` | `=1` | off | Usa `madvise` per leggere in anticipo i pesi non instradati del layer successivo mentre il layer corrente calcola. Può aiutare se il calcolo si sovrappone all'I/O; può nuocere se l'SSD è già saturato dal gather degli esperti. |
| `DS4_PREFETCH_EXPERTS` | intero | `0` | Con `DS4_PREFETCH=1`, prefetcha anche N esperti probabili dal prior d'uso. È speculativo e può sprecare banda SSD se il routing non è prevedibile. |
| `DS4_EXPERT_LOOKAHEAD` | intero | `0` | Look-ahead speculativo della slot cache: mentre il layer *i* calcola, PRE-RIEMPIE il pool del layer *i+1* con i suoi top-N esperti del prior d'uso (letture reali nella finestra di inattività dell'SSD, non hint di page cache — funziona con `DS4_EXPERT_PREAD`/bundle). I layer con routing a hash (0-2) vengono sempre pre-riempiti ESATTAMENTE (la loro selezione dipende solo dall'id del token). Provare `6`..`12`; una previsione sbagliata spreca solo banda inattiva. Richiede la slot cache. |
| `DS4_WILLNEED_EXPERTS` | `=0` disabilita | on | Hint di read-ahead non speculativo: dopo che il router seleziona gli esperti reali per un token, l'engine chiama `madvise(WILLNEED)` esattamente su quegli slab. Usare `=0` per confrontare con i fault puramente on-demand. |
| `DS4_RESIDENT_DENSE` | `=1` | off | Copia circa 5 GB di pesi non-expert in buffer wired invece di affidarsi alla residenza mmap/page cache. Aiuta quando lo streaming degli esperti sfratta i pesi densi; può nuocere su 16 GB aumentando la pressione sulla memoria. |
| `DS4_DENSE_AHEAD` | `1...3` | `1` | Profondità di read-ahead del ring di staging denso (richiede `DS4_DENSE_STREAM=1`). `2` tiene in volo i layer i+1 E i+2 mentre la GPU calcola il layer i, così l'SSD avvia la lettura successiva invece di restare inattivo quando la lettura di un layer finisce in anticipo. Costa uno slot di staging extra (~150 MB) per passo. Fare A/B rispetto al gather: un read-ahead denso più profondo CONTENDE anche con lo streaming degli esperti sullo stesso disco. |
| `DS4_DENSE_STREAM` | `=1` | off | STREAMING dei pesi densi a doppio buffer: invece di provare a tenere ~6 GB di pesi densi residenti, i tensori densi di ogni layer vengono letti con `pread`+`F_NOCACHE` in un ring di staging a 2 slot, avviato un layer in ANTICIPO così la lettura SSD del layer i+1 si sovrappone al calcolo GPU del layer i (il pattern di accesso denso è completamente sequenziale — nessuna speculazione). ~300 MB di staging invece di ~6 GB residenti, zero footprint di page cache. Ha precedenza su `DS4_RESIDENT_DENSE`. Numerica identica. La cura candidata quando `route/attn` domina su 16 GB. |
| `DS4_LAZY_IDX` | `=0` disabilita | on (richiede `DS4_DENSE_STREAM=1`) | Differisce le proiezioni di SCORING dell'indexer NSA (`indexer.attn_q_b` + `indexer.proj`) in base alle **chiavi vive usate**, non al `maxKeys` configurato. Prima del confine sparse sono assenti dallo stream denso; alla prima attivazione vengono caricate una volta in buffer residenti e riusate. Il risparmio di ~360 MB/token e gli esempi soglia-1024/top-K-512 sono misurazioni Flash; Pro fornisce top-K 1024 dalla propria geometria. Il compressore ricorrente dell'indexer gira sempre. LOSSLESS; `=0` ripristina il comportamento always-stage per A/B. |
| `DS4_RESIDENT_COMP` | `=0` disabilita | on (richiede `DS4_DENSE_STREAM=1`) | Mantiene RESIDENTI le quattro proiezioni del compressore NSA invece di farne streaming. Vengono lette a ogni token su 41/43 layer Flash e su tutti i 61 layer Pro; ~0.6 GB è il footprint Flash. Stessi byte, numerica identica. `=0` ripristina lo streaming completo (fallback a RAM stretta / A/B). |
| `DS4_COMP_Q8` | `=1` | off | **Lossy, sperimentale.** Converte le proiezioni residenti del compressore attention/indexer da F16 a Q8_0, dimezzando all'incirca la loro RAM e il traffico GPU. Il primo run crea un sidecar `.q8comp.Lx-y` in `DS4_Q4_CACHE_DIR` (o accanto al GGUF). |
| `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` | `64`, `128`, `256`, `512`, `1024`, `2048`, `4096` | `1024` | Numero di righe compresse sopra il quale l'attention in decode passa dalla scansione densa al percorso sparse dell'indexer — attorno alla frontiera ~2K il setup score/top-k sparse domina la scansione di attention più piccola. Cambia solo QUALE implementazione consuma le righe compresse (la selezione dell'indexer a 512 righe è un limite inferiore separato); stesso override d'ambiente e stessi valori ammessi dell'engine C. Determina anche il confine vivo al quale `DS4_LAZY_IDX` carica lo scorer residente. |
| `DS4_MLOCK` | `=1` | off | `mlock()` best-effort sui buffer residenti caldi: pool dell'expert cache, output head residente (con `DS4_DENSE_STREAM`), ring di staging denso e, con `DS4_RAW_RING=1`, ring del KV raw. I buffer Metal condivisi sono memoria anonima che macOS COMPRIME tra un uso e l'altro — un buffer toccato una volta per token si rilegge a ~2.4 GB/s attraverso il compressore invece che a velocità RAM (i ~235 ms misurati dell'output head su una copia "residente"). Blocca ~3.3 GB alle impostazioni di default; numerica identica. Fare A/B e osservare `output head` ed `experts` nel profilo. |
| `DS4_DENSE_Q4` | `=1` (richiede `DS4_DENSE_STREAM=1`) | off | **LOSSY.** Riquantizza le tre proiezioni giganti dell'attention (`q_b`, `output_a`, `output_b` — Q8_0, 107 di ~145 MB/layer) in Q4_K al caricamento e le mantiene RESIDENTI (~1.4 GB, bloccati con `DS4_MLOCK`): metà dei loro byte, letture a velocità RAM, ~4.6 GB/token rimossi dallo stream SSD — la maggiore riduzione di byte disponibile una volta che il decode è disk-bound. Usa i kernel matvec Q4_K già validati (matvec denso = kernel MoE a id con k=1; l'`output_a` raggruppato = k=8 con righe di attivazione per gruppo). Qualità: deriva dei logit ~0.02%, gli output greedy occasionalmente divergono ma restano coerenti. Il PRIMO caricamento paga un passaggio di requant parallelo e scrive una cache accanto al modello (`<gguf>.q4dense`, ~1.4 GB per questo trio base); le opzioni QKV/shared aggiungono record e ne aumentano la dimensione. I caricamenti successivi riusano la cache validata, con tempo di caricamento dipendente da SSD e pressione sulla memoria. Eliminare il file per forzare un nuovo requant. |
| `DS4_SHARED_Q4` | `=1` (richiede `DS4_DENSE_Q4=1`) | off | **LOSSY.** Riquantizza in Q4_K anche le proiezioni FFN dell'esperto condiviso (gate/up/down, Q8_0) e le mantiene residenti: i loro slab escono del tutto dallo stream denso per token, liberando banda disco per il gather degli esperti. Stessa cache `.q4dense` (il toggle ri-riquantizza una volta). Fare A/B sulla qualità prima di adottarla. |
| `DS4_QKV_Q4` | `=1` (richiede `DS4_DENSE_Q4=1`) | off | **LOSSY.** Riquantizza in Q4_K anche le restanti proiezioni di attention di media taglia (`q_a`, `kv` — Q8_0, ~16 MB/layer) e le mantiene residenti: ~0.7 GB/token in meno dallo stream per ~0.35 GB di RAM, e il matvec Q4 legge metà dei byte di quello Q8. Stessa cache `.q4dense` — una cache costruita senza questa manopola resta valida e solo i nuovi tensori vengono riquantizzati (i record corrispondono per chiave). Fare A/B sulla qualità prima di adottarla. |
| `DS4_SPEC_K` | `2`..`8` | off | **Decode greedy AUTO-SPECULATIVO** ([progetto e misurazioni](../../docs/SELF-SPECULATIVE.it.md)): per round, N-1 candidati generati con un draft economico e verificati in un solo passaggio batch a configurazione piena. Si attiva solo con temperatura `0` e penalità di ripetizione `<=1`; altrimenti la demo lo disabilita. Il percorso resta opt-in perché le misurazioni attuali non mostrano vantaggi prestazionali. |
| `DS4_SPEC_DRAFT_EXPERTS` | `1`..`k-1` | `2` | Esperti attivi del DRAFT speculativo (pesi di route rinormalizzati, stesso meccanismo di `DS4_ACTIVE_EXPERTS`): meno esperti = draft più economico ma minore accettazione. Fare A/B per modello. |
| `DS4_Q4_CACHE_DIR` | percorso di directory | non impostata (cache accanto al GGUF) | Dove la cache di requant `.q4dense` viene SCRITTA quando impostata (`<dir>/<gguf-name>.q4dense`). La lettura prova entrambi i posti: una cache prodotta dalla demo accanto al GGUF viene rilevata e PROMOSSA nella posizione primaria, così demo e app condividono una sola conversione. L'app sandboxed la imposta al proprio Application Support (non può scrivere accanto a un modello scelto dal picker). |
| `DS4_FUSED_MOE` | `=0` disabilita | on | Usa i kernel MoE fusi per default. `=0` seleziona il percorso non fuso per A/B numerici e debugging; può cambiare arrotondamento e output. |
| `DS4_FUSED_HC` | `=0` disabilita | on | Coda HC-reduce fusa: split+collapse+RMSNorm in UN solo dispatch invece di tre. Gira due volte per layer, quindi la fusione risparmia ~170 dispatch/token. Stessa matematica; differisce solo l'ordine di riduzione della RMSNorm (classe ±1 ulp). `=0` ripristina il percorso non fuso. |
| `DS4_ASYNC_FFN` | `=0` disabilita | on | Committa il command buffer della FFN instradata di ogni layer SENZA attesa CPU: il commit+wait della route del layer successivo atterra sulla stessa coda in-order, così la GPU resta alimentata mentre la CPU codifica e la bolla per layer (tempo di encoding × 43) sparisce. Attesa esplicita a fine token (prima dell'output head) e su ogni percorso d'errore; la correttezza è garantita dall'ordine di coda, numerica identica. `DS4_PROFILE_ROUTE` mantiene l'attesa sincrona per tempi di fase accurati. `=0` per A/B. |
| `DS4_PROFILE_ROUTE` | `=1` | off | Divide `route/attn` in sottofasi diagnostiche come compressore, proiezioni Q/KV, attention e output. Aggiunge overhead di sincronizzazione; guardare i rapporti, non i tok/s assoluti. |
| `DS4_Q8_NSG` | `1...8` | `4` | Simdgroup per threadgroup per i matvec densi Q8_0. Cambiano scheduling e occupancy, e la riduzione K è partizionata diversamente, quindi i bit floating-point finali non sono garantiti identici. Fare uno sweep di `2/4/6/8` sul Mac target e includere la validazione dell'output quando la parità conta. |
| `DS4_MOE_NSG` | `1`..`8` | `4` | Simdgroup per threadgroup nei kernel MoE instradati. Partizionati per righe: ogni valore è bit-identico, cambia solo l'occupancy; l'ottimo dipende dal numero di core GPU. Riletta alla creazione del decoder. |
| `DS4_DENSE_Q4_NSG` | `1`..`8` | eredita `DS4_MOE_NSG` | Occupancy di riga indipendente per le proiezioni Q4_K dense/raggruppate residenti. Bit-identica; consente a `q`/`out` di essere regolate separatamente dagli esperti instradati. |

## Cosa provare per primo?

| Sintomo nel profilo | Prime manopole da provare | Motivo |
|---|---|---|
| `expert gather` domina e la banda effettiva di gather è molto sotto il tetto SSD | Confrontare il default con `DS4_WILLNEED_EXPERTS=0`, poi provare `DS4_EXPERT_PREAD=1`, poi `DS4_EXPERT_BUNDLE=1` | Separa il beneficio del read-ahead, il churn della page cache e l'overhead delle letture sparse. |
| `route/attn`, `embed` o `head` restano lenti dopo il warm-up | `DS4_DENSE_STREAM=1`, `DS4_MLOCK=1`, poi `DS4_DENSE_Q4=1` se una velocità lossy è accettabile | I pesi densi probabilmente si rileggono da SSD o da buffer compressi in memoria. Streaming + pinning riducono il churn; Q4 rimuove le grandi proiezioni Q8 dal percorso caldo. |
| Lo hit-rate dell'expert cache è basso ma il routing è concentrato | Sweep di `DS4_EXPERT_CACHE_SLOTS=8/12/16/20/22`, con un `DS4_USAGE_FILE` fisso | Più slot possono trattenere gli esperti caldi e ridurre le letture SSD; il preset GUI misurato usa 22 su un M1 Pro 16 GB con il profilo Q4 completo, ma l'ottimo dipende dalla RAM libera. |
| L'expert cache è abilitata ma non aiuta | A/B con `DS4_EXPERT_CACHE_UNIFORM=1` | Verifica se la ridistribuzione guidata dall'uso sta aiutando o se il budget è sbagliato. |
| I run brevi sono rumorosi | `DS4_WARMUP=4` e `DS4_USAGE_FILE=<path>` fisso | Separa i costi a freddo dal regime e rende ripetibile la storia di routing. |
| Si vuole il miglior throughput su singola macchina | sweep di `DS4_Q8_NSG=2/4/6/8` | L'ottimo di scheduling del denso Q8 è specifico del SoC e della pressione sulla memoria. |

## Esempi

### 1. Solo bring-up di Metal

```sh
swift run DS4Demo
# DS4Demo: runtime Metal attivo su Apple M1 Pro, N kernel compilati
# DS4Demo: self-test GPU PASSED
```

### 2. Audit GGUF senza decode

```sh
DS4_TYPES_ONLY=1 swift run DS4Demo /path/DeepSeek-V4-Flash-...-imatrix.gguf
#   TYPE blk.2.ffn_gate_exps.weight = iq2_xxs (code ...)
#   SPECIAL bos=... eos=... user=... assistant=...
#   PROMPT ids = [...]
```

### 3. Solo smoke test forward

```sh
swift run DS4Demo /path/model.gguf 0
# DS4Demo: 1 forward in 3.2s - logits[...] finite=YES argmax=...
```

### 4. Generazione reale con un prompt personalizzato

```sh
swift run DS4Demo /path/model.gguf 32 "Explain RoPE briefly."
# log di prefill, risposta in streaming, tempi dei token, profilo di decode
```

### 5. Esperimento a bassa RAM

```sh
DS4_EXPERT_PREAD=1 DS4_DENSE_STREAM=1 DS4_MLOCK=1 DS4_EXPERT_CACHE_SLOTS=22 \
  swift run DS4Demo /path/model.gguf 16
```

Aggiungere `DS4_DENSE_Q4=1` quando si accetta un percorso di velocità lossy e
`DS4_EXPERT_BUNDLE=1` quando c'è abbastanza spazio su disco per il sidecar.

### 6. Confronto MoE fuso vs non fuso

```sh
swift run DS4Demo /path/model.gguf 8 "1+1?"
DS4_FUSED_MOE=0 swift run DS4Demo /path/model.gguf 8 "1+1?"
```

### 7. Diagnostica completa in streaming

Usare almeno 48 token generati quando si vogliono diagnostiche significative
sull'allocazione della cache.

```sh
DS4_DIAG=1 DS4_EXPERT_CACHE_SLOTS=8 \
  swift run DS4Demo /path/model.gguf 48 "Tell me the history of Rome."
```

Confronti A/B utili con `DS4_DIAG=1`:

- `DS4_WILLNEED_EXPERTS` di default vs `DS4_WILLNEED_EXPERTS=0`;
- allocazione guidata dall'uso di default vs `DS4_EXPERT_CACHE_UNIFORM=1`;
- `DS4_EXPERT_CACHE_SLOTS=8/12/16/20/22`;
- `DS4_Q8_NSG=2/4/6/8`.

### 8. A/B di correttezza + prestazioni per una manopola Metal

Usare un file di prompt, così entrambi i processi figli ricevono esattamente
gli stessi byte. L'harness sovrascrive la manopola scelta, forza il
campionamento greedy, disabilita il decode speculativo e impedisce che il run
baseline muti la storia d'uso consumata dal candidato:

```sh
scripts/metal_ab.sh /path/model.gguf prompt.txt DS4_ADAPTIVE_SPLITK 0 1 8
```

Il report distingue la parità bit-exact dalla parità numerica basata su
tolleranze, confronta gli id dei token generati e ogni logit trattenuto, poi
riporta i tok/s di prefill e decode. I file di trace sono limitati e analizzati
tramite `mmap`; il comparatore non carica mai tutti i vettori nella RAM di
Python. Per un'ottimizzazione che promette numerica esatta usare:

```sh
DS4_AB_ATOL=0 DS4_AB_RTOL=0 \
  scripts/metal_ab.sh /path/model.gguf prompt.txt DS4_FUSED_ROUTER_FINALIZE 0 1 8
```

Ripetere una volta con `DS4_AB_ORDER=candidate-first` prima di promuovere un
risultato: il calore della page cache, lo stato termico e la pressione sulla
memoria possono favorire l'uno o l'altro processo.

### 9. Profilo pulito a regime

```sh
DS4_WARMUP=4 DS4_USAGE_FILE=/tmp/ds4-demo-usage.json \
  swift run DS4Demo /path/model.gguf 32 "Write three sentences about Metal."

DS4_USAGE_FILE=off swift run DS4Demo /path/model.gguf 8 "Run without history."
```

## Stream di output

`stderr` riceve i log dell'engine: quantizzazione rilevata, tempi di prefill,
tempi dei token, report diagnostici e il profilo finale di decode. Il testo
generato va su `stdout` ed è trasmesso in streaming senza buffering.

```sh
swift run DS4Demo /path/model.gguf 8 > answer.txt
```

Il file riceve solo la risposta generata; i log restano a schermo.

## Struttura dei sorgenti

- [`Command/README.md`](Command/README.it.md) documenta l'entry point della CLI e
  il suo confine di dipendenze.
- [`Diagnostics/README.md`](Diagnostics/README.it.md) documenta il logging,
  l'ispezione dei GGUF e gli helper di misurazione del disco.

Mantenere questo README come riferimento autoritativo per comandi e variabili
d'ambiente. Quando una manopola o un argomento posizionale cambia, aggiornarlo
nella stessa modifica dell'implementazione.
