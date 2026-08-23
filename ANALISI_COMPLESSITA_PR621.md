# Analisi di complessità e ristrutturazioni prestazionali — PR #621

Data: 23 agosto 2026

Base analizzata: `84cc882`

HEAD dello snapshot originario: `167eb107f0f15aca922335718bfc7c69952aa815`

## Stato delle implementazioni successive

L'analisi delle funzioni e dei costi è stata redatta sullo snapshot indicato
sopra. Le ristrutturazioni seguenti sono state successivamente implementate
nel worktree e non sono più proposte future:

- CPU: dot Q4_K su coppie di token con decode dei pesi condiviso e
  quantizzazione Q8_K batch parallela (`6e0527aa`);
- CUDA: coppia MMQ Q4_K per il prefill con una sola quantizzazione Q8_1
  condivisa (`b4a106ec`);
- ROCm: singolo launch pair per decode e TILE8 prefill (`a02008f6`).

Le sezioni seguenti mantengono la fotografia completa dell'analisi originaria,
ma annotano questi punti come completati dove compaiono tra le priorità.

## Perimetro

La PR contiene 43.301 inserimenti, 5.161 rimozioni e 49 file modificati. Non tutto il diff appartiene al supporto AProjQ4: include DSpark, SSD streaming, tooling, test, documentazione e ottimizzazioni di altri quanti. Questa analisi copre **tutte le funzioni produttive direttamente coinvolte nel percorso Q4_K delle proiezioni dense di attenzione**, incluse le funzioni di dispatch e gli helper che ne determinano il costo. I test, il generatore GGUF e le funzioni puramente diagnostiche sono classificati a parte: il loro costo non incide sull’inferenza.

La complessità asintotica non cambia tra implementazioni corrette di una proiezione densa: il lavoro matematico minimo resta proporzionale agli elementi della matrice. Le ottimizzazioni utili riducono soprattutto traffico di memoria, quantizzazioni ripetute, lanci kernel e materiale intermedio.

## Simboli usati

| Simbolo | Significato |
|---|---|
| `L` | numero di layer |
| `T` | token nel batch/prefill chunk; `T=1` nel decode ordinario |
| `K` | dimensione di input della proiezione |
| `M` | dimensione di output della proiezione |
| `M0`,`M1` | output di due proiezioni accoppiate |
| `G` | gruppi/head group elaborati |
| `R` | rank per gruppo |
| `Q=256` | elementi per superblocco Q4_K/Q8_K |
| `B=K/Q` | superblocchi per riga |
| `S` | esperti selezionati per token |

Per una matrice Q4_K `[M,K]`:

- operazioni: `Θ(T·M·K)`;
- lettura pesi senza riuso tra token: `Θ(T·M·K)` elementi logici;
- lettura pesi con tiling su `τ` token: circa `Θ((T/τ)·M·K)` blocchi fisici, a parità di lavoro aritmetico;
- quantizzazione attivazioni F32→Q8_K: `Θ(T·K)` tempo e `Θ(T·K)` memoria temporanea, con costante compressa dal formato a blocchi.

## 1. CPU reference path (`ds4.c`)

Il backend CPU è dichiaratamente reference/debug; ottimizzarlo non accelera Metal/CUDA/ROCm. È comunque utile come baseline algoritmica.

| Funzione | Tempo | Memoria extra | Osservazioni |
|---|---:|---:|---|
| `q4_k_get_scale_min` | `Θ(1)` | `Θ(1)` | Decodifica scale/min di un sottoblocco. Inlinabile. |
| `ds4_vec_dot_q4_K_q8_K` | `Θ(K)` | `Θ(1)` | Un dot di una riga. Due passaggi per blocco: min correction e prodotto Q4×Q8. |
| `ds4_vec_dot_q4_K_f32` | `Θ(K)` | `Θ(1)` | Fallback diretto F32; evita quantizzazione ma moltiplica per valori dequantizzati. |
| `dense_q4_K_expect` | `Θ(1)` | `Θ(1)` | Validazione forma/tipo. |
| `matvec_q4_K_dense_worker` | `Θ((r1-r0)·K)` | `Θ(1)` | Una riga per iterazione; pesi letti una volta per chiamata. |
| `matvec_q4_K_prequant` | `Θ(M·K)` | `Θ(1)` locale | Dispatch parallelo usando un `xq` già pronto. |
| `matvec_q4_K` | `Θ(K + M·K)` | `Θ(K)` | Alloca e quantizza l’input a ogni chiamata. |
| `matvec_q4_K_decode_scratch` | `Θ(K + M·K)` | `Θ(1)` per chiamata | Riusa lo scratch; elimina malloc/free, non la quantizzazione. |
| `matvec_q4_K_grouped_expect` | `Θ(1)` | `Θ(1)` | Validazione. |
| `matvec_q4_K_grouped_worker` | `Θ((r1-r0)·K)` | `Θ(1)` | Totale `Θ(G·R·K)`. |
| `matvec_q4_K_grouped_rows_prequant` | `Θ(G·R·K)` | `Θ(1)` locale | Input Q8_K già pronto. |
| `matvec_q4_K_grouped_rows` | `Θ(G·K + G·R·K)` | `Θ(G·K)` | Quantizza ogni gruppo e poi proietta. |
| `matvec_q4_K_grouped_rows_decode_scratch` | `Θ(G·K + G·R·K)` | `Θ(1)` per chiamata | Scratch persistente. |
| `matmul_q4_K_batch_worker` | `Θ((r1-r0)·T·K)` | `Θ(1)` | Elabora coppie di token condividendo decode Q4_K; coda singola per `T` dispari. |
| `matmul_q4_K_batch` | `Θ(T·K + T·M·K)` | `Θ(T·K)` | Prefill CPU; quantizzazione per riga parallela oltre la soglia di lavoro. |
| `matmul_q4_K_grouped_batch_worker` | `Θ((r1-r0)·T·K)` | `Θ(1)` | Totale `Θ(T·G·R·K)`, con lo stesso microkernel a due token. |
| `matmul_q4_K_grouped_batch` | `Θ(T·G·K + T·G·R·K)` | `Θ(T·G·K)` | Analogo grouped. |
| `matvec_any`, `matvec_any_decode_scratch` | `Θ(K+M·K)` sul ramo Q4 | dipende dal ramo | Dispatcher `Θ(1)` più callee. |
| `layer_q_projection_*`, `layer_kv_projection_*` | costo della proiezione | nessuna propria rilevante | Wrapper; sul ramo Q4 ereditano `Θ(M·K)`. |
| `layer_grouped_out_*` | `Θ(G·R·K)` o `Θ(T·G·R·K)` | come callee | Wrapper output attention. |

### Ristrutturazioni CPU

1. **Un solo passaggio NEON nel dot Q4×Q8.** L’implementazione ARM costruisce `q4_u[32]` sullo stack e percorre ogni superblocco due volte. Decodificare min/scale e nibble direttamente in registri NEON elimina lo scratch locale e una parte dei load/store. Complessità invariata `Θ(K)`, costante più bassa.
2. **Tiling 2D per `matmul_q4_K_batch_worker`.** Il loop `row → token` rilegge l’attivazione Q8 per ogni riga e non esprime un microkernel. Un blocco `MR×NT` mantiene più accumulatori e riusa blocchi di peso/attivazione: ancora `Θ(T·M·K)`, ma migliore cache e SIMD.
3. **Quantizzazione fusionata con RMSNorm.** Se l’output F32 della norm serve solo alla proiezione Q4, produrre contemporaneamente Q8_K elimina una lettura e una scrittura F32: risparmio `Θ(T·K)` di traffico.

## 2. CUDA — primitive e kernel Q4_K (`ds4_cuda.cu`)

### Primitive per superblocco

| Funzione | Tempo per chiamata | Memoria | Nota |
|---|---:|---:|---|
| `dev_q4_K_get_scale_min` | `Θ(1)` | registri | Helper inlinabile. |
| `dev_dot_q4_K_q8_K_block` | `Θ(Q)` | registri | Un superblocco; costante perché `Q=256`, ma linearizzato come `Θ(Q)`. |
| `dev_dot_q4_K_q8_K_block_vec` | `Θ(Q)` | più registri | Nove load da 16 B; riduce istruzioni di load e migliora coalescenza. |
| `dev_dot_q4_K_q8_K_block8` | `Θ(8·Q)` | registri elevati | Riusa un blocco peso su fino a 8 token. È il nucleo concettuale del prefill tiled. |
| `quarter_warp_sum_f32` | `Θ(log 8)` = `Θ(1)` | registri | Riduzione intra-warp. |

### Dense projection e wrapper

| Funzione | Tempo | Memoria extra | Impatto |
|---|---:|---:|---|
| `matmul_q4_K_dense_kernel` | `Θ(T·M·K)` globale | `Θ(1)` per thread | Kernel bandwidth-oriented; ogni token rilegge tutti i pesi. Buono per `T=1`, pessimo per prefill grande. |
| `matmul_q4_K_dense_pair_kernel` | `Θ(T·(M0+M1)·K)` | due accumulatori | Condivide `xq`, non i pesi. Riduce un lancio/quantizzazione, non il termine dominante. |
| `cuda_matmul_q4_K_tensor` | `Θ(T·K + T·M·K)` | `Θ(T·K)` | Usa MMQ tiled quando accettato; fallback al kernel sopra. |
| `cuda_matmul_q4_K_pair_tensor_impl` | `Θ(T·K + T·(M0+M1)·K)` | `Θ(T·K)` | Quantizza una volta per due matrici. |
| `ds4_gpu_matmul_q4_K_pair_tensor` | come impl | `Θ(1)` propria | Wrapper. |
| `matmul_q4_K_kslice_kernel` | `Θ(M·Kslice)` | `Θ(1)` | Proiezione su slice K; decode/TP. |
| `cuda_matmul_q4_K_kslice_tensor` | `Θ(Kslice + M·Kslice)` | `Θ(Kslice)` | Quantizza la slice. |
| `ds4_gpu_attention_output_q4_K_batch_tensor` | `Θ(T·G·R·K + T·M·G·R)` circa | temporanei batch | Proietta A per gruppo e B globale; prova batch MMQ, poi fallback. |
| `ds4_gpu_attention_output_low_q4_K_slice_tensor` | `Θ(Gsel·R·K)` | `Θ(Gsel·K)` | Decode grouped attention-A. |

### HC expand e fusioni

| Funzione | Tempo | Memoria/traffico | Nota |
|---|---:|---:|---|
| `matmul_q4_K_hc_expand4_kernel` | `Θ(M·K + 4M)` | evita una rilettura di `block_out` | Fonde proiezione B e HC postprocess; ottima direzione per decode. |
| `q4_K_hc_expand4_rows_kernel` | `Θ(4M)` | legge/scrive intermedi | Esegue solo epilogo dopo MMQ canonico. |
| `q4_K_attn_hc_bitwise_compare_kernel` | `Θ(M)` | contatori diagnostici | Non deve essere nel release path. |
| `cuda_q4_K_hc_expand_canonical` | `Θ(M·K+4M)` | materiale `block_out` | MMQ + epilogo separato; due lanci. |
| `cuda_q4_K_hc_expand_q8k_launch` | `Θ(K+M·K+4M)` | Q8_K scratch | Fallback/fused. |
| `ds4_gpu_matmul_q4_K_hc_expand_available` | `Θ(1)` | `Θ(1)` | Query; dovrebbe essere precalcolata nel piano layer. |
| `ds4_gpu_matmul_q4_K_hc_expand_tensor` | `Θ(K+M·K+4M)` | scratch/oracle se attivo | Dispatcher con variabili diagnostiche. |

## 3. CUDA MMQ (`cuda/mmq/ds4_mmq.cu`)

Le funzioni MMQ sono il percorso che ha eliminato la regressione iniziale di prefill (~16×). La distinzione fondamentale è:

- decode/vector: `Θ(M·K)`, ottimizzato per latenza e bandwidth;
- prefill/matrix: `Θ(T·M·K)`, con pesi riusati su tile di token e tensor-core/microkernel quando possibile.

| Funzione | Complessità | Ruolo |
|---|---:|---|
| `ds4_mmq_dense_impl` | `Θ(T·M·K)` | Core generico tiled. |
| `ds4_mmq_q4_K_dense` | `Θ(T·M·K)` | Wrapper Q4_K prefill/MMQ. |
| `ds4_mmq_dense_vec_impl` | `Θ(M·K)` | Core decode vettoriale. |
| `ds4_mmq_q4_K_dense_vec` | `Θ(M·K)` | Wrapper decode Q4_K. |
| `ds4_mmq_dense_pair_vec_impl` | `Θ((M0+M1)·K)` | Due proiezioni, Q8 input condiviso. |
| `ds4_mmq_q4_K_dense_pair_vec` | stesso | Wrapper. |
| `ds4_mmq_q4_K_dense_pair` | `Θ(T·(M0+M1)·K)` | Prefill: due MMQ condividono una sola attivazione Q8_1 e il relativo scratch. |
| `ds4_mmq_q4_K_grouped_batch_vec_impl` | `Θ(T·G·R·K)` | Prefill grouped. |
| `ds4_mmq_q4_K_grouped_batch_vec` | stesso | Wrapper. |
| `ds4_mmq_q4_K_grouped_vec` | `Θ(G·R·K)` | Decode grouped. |
| `q4_K_k1024_bitwise_compare_kernel` | `Θ(M)` | Diagnostica. |
| `q4_k1024_env_flag`, report/counters | `Θ(1)` | Controllo/diagnostica; fuori dall’hot loop ideale. |
| `iq2_aligned_quantize_xn` | `Θ(T·K)` | Quantizzazione attivazioni per altri percorsi MMQ; può condividere scratch. |
| `q8_0_aligned_dense_vec_*` | `Θ(M·K)` | Q8, non AProjQ4 puro; incide sui tensori SExpQ8/OutQ8 della stessa rete. |

### Osservazione critica

La PR contiene più livelli di fallback che ripetono decisioni a runtime: capability, shape, env flag, disponibilità MMQ, path required/disabled. Ogni check è `Θ(1)`, ma viene ripetuto per proiezione e per layer. Il costo GPU domina sulle matrici grandi, tuttavia in decode a token singolo la latenza di launch/dispatch è visibile. Un **execution plan immutabile per layer**, costruito al load, può sostituire questa foresta di branch con puntatori a funzione e parametri già validati.

## 4. ROCm (`rocm/ds4_rocm_q4.cuh`)

| Funzione | Tempo | Memoria | Valutazione |
|---|---:|---:|---|
| `rocm_matmul_q4_K_dense_kernel` | `Θ(T·M·K)` | registri | Legacy: rilegge pesi per token. Decode. |
| `rocm_matmul_q4_K_dense_grouped_decode_kernel` | `Θ(G·R·K)` | registri | Decode grouped. |
| `rocm_matmul_q4_K_prefill_tile8_strided_kernel` | `Θ(T·M·K)` | LDS/register tile | Riusa pesi su 8 token; evidenza empirica +125–135% prefill. |
| `rocm_q4_K_dense_validate` | `Θ(1)` | `Θ(1)` | Validazione range/shape. |
| `rocm_q4_K_prequant_alloc` | `Θ(1)` logico | `Θ(T·K)` | Allocazione da scratch temporaneo. |
| `rocm_q4_K_dense_pair_requested` | `Θ(1)` | `Θ(1)` | Tre `getenv` nel path; da precalcolare. |
| `rocm_q4_K_prefill_tile8_scope/requested/required` | `Θ(1)` | `Θ(1)` | Policy. |
| report/note/result helpers | `Θ(1)` | `Θ(1)` | Diagnostica; atomiche/contatori host solo se abilitati. |
| `ds4_rocm_matmul_q4_K_tensor` | `Θ(T·K + T·M·K)` | `Θ(T·K)` | Quantizza, poi Tile8 per `T>8`, legacy altrimenti. |
| `ds4_gpu_matmul_q4_K_pair_tensor` | `Θ(T·K + T·(M0+M1)·K)` | `Θ(T·K)` | Condivide quantizzazione e usa un singolo launch pair sia in decode sia nel prefill Tile8. |
| `ds4_gpu_attention_output_low_q4_K_slice_tensor` | `Θ(G·R·K)` | `Θ(G·K)` | Grouped decode. |
| `rocm_q4_K_prefill_tile8_quant_launch` | `Θ(T·G·K + T·G·R·K)` | `Θ(T·G·K)` | Quant + grouped tile8. |
| `ds4_gpu_attention_output_q4_K_batch_tensor` | `Θ(T·G·R·K + T·M·G·R)` | temporanei | Due stadi A/B; Tile8 solo per `T>8`. |

### Ristrutturazioni ROCm ad alto valore

1. **Completato — pair Tile8 in un solo kernel.** Le due matrici condividono `xq` e un unico launch con domini di row-tile concatenati in `grid.x`. Il costo resta `Θ(T·(M0+M1)·K)` e l'ordine di accumulo per riga resta quello del kernel standalone.
2. **Token tile adattivo `{4,8,16}`.** La soglia fissa `T>8` lascia i microbatch 2–8 sul kernel che rilegge i pesi. Autotuning per shape/arch può usare tile4 per spec decode e tile16 quando LDS/occupancy lo consentono.
3. **Quantizzazione prodotta dal kernel precedente.** `q8_K_quantize_kernel` è sempre un lancio separato. Fusione RMSNorm→Q8_K o doppia uscita F32+Q8_K elimina `Θ(T·K)` traffico e un launch.

## 5. Metal (`ds4_metal.m`, `metal/moe.metal`, `metal/dsv4_misc.metal`)

### Funzioni host principali

| Funzione | Tempo host | Lavoro GPU | Nota |
|---|---:|---:|---|
| `ds4_gpu_matmul_q4_K_pair_tensor` | `Θ(1)` encode | `Θ(T·(M0+M1)·K)` | Pair small-batch; condivide input e command buffer. |
| `ds4_gpu_q4_K_pair_quad_compressor_store_tensor` | `Θ(1)` encode | `Θ((M0+M1)·K + compressor)` | Fonde Q/KV pair con compress/store; riduce intermedi e launch. |
| `ds4_gpu_attention_output_q4_K_ssd_prefill_exactn_tensor` | `Θ(1)` encode | `Θ(T·G·R·K)` sulle righe esatte | Ottimizzazione SSD prefill, vincolata agli exact rows. |
| `ds4_gpu_attention_output_q4_K_batch_tensor` | `Θ(1)` + branch | `Θ(T·G·R·K + T·M·G·R)` | Sceglie classic MV, MM tiled e exactn. |
| `ds4_gpu_attention_output_low_q4_K_slice_tensor` | `Θ(1)` encode | `Θ(G·R·K)` | Decode low projection. |
| `ds4_gpu_matmul_q4_K_hc_expand_available` | `Θ(1)` | nessuno | Query pipeline. |
| `ds4_gpu_matmul_q4_K_hc_expand_tensor` | `Θ(1)` encode | `Θ(M·K+4M)` | Fused decode tail. |

### Kernel Metal Q4_K direttamente rilevanti

| Funzione/kernel | Complessità globale | Caratteristica |
|---|---:|---|
| `ds4_glm_q4_K_value`, `glm_q4_K_scale_min` | `Θ(1)` | Decodifica elemento/scale. |
| `kernel_mul_mv_q4_K_f32_impl` | `Θ(M·K)` | Core matvec Q4. |
| `kernel_mul_mv_q4_K_dense_f32` | `Θ(T·M·K)` | Entry dense small batch. |
| `kernel_mul_mv_q4_K_dense_pair_f32` | `Θ(T·(M0+M1)·K)` | Pair. |
| `kernel_dsv4_q4_K_qkv_pair_quad_compressor_store` | `Θ((M0+M1)·K + compressor)` | Fusione verticale molto utile nel decode. |
| `kernel_dsv4_q4_K_hc_expand4` | `Θ(M·K+4M)` | Fusione output-B + HC. |
| `kernel_dsv4_attn_out_low_q4_K_f32` | `Θ(G·R·K)` | Low projection. |
| `kernel_mul_mv_q4_K_staged_exactn_impl` | `Θ(Eexact·M·K)` | Solo righe richieste; riduce lavoro se `Eexact << T`. |
| `kernel_dsv4_attn_out_q4_K_ssd_prefill_exactn_f32` | `Θ(Eexact·G·R·K)` | Prefill SSD specializzato. |
| `glm_q4_K_dot_row_tg_f32`, `glm_q4_K_dot_row_lane_f32`, `glm_quant_dot_row_*` | `Θ(K)` per riga | Varianti di riduzione/layout. |
| `kernel_glm_q4_K_pair_swiglu*` e varianti addr/mapped | `Θ(S·M·K)` | MoE Q4, non proiezione dense AProjQ4 ma incide sui modelli SExpQ4. |
| `kernel_glm_q4_K_down*` | `Θ(S·M·K)` | Down MoE. |

### Ristrutturazioni Metal

1. **Piano di encoding per layer.** Spostare selezione pipeline, numero simdgroup, controllo tipi/env e dimensioni dal token loop al caricamento. Il runtime esegue una lista di encoder prevalidati. Non cambia Big-O, riduce CPU latency e branch.
2. **Unificare fusioni verticali in una pipeline AProjQ4.** Il codice ha già due fusioni corrette: Q/KV pair→compressor e output-B→HC expand. Il passo successivo è evitare la materializzazione F32 tra RMSNorm e pair quantizzato, mantenendo un buffer Q8_K/f16 compatto condiviso.
3. **Tile prefill determinato dalla shape.** Per `T` medio, classic matvec e MM general-purpose possono entrambi essere subottimali. Serve un microbenchmark al load per scegliere `mul_mv_ext`, `mul_mm direct RHS` e tile N per ogni shape AProjQ4.
4. **Non applicare `FOR_UNROLL` indiscriminato.** Le review inline lo suggeriscono, ma l’unroll può aumentare pressione registri e ridurre occupancy. Usarlo solo se il metallib disassembly/benchmark mostra meno cicli senza spill.

## 6. Costo end-to-end per layer

Le cinque proiezioni dense AProjQ4 dominano come:

```text
Q_a     : Θ(T·Kqa·Mqa)
Q_b     : Θ(T·Kqb·Mqb)
KV      : Θ(T·Kkv·Mkv)
Out_a   : Θ(T·G·Koa·R)
Out_b   : Θ(T·Kob·Mob)
```

Il costo layer rimane la somma dei cinque termini. Sul decode (`T=1`) l’operazione è generalmente **memory-bandwidth bound**: ogni token deve rileggere le matrici dense. Sul prefill (`T≫1`) diventa possibile riusare i pesi su più token e usare microkernel/GEMM; questa è la ragione per cui il passaggio CUDA da fallback matvec a MMQ ha recuperato circa 15,5× e TILE8 ROCm ha guadagnato 125–135%.

## 7. Ristrutturazioni prioritarie

### P0 — prefill: garantire un vero path tiled per ogni backend e shape

**Problema:** il fallback `token × matvec` ha la stessa complessità `Θ(T·M·K)` ma traffico pesi circa `T` volte maggiore. È la causa già osservata del prefill ~16× più lento.

**Intervento:** creare una sola API semantica:

```c
q4_dense_run(plan, out[], weights[], input, T)
```

Il `plan` sceglie al load:

- decode vector per `T=1`;
- microbatch tile4/8 per `2≤T≤8/16`;
- MMQ/GEMM tiled per prefill;
- pair/multi-projection quando gli input sono identici.

**Guadagno atteso:** massimo sul prefill; nessun cambiamento numerico se si preserva l’ordine di accumulo dove richiesto. Il beneficio non è teorico: è già dimostrato da CUDA MMQ e ROCm TILE8.

### P0 — prefill e decode: eliminare la quantizzazione ridondante

**Problema:** F32→Q8_K è `Θ(T·K)` e spesso è un lancio separato. Pair Q_a/KV la condivide già, ma la rappresentazione quantizzata viene prodotta dopo che RMSNorm ha scritto F32 in memoria.

**Intervento:** RMSNorm con doppia uscita o uscita Q8_K nativa per le proiezioni Q4; conservare F32 solo se un consumer reale la richiede. Il piano layer dichiara la lifetime del Q8_K.

**Guadagno atteso:** riduzione di un kernel, una lettura F32 e una scrittura Q8_K per gruppo di proiezioni; più importante per decode latency che per il termine `M·K`.

### P1 — decode: execution plan immutabile

**Problema:** capability/env/shape/type checks e fallback sono ripetuti dentro il percorso per token. Il codice è difficile da verificare e impedisce al compilatore/CPU di avere un percorso lineare.

**Intervento:** al model load costruire per ogni layer:

- puntatore encoder/kernel;
- tile e geometry;
- offset/row bytes già validati;
- scratch richiesto;
- fusioni disponibili;
- fallback definitivo.

Il token loop non chiama `getenv`, non rivalida i range e non cerca pipeline.

**Guadagno atteso:** piccolo ma sistematico sul decode; grande vantaggio di manutenibilità e minore rischio di false fallback.

### P1 completata — ROCm: pair Tile8 in un lancio

Il path pair ora condivide la quantizzazione ed elabora entrambe le matrici in
un solo launch, concatenando i due domini di tile. La complessità resta
invariata; viene eliminato un launch per layer senza modificare l'ordine delle
riduzioni.

### P1 — Metal: command-buffer fusion e buffer intermedi

Completare la catena di fusioni già iniziata:

```text
RMSNorm → Q8_K → {Q_a, KV} → compressor/store
attention heads → Out_a → Out_b → HC expand
```

Non è necessario fondere tutto in un solo mega-kernel: bastano due o tre kernel verticali con lifetime esplicite e nessuna round-trip F32 non necessaria.

### P2 — decode: cache dequantizzata selettiva, solo con budget

Una cache FP16 delle sole cinque matrici dense può abilitare GEMV/GEMM più semplici ma aumenta i byte per peso di circa 3–4× rispetto a Q4_K. Sul decode bandwidth-bound può essere peggiore. Va considerata solo:

- su macchine con ampia memoria residua;
- se il backend dispone di una primitive tensor-core realmente più rapida;
- dopo benchmark A/B che includa il costo di warmup e pressione cache.

Non è una raccomandazione di default.

## 8. Proposte da evitare

- **Ridurre Big-O con pruning non validato.** Saltare righe/pesi cambia il modello; non è una ristrutturazione semantics-preserving.
- **Fondere matrici concatenandole permanentemente nel GGUF senza misure.** Q_a e KV hanno output diversi; la concatenazione semplifica un launch ma può peggiorare locality e streaming.
- **Cache FP16 globale dei pesi Q4.** Aumenta memoria e bandwidth; contraddice il motivo di AProjQ4.
- **Flag permanenti per ogni variante.** Il file AGENT.md richiede un solo release path. Usare flag solo per diagnosi/rollback, poi scegliere il vincitore.
- **Unroll aggressivo non misurato.** Può ridurre occupancy e peggiorare sia prefill sia decode.

## 9. Piano di benchmark per validare le ristrutturazioni

Per ogni proposta misurare separatamente:

1. kernel time e numero di launch per layer;
2. byte letti/scritti e cache hit, se il profiler lo espone;
3. prefill a `T={1,2,4,8,16,128,512,2048,4096}`;
4. decode a contesti 2k/4k/8k e almeno 128 token;
5. `--decode-consistency` e `score_official`;
6. memoria di picco e startup spans;
7. AProjQ8 control per impedire regressioni del percorso esistente.

Ordine consigliato degli esperimenti:

1. execution plan senza cambiare kernel;
2. RMSNorm→Q8_K fusion;
3. misurazione hardware del ROCm pair Tile8 a singolo launch (implementato);
4. tile adattivo per microbatch;
5. Metal vertical fusion;
6. solo infine cache dequantizzata selettiva.

## 10. Conclusione

Non esiste una ristrutturazione semantics-preserving che trasformi la proiezione densa da `Θ(T·M·K)` a un ordine inferiore. I guadagni reali vengono da quattro leve:

1. **riuso dei pesi tra token nel prefill**;
2. **riuso della quantizzazione tra proiezioni**;
3. **fusione degli epiloghi e rimozione degli intermedi**;
4. **riduzione di launch e dispatch nel decode**.

La priorità più solida è rendere il percorso tiled obbligatorio per tutte le shape di prefill, poi fondere RMSNorm→Q8_K e costruire un execution plan per layer. Queste modifiche attaccano costi già visibili nel codice e già confermati dai benchmark della PR, senza introdurre approssimazioni sul modello.
