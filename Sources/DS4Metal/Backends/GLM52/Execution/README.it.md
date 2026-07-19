[English](README.md) | **Italiano**

# Esecuzione dei layer GLM 5.2

Prima composizione GPU di un layer GLM: il forward del primo token con ogni
matvec quantizzato dispatchato attraverso i kernel validati (proiezioni di
attenzione Q8_0 e blocchi densi/condivisi, esperti instradati K-quant, router
su GPU), giudicato rispetto a `GLM52LayerCPUReference` sui pesi dequantizzati.

Divisione deliberata a livello di validazione: la colla economica — RMSNorm su
una singola riga di embedding, le somme residuali, il minuscolo matvec F32 del
router (il GGUF memorizza `ffn_gate_inp` in F32) — resta sulle implementazioni
oracolo su CPU, così ogni divergenza dall'oracolo è attribuibile a un kernel
GPU. Gli esperti arrivano tramite una closure provider (cache a slot, lettore
di payload o byte di fixture).

`glm52FirstTokenForward` concatena i layer e termina con l'output head.
Questa è la baseline di correttezza per il grafo persistente a GPUTensor
(attivazioni residenti, cache compatta, prefill/decode), che verrà dopo — non
è un loop di decode e non abilita alcuna capacità di catalogo.

# Esecuzione del decode GLM 5.2

`glm52DecodeAttention`/`glm52DecodeLayer` compongono un passo di DECODE su GPU
sopra le primitive validate, giudicato rispetto a `GLM52DecodeCPUReference`:
matvec Q8_0 per ogni proiezione, norma KV-LoRA su GPU che alimenta lo store
F16 della cache compatta (prefisso normalizzato, coda RAW — scritta PRIMA
della selezione e dell'attenzione), lo store GPU delle chiavi dell'indexer
(LayerNorm + RoPE del prefisso), i kernel RoPE di coda/prefisso, lo scoring
dell'indexer su GPU più il top-k multi-blocco quando l'intervallo visibile
supera il top-k (altrimenti intervallo di riempimento causale, con il token
corrente sempre incluso), e l'attenzione indicizzata con rotazione della coda
per riga (`rotateTailByRowPosition`). I layer IndexShare riutilizzano
letteralmente la selezione del full-indexer precedente e non memorizzano mai
chiavi dell'indexer.

Stessa divisione dell'esecutore del primo token: la colla RMSNorm, le somme
residuali e i minuscoli matvec F32 di router/indexer-proj restano sulle
implementazioni oracolo su CPU, così ogni divergenza è attribuibile a un
kernel GPU. I kernel di attenzione e dell'indexer fissano la geometria delle
teste v5_2; larghezza dell'embedding, rango Q-LoRA, larghezze FFN e top-k
restano liberi per fixture economiche. La metà FFN è lo `glm52LayerFFNStage`
condiviso. Gli array host si muovono attraverso buffer condivisi a ogni
dispatch — il grafo di decode persistente a GPUTensor che mantiene residenti
attivazioni e cache verrà dopo, con questa composizione come sua baseline di
correttezza. Ancora non un loop di decode eseguibile; nessuna modifica alle
capacità di catalogo.

# Grafo di decode residente GLM 5.2

`GLM52ResidentDecodeWeights` carica UNA SOLA VOLTA i pesi quantizzati di un
layer in MTLBuffer; `GLM52ResidentDecodeCaches` mantiene residenti sulla GPU
la cache compatta (righe F16 interlacciate `[capacity][576]`) e il piano
delle chiavi dell'indexer, con append sul posto. `glm52ResidentDecodeAttention`
codifica l'intero passo di attenzione su buffer concatenati: nel percorso
fill-range un SINGOLO command buffer copre le norme (nuovo
`kernel_glm52_rms_norm_f32`), le proiezioni LoRA, lo store della riga compatta
(nuovo `kernel_glm52_store_compact_row_f16`), entrambi i kernel RoPE, lo store
delle chiavi dell'indexer, qk_lowrank, l'attenzione indicizzata ruotata e la
proiezione di output; il percorso top-k si divide solo attorno al readback dei
punteggi che alimenta il top-k multi-blocco orchestrato dall'host. La CPU
mantiene le somme residuali, il router e il matvec F32 `indexer.proj` a 32
righe.

Ancora di correttezza: parità con `glm52DecodeAttention`, a sua volta
giudicato da `GLM52DecodeCPUReference`; l'unica differenza aritmetica
intenzionale è la RMSNorm GPU con riduzione float che sostituisce la colla CPU
con accumulo in Double.

Il livello stack completa la storia della residenza: `GLM52ResidentFFN` carica
una sola volta la norma FFN di ogni layer più i pesi densi/condivisi (gli
esperti instradati restano un flusso di byte per token attraverso il provider
— inerente allo streaming, non una lacuna di residenza),
`glm52ResidentDecodeLayer` esegue la metà FFN residuale su buffer residenti
con accumulo su GPU (`kernel_glm52_add_f32`), riportando `ffnIn` all'host una
volta per layer sparso per il router F32, e `glm52ResidentDecodeForward`
concatena uno stack di valori `GLM52ResidentStackLayer` sotto la VERA politica
IndexShare (indici di layer assoluti; i layer full pubblicano la selezione, i
layer intermedi devono corrispondere alla loro sorgente di politica e
riutilizzarla letteralmente), terminando nell'output head residente
(`GLM52ResidentOutputHead`: RMSNorm finale + matvec del vocabolario). Giudicato
rispetto alla composizione per-dispatch. Il prefill su prompt reali e la
parità dei logit sul GGUF reale sono i gate rimanenti.
