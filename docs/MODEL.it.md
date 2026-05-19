# Il modello

Riferimento completo del transformer DeepSeek-V4 così come è
implementato in questo repo. Leggilo quando vuoi capire cosa fa ogni
componente, come si incastrano, quali dtype si scambiano, e dove vive
ciascuna parte sotto `Sources/DeepSeekKit/`.

I documenti complementari coprono argomenti collegati:

- [`PYTHON-MAPPING.md`](PYTHON-MAPPING.md) — cross-walk linea-per-linea
  Swift ↔ Python.
- [`MODULES.md`](MODULES.md) — indice per-file di `Sources/`.
- [`KERNELS.md`](KERNELS.md) — riferimento per ogni kernel `.metal`.
- [`DTYPES.md`](DTYPES.md) — layout bit-a-bit di FP8 / FP4 / E8M0 /
  BF16.
- [`MEMORY.md`](MEMORY.md) — mmap, ciclo di vita della KV cache,
  stima del working set.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — data flow dell'engine + il
  contorno dell'app desktop.
- [`GLOSSARY.md`](GLOSSARY.md) — definizioni one-liner di ogni
  acronimo che usi qui sotto.

> 🇬🇧 La versione inglese è [`MODEL.md`](MODEL.md).

---

## 1. Cos'è il modello

DeepSeek-V4 è un transformer autoregressivo decoder-only con
feed-forward **Mixture-of-Experts** e una pila di scelte non standard
sull'attenzione e sul ramo residuale. Il port Swift rispecchia 1:1
l'implementazione PyTorch di riferimento in
`Reference/inference/model.py`; quel file è la fonte di verità in caso
di dubbio.

Gli ingredienti caratteristici, dall'alto verso il basso:

| Componente | Note |
|---|---|
| **MLA** (Multi-head Latent Attention) | Q low-rank via `wq_a → q_norm → wq_b`, KV single-head condivisa via `wkv → kv_norm`, O low-rank grouped via `wo_a → wo_b`. Scalare `attn_sink` per ogni testa, appreso. |
| **Sliding window + sparse attention** | La KV per-token vive in un ring buffer da 128 slot; opzionale coda di KV compressi per layer. Online-softmax stile FlashAttention su un sottoinsieme top-k. |
| **Compressor** | Pooling softmax gated su `compress_ratio` token consecutivi, emette un KV compresso ogni `ratio` passi. Overlap a `ratio == 4`. |
| **Indexer** | Selettore top-k appreso: sceglie quali posizioni compresse l'attenzione sparse deve guardare. Solo nei layer con `ratio == 4`. |
| **Hyper-Connections (HC)** | Lo stato nascosto è tenuto come `hc_mult = 4` copie parallele; ogni block usa un mixing Sinkhorn-normalizzato invece di un residuo classico. |
| **MoE FFN** | Top-2 routed expert con `sqrtsoftplus` + 1 shared expert. Hash routing opzionale sui primi `n_hash_layers`. |
| **RoPE + YaRN** | RoPE solo sugli ultimi `rope_head_dim = 64` di ogni testa. YaRN per estendere il contesto oltre la lunghezza di training. |
| **MTP** (Multi-Token Prediction) | Block speculativo in coda che predice il *prossimo* token dato lo stato corrente + il nuovo embedding, condividendo `ParallelEmbedding` e `ParallelHead`. |

### Le due taglie pubbliche

I checkpoint rilasciati esistono in due varianti. Il runtime è
identico — quello che cambia è `ModelConfig` e di conseguenza le shape
dei tensori.

| | V4-Pro | V4-Flash |
|---|---|---|
| Parametri totali | ≈ 1,6 T | 284 B |
| Attivi per token | ≈ 50 B | 13 B |
| `n_layers` | maggiore | tipicamente intorno ai 7 nel pubblico |
| Disco @ FP8/FP4 nativo | ≈ 800 GB | ≈ 142 GB |
| Disco @ BF16 fuso | ≈ 1,6 TB | ≈ 600 GB |
| Sta in un Mac da 192 GB? | **no** | sì (mmap) |
| Sta in un Mac da 16 GB? | no | sì (streaming) |

Il target realistico on-device è **V4-Flash**. V4-Pro non entra nella
memoria unificata di nessun Mac. Vedi [`MEMORY.md`](MEMORY.md) per il
loader streaming.

---

## 2. Gli iperparametri: `ModelConfig`

`ModelConfig` (`Sources/DeepSeekKit/Config.swift:5`) rispecchia la
dataclass `ModelArgs` di `Reference/inference/model.py:34`. I nomi dei
campi usano lo `snake_case` Python via `CodingKeys`; il path
`init(fromDict:)` accetta anche gli alias HuggingFace `transformers`
(`hidden_size`, `num_hidden_layers`, `num_attention_heads`,
`rms_norm_eps`, `rope_scaling.factor`, …) così lo stesso codice legge
sia il `config.json` post-converter di questo repo sia quello del
release-card upstream.

### Parametri di forma

| Campo | Default (V4-Flash) | Significato |
|---|---|---|
| `vocabSize` | 129 280 | Dimensione del vocabolario di token. |
| `dim` | 4096 | Hidden size / dimensione del residual stream. |
| `nLayers` | 7 | Numero di transformer block principali. |
| `nMtpLayers` | 1 | Block MTP in coda per predizione speculativa. |
| `nHashLayers` | 0 | I primi N layer usano una tabella di hash precomputata invece del gate. |
| `nHeads` | 64 | Teste di attenzione. |
| `headDim` | 512 | Dimensione per testa (Q, K, V condividono questo size). |
| `ropeHeadDim` | 64 | Tail di ogni testa a cui si applica RoPE. I primi `headDim - ropeHeadDim = 448` sono *nope* (no positional encoding). |
| `maxBatchSize` | 4 | Usato per dimensionare la KV cache, non per il batching runtime. |
| `maxSeqLen` | 4096 | Lunghezza del ring KV + slice compressa. L'inference long-context lo deve alzare. |

### Attenzione low-rank

| Campo | Default | Significato |
|---|---|---|
| `qLoraRank` | 1024 | Bottleneck sul path Q (`wq_a: dim → qLoraRank`, `wq_b: qLoraRank → nHeads·headDim`). |
| `oLoraRank` | 1024 | Bottleneck per-gruppo sul path O. |
| `oGroups` | 8 | Numero di gruppi low-rank della proiezione di output. Ogni gruppo proietta `nHeads·headDim / oGroups` colonne attraverso il proprio rank `oLoraRank`. |
| `windowSize` | 128 | Dimensione della sliding window (le righe del ring buffer). |

### Politica di compressione per-layer

| Campo | Default | Significato |
|---|---|---|
| `compressRatios` | `[0, 0, 4, 128, 4, 128, 4, 0]` | Una entry per layer (main + MTP, in quest'ordine). `0` = pura sliding window; `4` = window + Compressor (con overlap) + Indexer; `128` = window + Compressor (senza overlap, senza Indexer). |

La lunghezza dell'array deve uguagliare `nLayers + nMtpLayers`.
`ModelConfig.inferred(from: loader)` corregge config non coerenti
inferendo `nLayers = compressRatios.count - nMtpLayers`.

### MoE feed-forward

| Campo | Default | Significato |
|---|---|---|
| `nRoutedExperts` | 8 (V4-Flash production: 256) | Pool di expert su cui il gate fa routing. |
| `nActivatedExperts` | 2 | Top-K expert attivi per token. |
| `nSharedExperts` | 1 | Expert "shared" sempre attivo, sommato sopra al contributo routed. |
| `moeInterDim` | 4096 | Dim interna della SwiGLU FFN per expert. |
| `scoreFunc` | `"sqrtsoftplus"` | Una di `softmax / sigmoid / sqrtsoftplus`. V4 usa `sqrtsoftplus`. |
| `routeScale` | 1.0 | Scalare moltiplicato ai pesi per-expert dopo la normalizzazione. |
| `swigluLimit` | 0.0 | Se non-zero, clampa il gate SwiGLU. Zero disabilita il clamp. |

### RoPE / YaRN

| Campo | Default | Significato |
|---|---|---|
| `ropeTheta` | 10 000 | Base θ del RoPE non-compressor (layer con `ratio == 0`). |
| `compressRopeTheta` | 40 000 | Base θ per i layer con compressor (`ratio > 0`); base maggiore = frequenze più lente → contesto effettivo più lungo. |
| `originalSeqLen` | 0 | Lunghezza di "training context" per YaRN. `0` disabilita la correzione di frequenza YaRN. |
| `ropeFactor` | 40 | Fattore di estrapolazione YaRN. |
| `betaFast`, `betaSlow` | 32, 1 | Boundary del ramp YaRN. |
| `mscale` | 1.0 | Correzione opzionale del softmax-scale stile V3. V4 lo lascia a 1 (vedi Attention.softmaxScale più avanti). |

### Indexer

| Campo | Default | Significato |
|---|---|---|
| `indexNHeads` | 64 | Teste di attenzione dell'Indexer. |
| `indexHeadDim` | 128 | Dim per-testa dell'Indexer (separata da `headDim` principale). |
| `indexTopk` | 512 | Top-K posizioni compresse selezionate per query. |

### Hyper-Connections

| Campo | Default | Significato |
|---|---|---|
| `hcMult` | 4 | Numero di copie parallele dello stato nascosto. |
| `hcSinkhornIters` | 20 | Iterazioni Sinkhorn quando si normalizza la comb matrix. |
| `hcEps` | 1e-6 | Epsilon di stabilità numerica per il clamp `sigmoid + eps`. |

### Dtype di quantizzazione + scale

| Campo | Default | Significato |
|---|---|---|
| `dtype` | `"fp8"` | Dtype nativo del checkpoint per i linear non-expert. `"fp8"` (E4M3) o `"bf16"`. |
| `expertDtype` | `nil` (= come `dtype`) | Override per gli expert routed. V4-Flash usa `"fp4"` (E2M1, packed two-per-byte). |
| `scaleFmt` | `"ue8m0"` | Dtype della block-scale compagna dei pesi FP8/FP4. |
| `scaleDtype` | `"fp8"` | Dtype della scale dell'activation-quant, usata da `act_quant`. |

### Helper diagnostici

- `ModelConfig.summary` — dump human-readable di ogni campo, usato da
  `--print-config`.
- `ModelConfig.inferred(from:)` — corregge `n_layers`, `vocab_size`,
  `dim`, `q_lora_rank`, `n_heads`, `o_lora_rank`, `moe_inter_dim` e
  `index_n_heads` dalle shape effettive dei tensori quando il
  `config.json` su disco è incompleto o stale.
- `ModelConfig.projectedKVCacheBytes` — bound grossolano sulla memoria
  che la KV cache occuperà ai valori `(max_seq_len, max_batch_size)`
  scelti. Il loader rifiuta in anticipo se sforerebbe il budget.
- `ModelConfig.compressRatioLCM` — minimo comune multiplo dei ratio
  non-zero. Usato dal KV-rewind per imporre una `pos` allineata alla
  window (vedi §10.3).
- `ModelConfig.nopeHeadDim` — derivato come `headDim - ropeHeadDim`.

---

## 3. Forward pass a colpo d'occhio

```
input_ids: [[Int]]  (outer = batch, inner = seqlen, tutti uguali)
        │
        ▼
ParallelEmbedding.lookup     →  h: [B·S, dim] f32
        │
        ▼
hc_expand_f32 kernel         →  h: [B, S, hc_mult, dim]
        │
        ▼
for layer i in 0 ..< n_layers:                     ┐
    block_i(h, start_pos, input_ids)               │  Block:
        │                                          │   HC.pre  → attn_norm → MLA → HC.post
        ▼                                          │   HC.pre  → ffn_norm  → MoE → HC.post
   h: [B, S, hc_mult, dim]                         ┘
        │
        ▼
ParallelHead(h, hc_head_*, norm)
        │  collapse hc → norm → ultima posizione → matmul lm_head
        ▼
logits: [B, vocab_size] f32   →  Sampler.sample(...)
```

Ogni main block gira sul proprio command buffer (`cmd.commit +
waitUntilCompleted`) così che il loader streaming-pool possa ruotare
uno shard alla volta e i trace numerici per layer (sotto
`--trace-norms`) abbiano un boundary pulito.

La decode path single-token collassa `S = 1` e aggiunge una scrittura
ring-buffer nella KV cache della sliding window; il prefill con `S >
1` gira l'intera sequenza in un colpo solo. Entrambi i path
condividono lo stesso codice in `MLA.callAsFunction` e
`Compressor.callAsFunction`.

Per una vista per-componente e per-linea della decode path apri
`Sources/DeepSeekKit/Model.swift:214` (`Transformer.forward`).

---

## 4. I componenti

### 4.1 Tokenizer ed embedding

Il tokenizer è una preoccupazione separata (vive in
`Sources/DeepSeekKit/{BPETokenizer,SentencePieceTokenizer,WordPieceTokenizer}.swift`
ed è documentato in [`MODULES.md`](MODULES.md)). L'engine vede solo gli
id interi che il tokenizer emette.

La tabella di embedding è una singola matrice `[vocab, dim]` in F32 o
BF16. `ParallelEmbedding.lookup` (`Sources/DeepSeekKit/Model.swift:6`)
legge una riga per ogni id con il kernel `embed_lookup_f32` o
`embed_lookup_bf16_to_f32` e ritorna un Tensor F32 `[N, dim]` con `N =
batch · seqlen`.

La classe si chiama `ParallelEmbedding` per mantenere il nome Python;
questo port è single-rank, quindi non c'è vero sharding.

### 4.2 Hyper-Connections (HC)

Il residual classico è `x = x + sublayer(norm(x))`. HC lo sostituisce
con un mixing su `hc_mult = 4` copie parallele dello stato nascosto,
con la matrice di mixing doppiamente stocastica ottenuta via
iterazioni Sinkhorn. Conseguenza: ogni block ha *due* passaggi HC —
uno che avvolge l'attenzione, uno che avvolge l'FFN — e il residual
stream vive come `[B, S, hc_mult, dim]` invece di `[B, S, dim]`.

`HyperConnections` (`Sources/DeepSeekKit/Layers/HyperConnections.swift:7`)
ha due fasi:

**`pre(x, hcFn, hcScale, hcBase)` →  `(y, post, comb)`**

1. `rsqrt(mean(x²) + normEps)` per riga sull'asse appiattito `[N,
   hc·dim]` (kernel `rsqrt_mean_square_f32`).
2. `mixes = x_flat @ hcFnᵀ` (linear F32, `hcFn: [(2+hc)·hc, hc·dim]`).
3. `mixes *= rsqrt` (broadcast sulla riga, `broadcast_row_mul_f32`).
4. Split Sinkhorn (`HCSinkhorn.split`, kernel `hc_split_sinkhorn_f32`):
   reshape di `mixes` in un tile `[N, (2+hc), hc]`, applica
   `sigmoid(... · hcScale + hcBase) + hcEps`, poi itera
   `hc_sinkhorn_iters` round di normalizzazione colonna/riga per
   produrre una matrice `comb` doppiamente stocastica; le prime due
   "slab" si separano in `pre` e `post`.
5. `y[n, d] = Σ_h pre[n, h] · x[n, h, d]` — kernel `hc_collapse_f32`.
   Questo è l'"input che il sublayer vede", una vista `[N, dim]` delle
   copie `hc` dopo il collapse pesato.

**`post(out, residual, post, comb)` → `[N, hc, dim]`**

Kernel `hc_post_compose_f32`:

```
y[n, j, d] = post[n, j] · out[n, d]
           + Σ_k comb[n, k, j] · residual[n, k, d]
```

Il `out` del sublayer viene riportato sulle `hc` copie pesato per
`post`, e ogni copia riceve un contributo mixato Sinkhorn dal residual
pre-sublayer via `comb`.

I parametri HC caricati per ogni block sono sei tensori:
`hc_attn_fn / hc_attn_base / hc_attn_scale` (per il sublayer
attenzione) e `hc_ffn_fn / hc_ffn_base / hc_ffn_scale` (per il
sublayer FFN).

`ParallelHead` finale usa un **collapse sigmoid-only** più semplice
(niente Sinkhorn): le gating `hcFn / hcBase / hcScale` sono lette da
`hc_head_fn / hc_head_base / hc_head_scale` e usate direttamente per
collassare `[B, S, hc, dim]` → `[B, S, dim]` prima della matmul LM
head.

### 4.3 RMSNorm

`RMSNorm(weight, eps)` (`Sources/DeepSeekKit/Layers/RMSNorm.swift:4`):

```
y[r, d] = x[r, d] · rsqrt(mean(x[r]²) + eps) · weight[d]
```

Il kernel viene in due varianti scelte dal dtype del gain:
`rmsnorm_f32` (gain F32) e `rmsnorm_bf16w_f32` (gain BF16, dispatchato
quando il loader ha caricato un peso BF16 da un checkpoint HF nativo).

`eps = norm_eps = 1e-6` ovunque appaia.

Il modello usa RMSNorm in sette punti per block: `attn_norm`, `q_norm`
(sull'intermedio low-rank Q), `kv_norm` (sulla proiezione KV),
`ffn_norm`, il `norm` del Compressor (sull'output pooled), più il
`norm` finale prima della LM head. C'è anche una rsqrt-by-row "calda"
dentro sia il path Q di MLA che HC.pre — quelle usano direttamente il
kernel `rsqrt_mean_square_f32` (senza moltiplicare un gain),
producendo solo il fattore inverso che poi viene broadcast.

### 4.4 Multi-head Latent Attention (MLA)

`MLA` (`Sources/DeepSeekKit/Layers/Attention.swift:16`) è la variante
di attenzione di V4. Le differenze rispetto all'attenzione multi-head
da manuale:

- **Q è low-rank**: `wq_a: dim → q_lora_rank` → `q_norm` → `wq_b:
  q_lora_rank → n_heads·head_dim`. Più una rsqrt re-norm per testa
  (scaling inverse-sqrt, senza gain appreso), così che il vettore di
  ogni testa finisca con magnitudine limitata.
- **KV è condivisa, single-head**: una singola matrice `wkv: dim →
  head_dim`. Tutte le teste leggono la stessa K e V. Questa è la
  "latent" di MLA — la capacità KV non scala con `n_heads`.
- **O è low-rank grouped**: `nHeads · headDim` reshapato come
  `[nGroups, nHeads·headDim / nGroups]`, proiettato a `[nGroups,
  oLoraRank]` via `wo_a`, poi `nGroups · oLoraRank → dim` via `wo_b`.
  Implementato come einsum esplicito (`Einsum.bsgdGrd`) seguito da una
  matmul `wo_b`.
- **Attention sink per testa**: uno scalare appreso `attn_sink[h]`
  viene piegato nel denominatore del softmax (vedi §4.5).
- **Nessuna correzione mscale del softmax**: `softmax_scale =
  head_dim^(-0.5)`, inverse-sqrt liscio. La correzione `mscale·mscale`
  di V3 *non* è applicata — provare a importarla ha peggiorato gli
  output di V4.

Step-by-step (decode-style; il prefill segue lo stesso path ma con più
righe di sequenza):

1. `xFlat = x.reshape([B·S, dim])`.
2. **Path Q**:
   - `qrFlat = q_norm(wq_a(xFlat))`  → `[B·S, q_lora_rank]`.
   - `q = wq_b(qrFlat).reshape([B·S, nHeads, headDim])`.
   - `q *= rsqrt(mean(q²) + eps)` sull'asse headDim (re-norm per testa).
   - **RoPE** sul tail `ropeHeadDim` di ogni testa (vedi §4.6).
3. **Path KV**:
   - `kvFlat = kv_norm(wkv(xFlat))` → `[B·S, headDim]` (single-head).
   - RoPE sul tail `ropeHeadDim`.
   - **FP8 QAT noise** sui primi `nopeHeadDim` (le dim non-RoPE):
     `ActQuant.partialInplaceQuant(..., blockSize=64)` rispecchia
     `act_quant(kv[..., :-rd], 64, ..., True)` a `model.py:506`. Senza
     questo step le dim nope di KV escono dal range FP8 nei layer
     profondi, gli score di attenzione crescono e il residual stream
     si amplifica fuori controllo (osservato: norma L2 layer 0 → 42 da
     75 a 615 000 prima del ripristino del QAT noise).
4. **Costruzione degli indici top-k**: la sparse attention legge solo
   `kWin + kComp` posizioni KV per query. `AttnIndicesGPU.window`
   riempie la slice di window (sempre presente); quando
   `compressRatio > 0`, l'Indexer (`ratio == 4`) o
   `AttnIndicesGPU.compressedDeterministic` (`ratio == 128`) riempie
   la slice compressa.
5. **Scrittura della KV cache**:
   - Decode (`startPos > 0`, `seqlen == 1`): scrittura single-row
     ring-buffer `kvCache[:B, startPos % windowSize] = kv[:, 0]`.
   - Prefill, `seqlen ≤ windowSize`: fill contiguo dalla riga 0.
   - Prefill, `seqlen > windowSize`: tieni solo le ultime `windowSize`
     righe, con cutoff/wrap così il ring finisce a `(S - 1) %
     windowSize`.
6. **Compressor** gira per side-effect: aggiorna il rolling `kvState /
   scoreState` e può emettere un nuovo token compresso nella slice di
   coda della KV cache (`§4.7`). Il Tensor `kvCache` del Compressor è
   un *alias* (offset diverso, stesso `MTLBuffer`) della trailing
   slice del `kvCache` di MLA, così la scrittura è visibile a MLA
   senza copia.
7. **Sparse attention** — `SparseAttention.apply(q, kvFull, sink,
   topkIdxs, scale)` — vedi §4.5.
8. **RoPE inverso** sull'output `o`, così il riferimento ruotato viene
   annullato prima della proiezione di output (allenata su output
   non ruotati).
9. **Output grouped**:
   - `o.reshape([B, S, nGroups, nHeads·headDim / nGroups])`.
   - `oR = Einsum.bsgdGrd(o, woA: woA.weight.reshape([nGroups, oLoraRank,
     perGroupD]))` → `[B, S, nGroups, oLoraRank]`. Se `woA` è FP8 su
     disco il kernel einsum dequantizza inline con `woA.scale`; per
     modelli INT-quantizzati o BF16-fusi il path è automatico.
   - `result = wo_b(oR.reshape([B·S, nGroups·oLoraRank])).reshape([B, S, dim])`.

`MLA.callAsFunction` prende il command buffer come `inout`: quando
l'indexer è attivo bisogna committare e attendere in un punto per
leggere output GPU su host, e al ritorno consegna un command buffer
fresco.

### 4.5 Sliding-window sparse attention

`SparseAttention.apply`
(`Sources/DeepSeekKit/Layers/SparseAttention.swift:13`) esegue
online-softmax stile FlashAttention. Un thread per `(b, m, h)`:

- Itera sui `K = kWin + kComp` indici top-k per questa query.
- Gather della riga KV corrispondente da `kvFull[:, idx, :]`.
- Calcola `score = q · kv * scale` (con `scale = head_dim^(-0.5)`).
- Online-softmax accumula `acc = Σᵢ e^(scoreᵢ - max) · kvᵢ`,
  riscalando sugli update di `max`.
- Dopo il loop, piega `attn_sink[h]` nel denominatore: `sumExp +=
  exp(sink[h] - sMax)`.
- Scrive `o[b, m, h, :] = acc / sumExp`.

Il trucco del sink viene dal paper V4: dà al softmax una "null
position" appresa che assorbe massa di probabilità quando nessun token
reale è un buon match. Il modello può scegliere di non attendere a
nulla.

Il meccanismo "top-k" è ciò che rende l'attenzione sparse:

- La slice di window è `topkIdxs[b, s, 0..kWin-1] = [windowStart + 0,
  ..., windowStart + kWin - 1]`. Durante la decode lo start è
  `startPos % windowSize` (ring wrap); durante il prefill è 0. Gli
  indici che punterebbero oltre il prompt reale sono paddati a `-1`,
  che il kernel salta.
- La slice compressa è:
  - **Dall'Indexer** (`ratio == 4`): l'Indexer scora ogni posizione
    compressa contro la query corrente e ritorna i top `indexTopk =
    512`. Heavy-learned.
  - **Deterministico** (`ratio == 128`): il kernel
    `attn_compressed_indices_i32` produce `[compOffset, compOffset+1,
    ...]` coprendo i token compressi disponibili (capped da `endPos /
    ratio`).
- Per i layer con `ratio == 0` la slice compressa ha lunghezza zero —
  pura sliding-window.

### 4.6 RoPE (con correzione YaRN)

`RoPE(ropeHeadDim, freqs)`
(`Sources/DeepSeekKit/Layers/RoPE.swift:7`) applica rotazione in-place
alle ultime `ropeHeadDim` colonne di ogni testa di un tensor `[tokens,
heads, head_dim]`. Le prime `headDim - ropeHeadDim` colonne sono
intoccate (split "no-positional-encoding" / nope).

La tabella di freqs è precomputata da `YaRN.precomputeFreqsCis`
(`Sources/DeepSeekKit/YaRN.swift:11`):

1. Frequenze base `f_i = 1 / base^(2i / dim)` per `i = 0 ..
   ropeHeadDim/2 - 1`.
2. **Correzione YaRN** quando `originalSeqLen > 0`:
   - Calcola il range di correzione `[lo, hi]` da `betaFast` /
     `betaSlow`.
   - Per ogni `i`, fattore di ramp `s_i ∈ [0, 1]`, blend
     `f_i := f_i / factor · (1 - s_i) + f_i · s_i`.
   - Questo estrapola le frequenze di rotazione così che le posizioni
     più lunghe-del-training producano comunque rotazioni non
     collidenti.
3. Materializza `[seqlen][rope_dim/2][2]` di coppie (cos, sin) in F32.

Il Tensor `freqs` è per-layer perché ogni layer sceglie una di due
basi RoPE:

- Layer con `ratio > 0` usano `compress_rope_theta = 40 000` e
  `useYarn = true`.
- Layer con `ratio == 0` usano `rope_theta = 10 000` e `useYarn =
  false`.

L'Indexer riusa la stessa istanza RoPE (`Indexer.rope` è collegato al
`RoPE` del MLA parent al primo forward).

`RoPE.apply(_, startPos, inverse)`:
- `inverse = false` ruota per angolo `+freq[t]`.
- `inverse = true` ruota per `-freq[t]` (componente sin negata). Usato
  da MLA sull'output di attenzione per annullare la rotazione prima
  di `wo_a`.

### 4.7 Compressor

`Compressor` (`Sources/DeepSeekKit/Layers/Compressor.swift:15`) fa
pooling di `compressRatio` token consecutivi in una riga KV compressa.
Due modalità:

- **`ratio == 128`**: senza overlap. Ogni finestra di 128 token
  produce un token compresso.
- **`ratio == 4`**: overlap on. Ogni emit fonde la finestra precedente
  e quella corrente (`coff = 2`), dando boundary più lisci.

Parametri interni per layer (caricati sotto
`layers.<i>.attn.compressor.*`):

- `ape: [ratio, coff·head_dim]` — additive positional encoding per i
  pesi di pooling.
- `wkv: dim → coff·head_dim` — Linear che produce il contributo KV
  per-token.
- `wgate: dim → coff·head_dim` — Linear che produce lo score di
  pooling.
- `norm: RMSNorm(head_dim)` — normalizzazione post-pooling.

Il `kvCache` del Compressor *non* è un suo buffer — è una vista
zero-copy sulla trailing slice del `kvCache` parent MLA, collegata al
primo forward:

```swift
comp.kvCache = Tensor(shape: [B, compRows, headDim], dtype: .f32,
                      buffer: kvCache.buffer,
                      offset: kvCache.offset + win * bytesPerRow)
```

(`Sources/DeepSeekKit/Layers/Attention.swift:195`.) Simmetricamente, il
compressor interno dell'Indexer aliasa il `kvCache` dell'Indexer.

#### Prefill (`startPos == 0`)

Si scorre tutto il prompt in un colpo. Riferimento a `model.py:325`.

1. Proietta l'intero prompt tramite `wkv` e `wgate` → `[B·S,
   coff·head_dim]` ciascuno.
2. **Stashing dello state** così che un decode successivo che attraversa
   un boundary di compressione abbia disponibili i token del prompt:
   - Con overlap e `cutoff >= ratio`: copia `kv[cutoff-ratio:cutoff]` e
     `score[cutoff-ratio:cutoff] + ape` in `kvState / scoreState
     [:, :ratio]`.
   - Per ogni `remainder = S % ratio > 0`: copia il tail
     `kv[cutoff:]` e `score[cutoff:] + ape[:remainder]` negli slot di
     state opportuni (`[:, ratio:ratio+remainder]` con overlap;
     `[:, 0:remainder]` senza).
3. Se `numBlocks = S / ratio > 0`, reshape `kv[:cutoff]` e
   `score[:cutoff]` in `[B, numBlocks, ratio, coff·head_dim]`.
4. Broadcast-add `ape` sul tensor di score.
5. **Overlap transform** se `ratio == 4`: shuffle in `[B, numBlocks,
   2·ratio, head_dim]` così ogni block pooled vede anche le metà
   precedenti. Pad value `-inf` lato score, `0` lato KV.
6. `softmax` sull'asse del ratio.
7. Somma pesata: `pooled[b, n, d] = Σ_r kv[b, n, r, d] · score[b, n,
   r, d]`. Produce `[B, numBlocks, head_dim]`.
8. **Post-process** (vedi sotto).
9. Blit `result` in `self.kvCache[:B, :numBlocks]`.

#### Decode (`startPos > 0`)

Incrementale, singolo token. Riferimento a `model.py:343`.

1. Proietta il nuovo token tramite `wkv` e `wgate` → righe `[B,
   coff·head_dim]`.
2. Somma `ape[startPos % ratio]` alla riga di score.
3. Scrivi la nuova riga nel rolling state:
   - Overlap: slot `ratio + startPos % ratio`.
   - Senza overlap: slot `startPos % ratio`.
4. Se `(startPos + 1) % ratio != 0`: la finestra non è ancora piena,
   ritorna `nil`. Il caller lo tratta come "nessun nuovo token
   compresso a questo step".
5. Altrimenti:
   - **Path overlap**: costruisce `pooledKV` e `pooledScore` come
     `[B, 2·ratio, head_dim]` via `overlapConcat` — first-half-low
     (le prime `head_dim` colonne della prima metà dello state)
     concatenato con second-half-high (le ultime `head_dim` della
     seconda metà).
   - **Path senza overlap**: aliasa `kvState` / `scoreState`
     direttamente come `[B, ratio, head_dim]`.
6. `softmax` sull'asse ratio, somma pesata, ottieni il singolo
   token emesso `[B, 1, head_dim]`.
7. **State shift-down** (solo overlap): copia `state[:, ratio:]` in
   `state[:, :ratio]` così che lo slot della seconda metà diventi la
   prima per la window successiva.
8. Post-process.
9. Blit `result` in `self.kvCache[:B, startPos / ratio]`.

#### Post-process (condiviso)

```
result := norm(result)
RoPE.apply(result, startPos: <ratio-adjusted>, inverse: false)
if rotate:
    Hadamard.apply(result)           # in-place
    ActQuant(.fp4).quant(result, inplace: true)   # FP4 QAT noise (Indexer)
else:
    ActQuant.partialInplaceQuant(    # FP8 QAT noise su nope dims (MLA)
        result, colStart: 0, colEnd: nopeHeadDim,
        blockSize: actBlockSizeFP8KVNope=64)
```

`rotate = true` è il path del Compressor posseduto dall'Indexer (FP4
con rotazione Hadamard); `rotate = false` è il path lato MLA (FP8 QAT
sulle dim nope).

#### Rewind dello stato

`Compressor.rewindStateTo(pos:)` resetta il rolling state a un confine
window pulito. Necessario per ogni KV rewind attraverso il modello:

- Ritorna `false` se `pos % compressRatio != 0` — lo state a metà
  window non è ricostruibile senza ri-eseguire il prompt.
- Al successo, azzera `kvState` e setta `scoreState` a
  `-Float.infinity` (così gli slot inutilizzati contribuiscono 0 massa
  tramite softmax). Il `kvCache` principale resta intoccato.

Questa è metà del macchinario di resume cross-restart + KV-snapshot
delle delegazioni — vedi §10.

### 4.8 Indexer

`Indexer` (`Sources/DeepSeekKit/Layers/Indexer.swift:10`) è un
selettore top-k appreso, usato solo nei layer con `ratio == 4`. Dove
il Compressor collassa token, l'Indexer **sceglie quali posizioni
compresse** la sparse attention deve guardare.

Parametri:

- `wqB: q_lora_rank → index_n_heads · index_head_dim` — proiezione
  dall'intermedio low-rank Q di MLA alla query multi-head propria
  dell'Indexer.
- `weightsProj: dim → index_n_heads` — pesi di scoring per testa.
- `compressor: Compressor` — l'Indexer mantiene un **suo** Compressor
  (`rotate = true`, `head_dim = index_head_dim`), con i propri state
  buffer e il proprio kvCache. Stessa logica prefill/decode del
  Compressor di MLA ma con quant FP4 + Hadamard al post-process.
- `kvCache: [maxBatch, maxSeqLen/ratio, index_head_dim]` — distinto
  dal `kvCache` di MLA.

Forward (chiamato da MLA con l'intermedio low-rank Q `qr` condiviso):

1. `q = wq_b(qr) → [B, S, index_n_heads, index_head_dim]`.
2. **RoPE** sul tail rope di `q` (usando l'istanza RoPE condivisa con
   MLA).
3. **Rotazione Hadamard** sull'asse head dim di `q` (per testa,
   in-place).
4. **FP4 QAT noise** su `q` per rispecchiare il round-trip
   `fp4_act_quant` al training.
5. Esegue il Compressor interno sul layer `x` → stasha state e scrive
   un nuovo token compresso in `Indexer.kvCache` quando una window
   chiude.
6. `weights = weightsProj(x) * (softmaxScale * n_heads^(-0.5))`.
7. **Score**: `score = einsum("bshd,btd→bsht", q, kvCache[:T])` →
   `[B, S, index_n_heads, T]`, con `T = endPos / ratio`.
8. **Reduce**: `y[b, s, t] = Σ_h max(0, score[b, s, h, t]) · weights[b, s,
   h]`. `relu(score)` per testa pesato dai `weights` per testa. Con
   maschera causale al prefill (`startPos == 0`).
9. **Top-K**: `topkIdxs = TopK(y, k=min(index_topk, T))` → `[B, S, K]`.
10. **Post-process**: maschera slot invalidi (`-1`) e somma l'offset
    (`compOffset = isDecode ? windowSize : S`) così che gli indici
    ritornati siano posizioni assolute nella tabella merged
    `[window | compressed]` che la sparse attention legge.

`releaseCache()` / `restoreKVCacheBytes(...)` propri dell'Indexer
rispecchiano quelli di MLA: ARC libera il buffer; il restore riscrive
la cache da uno snapshot e riallinea l'alias interno del Compressor.

### 4.9 MoE feed-forward

Il sublayer FFN è un Mixture-of-Experts top-K più un shared expert
sempre attivo. Tre classi in `Sources/DeepSeekKit/Layers/MoE.swift`:

**`Gate(config, layerId, weight, bias, tid2eid)`** — routing top-K.

- `weight: dim → n_routed_experts` — Linear (tenuto in F32, vedi sotto).
- `bias: [n_routed_experts]?` — bias additivo opzionale sui logits
  prima dello scoring.
- `tid2eid: [vocab, top_k]?` — lookup token-id → expert-id per hash
  routing.

Due path:

- **Routing per score** (`layerId >= n_hash_layers`): il kernel
  `moe_gate` calcola `logits = x @ weight^T + bias`, applica la
  funzione di scoring (specializzata alla creazione della pipeline via
  la function constant `SCORE`: 0 = softmax, 1 = sigmoid, 2 =
  sqrtsoftplus), e sceglie i top `topK` expert per token. I pesi
  finali sono rinormalizzati per sommare a 1 e moltiplicati per
  `route_scale`.
- **Hash routing** (`layerId < n_hash_layers`): gli id expert per
  token vengono da `tid2eid[input_id, :]` (una tabella precomputata);
  il peso per-expert è `sqrt(softplus(logits[expert])) / Σ`,
  moltiplicato per `route_scale`. In passato questo branch usava un
  peso uniforme `1/topK`, che degradava silenziosamente i primi tre
  layer di V4-Flash sostituendo il gating appreso con una media piatta
  — corretto in `Sources/DeepSeekKit/Layers/MoE.swift:75`.

**Nota numerica critica**: il `Linear` del gate è costruito con
`castOutputToBF16: false`. Il riferimento `model.py:566` gira il gate
esplicitamente in F32:

```python
scores = linear(x.float(), self.weight.float())
```

Quantizzare i logits a BF16 (7 bit di mantissa) prima di
`sqrt(softplus) + topk` perturba quali expert vengono selezionati; su
V4-Flash quella perturbazione produce un'amplificazione 8.4× del
residual stream al primo layer score-routed (`= il primo layer dopo
n_hash_layers`). I layer hash-routed sono risparmiati perché i loro
indici vengono dalla tabella precomputata.

**`Expert(w1, w2, w3, swigluLimit)`** — singola SwiGLU FFN.

```
g = w1(x)
u = w3(x)
h = silu(g) · u                      # clamp opzionale via swigluLimit
y = w2(h)
```

Per V4-Flash i pesi degli expert vivono in **FP4** (E2M1) su disco,
packed two-per-byte, con scale group E8M0 (una scale per K-block
`[1, 32]`). Il kernel FP4 GEMM dequantizza inline; con
`--target-dtype bf16` il converter fonde FP4 + E8M0 → BF16 a tempo di
conversione.

`MoEFFN(gate, experts, shared)`:

1. Gate: `(weights, indices) = gate(x, inputIds)`.
2. Leggi indices e weights su host per costruire il piano di dispatch
   (`MoEDispatch.prepare`): per ogni expert routed, la lista di righe
   token assegnate.
3. Gather: pacchettizza i token per expert in `gathered: [T_total,
   dim]`.
4. Forward di ogni expert attivo sulla sua slice, scrivendo nell'offset
   giusto di `outs: [T_total, dim]`.
5. Scatter di ritorno in `y: [N, dim]` con somma pesata: `y[n, d] =
   Σ_assignments weight · outs[t, d]`.
6. Aggiungi il contributo dello shared expert: `y += shared(x)`.

Lo shared expert esiste una volta per layer e gira su *tutti* i token;
il suo output viene sommato in `y` indipendentemente dal routing.
V4-Flash ha `n_shared_experts = 1`.

### 4.10 Il block transformer (DecoderLayer)

`Block` (`Sources/DeepSeekKit/Layers/DecoderLayer.swift:6`) incolla
attenzione e FFN con i loro HC pre/post:

```
x : [B, S, hc, dim]

# Sublayer attenzione
(yA, postA, combA) = HC.pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)
yA = attn_norm(yA)
oA = attn(yA, start_pos)
x  = HC.post(oA, residual=x, post=postA, comb=combA)

# Sublayer FFN
(yF, postF, combF) = HC.pre(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
yF = ffn_norm(yF)
oF = ffn(yF, input_ids)
x  = HC.post(oF, residual=x, post=postF, comb=combF)

return x
```

Il block prende il command buffer come `inout`: MLA (quando l'indexer
è attivo) e MoE entrambi devono committare-e-attendere a metà strada
per leggere output GPU in memoria host, e consegnano un command buffer
fresco al ritorno. Il block continua a encodare nel buffer scambiato.

### 4.11 Multi-Token Prediction (MTP)

`MTPBlock` (`Sources/DeepSeekKit/Layers/MTPBlock.swift:21`) sta in
coda allo stack principale. Il suo scopo è il decoding speculativo:
dato lo stato nascosto corrente e l'embedding del *prossimo* token in
input, predire i logits di quel token. Se la predizione coincide con
la scelta del sampler al decode time, il forward successivo può
saltare un passaggio completo.

L'MTP block possiede:

- Un `Block` interno (stessa classe Block dello stack principale, con
  la propria attenzione + MoE).
- `e_proj`, `h_proj` — due proiezioni linear che fondono l'embedding
  del nuovo token con lo stato nascosto del block precedente.
- `enorm`, `hnorm` — RMSNorm di pre-proiezione.
- `norm` — RMSNorm finale prima della LM head.
- `hc_head_fn`, `hc_head_base`, `hc_head_scale` — parametri propri di
  collapse HC della head.
- Riferimenti non-possesori a `ParallelEmbedding` e `ParallelHead`
  condivisi.

Forward (rispecchia `model.py:756`):

```
e = enorm(embed(input_ids))           # [N, dim]
xN = hnorm(x.flatten(2)).reshape([N, hc, dim])
combined[N, hc, dim] = e_proj(e)[N, 1, dim] + h_proj(xN)[N, hc, dim]
after = inner_block(combined.reshape([B, S, hc, dim]), start_pos, input_ids)
logits = head(after, hc_head_fn, hc_head_scale, hc_head_base, norm)
```

I layer MTP contribuiscono a `Transformer.layers` indirettamente: le
loro entry `compress_ratio` vivono in coda a `compress_ratios`
(`compressRatios[nLayers .. nLayers+nMtpLayers-1]`). Il block MTP
**non** è ancora integrato nel decode loop della CLI oggi —
l'infrastruttura c'è ma `Sources/deepseek/main.swift` gira solo la
plain head. Vedi `docs/ROADMAP.md` per il piano di integrazione.

### 4.12 LM head (ParallelHead)

`ParallelHead` (`Sources/DeepSeekKit/Model.swift:45`) chiude il
forward pass. Il collapse HC più semplice del block è un gating
*sigmoid-only* (no Sinkhorn), usato una sola volta alla fine:

1. `rsqrt(mean(x²) + eps)` per riga sul flatten `hc·dim`.
2. `mixes = x_flat @ hc_head_fn^T` (linear F32).
3. `mixes *= rsqrt`.
4. `pre[n, h] = sigmoid(mixes[n, h] · hc_head_scale + hc_head_base[h]) +
   hc_eps`. Fatto host-side via broadcast scalare (operazione poco
   costosa e i valori servono subito).
5. `y[n, d] = Σ_h pre[n, h] · x[n, h, d]` — kernel `hc_collapse_f32`.
6. `y = norm(y)` — RMSNorm finale.
7. **Slice dell'ultima riga di sequenza per batch**: `last[b, :] = y[b, S-1,
   :]`. La LM head produce logits solo per l'ultima posizione.
8. `logits = last @ lm_head_weight^T` → `[B, vocab_size]` F32.

Il `Linear` della LM head ha `castOutputToBF16 = false` — perdere i 16
bit bassi di mantissa collasserebbe near-ties e snaturerebbe lo
scaling di temperatura a valle.

---

## 5. Il data flow completo al decode

Per ogni token generato, la CLI (`Sources/deepseek/main.swift`) chiama
`Transformer.forward(inputIds: [[id]], startPos: pos)` una volta.
Dentro:

1. **Embed lookup**: `flatIds = inputIds.flatMap { $0.map(Int32.init) }`
   → `embed.lookup(flatIds, in: cmd)` → `[1, dim]` F32.
2. **HC expand**: `hc_expand_f32` tile a `[1, hc_mult, dim]`. Il
   residual stream ora vive in shape `[B, S, hc_mult, dim]`.
3. **Hint di streaming**: per ogni layer K,
   `loader.ensureLayer(K)` (no-op fuori da `.streaming`) — page-in
   dello shard del layer prima di referenziare i suoi Tensor.
4. **Block K** gira sul proprio command buffer:
   - `HC.pre` → `attn_norm` → `MLA(start_pos)` → `HC.post`.
   - `HC.pre` → `ffn_norm` → `MoEFFN(input_ids)` → `HC.post`.
   - Dentro MLA: l'**alias del kvCache** del Compressor sulla trailing
     slice di MLA viene collegato la prima volta. Da lì, il path pure
     window o window+compressed top-k gira secondo `compressRatios[K]`.
   - Al decode, `MLA` scrive una riga in `kvCache[:, startPos %
     windowSize]` e il Compressor (se presente) accumula nel rolling
     state, possibilmente emettendo una riga compressa in
     `kvCache[:, windowSize + startPos / ratio]`.
5. `cmd.commit()` + `cmd.waitUntilCompleted()` tra block.
6. `loader.releaseLayer(K)` — in modalità `.streaming` segna le pagine
   del layer con `MADV_DONTNEED` così lo shard del layer successivo ha
   spazio.
7. **Head**: `head(x, hc_head_fn, hc_head_scale, hc_head_base, norm)`
   su un command buffer finale; commit + wait.
8. Il sampler sceglie il prossimo id dai logits `[1, vocab]`.
9. Ripeti con `startPos += 1`, alimentando il nuovo id appena
   campionato.

Il primo forward in una sessione è "prefill" — `startPos = 0`, `S =
lunghezza prompt`. Dopo che il prompt è stato digerito, il loop passa
alla decode single-token.

---

## 6. Dtype numerici per componente

Il modello è un mix fluido di dtype. Apple Silicon supporta nativamente
F32, F16, BF16 (Metal 3+), I32, I8; FP8 / FP4 / E8M0 non sono nativi e
vengono spacchettati in shader. Vedi [`DTYPES.md`](DTYPES.md) per i
layout bit-a-bit e la matematica di fusion del converter.

### Su disco

| Famiglia di tensor | Release HF nativo | Post-`--target-dtype bf16` (default) | Post-INT8 |
|---|---|---|---|
| Embedding | BF16 | BF16 | BF16 |
| Linear di attenzione (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`) | FP8-E4M3 + E8M0 scale per 128×128 | BF16 | INT8 + scale F16 per riga |
| Linear degli expert routed (`w1`, `w2`, `w3`) | FP4-E2M1 (packed) + E8M0 scale per row × 32-K-block | BF16 (4× disco!) | INT8 |
| Linear shared expert | FP8 | BF16 | INT8 |
| Gate MoE (`ffn.gate`) | F32 in codice, BF16 su disco | F32 | F32 |
| Parametri HC (`hc_*_fn`, `hc_*_base`, `hc_*_scale`) | F32 (piccoli) | F32 | F32 |
| Gain di RMSNorm (`*_norm.weight`) | BF16 | BF16 | BF16 |
| Compressor `wkv`, `wgate`, `ape` | BF16 / FP8 | BF16 | BF16 |
| Indexer `wqB`, `weightsProj` | FP8 | BF16 | INT8 |
| LM head (`lm_head.weight`) | BF16 | BF16 | BF16 |

### In transito (durante un forward)

| Tensor | Dtype |
|---|---|
| Output di embedding | F32 |
| Residual stream | F32 |
| Output di tutti gli RMSNorm | F32 |
| Output di Linear | F32 (BF16 round-trip via `castOutputToBF16: true` per tutto *tranne* il gate e la LM head) |
| Freqs RoPE | F32 |
| KV cache | F32 |
| `kvState` / `scoreState` del Compressor | F32 (scoreState inizializzato a `-Float.infinity`) |
| Input a GEMM FP8 / FP4 (activations) | Quantizzati al volo via `ActQuant` — FP8 block 128, FP4 block 32 |
| Q interno dell'Indexer (post-Hadamard) | F32 round-tripped attraverso FP4 |

### Noise quantization-aware (QAT)

DeepSeek-V4 è stato allenato con attivazioni FP8/FP4 su un paio di
path specifici, e l'engine di inference deve riprodurre lo stesso
round-trip noise in inferenza:

- **Dim nope KV di MLA**: `ActQuant.partialInplaceQuant(kv, 0,
  nopeHeadDim, blockSize=64)` — round-trip FP8, block size 64.
  Rispecchia `act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype,
  True)` a `model.py:506`.
- **Post-process del Compressor lato MLA**: stessa call sul token
  compresso emesso.
- **Q dell'Indexer**: `ActQuant(.fp4).quant(q, inplace: true)`
  rispecchia `fp4_act_quant(q, fp4_block_size, True)`.
- **Post-process del Compressor dell'Indexer**: quant FP4 + Hadamard.

Senza queste iniezioni di noise il residual stream si amplifica
nell'ordine di 1e5 nei layer profondi e gli output diventano spazzatura.

---

## 7. Convenzione di naming dei pesi

L'albero dei pesi che il loader percorre
(`Sources/DeepSeekKit/Assembly.swift:152`). I nomi sono scritti
esattamente come `WeightLoader.tryLoad(_:)` li cerca, con fallback
elencati dove il loader accetta più alias.

```
embed.weight                                 # oppure model.embed.weight
norm.weight
head.weight                                  # oppure lm_head.weight
hc_head_fn
hc_head_base
hc_head_scale

# Per ogni main layer i ∈ [0, n_layers)
layers.<i>.attn_norm.weight
layers.<i>.ffn_norm.weight

# MLA
layers.<i>.attn.wq_a.weight  + .scale | .weight_scale_inv      # quantizzato
layers.<i>.attn.q_norm.weight                                  # bf16/f32
layers.<i>.attn.wq_b.weight  + .scale | .weight_scale_inv
layers.<i>.attn.wkv.weight   + .scale | .weight_scale_inv
layers.<i>.attn.kv_norm.weight
layers.<i>.attn.wo_a.weight  + .scale | .weight_scale_inv      # bf16 dopo il converter
layers.<i>.attn.wo_b.weight  + .scale | .weight_scale_inv
layers.<i>.attn.attn_sink                                      # [n_heads] f32

# Compressor (quando compress_ratios[i] > 0)
layers.<i>.attn.compressor.ape                                 # [ratio, coff·head_dim]
layers.<i>.attn.compressor.wkv.weight     + .scale
layers.<i>.attn.compressor.wgate.weight   + .scale
layers.<i>.attn.compressor.norm.weight

# Indexer (quando compress_ratios[i] == 4)
layers.<i>.attn.indexer.wq_b.weight       + .scale
layers.<i>.attn.indexer.weights_proj.weight + .scale
layers.<i>.attn.indexer.compressor.ape
layers.<i>.attn.indexer.compressor.wkv.weight     + .scale
layers.<i>.attn.indexer.compressor.wgate.weight   + .scale
layers.<i>.attn.indexer.compressor.norm.weight

# MoE
layers.<i>.ffn.gate.weight                  # f32 (nessuna scale anche se gli altri sono quantizzati)
layers.<i>.ffn.gate.bias                    # opzionale, quando i >= n_hash_layers
layers.<i>.ffn.gate.tid2eid                 # [vocab, top_k] i32, quando i < n_hash_layers

layers.<i>.ffn.experts.<j>.w1.weight  + .scale     # j ∈ [0, n_routed_experts)
layers.<i>.ffn.experts.<j>.w2.weight  + .scale
layers.<i>.ffn.experts.<j>.w3.weight  + .scale

layers.<i>.ffn.shared_experts.w1.weight  + .scale
layers.<i>.ffn.shared_experts.w2.weight  + .scale
layers.<i>.ffn.shared_experts.w3.weight  + .scale

# Hyper-Connections (un set per sublayer)
layers.<i>.hc_attn_fn        # [(2+hc)·hc, hc·dim] f32
layers.<i>.hc_attn_base      # [(2+hc)·hc]         f32
layers.<i>.hc_attn_scale     # [3]                 f32
layers.<i>.hc_ffn_fn
layers.<i>.hc_ffn_base
layers.<i>.hc_ffn_scale
```

I **block MTP** (quando `nMtpLayers > 0`) vivono sotto lo stesso
albero `layers.*` agli indici `[n_layers, n_layers + n_mtp_layers)`.
Hanno la stessa substruttura MLA + MoE più i tensori MTP-specifici:
`mtp.<k>.e_proj.weight`, `mtp.<k>.h_proj.weight`,
`mtp.<k>.enorm.weight`, `mtp.<k>.hnorm.weight`, `mtp.<k>.norm.weight`,
e i propri `hc_head_fn`, `hc_head_base`, `hc_head_scale`.

Il loader è **indulgente**: qualsiasi tensor che non trova viene
riempito con init random via `MiniRNG`
(`Sources/DeepSeekKit/Assembly.swift:502`) e riportato su stderr alla
fine. Questo permette a un checkpoint parzialmente convertito o
prunato di produrre comunque un forward — utile in fase di porting,
debug del loader, o iterazione sulle shape di layer prima di un
release definitivo.

### Stranezze di storage FP4 / FP8

- **Gli expert routed FP4** sono salvati come **byte `i8` grezzi** nel
  release HuggingFace (safetensors non ha dtype FP4 nativo). Il loader
  riconosce il pattern (`.experts.` nel nome + dtype `.i8`) e
  reinterpreta come `.fp4E2M1` con l'ultima dim raddoppiata
  (`Sources/DeepSeekKit/Assembly.swift:472`).
- **Nomi della scale companion**: `weight_scale_inv` sul release HF
  nativo vs `scale` post-converter. Il loader prova entrambi
  (`Sources/DeepSeekKit/Assembly.swift:492`).
- **Tabella hash routing (`tid2eid`)** è tipicamente salvata come
  `i64` su disco; il loader fa cast a `i32` via
  `AssemblyHelpers.castIntToI32` per il consumo downstream.

---

## 8. KV cache: layout, ciclo di vita, snapshot

La KV cache è l'unica grande allocazione dinamica che il modello
possiede. Deve restare residente in memoria unificata (viene letta e
scritta a ogni token, quindi memory-mapparla da disco non è
un'opzione), quindi la sua dimensione moltiplicata per `max_seq_len` ×
`max_batch_size` è il numero "entra?" più importante dopo i pesi
stessi.

### 8.1 Shape per layer

Per un layer con `compress_ratios[i] = r`:

```
shape kvCache = [maxBatchSize, kvCacheRows, headDim]   dtype f32
kvCacheRows   = windowSize + (r > 0 ? maxSeqLen / r : 0)
```

Per i default V4-Flash (`maxBatchSize=4`, `windowSize=128`,
`maxSeqLen=4096`, `headDim=512`):

| Ratio del layer | Righe | Byte per layer |
|---|---|---|
| 0 (pure window) | 128 | 4 × 128 × 512 × 4 B = ~1 MB |
| 4 (window + compresso) | 128 + 1024 = 1152 | ~9 MB |
| 128 (window + compresso pesante) | 128 + 32 = 160 | ~1,3 MB |

Più il `kvCache` proprio dell'Indexer (solo nei layer con `ratio == 4`)
di shape `[maxBatchSize, maxSeqLen/ratio, indexHeadDim]`, più gli
state buffer del Compressor `[maxBatchSize, coff·ratio,
coff·headDim]`. Il totale sui 7 layer di V4-Flash sta sotto qualche
GB.

Per inference long-context (`maxSeqLen = 1 M`), scala linearmente
nella slice `compressed` — `maxSeqLen / 4 = 256 K` righe per layer con
`ratio=4`, ~512 MB per layer. Ecco perché
`ModelConfig.projectedKVCacheBytes` rifiuta in anticipo se la
richiesta supera il budget di memoria del sistema.

### 8.2 Layout dentro ogni slot di batch

```
kvCache[b]:  [ window_size righe       ][ compress_rows                ]
             [ ring sliding-window     ][ compressor.kvCache (alias)   ]
              ↑                          ↑
              MLA scrive qui ad ogni     Compressor scrive qui quando
              decode step (`pos%win`)    emette un nuovo token compresso
                                         (`pos // ratio`)
```

Compressor e MLA scrivono regioni non-sovrapposte dello stesso buffer
(offset diversi). MLA legge l'intero buffer al tempo dell'attenzione,
con gli indici top-k che gli dicono quali slot caricare davvero.

### 8.3 Sito di allocazione

`Sources/DeepSeekKit/Assembly.swift:336` è l'unico posto dove la KV
cache viene allocata:

```swift
let kvCacheRows = config.windowSize +
    (ratio > 0 ? config.maxSeqLen / ratio : 0)
let kvCacheShape = [config.maxBatchSize, max(kvCacheRows, 1),
                    config.headDim]
let kvCache = kvCacheFile != nil
    ? kvFile.tensor(at: off.attnKVCache, shape: kvCacheShape, dtype: .f32)
    : Tensor.empty(shape: kvCacheShape, dtype: .f32)
```

Il path `kvCacheFile` è un layer opzionale di persistenza
cross-restart (`Sources/DeepSeekKit/KVCacheFile.swift`): quando
fornito, l'allocazione è una slice zero-copy di un file di backing
mmappato invece di un fresh MTLBuffer. La chat surface lo usa per
mantenere lo stato KV di una chat attraverso un quit-and-relaunch.

### 8.4 Ciclo di vita

- **Lazy alloc al primo forward**: `MLA.ensureKVCache()` /
  `Indexer.ensureKVCache()` / `Compressor.ensureKVState()`
  ri-allocano dalla shape salvata se la call precedente ha rilasciato
  il buffer. Questo rende `releaseCache()` cheap e non-distruttivo.
- **Release per-layer**: `releaseCache()` droppa la cache + il rolling
  state del Compressor + la cache dell'Indexer. ARC libera i
  `MTLBuffer` e le pagine di memoria unificata tornano al sistema.
- **Release Transformer-wide**: `Transformer.releaseCache()`
  (`Sources/DeepSeekKit/Model.swift:314`) percorre ogni Block e ogni
  MTP block e chiama il loro `releaseCache()`. Utile tra prompt
  scollegati o sotto memory pressure.
- **Restore da snapshot**: `MLA.restoreKVCacheBytes(shape:, dtype:,
  bytes:)` ri-alloca da un blob Data; `Indexer.restoreKVCacheBytes` è
  simmetrico e aggiuntivamente riallinea l'alias del Compressor
  interno al buffer appena ripristinato (altrimenti il prossimo
  forward leggerebbe dati stale).

### 8.5 Rewind

Per il prefill incrementale — quando l'utente edita l'ultimo messaggio
e il nuovo prompt condivide un prefisso lungo con il precedente —
l'engine può rewindare lo stato KV a una position `P` e fare prefill
solo del delta, invece di un cold-prefill dell'intero nuovo prompt.

`Transformer.rewindKVTo(pos:)` (`Sources/DeepSeekKit/Model.swift:342`):

- Ritorna `true` se ogni `MLA.rewindKVTo` e ogni rewind del compressor
  dell'Indexer ha avuto successo.
- Il caller deve arrotondare `pos` in basso al multiplo di
  `compressRatioLCM` (= 128 per i ratio V4 di default), così il rewind
  è allineato a window per tutti i compressor insieme.
- In caso di fallimento (o quando l'alignment non è onorabile), il
  fallback safe è `releaseCache()` + cold prefill da `startPos = 0`.

Perché funziona: i layer ratio-0 non hanno rolling state da
resettare, il ring buffer si auto-overwrite al prossimo forward. I
layer con compressor hanno bisogno che `kvState` / `scoreState` siano
azzerati a un confine di window pulito; gli entry alle position `[0,
pos)` nel `kvCache` principale restano validi (sono il prefisso
preservato). Gli entry oltre `pos` saranno sovrascritti dal forward
successivo.

### 8.6 Snapshot/restore (delegazione sub-agent)

Quando l'app desktop delega un sub-task a un altro agente, fa lo
snapshot dello stato KV corrente del modello, gira il sub-agent (che
muta la cache), poi ripristina la cache originale. Questo evita di
pagare un cold re-prefill al ritorno.

L'implementazione vive in `Sources/DeepSeekKit/Model+KVSnapshot.swift`
(byte di cache MLA + Compressor + Indexer + state serializzati in un
`KVCacheSnapshot`), e il lato InferenceService collega
`beginDelegation()` / `endDelegation()` (vedi
[`ARCHITECTURE.md`](ARCHITECTURE.md#agents-delegation-kv-snapshots)).

Lo snapshot è una copia "congelata"; il modello in esecuzione
continua a mutare la propria cache dopo che la call ritorna l'id dello
snapshot. La call di restore riscrive i byte salvati in buffer
nuovamente allocati e riallinea gli alias del Compressor / Indexer.

---

## 9. Qualche numero su V4-Flash

Per il config V4-Flash di default (`n_layers = 7`, `dim = 4096`,
`n_routed_experts = 256` in produzione, `n_activated_experts = 2`,
`moe_inter_dim = 4096`, `n_heads = 64`, `head_dim = 512`):

### Conteggio parametri per layer

| Tensor | Shape | Byte FP4/FP8 | Byte BF16 |
|---|---|---|---|
| `attn.wq_a` | [4096, 1024] | 4 MB | 8 MB |
| `attn.wq_b` | [1024, 64·512=32768] | 32 MB | 64 MB |
| `attn.wkv` | [4096, 512] | 2 MB | 4 MB |
| `attn.wo_a` | [4096·512/8, 8·1024=8192] | sempre BF16 | 64 MB |
| `attn.wo_b` | [8·1024, 4096] | 8 MB | 16 MB |
| `attn.attn_sink` | [64] | 256 B | 256 B |
| Tutto compressor / indexer per-layer | varia | ~10–20 MB | ~20–40 MB |
| Un expert (`w1+w2+w3`) | 3 × [4096, 4096] | 12 MB (FP4) | 96 MB (BF16) |
| 256 expert | — | ~3 GB (FP4) | ~24 GB (BF16) |
| Shared expert | — | 48 MB | 96 MB |
| Gate (`ffn.gate.weight`) | [256, 4096] | 1 MB | (sempre F32 in codice) |
| `attn_norm` + `ffn_norm` + …norms | — | decine di KB | — |
| Parametri HC per block | 6 piccoli | qualche KB | qualche KB |

Un block V4-Flash con tutti i 256 expert in FP4 + attenzione in FP8
sta intorno ai **3-4 GB su disco**. Con 7 block più MTP e
embed/head, fa ~28 GB — ma il checkpoint rilasciato è **~142 GB**
perché il V4-Flash completo ha più layer (il default del campo
`n_layers = 7` è un toy default; il config reale del release porta il
vero valore).

### Parametri attivi per token

Con `n_activated_experts = 2 + n_shared_experts = 1` di 256 expert
attivi per token, il workload MoE per token è `3 / 256 ≈ 1,2%` del
peso totale degli expert. Da lì il titolo "284 B parametri, 13 B
attivati".

### Proiezione KV cache (config di default)

`ModelConfig.projectedKVCacheBytes`:

```
per ogni layer i con ratio r:
    cacheRows = (r > 0) ? windowSize + maxSeqLen / r : windowSize
    + attention kvCache:  maxBatchSize · cacheRows · headDim · 4 byte
    + indexer kvCache (solo se r == 4): stessa shape
    + compressor kvState (solo se r > 0):
        hcMult · ratio · maxBatchSize · headDim · 4 byte
```

Per 7 layer ai default: ~30-60 MB totali. Per long-context
(`max_seq_len = 1 M`): ~decine di GB. Il loader rifiuta in anticipo
quando questo eccede `SystemProbe.effectiveProcessBudget()` così il
processo non crasha a metà allocazione.

---

## 10. Concerne operative

### 10.1 Strategie del loader

Il loader sceglie una delle tre strategie in base alla RAM disponibile
(`Sources/DeepSeekKit/LoadStrategy.swift`):

- **`preload`** (≥ 192 GB RAM): ogni shard slurpato in memoria upfront.
  Steady-state più veloce, cold-start più costoso.
- **`mmap`** (32-192 GB): ogni shard mmappato, l'OS pagina i pesi on
  demand. Il primo forward triggera ~13 GB di letture sequenziali SSD
  (~2 s su SSD Apple a 7 GB/s).
- **`streaming`** (16-32 GB): uno shard di un layer alla volta. Il
  `Transformer.forward(...)` chiama `loader.ensureLayer(K)` prima che
  il block K giri e `loader.releaseLayer(K)` dopo, limitando il
  working set a "il layer che la GPU sta leggendo ora".

Vedi [`MEMORY.md`](MEMORY.md) per i dettagli completi di mmap e le
stime di footprint per fase.

### 10.2 Streaming e commit per-layer

La scelta di committare il command buffer di ogni block separatamente
(invece di batchare l'intero forward in un singolo buffer) è un
trade-off deliberato:

- ✅ Il loader streaming-pool può ruotare gli shard layer per layer
  (working set bounded).
- ✅ I trace numerici per-layer sotto `--trace-norms` hanno un punto
  di sync pulito.
- ❌ Si perdono alcune opportunità di concorrenza GPU.

Per la taglia di V4-Flash su Apple Silicon questo è il bilanciamento
giusto: il modello è prevalentemente memory-bound, l'overhead del
commit per-layer è piccolo rispetto al costo del GEMM per-layer, e
l'alternativa (un grosso buffer) o OOMmerebbe in streaming o
bloccherebbe finché l'intero forward non è completo prima che
qualunque progresso sia visibile al caller.

### 10.3 Alignment del rewind e LCM

`compressRatioLCM` è il minimo comune multiplo di ogni entry non-zero
di `compressRatios`. Per il default V4 `[0, 0, 4, 128, 4, 128, 4, 0]`,
è 128.

Perché `Transformer.rewindKVTo(pos:)` abbia successo, `pos` deve
essere multiplo di `compressRatioLCM`. Concretamente:

- Position 0 ✓ (reset completo).
- Position 128 ✓ (una window).
- Position 256 ✓.
- Position 100 ✗ — mid-window per `ratio == 128`.
- Position 132 ✗ — mid-window per `ratio == 4`.

Il caller è atteso ad arrotondare giù al multiplo LCM più vicino prima
della call. Se il valore arrotondato è troppo indietro per essere
utile, fallback su `releaseCache()` + cold prefill.

### 10.4 Init random per tensori mancanti

Il loader logga un warning e riempie pesi random F32 quando un tensor
nominato manca. Voluto: un checkpoint parzialmente prunato può
comunque girare un forward, che è utile in fase di porting, debug del
path del loader, o iterazione sulle shape dei layer contro un release
non ancora finalizzato.

In assenza di tutti i pesi, `Transformer.randomInit(config:)` costruisce
la stessa struttura con random F32 ovunque — usato dagli smoke test in
`Tests/DeepSeekKitTests/` per verificare la catena forward end-to-end
senza alcun checkpoint su disco.

---

## 11. Cross-walk Python ↔ Swift

La tabella completa linea-per-linea vive in
[`PYTHON-MAPPING.md`](PYTHON-MAPPING.md). Le voci rilevanti al modello
(ogni classe e ogni funzione major di
`Reference/inference/model.py`):

| Python | Linee | Swift |
|---|---|---|
| `ModelArgs` dataclass | 34–81 | `Sources/DeepSeekKit/Config.swift` (`ModelConfig`) |
| `ParallelEmbedding` | 83–105 | `Sources/DeepSeekKit/Model.swift` (`ParallelEmbedding`) |
| `linear` dispatch fn | 108–120 | `Sources/DeepSeekKit/Layers/Linear.swift` |
| `Linear` class | 123–152 | stesso file |
| `RMSNorm` | 183–196 | `Sources/DeepSeekKit/Layers/RMSNorm.swift` |
| `precompute_freqs_cis` (YaRN) | 199–229 | `Sources/DeepSeekKit/YaRN.swift` |
| `apply_rotary_emb` | 232–244 | `Sources/DeepSeekKit/Layers/RoPE.swift` + `Kernels/rope.metal` |
| `rotate_activation` (Hadamard) | 247–251 | `Sources/DeepSeekKit/Layers/Hadamard.swift` + `Kernels/hadamard.metal` |
| `get_window_topk_idxs` | 254–265 | `Sources/DeepSeekKit/Layers/AttentionIndices.swift` |
| `get_compress_topk_idxs` | 268–276 | stesso file |
| `Compressor` | 279–377 | `Sources/DeepSeekKit/Layers/Compressor.swift` |
| `Indexer` | 380–433 | `Sources/DeepSeekKit/Layers/Indexer.swift` |
| `Attention` (MLA) | 436–543 | `Sources/DeepSeekKit/Layers/Attention.swift` |
| `Gate` | 546–584 | `Sources/DeepSeekKit/Layers/MoE.swift` (`Gate`) |
| `Expert` | 587–606 | stesso file (`Expert`) |
| `MoE` | 609–644 | stesso file (`MoEFFN`) |
| `Block` | 647–700 | `Sources/DeepSeekKit/Layers/DecoderLayer.swift` |
| `Block.hc_pre / hc_post` | 673–686 | `Sources/DeepSeekKit/Layers/HyperConnections.swift` |
| `ParallelHead` | 703–735 | `Sources/DeepSeekKit/Model.swift` (`ParallelHead`) |
| `MTPBlock` | 738–766 | `Sources/DeepSeekKit/Layers/MTPBlock.swift` |
| `Transformer` | 769–809 | `Sources/DeepSeekKit/Model.swift` (`Transformer`) |

Apri prima il Python per rispondere a una domanda architetturale; il
Swift lo rispecchia. Dove il Swift sembra divergere — per esempio le
call esplicite di FP8/FP4 QAT noise, il pattern `inout cmd`, o i
guard di allineamento per-batch dentro il Compressor — i commenti nel
sorgente Swift spiegano perché.

---

## 12. Source map

Per topic, ecco dove aprire per primo.

| Topic | File |
|---|---|
| Config + alias di campo + proiezione budget KV | `Sources/DeepSeekKit/Config.swift` |
| Tensor + enum DType | `Sources/DeepSeekKit/Tensor.swift` |
| Embed + Head + Transformer | `Sources/DeepSeekKit/Model.swift` |
| Assembly (albero dei nomi, init random, path di load) | `Sources/DeepSeekKit/Assembly.swift` |
| Forward MLA | `Sources/DeepSeekKit/Layers/Attention.swift` |
| MoE gate + expert + scatter/gather | `Sources/DeepSeekKit/Layers/MoE.swift` + `MoEDispatch.swift` |
| Hyper-Connections (pre/post + Sinkhorn) | `Sources/DeepSeekKit/Layers/HyperConnections.swift` + `HCSinkhorn.swift` |
| Compressor (prefill, decode, overlap) | `Sources/DeepSeekKit/Layers/Compressor.swift` |
| Indexer (selettore top-k) | `Sources/DeepSeekKit/Layers/Indexer.swift` |
| Block (HC + composizione sublayer) | `Sources/DeepSeekKit/Layers/DecoderLayer.swift` |
| Multi-Token Prediction | `Sources/DeepSeekKit/Layers/MTPBlock.swift` |
| RoPE + YaRN | `Sources/DeepSeekKit/Layers/RoPE.swift` + `Sources/DeepSeekKit/YaRN.swift` |
| RMSNorm | `Sources/DeepSeekKit/Layers/RMSNorm.swift` |
| Sparse attention (stile FlashAttention) | `Sources/DeepSeekKit/Layers/SparseAttention.swift` + `Kernels/sparse_attn.metal` |
| Linear (dispatch BF16 / FP8 / FP4 / INT8/4/2) | `Sources/DeepSeekKit/Layers/Linear.swift` |
| Act quant FP8 / FP4 (QAT noise) | `Sources/DeepSeekKit/Layers/ActQuant.swift` + `Kernels/act_quant.metal` |
| KV cache snapshot/restore + delegazione | `Sources/DeepSeekKit/Model+KVSnapshot.swift` |
| Persistenza KV cache (cross-restart) | `Sources/DeepSeekKit/KVCacheFile.swift` + `KVCacheLayout.swift` |
| Loader + strategia di streaming | `Sources/DeepSeekKit/WeightLoader.swift` + `LoadStrategy.swift` + `StreamingPool.swift` |
| Riferimento per ogni kernel Metal | [`KERNELS.md`](KERNELS.md) |
| Indice per-file di tutto | [`MODULES.md`](MODULES.md) |

---

## 13. Limitazioni note e lavoro deferred

Tracciate in dettaglio in `TODO.md` (root del progetto) e
[`ROADMAP.md`](ROADMAP.md). Voci modello-specifiche a colpo d'occhio:

- **MTP non è collegato al decode loop.** La classe del block esiste,
  il loader la costruisce, ma `Sources/deepseek/main.swift` gira solo
  la plain LM head. Decoding speculativo richiederebbe: prendi i
  `logits` dell'MTP, controlla se la scelta del sampler matcha,
  condizionatamente salta il prossimo prefill.
- **QAT noise di `act_quant` sulle dim non-rope KV usa un kernel
  contiguo per partial-block.** Una vista Tensor strided permetterebbe
  allo stesso kernel di toccare slice arbitrarie senza copia. Tier 3 —
  deferred perché il path contiguo copre le call site reali.
- **Validazione numerica vs Python è hand-tested, non automatizzata.**
  Un harness esiste in `Tests/` ma una run completa di riferimento
  equivalente PyTorch + CUDA non è in CI (serve CUDA).
- **`cast_e2m1fn_to_e4m3fn` nel converter non è portato.** Il path
  `--expert-dtype fp8` fa fallback a relabel-only; il path di fusion
  BF16 (`--target-dtype bf16`, default) non ne ha bisogno.
- **Prefill multi-batch del Compressor** con `S % ratio != 0` è
  guardato: per ora i prompt con B > 1 devono avere `S` divisibile per
  `ratio`. Il path single-batch supporta qualsiasi remainder.

Per le ottimizzazioni perf (simdgroup matrix, tweak di layout fp8)
vedi [`PERFORMANCE.md`](PERFORMANCE.md).
