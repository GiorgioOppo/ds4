# Kernel GLM 5.2

Qui risiedono i wrapper Swift e gli oracoli scalari di correttezza per i kernel
in `metal/glm5.2/glm52.metal`. Le pipeline vengono richieste in modo lazy a
`MetalRuntime`; la semplice ispezione o il download di un modello GLM non
compila un set di pipeline separato né alloca buffer GPU.

Ogni nuovo kernel richiede un confronto CPU deterministico prima di poter
essere usato dal grafo GLM. I test devono includere il comportamento dei
pareggi e i valori limite dove questi influenzano l'output greedy.

I confini atomici attualmente validati sono:

- `GLM52Router`: routing top-8 sigmoid-più-bias con pesi normalizzati non
  distorti;
- `GLM52CompactKV`: posizionamento e conversione binary16 di righe cache-ready
  larghe 576 in piani separati KV-LoRA larghi 512 e K-RoPE larghi 64;
- `GLM52KVLoRANorm`: RMSNorm del prefisso KV-LoRA largo 512 con epsilon
  `1e-5`, preservando in F32 il payload K-RoPE grezzo largo 64;
- `GLM52IndexerKeyStore`: LayerNorm centrata compatibile con upstream più
  peso/bias affini, RoPE su `0..<64` e posizionamento in cache F16 con
  controllo di capacità;
- `GLM52IndexerScores`: scoring causale token-major di query 32x128 contro una
  cache condivisa binary16 di chiavi dell'indexer usando ReLU e pesi per head.
  La sua scala predefinita è il valore architetturale
  `1/sqrt(32*128) = 1/64`;
- `GLM52CompactAttention`: il core di attention in decode a stadi sulla cache
  compatta — `qk_lowrank` (assorbe la query nope larga 192 in `attn_k_b`),
  `attention_indexed` (softmax sulle righe selezionate, accumulata nel dominio
  KV-LoRA, con selezione limitata al top-2048 dell'architettura) e
  `value_project` (`attn_v_b`), ciascuno dispatchato in isolamento più un
  percorso di validazione concatenato confrontato con
  `GLM52AttentionCPUReference`. Le due proiezioni esistono anche come varianti
  Q8_0 che leggono direttamente i byte dei pesi GGUF (blocchi da 34 byte,
  scala fuori dal prodotto int8 come `dot_q8_0_row_f32_ref` upstream); la loro
  baseline è l'oracolo F32 sui pesi dequantizzati. `rotateTailByRowPosition`
  seleziona la semantica della coda in decode: la cache compatta mantiene code
  K-RoPE GREZZE e il kernel ruota ogni riga selezionata con la posizione
  assoluta della RIGA stessa al momento dell'attention (upstream
  `kernel_glm_attention_indexed_decode`); senza il flag la coda viene
  consumata così com'è memorizzata — il percorso delle fixture pre-ruotate;
- `GLM52IndexerTopK`: top-k discendente multi-blocco su righe di score
  token-major (il dispatch `ds4_gpu_indexer_topk_tensor`: argsort bitonico per
  blocco, poi merge iterativi con ricerca binaria), riutilizzando i kernel
  argsort DeepSeek vendorizzati. Le righe future causali arrivano come score
  `-INFINITY` e affondano in fondo; i pareggi seguono la rete bitonica, non la
  regola dell'indice più basso dell'oracolo;
- `GLM52RopeTail`: la RoPE lineare di GLM (coppie adiacenti, freq base 8e6,
  senza YaRN — upstream `rope_tail_ext_inplace` con costanti GLM) in entrambe
  le convenzioni di span: `glm52RopeTail` ruota gli ULTIMI `n_rot` di ogni
  head (le code delle query MLA), `glm52RopePrefix` ruota i PRIMI `n_rot` (le
  query dell'indexer — lì upstream forza `rot_offset = 0`). Oracolo CPU con
  theta iterativo (`rotate`/`rotatePrefix`), kernel con theta in forma chiusa,
  solo forward su GPU;
- primitive del grafo residente (i wrapper vivono accanto al grafo in
  `Execution/GLM52DecodeGraph.swift`): `kernel_glm52_rms_norm_f32`, una
  RMSNorm pesata a larghezza generica (riduzione float a 256 thread) per gli
  stadi attn_norm/q_a_norm del grafo di decode residente, e
  `kernel_glm52_store_compact_row_f16`, lo store F16 interleaved `[pos][576]`
  di righe compatte che rispecchia il layout letto dal kernel di attention
  indicizzata (lo store a due piani resta per le cache con forma upstream);
- `GLM52MoE`: kernel di validazione per gli stadi matvec dell'FFN quantizzato
  — SwiGLU gate/up fuso (peso di route sul mid, prima del down) e la
  proiezione down — che leggono righe Q8_0 e Q2_K/Q4_K/Q5_K/Q6_K esattamente
  come memorizzate nel GGUF, con un thread per riga di output e
  l'accoppiamento di elementi di riferimento. Q8_0 copre i blocchi densi,
  l'esperto condiviso e il matvec dell'output-head (`glm52FFNBlock`,
  `glm52OutputHeadLogits`); i K-quant coprono gli esperti instradati
  (`glm52RoutedFFN`). Baseline: `GLM52FFNCPUReference` sui pesi dequantizzati.
  Le famiglie ottimizzate per quant (slot/addr/batch mascherati) arriveranno
  più avanti accanto a queste.

L'input dello store compatto è intenzionalmente *cache-ready*: i suoi primi
512 valori sono già passati attraverso `GLM52KVLoRANorm` e i suoi ultimi 64
valori sono il payload K-RoPE intatto. Il nome GGUF `indexer.k_norm` è
potenzialmente fuorviante: il kernel di riferimento sottrae la media, quindi
il suo contratto è LayerNorm anziché RMSNorm. Analogamente, l'indexer GLM
ruota le prime 64 colonne; ciò segue deliberatamente upstream anche se un
helper generico lato query chiama la sua operazione `rope_tail`.

Questi wrapper allocano buffer condivisi per test deterministici; non rendono
eseguibile il backend GLM.
