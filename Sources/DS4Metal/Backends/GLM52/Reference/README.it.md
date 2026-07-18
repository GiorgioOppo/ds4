# Riferimento del layer GLM 5.2

Oracolo CPU di un layer GLM completo e della catena forward del primo token,
il port del percorso di riferimento F32 upstream
(`layer_glm_first_token_one_f32_ref`, `forward_glm_first_token_cpu_f32_ref`):
token singolo in posizione 0, nessuna cache, dove l'attenzione su una sola
riga visibile collassa nella value projection del token stesso (la softmax di
un singolo score è 1 — il percorso Q non viene mai valutato).

`GLM52LayerCPUReference` compone gli oracoli FFN/router fissati nella
struttura residuale pre-norm (`afterAttn = x + attn(x)`,
`out = afterAttn + ffn(rmsNorm(afterAttn))`), densa per i blocchi iniziali e
sparsa con il router integrato altrove. I layer sparsi recuperano SOLO gli
esperti selezionati dal router attraverso una closure provider — rispecchiando
lo streaming, dove gli esperti non selezionati non vengono mai letti.
`firstTokenForward` concatena i layer; la testa di output resta
`GLM52FFNCPUReference.outputHead`.

Questa è la baseline indipendente per il confronto tensore per tensore della
roadmap (passo 4): il futuro grafo GPU deve corrisponderle layer per layer,
dall'embedding ai logit. Deliberatamente non è un percorso di decode.

# Riferimento del decode GLM 5.2

`GLM52DecodeCPUReference` è l'oracolo CPU di un singolo passo di DECODE — il
port del cablaggio di `glm_graph_forward_token` upstream sul percorso di
attenzione indicizzata, composto dagli oracoli delle primitive fissate.
L'ordine di binding che codifica:

- `attn_norm = rmsNorm(cur)` alimenta ENTRAMBE le down-projection LoRA
  (`attn_q_a`, `attn_kv_a_mqa`); le cache vengono scritte PRIMA della
  selezione e dell'attenzione (lo store fuso upstream). La riga compatta
  memorizza il prefisso KV-LoRA normalizzato largo 512 più la coda K-RoPE
  GREZZA larga 64 — nessuna norm, nessuna rotazione;
- la chiave dell'indexer (solo layer full-indexer) proietta il residuo GREZZO,
  applica una LayerNorm centrata (eps `1e-6`, weight+bias), la RoPE sul
  PREFISSO con la posizione del token, e finisce nella key cache — anch'essa
  prima della selezione;
- la RoPE della coda della query (gli ultimi 64 di ogni testa larga 256)
  precede `qk_lowrank`; la query dell'indexer da `q_a_norm` ruota invece il
  suo PREFISSO. I pesi delle teste dell'indexer arrivano grezzi (senza
  softmax) da `indexer.proj` del residuo GREZZO;
- `visible = position + 1`: il token corrente partecipa sempre. Quando
  `visible <= topK` la selezione è l'intervallo di riempimento causale;
  altrimenti gli score su tutte le righe visibili alimentano il top-k con
  parità risolte a favore dell'indice più basso. I layer IndexShare riusano
  alla lettera l'ultima selezione full-indexer (righe assolute, nessun offset)
  e non possiedono mai chiavi dell'indexer;
- l'attenzione consuma le code grezze in cache e ruota ogni riga selezionata
  con la posizione assoluta della RIGA stessa al momento dell'attenzione
  (`GLM52AttentionCPUReference` con `rotateTailByRowPosition`).

Gli store in cache vengono arrotondati passando per IEEE binary16 perché le
cache reali sono F16 — numerica upstream, non un artefatto GPU. `decodeLayer`
condivide lo stadio FFN residuale con l'oracolo del primo token (`ffnStage`).
La geometria è parametrizzata così che i test possano ridurre ogni larghezza;
`GLM52DecodeGeometry.v5_2` fissa l'architettura reale (Q-LoRA 2048, nope 192,
indexer 32x128 rot 64, top-2048).
