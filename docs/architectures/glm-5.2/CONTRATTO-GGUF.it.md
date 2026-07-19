# Contratto GGUF GLM 5.2

Il backend accetta soltanto la geometria GLM 5.2 verificata. Non tenta di
interpretare varianti future con costanti compatibili “per caso”.

## Metadata essenziali

| Campo | Valore |
|---|---:|
| `general.architecture` | `glm-dsa` |
| `glm-dsa.block_count` | 79 |
| `glm-dsa.context_length` | 1.048.576 |
| `glm-dsa.embedding_length` | 6.144 |
| `glm-dsa.vocab_size` | 154.880 |
| `glm-dsa.attention.head_count` | 64 |
| `glm-dsa.attention.key_length` | 576 |
| `glm-dsa.attention.value_length` | 512 |
| `glm-dsa.attention.q_lora_rank` | 2.048 |
| `glm-dsa.attention.kv_lora_rank` | 512 |
| `glm-dsa.rope.dimension_count` | 64 |
| `glm-dsa.expert_count` / `expert_used_count` | 256 / 8 |
| `glm-dsa.leading_dense_block_count` | 3 |
| `glm-dsa.nextn_predict_layers` | 1 |
| `glm-dsa.attention.indexer.head_count` | 32 |
| `glm-dsa.attention.indexer.key_length` | 128 |
| `glm-dsa.attention.indexer.top_k` | 2.048 |

Anche epsilon RMS, frequenza RoPE, scala/norma del router, dimensioni FFN e
campi MLA vengono confrontati con `GLM52Shape.v5_2`.

## Tensor directory

I tre tensori globali sono embedding Q8_0, output norm F32 e output head Q8_0.
Ogni blocco contiene RMS, Q-LoRA, KV-LoRA, proiezioni MLA, indexer e FFN. I
blocchi 0–2 hanno FFN denso Q8_0; i blocchi 3–78 contengono router, 256 esperti
routed e un esperto condiviso. Il blocco 78 conserva inoltre i tensori `nextn` e
non viene eseguito nel normale percorso autoregressivo.

Quantizzazioni routed ammesse dal contratto del grafo di riferimento:

- gate/up: IQ2_XXS, Q2_K, Q4_K o Q5_K, con tipo identico nello stesso layer;
- down: IQ2_XXS, Q2_K, Q4_K, Q5_K o Q6_K;
- pesi di controllo/densi: Q8_0 o F32 secondo il singolo tensore.

La validazione controlla nomi, rango, forme, tipi e coerenza gate/up prima di
allocare risorse GPU. Il futuro loader di streaming deve inoltre verificare che
ogni slice expert rispetti il blocco della quantizzazione e i limiti del file.

## Tokenizer e wire format

- GPT-2 byte-level BPE con pretokenizer `glm4`;
- sequenza iniziale `[gMASK]<sop>`;
- ruoli `<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`;
- thinking `<think>…</think>`;
- tool call
  `<tool_call>nome<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`;
- tool result sotto
  `<|observation|><tool_response>…</tool_response>`.

`<|tool|>` non esiste nel vocabolario verificato e non deve essere introdotto
come ruolo sintetico.
