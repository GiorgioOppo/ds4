[English](C-PARITY-REVIEW.md) | **Italiano**

# Laguna S 2.1 — revisione di parità col C (branch `laguna-s2.1` @ `448d569`)

Revisione riga per riga del port Swift rispetto al motore C di riferimento
(`antirez/ds4`, feature branch `laguna-s2.1`, head `448d569` — la stessa
baseline pinnata in [`UPSTREAM-SYNC.md`](../../UPSTREAM-SYNC.md)). Sono stati
confrontati sei assi (geometria/config, tokenizer, protocollo chat/tool,
schema tensori/catalogo, percorso di attenzione, MoE/FFN/dequant) e ogni
divergenza segnalata è stata ri-verificata in modo indipendente sul codice di
entrambi i lati prima di intervenire.

## Verificato identico (nessun intervento)

- **Geometria e configurazione** — tutte le 29 chiavi GGUF `laguna.*`
  validate, i valori attesi, i tipi richiesti, l'alternanza teste 48/72
  (`il % 4 == 0`), l'obbligo YaRN e la tolleranza float relativa 1e-6
  (`config_validate_laguna_model` ↔ `LagunaConfiguration`). Differiscono solo
  ordine e classificazione della diagnostica; nessun file viene accettato o
  rifiutato diversamente.
- **Nucleo del tokenizer** — pre-split dei run di LF più segmento GLM4 a
  cifre singole identici carattere per carattere, incluse tutte le tabelle
  Unicode; BPE byte-level, caricamento dei token speciali e politica di stop
  EOS/`</assistant>` esattamente coincidenti.
- **Schema tensori, layout quant, catalogo** — nomi, shape e ordine delle
  dimensioni, i marker di layout (Q8_0 signal vs legacy Q4_K/F16), i tipi
  ammessi per gruppo inclusa l'eccezione legacy del down Q6_K, e i tre
  artefatti di catalogo (repo, revision pin, nomi file, dimensioni) sono
  identici nei valori. Lo Swift è più severo solo nella diagnostica (tensori
  duplicati, `partialOutputHead`, errori per singolo tensore).
- **Percorso di attenzione** — formule YaRN complete (corr-dims, ramp con
  divisione intera, mscale ≈ 1.34657 su cos/sin di Q e K nei layer full),
  rotazione NeoX sul prefisso 64/128, RMS-norm Q/K per head prima del RoPE a
  eps 1e-6, gate softplus (guardia a 20) per head dopo la normalizzazione e
  prima della proiezione di uscita, scala softmax 1/√128 senza mscale,
  finestra scorrevole di 512 chiavi inclusa la posizione corrente, store nel
  ring F16 con RNE — identici operazione per operazione su oracolo, motore e
  kernel. `metal/laguna/laguna.metal` è byte-identico all'upstream a parte un
  preludio locale documentato (`block_q6_K`). Laguna **non** ha attention
  sink upstream; nessuno dei due lati lo implementa.
- **MoE/FFN/routing/dequant** — routing sigmoid con bias solo in selezione,
  top-10 bitonico con tie sul id più basso, normalizzazione sulle probabilità
  non biasate con clamp 2⁻¹⁴, scala ×2.5, SwiGLU senza clamp col peso di
  route applicato al vettore mid prima del down, esperto condiviso non
  pesato, ordine dei residui `add3`, embedding e LM head Q8_0, e layout dei
  blocchi Q8_0/Q2_K/Q3_K/Q4_K byte-exact (84/110/144/34 byte, packing di
  scale/min verificato bit per bit).

## Divergenze trovate e corrette in questo passaggio

| # | Gravità | Divergenza | Correzione |
|---|---------|------------|------------|
| F3 | alta | Storia in think mode: il server rende `<assistant><think>reasoning</think>content` (reasoning vuoto incluso, scartato in nothink); il renderer Swift emetteva un `</think>` secco e non aveva il campo reasoning | `LagunaChatMessage` (la forma di `chat_msg`: reasoning separato + raw tool text) con resa server-exact; la mappatura da `ChatTurn` divide i prefissi think incorporati come `split_reasoning_content` |
| F4 | alta | Argomenti dei tool call resi in ordine alfabetico; il riferimento usa l'ordine di *dichiarazione* delle properties dello schema, poi le chiavi non mappate nell'ordine JSON originale, con i valori non-stringa come JSON raw minificato | Scansione JSON ordinata (port di `parse_schema_properties`/`json_args_parse`/`json_minify_raw_value`) in `LagunaToolCodec.renderToolCalls` |
| F5 | media | I parser non decodificavano le entity XML del renderer (`dsml_unescape_text`): il round trip render→parse non era stabile | `dsmlUnescape` applicato a chiavi e valori in entrambi i parser |
| — | — | Mancava il port del parser del server (permissivo: scoping dopo l'ultimo `</think>`, nome qualunque, chiavi duplicate, coda scartata, argomenti tutti-stringa con spaziatura C, cattura di `raw_tool_text`) | `LagunaToolCodec.parseServer`, port esatto di `parse_glm_generated_message_ex`; `parseStrict` mantiene le validazioni agent-grade sulla stessa grammatica |
| — | — | Il parser streaming era solo accumula-e-chiudi; l'agente di riferimento fallisce subito sulla malformazione completata ed espone le call man mano | `LagunaIncrementalToolParser` ora distingue incompleto/malformato per chunk ed espone `completedCalls` (più un fix di confine: fine buffer su un bordo di tag ora attende invece di fallire) |
| F6 | media | I default di sampling (0.7/20/0.95/0.05) erano definiti e testati ma non cablati in alcun percorso di inferenza | Il ramo Laguna del demo ora campiona con i default di famiglia (gli override d'ambiente vincono, `DS4_DEMO_TEMPERATURE=0` per i run di parità greedy) |
| F1 | media | Lo scanner upstream del testo renderizzato tiene attivi per Laguna cinque letterali cross-famiglia (`[gMASK]`, `<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`, `<｜Assistant｜>`, `<|assistant|>`); lo Swift scandiva solo i sette nativi | Letterali aggiunti allo scanner (e all'insieme di neutralizzazione, così il contenuto non fidato non può iniettarli) |
| F2 | bassa | `encodeChatPrompt` rendeva il template server e lo scandiva (id EOS per primo, system di default); il percorso CLI di riferimento inserisce l'id BOS e non ha system di default | `encodeChatPrompt` ora rispecchia esattamente `encode_chat_prompt`/`laguna_chat_append_wrapped`; il demo passa il system Poolside di default come fa `ds4_cli.c` |
| — | bassa | Mancava la politica di stop think-aware (`ds4_token_is_stop_for_think_mode`) | `LagunaTokenizer.isStopToken(_:reasoning:)`, usata dal demo |
| — | — | Mancava il port del live tool tail e del suffisso di recovery per tool call malformate (`render_laguna_live_tool_tail`, `build_invalid_laguna_tool_error_suffix`) | `LagunaChatRenderer.liveToolTail` e `invalidToolCallRecoverySuffix` (helper puri; il cablaggio nel loop server resta aperto, vedi sotto) |
| F7 | bassa | La spec sliding-window dell'oracolo CPU fissava il ring a 512 righe senza il clamp upstream `min(512, n_ctx)` (il motore già clampava) | Clamp aggiunto a `LagunaAttentionSpec.slidingWindow` |
| F8 | bassa | Header del motore stantio: dichiarava rifiutato il file misto Q2_K/Q3_K (accettato da quando i matvec K-quant sono cablati) | Commento corretto |
| — | bassa | Whitespace ASCII vs Unicode nel renderer/parser (il C usa `isspace`) | Helper ASCII usati in tutto il backend Laguna |

Inoltre `LagunaRuntimeGate` ora onora `DS4_LAGUNA_RUNTIME=1` come override di
bring-up per processo; il default committato resta spento finché il gate di
parità dei logits non passa.

## Deviazioni deliberate mantenute (documentate, non bug)

- **Neutralizzazione del contenuto non fidato attiva di default** (U+2060
  dentro i letterali di controllo). Il C rende verbatim il contenuto
  utente/tool, quindi lì un `</assistant>` letterale nel testo utente diventa
  un vero token di controllo. La parità byte col C è disponibile con
  `neutralizeUntrustedContent: false`.
- **Extra di `parseStrict`** — tool/argomenti non dichiarati, chiavi
  duplicate, charset degli identificatori, argomenti required e valori
  tipizzati da schema sono validazioni che il server C non fa; `parseServer`
  fornisce il comportamento di riferimento esatto.
- **Ri-codifica deterministica di `<available_tools>`** (chiavi ordinate)
  contro il JSON client verbatim del server C: il port non ha un JSON client
  HTTP da riprodurre, e un prefisso di prompt stabile vale più dell'emulazione
  di byte che non riceve mai.

## Ancora aperto (serve un Mac e/o lavoro a forma di upstream)

- I passi hardware del Gap 4: prima compilazione, test di parità dei kernel,
  parità end-to-end dei logits contro `ds4 --temp 0`, prefill batched e il
  dispatch del decode flash split-K (`kernel_laguna_flash_attn_reduce_gate_f32`
  è portato ma non ancora dispatchato), la ricetta legacy F16/Q6_K, DFlash, i
  digest del catalogo.
- Comportamenti Laguna a livello server trovati dallo sweep di completezza e
  non ancora dichiarati altrove: il sistema di alias modello
  (`laguna-s-2.1-chat/-nothink/-reasoner` con mappatura del think-mode e
  listing `/v1/models`), il cablaggio del suffisso di recovery nel loop del
  server, i suffissi della cache di sessione Laguna (checkpoint
  `</assistant>\n` + live tool tail), e il flusso QA upstream
  `laguna-openrouter-100`.
