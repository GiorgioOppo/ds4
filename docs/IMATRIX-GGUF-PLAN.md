# Piano — imatrix + quantizzazioni mancanti + bridge nomi GGUF

Bozza operativa e tracker d'implementazione per il filone tracciato in
[TODO §12](../TODO.md).

**Scope confermato**: imatrix import + export, lettura dei dtype GGUF
mancanti, quantizzatore imatrix-driven, GGUF writer, e larghezze
3/5/6-bit native.

> **Stato del documento**: bozza — nessun codice ancora scritto. Man
> mano che le fasi partono, aggiornare la checklist *Stato
> implementazione* qui sotto seguendo la convenzione di `TODO.md`:
> `[~]` + branch link mentre è in corso, `[x]` + commit hash quando è
> chiuso.

## Stato implementazione

- [ ] **Pre-flight** — riconciliazione dei doc stantii
- [ ] **Fase 0** — Fondamenta condivise (bridge nomi + statistiche)
- [ ] **Fase 1** — imatrix import + export
- [ ] **Fase 2** — Quantizzatore imatrix-driven
- [ ] **Fase 3** — Lettura GGUF dtype mancanti
  - [ ] Livello 1 — `Q4_1`, `Q5_0`, `Q5_1`, `Q8_1`
  - [ ] Livello 2 — `Q2_K`, `Q3_K`
  - [ ] Livello 3 — `IQ4_NL`, `IQ4_XS`
  - [ ] Livello 4 — griglie `IQ1`/`IQ2`/`IQ3`
- [ ] **Fase 4** — GGUF writer
- [ ] **Fase 5** — Larghezze 3/5/6-bit native

---

## Stato attuale (baseline accertata)

| Area | Stato nel codebase |
| --- | --- |
| **imatrix** | Assente (zero riferimenti nel codice). |
| **Calibrazione propria** | `CalibratedQuant.swift`: `ActivationObserver` (absmax/mean), `HessianObserver` (Hessiana piena), formato `stats.json` + `hessians/*.f64`. |
| **GGUF read** | `GGUF.swift` / `GGUFLoader.swift`: dequant per `Q8_0`/`Q4_0`/`Q4_K`/`Q5_K`/`Q6_K`. Mancano `Q2_K`/`Q3_K`, i legacy `Q4_1`/`Q5_0`/`Q5_1`/`Q8_1`, e tutta la famiglia `IQ` (sollevano `unsupportedType`). |
| **GGUF write** | Assente (solo reader). |
| **Bridge nomi** | `Rename.swift`: solo HF→V4. Nessun mapping llama.cpp↔V4. |
| **Bit-width nostri** | INT2/4/8 (`Int{2,4,8}Quant.swift` + kernel). Mancano 3/5/6. |

## Pre-flight (½ giornata)

Alcuni doc sono stantii e vanno riconciliati *prima* di iniziare, per
non lavorare su premesse sbagliate:

- `docs/GGUF.md` §"Cosa NON funziona" dice che i kernel dequant non
  esistono, mentre TODO §10.2 (e §9) li danno per landed.
- `converter/main.swift:40` dice che `inverseChannelScale` è scartato,
  mentre TODO §0 dà il pre-mul AWQ come chiuso.

Verificare lo stato reale del wiring runtime AWQ **prima della Fase 2**.

## Fase 0 — Fondamenta condivise (~3-4 gg)

Prerequisito di Fase 1, 2 e 4.

### 0a. Bridge nomi llama.cpp ↔ V4

**Nuovo**: `Sources/DeepSeekConverter/GGUFNameMap.swift`

Tabella leaf bidirezionale per la convenzione GGUF, includendo sia i
nomi Llama densi sia i leaf MLA DeepSeek (llama.cpp ha già la
nomenclatura MLA per DeepSeek-V2/V3):

- `attn_q_a`↔`attn.wq_a`, `attn_q_b`↔`attn.wq_b`,
  `attn_kv_a_mqa`↔`attn.wkv_a`, `attn_kv_b`↔`attn.wkv_b`,
  `attn_q_a_norm`↔`attn.q_norm`, `attn_kv_a_norm`↔`attn.kv_norm`,
  `attn_output`↔`attn.wo`
- `ffn_gate`↔`ffn.w1`, `ffn_up`↔`ffn.w3`, `ffn_down`↔`ffn.w2`
- `token_embd`↔`embed`, `output_norm`↔`norm`, `output`↔`head`
- Trasformazione indice `blk.{N}.` ↔ `layers.{N}.`

**API**: `ggufToV4(_:) -> String`, `v4ToGGUF(_:) -> String?`

**Caso non-1:1 da documentare**: gli expert MoE in GGUF sono *stacked*
in un tensore 3D (`ffn_gate_exps` `[n_expert, n_ff, n_embd]`), mentre V4
li tiene per-expert (`ffn.experts.{E}.w1`). Il rename da solo non
basta: serve uno strato di slicing. La tabella completa va verificata
contro `gguf-py/gguf/tensor_mapping.py` di llama.cpp.

### 0b. Estensione statistiche di calibrazione

**Modifica**: `CalibratedQuant.swift`

- `CalibrationStats`: aggiungere `perChannelMeanSq: [Float]?` (media di
  x² per canale = ciò che l'imatrix rappresenta).
- `ActivationObserver`: accumulare anche la somma dei quadrati (un solo
  `[Float]` extra per layer, costo trascurabile) ed esporla in
  `finalize`.

**Test**: `GGUFNameMapTests` (round-trip su nomi noti), estensione di
`CalibratedQuantTests` per `meanSq`.

## Fase 1 — imatrix import + export (~6-7 gg)

### 1a. Reader imatrix

**Nuovo**: `Sources/DeepSeekKit/IMatrix.swift`

- Struct `IMatrix`: `[String: [Float]]` (per-tensor, dato = media di x²
  per colonna = diagonale di E[xxᵀ]), conteggi chiamate, nome dataset.
- `IMatrix.readLegacyDat(url:)` — formato binario classico
  (`n_entries`, poi per entry: `len`+`name`+`ncall`+`nval`+`float[nval]`,
  infine `m_last_call`+`dataset`). Da verificare byte-per-byte contro
  `tools/imatrix/imatrix.cpp` della versione llama.cpp target.
- `IMatrix.readGGUF(url:)` — riusa `GGUFFile`/`GGUFHeader.parse`
  esistente; legge i tensori per-nome `*.in_sum2` / `*.counts` del
  formato imatrix-GGUF recente.
- `IMatrix.read(url:)` — auto-detect via magic (`"GGUF"` vs legacy).

### 1b. Bridge imatrix → calibrazione interna

- `IMatrix.toCalibrationStats(nameMap:)`: traduce le chiavi
  llama.cpp→V4 (Fase 0a) e costruisce `[String: CalibrationStats]`,
  popolando `perChannelMeanSq` e derivando
  `perChannelAbsMax ≈ sqrt(meanSq)` (proxy RMS — vedi *Note*).
- **Caso MoE**: l'imatrix può avere una entry unica per gli `exps`
  stacked; mappare verso le N entry per-expert.

### 1c. Writer imatrix (export)

- `IMatrix.writeLegacyDat(url:)` e `IMatrix.writeGGUF(url:)`, con
  conversione nomi V4→llama.cpp (Fase 0a) così che llama.cpp lo possa
  consumare.
- **Sorgente dati**: `ActivationObserver.perChannelMeanSq` (Fase 0b).
  In alternativa la diagonale di `HessianObserver`.

### 1d. Wiring CLI

- `deepseek_calibrate`: flag `--emit-imatrix <file>`
  `[--imatrix-format dat|gguf]`.
- `converter`: flag `--imatrix <file>` come alternativa/complemento a
  `--calib-stats`.

> **Note**: l'imatrix è una statistica del 2° momento (mean-square),
> mentre AWQ/SmoothQuant qui consumano `perChannelAbsMax` (L∞) →
> `sqrt(meanSq)` è un proxy RMS ragionevole (l'AWQ originale usa
> comunque la magnitudine media). L'imatrix è solo la *diagonale*
> dell'Hessiana: non può alimentare il GPTQ a Hessiana piena —
> alimenta invece il quantizzatore di Fase 2.

**Test**: round-trip write→read (`.dat` e GGUF), import di un imatrix
reale piccolo di llama.cpp, fixture GGUF-imatrix sintetica.

## Fase 2 — Quantizzatore imatrix-driven (~4-6 gg)

È il motivo per cui esiste l'imatrix. RTN minimizza Σ(w−deq(q))²; la
versione *importance-weighted* minimizza Σ imp[c]·(w[c]−deq(q[c]))².

**Nuovo**: `Sources/DeepSeekKit/ImatrixQuant.swift`

- `quantizeImportanceWeighted(srcURL:…, importance:[Float], bits: 2|3|4|5|6|8)`
  con due leve (come `make_qx_quants` di llama.cpp):
  1. Ricerca della scala di blocco che minimizza l'errore pesato (non
     absmax/maxLevel).
  2. Ricerca dell'arrotondamento per-elemento (round vs round±1) —
     impatto forte a basso bit-width.
- Output nei layout INT2/4/8 esistenti → i GEMM kernel attuali restano
  invariati (**vantaggio chiave**).
- `QuantMethod.imatrix` nuovo case in `CalibratedQuant.swift`; dispatch
  in `quantizeBF16ToInt8Calibrated`.
- Converter: `--quant-method imatrix` (consuma `--imatrix`).
- Ortogonale ad AWQ/SmoothQuant: opzionalmente comporre (scale-search
  pesata dentro AWQ) come follow-up.

**Test**: verificare che l'errore pesato sia ≤ RTN su pesi sintetici;
test di accuratezza end-to-end su un layer reale.

## Fase 3 — Lettura GGUF dtype mancanti (~16-21 gg, a livelli)

Tutti puramente **additivi e non-breaking**: kernel in
`dequant_gguf.metal` + un case in `GGUFLoader.load`. Possono andare in
PR separate, una per dtype.

| Livello | Dtype | Difficoltà | Stima |
| --- | --- | --- | --- |
| 1 | `Q4_1`, `Q5_0`, `Q5_1`, `Q8_1` | Bassa (lineari, come `Q4_0`) | ~2 gg |
| 2 | `Q2_K`, `Q3_K` | Media (superblocchi K-quant, come `Q4_K`) | ~3-4 gg |
| 3 | `IQ4_NL`, `IQ4_XS` | Alta (LUT non-lineare 16 entry) | ~3 gg |
| 4 | `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S` | Molto alta (codebook a griglia 256/512 entry) | ~8-12 gg |

Il Livello 4 (griglie IQ) domina il costo: richiede le tabelle-codebook
costanti portate da `ggml-quants.c`. Si può differire o fare in modo
incrementale (`IQ4` prima).

**Test**: estendere `GGUFTests` con block-sizing e cross-check valori
vs riferimento per ogni dtype.

## Fase 4 — GGUF writer (~5-8 gg)

**Nuovo**: `Sources/DeepSeekKit/GGUFWriter.swift`

- Scrittura magic + v3, KV metadata, tabella tensor-info, dati
  allineati (`general.alignment`).
- **Quantize-on-write**: encoder per `Q4_0`/`Q4_K`/`Q6_K`/`Q8_0`…; in
  modalità imatrix usa l'errore pesato di Fase 2 (encoder CPU
  sufficienti per un tool one-shot).
- Conversione nomi V4→GGUF (Fase 0a) + emissione metadata
  architettura/iperparametri.
- Converter: `--output-format safetensors|gguf`.

> **Caveat onesto da documentare**: llama.cpp può eseguire solo
> architetture che conosce. L'architettura V4 (MLA + MoE +
> hyper-connections) non è in llama.cpp, quindi un GGUF V4 sarebbe
> leggibile *strutturalmente* ma non eseguibile da llama.cpp. Il writer
> è utile soprattutto per: modelli Llama-family già ingeriti,
> round-trip dei nostri quant, e fixture di test.

**Test**: round-trip writer→`GGUFFile` reader; cross-check di un file
piccolo con `gguf-dump` di llama.cpp.

## Fase 5 — Larghezze 3/5/6-bit native (~6-9 gg, priorità più bassa)

Mirror di `Int4Quant.swift`. Per ciascuna larghezza:
`Int{3,5,6}Quant.swift`, layout di packing, kernel
`int{3,5,6}_gemm.metal`, `shouldQuantizeToIntN`,
`--target-dtype int3|int5|int6`, test.

- Packing non-potenza-di-2 fastidioso (INT3: 8 valori/3 byte; INT5: 8/5
  byte; INT6: 4/3 byte).
- **Nota di valore**: un INT4 imatrix-driven (Fase 2) batte spesso un
  INT5/6 RTN — valutare se INT3/5/6 vale rispetto a "INT4 + imatrix".
  Tenuta come workstream reale (richiesta esplicita) ma con priorità
  più bassa.

## Sequenza e dipendenze

```
Fase 0 (fondamenta) ──┬─→ Fase 1 (imatrix I/O) ──→ Fase 2 (quantizzatore)
                      └─→ Fase 4 (GGUF writer)  ←──┘ (beneficia di Fase 2)
Fase 3 (read dtype)  ── indipendente, in parallelo da subito
Fase 5 (3/5/6-bit)   ── indipendente, in parallelo da subito
```

**Ordine consigliato**: 0 → 1 → 2 → 4, con 3 e 5 in parallelo quando
c'è banda. Critical path ≈ Fasi 0+1+2+4.

## Stima complessiva

| Fase | Stima | Note |
| --- | --- | --- |
| 0 — Fondamenta | 3-4 gg | Sblocca tutto il resto |
| 1 — imatrix I/O | 6-7 gg | |
| 2 — Quantizzatore imatrix | 4-6 gg | |
| 3 — Read dtype GGUF | 16-21 gg | IQ-grids dominano; differibili |
| 4 — GGUF writer | 5-8 gg | |
| 5 — 3/5/6-bit | 6-9 gg | Priorità più bassa |
| **Totale** | **~40-55 gg** | Ordine di grandezza, non impegno |

## Rischi principali

- **Formato imatrix `.dat`**: storicamente evoluto — verificare
  byte-per-byte contro la versione llama.cpp target prima di codificare
  il parser.
- **MoE imatrix/nomi**: gli expert stacked richiedono slicing, non solo
  rename — il punto più fragile delle Fasi 0/1.
- **Griglie IQ (Fase 3 liv. 4)**: le tabelle-codebook sono voluminose e
  facili da sbagliare; pianificare cross-check numerici.
