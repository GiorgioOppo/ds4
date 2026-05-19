# Il converter

Transcoder offline che prende una release HuggingFace di DeepSeek-V4
e produce una directory che il runtime Swift può caricare. Rinomina
i tensor, fonde FP8/FP4 + scale in BF16/F16, o quantizza i pesi
Linear a INT8/INT4/INT2 — a seconda del target dtype.

Documenti complementari:

- [`DTYPES.md`](DTYPES.md) — layout bit-a-bit di FP4 / FP8 / E8M0 /
  BF16 / F16, la matematica di fusion, e gli excerpt del codice di
  conversione.
- [`MODEL.md`](MODEL.md) — cosa l'output del converter viene caricato
  come.
- [`LOADING.md`](LOADING.md) — come il loader legge gli shard.
- [`MODULES.md`](MODULES.md) — indice per-file.
- [`USAGE.md`](USAGE.md) — riferimento dell'operatore (esecuzione del
  binario, tabella dei flag).

> 🇬🇧 La versione inglese è [`CONVERTER.md`](CONVERTER.md).

---

## 1. Cosa fa il converter

Tre lavori:

1. **Rinominare** i nomi dei tensor HuggingFace nella convenzione di
   naming canonica inference-side.
2. **Transcodare** dtype non-native in formati per cui la GPU Apple
   Silicon ha effettivamente hardware: FP8 + E8M0 scale → BF16, FP4
   + E8M0 scale → BF16, o quantizzazione fresca INT8/INT4/INT2
   quando l'utente vuole ridurre l'impronta su disco.
3. **Ri-shardare** l'output in file safetensors layer-aligned di
   dimensione configurabile, più un `model.safetensors.index.json`
   così altre tool possono trovare ogni tensor.

I file del tokenizer (`tokenizer.json`, `tokenizer_config.json`,
`config.json`, `generation_config.json`, `special_tokens_map.json`)
sono copiati verbatim accanto.

La CLI è `Sources/converter/main.swift` (script straight-line
top-level — nessun wrapper a classi). I pezzi riusabili (logica
rename, packer dtype, funzioni di fusion, ConversionSpec) vivono
nella libreria `DeepSeekConverter` (`Sources/DeepSeekConverter/`).

### Perché convertire

Le GPU Apple gestiscono questi nativamente in MSL:

- `F32`, `F16`, `BF16` (Metal 3+ / macOS 14+)
- Tipi interi (`i8`, `i16`, `i32`)
- `simdgroup_matrix<float|half|bfloat>` su M1+ / M3+

**Non** hanno tipi nativi o aritmetica per:

- FP8-E4M3, FP4-E2M1, E8M0 (MX scale)

L'inference su FP8/FP4 ha bisogno di dequant per-elemento in shader
su ogni GEMM. Più economico pagare il costo una volta in fase di
convert e spedire pesi in dtype native — al prezzo di file più
grandi (FP8 → BF16 raddoppia, FP4 → BF16 quadruplica).

I path INT8/INT4/INT2 vanno in direzione opposta: paga costo extra
di quantizzazione upfront per *ridurre* l'impronta su disco vs BF16,
trading un po' di accuratezza.

---

## 2. Invocazione

```
swift run -c release converter \
    --hf-ckpt-path /path/V4-Flash-HF \
    --save-path /path/V4-Flash-converted \
    --n-experts <N> \
    [--model-parallel 1] \
    [--target-dtype bf16|f16|int8|int4|int2|keep] \
    [--shard-size-gb 5]
```

Flag richiesti:

| Flag | Significato |
|---|---|
| `--hf-ckpt-path` | Directory sorgente HuggingFace. Deve contenere shard `*.safetensors` + `model.safetensors.index.json`. |
| `--save-path` | Directory di output. Creata se mancante; resume-safe (vedi §9). |
| `--n-experts` | Conteggio totale degli expert. Verificato contro l'input; mismatch è un errore hard. |

Flag opzionali:

| Flag | Default | Significato |
|---|---|---|
| `--model-parallel` | 1 | Ereditato dal reference Python; il port Swift è solo single-rank, quindi resta 1. |
| `--target-dtype` | `bf16` | Modalità dtype di output. Vedi §3. |
| `--shard-size-gb` | 5 | Cap soft per shard di output. Auto-cappato a ~95% di `MTLDevice.maxBufferLength` così il runtime può mmappare ogni shard come un MTLBuffer. |

### Matrice del target dtype

| `--target-dtype` | Pesi Linear | Altri tensor | Disco vs HF native | Velocità inference |
|---|---|---|---|---|
| `keep` | FP8 / FP4 preservati | invariati | più piccolo | più lento (dequant in shader) |
| `bf16` (default) | BF16 fusi | BF16 | ~2× input FP8, ~4× input FP4 | più veloce (simdgroup_matrix nativo) |
| `f16` | F16 fusi | F16 | come bf16 | più veloce |
| `int8` | INT8 W8A16 + scale F16 di gruppo | BF16 (fallthrough) | ~½ × BF16 | leggermente più lento (dequant in shader) |
| `int4` | INT4 W4A16 + scale F16 di gruppo, packed 2-per-byte | BF16 | ~¼ × BF16 | simile a int8 |
| `int2` | INT2 W2A16 + scale F16 di gruppo, packed 4-per-byte | BF16 | ~⅛ × BF16 | simile; perdita di accuratezza brutale |

La modalità `keep` è il "path più corto" — solo `wo_a` viene fuso
(dato che MLA lo legge tramite `Einsum.bsgdGrd` direttamente senza
passare per un dispatch `Linear`, il kernel einsum si aspetta un
input in stile BF16). Tutto il resto è al massimo rinominato.

---

## 3. Convenzione di naming

`Sources/DeepSeekConverter/Rename.swift` — `renameKey(_:)`.

Tre fasi:

1. Strip del prefisso `model.`.
2. Sostituisci frasi comuni:
   - `self_attn → attn`
   - `mlp → ffn`
   - `weight_scale_inv → scale`
   - `e_score_correction_bias → bias`
3. Riscrivi il leaf via `leafMapping[String:String]`:
   - `embed_tokens → embed`
   - `input_layernorm → attn_norm`
   - `post_attention_layernorm → ffn_norm`
   - `q_proj → wq`
   - `q_a_proj → wq_a`
   - `q_a_layernorm → q_norm`
   - `q_b_proj → wq_b`
   - `kv_a_proj_with_mqa → wkv_a`
   - `kv_a_layernorm → kv_norm`
   - `kv_b_proj → wkv_b`
   - `o_proj → wo`
   - `gate_proj → w1`, `down_proj → w2`, `up_proj → w3`
   - `lm_head → head`
   - Entry identity per nomi già canonici così una ri-conversione è
     un no-op.

C'è anche un guard di detection del leaf: tensor il cui parent è
`hc`, `attn_sink`, `tie2eid`, `ape`, o inizia con `hc_` usano
l'ultima componente del path come leaf invece della penultima.

`shouldSkip(_:)` droppa gli alias MTP-tied: `mtp.*.emb*` e
`mtp.*.head.weight` esistono nel checkpoint HF come reference
all'embed/head principale, quindi le scartiamo in fase di convert.
Il `MTPBlock` del runtime tiene weak reference all'embed/head
condivisi invece.

### Companion di scale

`scaleNameFor(_:)` ritorna `<base>.scale` per `<base>.weight`. La
release HF nativa li nomina `<base>.weight_scale_inv` (nota il
suffisso `_inv`); il rename pass strippa la parte `_inv`. Il loader
runtime accetta entrambe le forme via `WeightLoader.tryLoad`.

---

## 4. Indexing dell'input

`Sources/converter/main.swift:122` (walk + collect):

```swift
for inputURL in inputs {
    try autoreleasepool {
        let stf = try SafeTensorsFile(url: inputURL)
        for (origName, entry) in stf.entries {
            if shouldSkip(origName) { continue }
            let newName = renameKey(origName)
            let absOffset = try absoluteOffset(of: entry, in: inputURL)
            plan[newName] = PendingTensor(
                url: inputURL, offset: absOffset,
                byteCount: entry.dataOffsets[1] - entry.dataOffsets[0],
                dtype: entry.dtype, shape: entry.shape)
        }
    }
}
```

`autoreleasepool` conta: ogni `SafeTensorsFile` mmappa l'intero
shard e lo wrappa come MTLBuffer. Senza drainare il pool tra shard,
quelle mappature restano vive per tutta la fase di indexing e la
memoria virtuale del processo cresce di ~14 GB per shard indicizzato.

L'output è una mappa flat `[String: PendingTensor]`: nuovo nome →
input file + byte range + dtype + shape. Nessun dato è stato ancora
letto — la mappa del file è referenziata per il pass di parse e
rilasciata.

Tutti i tensor finiscono nella stessa mappa. Un refactor precedente
segregava `wo_a.scale` in un dict separato ma non lo guardava mai di
nuovo, il che silenziosamente rompeva la fusion di wo_a in ogni
modalità (wo_a FP8 era passato verbatim invece di essere fuso in
BF16 / INT8). Trattarlo come qualsiasi altra scale lascia che i path
di fusion esistenti lo trovino via `plan[scaleName]`.

---

## 5. Costruire le write entry

Per ogni tensor nel plan, il converter decide cosa emettere. La
struttura di output è `WriteEntry { name, dtype, shape, byteCount,
source: SafeTensorsWriter.Source }`. `Source` è uno di:

- `.data(Data)` — byte pre-computati in memoria.
- `.file(url:offset:byteCount:)` — copia direttamente da un altro
  file.
- `.compute(byteCount:closure)` — producer lazy, chiamato quando il
  writer ne ha bisogno.

L'uso di `.compute` è essenziale per i path di fusion / quantization:
il closure è chiamato dal writer appena prima di streamare quel
tensor su disco, così la memoria di picco resta bounded a `nCore ×
dimensione tensor`.

L'albero di decisione (`Sources/converter/main.swift:236-622`):

```
Per ogni tensor t nel plan, in ordine di nome:
  if t è un tensor .scale standalone e il suo parent .weight è nel plan:
    skip (il suo parent lo consumerà)
    
  if target == int8:
    if t è un peso Linear che dovrebbe essere quantizzato:
      emetti due write entry:
        <name>.weight  (INT8 packed)
        <name>.scale   (scale F16 di gruppo)
      scegli il compute closure source-dependent:
        FP8 input → quantizeFP8ToInt8
        FP4 input → quantizeFP4ToInt8
        BF16 input → quantizeBF16ToInt8
        F32 input → quantizeF32ToInt8
      continue
      
  if target == int4: …stessa shape, packed two-per-byte, range INT4
  if target == int2: …stessa shape, packed four-per-byte, range INT2

  effectiveTarget = bf16 if target ∈ {int8, int4, int2} else target

  if effectiveTarget != keep:
    if t è FP8 con un companion di scale:
      emetti <name> come BF16 (o F16), source = fuseFP8ToNative (compute)
      marca la scale come consumata
      continue
    if t è FP4 (o I8/U8 packed in una experts dir) con scale companion:
      emetti <name> come BF16 (o F16), source = fuseFP4ToNative (compute)
      marca la scale come consumata
      continue

  if effectiveTarget == keep:
    fondi solo wo_a (BF16); rinomina I8/U8 degli experts come F4_E2M1
    tutto il resto passa verbatim via .file(url:offset:byteCount:)
    continue

  default: pass-through .file
```

### Quali pesi Linear vengono quantizzati

`Sources/DeepSeekKit/Int8Quant.swift` espone
`shouldQuantizeToInt8(name, lastDim:)`. La whitelist default (quando
nessun `--int8-whitelist` è dato):

- Ogni `layers.*.attn.*.weight` (MLA: wq_a, wq_b, wkv, wo_a, wo_b).
- Ogni `layers.*.ffn.experts.*.{w1,w2,w3}.weight` (experts routed).
- Ogni `layers.*.ffn.shared_experts.{w1,w2,w3}.weight`.
- Ogni `layers.*.attn.compressor.{wkv,wgate}.weight`.
- Ogni `layers.*.attn.indexer.{wq_b,weights_proj}.weight`.
- I tensor analoghi di ogni layer MTP.

Escluse (sempre BF16): embed, head, gain RMSNorm, attn_sink, peso del
gate (deve restare F32 — vedi
[`MODEL.md`](MODEL.md#48-moe-feed-forward)), HC fn/base/scale, bias.
Le esclusioni contano perché la quantizzazione INT8 abbassa i logit
del gate sotto la precisione che la gating sqrt(softplus) + topK
tollera; lo stesso razionale si applica in `--target-dtype bf16` dove
`Linear.castOutputToBF16` è settato a `false` solo sul gate e LM
head.

`shouldQuantizeToInt4` e `shouldQuantizeToInt2` rispecchiano questo
con i loro constraint group-K (devono essere divisibili per
`kInt4GroupK = 128` e `kInt2GroupK = 128` rispettivamente).

### Kernel di quantizzazione

`Sources/DeepSeekConverter/` impacchetta le routine per source dtype:

| Funzione | Cosa fa |
|---|---|
| `quantizeFP8ToInt8(weightURL:, scaleURL:, ...)` | Legge i byte FP8 + scale E8M0 per-(128,128), dequant inline (LUT), poi RTN simmetrico a INT8 con scale F16 per-(row, K-128). |
| `quantizeFP4ToInt8(...)` | Stessa shape ma per FP4 packed two-per-byte + scale E8M0 per-(row, K-32) (il formato storage FP4). |
| `quantizeBF16ToInt8(...)` | RTN simmetrico diretto da BF16. Usato quando l'input è già una directory BF16 convertita e vuoi ri-quantizzare. |
| `quantizeF32ToInt8(...)` | Lo stesso per input F32. |
| `quantizeFP8ToInt4` / `quantizeFP4ToInt4` / `quantizeBF16ToInt4` / `quantizeInt8ToInt4` / `quantizeF32ToInt4` | Varianti INT4. Il path `Int8ToInt4` è utile per "ri-quantizza una directory INT8 esistente" senza tornare al HF native. |
| Stessa serie per `…ToInt2` | Varianti INT2, packing 4-per-byte. |
| `fuseFP8ToNative(..., target: TargetDType)` | FP8 + E8M0 → BF16 (o F16) per elemento. Il loop caldo usa LUT precomputate (`e4m3LUT[256]`, `e8m0LUT[256]`); la scale per-(block_o, block_i) è issata fuori dal loop interno. |
| `fuseFP4ToNative(...)` | FP4 + E8M0 → BF16 (o F16). Scale per-row su K-block `[1, 32]`. |

Tutta la quantizzazione usa **RTN simmetrico** (round-to-nearest):
il max-assoluto del gruppo è calcolato, lo scale è `max_abs / int_max`
(dove `int_max` è 127 per INT8, 7 per INT4, 1 per INT2), i valori
sono arrotondati e clampati al range. Scale per-(row × K-128) per
INT4 e INT8, per-(row × K-128) per INT2.

Non c'è calibration oggi (no AWQ / GPTQ / SmoothQuant). L'output è
"veloce e ragionevole" fino a INT4; INT2 prende una perdita di
accuratezza marcata perché il range simmetrico `[-2, 1]` è troppo
grossolano per molti outlier di attivazione. La CLI stampa un
warning per INT2 che "calibration recommended for production".

### Coppie lazy-compute

Per `--target-dtype int8|int4|int2`, ogni peso Linear produce *due*
record `WriteEntry` (il peso packed + la scale F16). Entrambi i
closure catturano per riferimento lo stesso `var cached: (weight:
Data, scale: Data)?`. Il writer processa le entry in ordine di
declaration:

1. La entry del peso esegue il closure di quantizzazione → cache
   entrambi i risultati → ritorna i byte del peso.
2. La entry della scale consuma i risultati cachati → nilla la
   cache (rilasciando il buffer del peso immediatamente) → ritorna
   i byte della scale.

Così il converter non tiene mai entrambi il peso e la scale
simultaneamente dopo che sono stati scritti, il che conta a picco
tensor di 4-core × 8 GB.

---

## 6. Shardare l'output

`Sources/converter/main.swift:638` (depth bucketing).

Strategia: ogni transformer layer va nel suo proprio shard quando
possibile. Se un layer è più grande del cap di shard, splittalo su
shard consecutivi (il layer successivo inizia sempre uno shard
fresco). I tensor top-level (embed, head, norm, hc_head_*) vanno
nello shard 0.

```
depthKey(name):
  if name inizia con "layers." → indice layer dalla seconda componente
  if name inizia con "mtp."    → 100_000 + indice    (sort MTP dopo i main)
  altrimenti                   → -1                  (top-level)
```

Dentro ogni bucket di depth, i tensor sono ordinati per nome per
determinismo. Il packer li percorre in ordine e inizia un nuovo
shard quando aggiungere il prossimo tensor spingerebbe il totale in
corso oltre `shard-size-gb`.

### Perché layer-aligned

Il forward pass tocca i layer in ordine stretto di depth. Con shard
layer-aligned, un forward pass legge lo shard 0, poi 1, poi 2, …,
mai indietro. La page cache dell'OS prefetcha lo shard successivo
mentre la GPU sta leggendo quello corrente, e evicta gli shard già
consumati sotto memory pressure senza disturbare il layer che viene
letto attivamente. Il loader streaming-pool costruisce su questo:
ogni shard per-layer è permanentemente assegnato a uno slot
rotating, e il pool kicka i `pread` davanti al layer di cui sta per
aver bisogno.

### Cap di dimensione shard

`--shard-size-gb 5` è il default. La CLI cappa anche questo a ~95%
di `maxBufferLength` del device (MTLDevice.maxBufferLength): il
runtime mmappa ogni shard come un MTLBuffer, quindi uno shard più
grande di questo è non-caricabile su quella macchina. Il margine
del 95% lascia spazio per l'header safetensors + arrotondamento di
page alignment.

### Nomi file di output

`expectedFilename(i:total:)` ritorna
`model-<i+1 zero-padded 5>-of-<total zero-padded 5>.safetensors`.
Insieme a `model.safetensors.index.json` matcha il formato
HuggingFace che altre tool si aspettano.

---

## 7. Scrivere ogni shard

`SafeTensorsWriter` (`Sources/DeepSeekKit/SafeTensorsWriter.swift`)
è il writer streaming. Per ogni tensor:

- Costruisce l'header JSON safetensors lazy mentre
  `add(name:dtype:shape:source:)` è chiamato.
- Quando `write(to: URL)` è invocato, scrive l'header (dopo padding
  a 8 byte), poi streama il payload di ogni tensor dalla sua
  `Source`:
  - `.data(Data)` — scrivi direttamente.
  - `.file(...)` — apri il sorgente, copia in chunk da 64 MB.
  - `.compute(byteCount:closure)` — chiama il closure, scrivi il
    `Data` ritornato. Il closure è responsabile della gestione di
    memoria (la coppia lazy-compute sopra assicura che il payload
    cachato sia rilasciato dopo che il secondo consumer lo legge).

`autoreleasepool` wrappa ogni scrittura di shard così le allocazioni
transient da uno shard non sopravvivono nel successivo.

---

## 8. `model.safetensors.index.json`

Dopo che tutti gli shard sono scritti, il converter assembla:

```json
{
  "metadata": { "total_size": <bytes> },
  "weight_map": {
    "embed.weight": "model-00001-of-00046.safetensors",
    "norm.weight": "model-00001-of-00046.safetensors",
    "layers.0.attn.wq_a.weight": "model-00001-of-00046.safetensors",
    "layers.0.attn.wq_a.scale": "model-00001-of-00046.safetensors",
    ...
  }
}
```

Il loader Swift non ne ha strettamente bisogno (percorre i file
`*.safetensors` direttamente), ma altre tool HF sì. Stesso formato
dell'index della directory HF di input.

---

## 9. Resume

La conversione di V4-Flash può richiedere 30+ minuti su un Mac
16-core. Il converter è resume-safe:

`Sources/converter/main.swift:735`:

```swift
var resumeFromShard = 0
for (i, shard) in shards.enumerated() {
    let fileName = expectedFilename(i: i, total: total)
    let url = saveDir.appendingPathComponent(fileName)
    guard let attrs = try? fm.attributesOfItem(atPath: url.path),
          let size = attrs[.size] as? Int,
          size >= shard.totalBytes else { break }
    for e in shard.entries { weightMap[e.name] = fileName }
    resumeFromShard = i + 1
}
```

Uno shard è "completo" iff:
- Il suo filename matcha `model-(i+1)-of-(total).safetensors`
  esattamente (così una scrittura partial sotto un `total` diverso
  *non* è skippata).
- La sua dimensione è almeno `shard.totalBytes` (l'header aggiunge
  qualche KB di JSON, quindi `>=` implica scrittura completa).

Il `weightMap` per gli shard skippati è popolato così che il finale
`index.json` li rifletta correttamente.

Lo sharding è deterministico dato lo stesso input + stessi flag
(assegnazione deterministica nome → depth bucket + sort
deterministico dentro ogni bucket), quindi rieseguire con argomenti
identici produce file identici a posizioni identiche. Cambiare
`--shard-size-gb` o `--target-dtype` invalida il resume — rimuovi
manualmente la directory di output prima.

---

## 10. Fusion concorrente

Il closure di compute di ogni tensor è sequenziale, ma gli shard di
input sono letti in parallelo. Il bottleneck durante la fusion è il
loop SIMD per-tensor (op Float su ogni elemento); su un M3 Max
16-core, la fusion satura intorno ai 4 thread concorrenti prima che
la contention SSD inizi a dominare.

Memoria residente di picco durante la fusion ≈
`nThread × dimensione max tensor`. Per V4-Flash sono qualche GB.

---

## 11. Kernel di quantizzazione (host-side)

I kernel di quantizzazione INT8 / INT4 / INT2 vivono in
`Sources/DeepSeekKit/{Int8Quant,Int4Quant,Int2Quant,CalibratedQuant}.swift`
più `Sources/DeepSeekConverter/DTypePacking.swift`. Loop caldi:

### RTN simmetrico a INT8

```
per ogni row r in [0, outDim):
    per ogni K-block bk in [0, inDim/128):
        block_start = bk * 128
        // Trova max-abs nel K-block.
        maxAbs = 0
        per k in block_start..<block_start+128:
            maxAbs = max(maxAbs, abs(w[r, k]))
        scale = maxAbs / 127        // f16
        // Arrotonda e clamp.
        per k in block_start..<block_start+128:
            q = round(w[r, k] / scale)
            q = clamp(q, -127, 127)  // -128 riservato per il "no-value" simmetrico
            packed_weight[r, k] = q
        weight_scale[r, bk] = scale.bf16ToF16()
```

Il `clamp(-127, 127)` è simmetrico (lasciando -128 inutilizzato) così
che il kernel GEMM possa usare aritmetica unsigned per il range di
valore `|x|`.

INT4 packa nibble two-per-byte: nibble basso = colonna `2k`, nibble
alto = colonna `2k+1`. INT2 packa quattro valori 2-bit per byte:
`[7:6]` = col `4k+3`, `[5:4]` = col `4k+2`, `[3:2]` = col `4k+1`,
`[1:0]` = col `4k`.

### Dequant FP8 / FP4 LUT-driven

`fuseFP8ToNative` e `fuseFP4ToNative` usano LUT precomputate per
evitare bit twiddling per-elemento:

```swift
let e4m3LUT: [Float] = (0..<256).map { dequantE4M3(UInt8($0)) }
let e2m1LUT: [Float] = (0..<16).map  { dequantE2M1(UInt8($0)) }
let e8m0LUT: [Float] = (0..<256).map { dequantE8M0(UInt8($0)) }
```

Ogni LUT sta in L1. Il loop caldo diventa "carica byte FP8 → lookup
LUT → moltiplica per scale → scrivi BF16". La scale per-(block_o,
block_i) è issata fuori dal loop interno così è ammortizzata su 128²
pesi.

Vedi [`DTYPES.md`](DTYPES.md) per i dettagli di encoding.

---

## 12. Source map

| Topic | File |
|---|---|
| Entry point CLI | `Sources/converter/main.swift` |
| Rename + leaf mapping | `Sources/DeepSeekConverter/Rename.swift` |
| Helper di packing dtype (conversioni BF16 / F16) | `Sources/DeepSeekConverter/DTypePacking.swift` |
| Fusion FP8 / FP4 → BF16/F16 | `Sources/DeepSeekConverter/NativeFusion.swift` |
| Quant INT8 + whitelist | `Sources/DeepSeekKit/Int8Quant.swift` |
| Quant INT4 | `Sources/DeepSeekKit/Int4Quant.swift` |
| Quant INT2 | `Sources/DeepSeekKit/Int2Quant.swift` |
| Helper di calibration (stub oggi) | `Sources/DeepSeekKit/CalibratedQuant.swift` |
| ConversionSpec / ConversionTarget | `Sources/DeepSeekConverter/ConversionSpec.swift` |
| Writer streaming safetensors | `Sources/DeepSeekKit/SafeTensorsWriter.swift` |
| Reader safetensors (input side) | `Sources/DeepSeekKit/SafeTensors.swift` |
| ConvertSheet (driver GUI) | `Sources/DeepSeekUI/Views/Convert/ConvertSheet.swift` |
| Riferimento Python | `Reference/inference/convert.py` |

---

## 13. La variante GUI

L'app desktop espone il converter dal menu **Convert** della toolbar
(icona bacchetta) e Settings:

- `Sources/DeepSeekUI/Views/Convert/ConvertSheet.swift` — il sheet
  SwiftUI.
- `Sources/DeepSeekUI/State/ConvertViewModel.swift` — driver:
  assembla argv, spawna il binario converter, streama stdout/stderr
  nel pane log del sheet.

Il binario che la GUI esegue è lo stesso output di
`swift run -c release converter`; la GUI fornisce solo una UI per
scegliere directory di input / output e i flag. Nessun codice è
condiviso tra il lato SwiftUI e il converter — la GUI è un wrapper
attorno alla CLI.

---

## 14. Limitazioni e lavoro deferred

Tracciato in `TODO.md` (§0 Quantizzazione + §1 Parità). A colpo
d'occhio:

- **`cast_e2m1fn_to_e4m3fn`** nel reference Python
  (`Reference/inference/convert.py:17-52`) NON è portato. Il path
  `--target-dtype keep` con `--expert-dtype fp8` fa fallback a
  relabel-only. Con il path di fusion `bf16` default questo è
  irrilevante — gli experts vengono fusi a BF16 comunque.
- **Calibration (AWQ / GPTQ / SmoothQuant)** non è implementata. I
  path INT4 e INT2 usano RTN simmetrico, che è lo schema più
  semplice; calibration stringerebbe le scale per-row e recupererebbe
  un po' di accuratezza.
  `Sources/DeepSeekKit/CalibratedQuant.swift` è lo scaffolding.
- **Modalità di conversione W8A8** (separata dal flag runtime
  `useW8A8Activations` di Linear): il converter non emette un layout
  W8A8-specific perché il runtime può prendere qualsiasi layout W8A16
  e optare in W8A8 al dispatch time via `Linear.useW8A8Activations`.
- **Mixing dtype per-layer** (es. INT4 sugli experts, BF16
  sull'attenzione) non è un'opzione CLI ma sarebbe utile per layer
  accuracy-sensitive. Aggiungi una whitelist pattern-based a
  `shouldQuantizeToInt4` se l'use case salta fuori.
- **Sharding multi-GPU** (il flag `--model-parallel` dal Python
  convert.py) non è implementato; il port Swift è single-rank.

---

## 15. Esempio end-to-end

```bash
# 1. Scarica la release HF (una volta sola, ~142 GB).
pip install --upgrade huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
    --local-dir ~/Downloads/V4-Flash-HF

# 2. Converti a BF16 (default). Richiede ~30 min su M3 Max.
swift run -c release converter \
    --hf-ckpt-path ~/Downloads/V4-Flash-HF \
    --save-path ~/Downloads/V4-Flash-BF16 \
    --n-experts 256

# 3. Il runtime ora legge la directory convertita direttamente.
swift run -c release deepseek ~/Downloads/V4-Flash-BF16 \
    "Spiega la fusione nucleare in due frasi." \
    --mode chat --temperature 0.7 --max-tokens 256

# Impronta su disco: ~600 GB (BF16). Quadruplata se invece passi
# --target-dtype int4. Il runtime auto-rileva il dtype dall'header
# safetensors di ogni tensor — nessun flag lato loader è necessario.
```
