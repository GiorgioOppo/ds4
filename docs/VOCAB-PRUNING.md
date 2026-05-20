# Vocab + Expert pruning per DeepSeek-V4

DeepSeek-V4 ship con un tokenizer da **129'280 token** pensato per multilingua
(con copertura forte di CJK, hangul, arabo, devanagari, ecc.). Se il caso d'uso
reale è italiano-only (o più in generale multilingua latino), la maggior parte
di questi token resta "morta": occupa righe in `embed.weight` e in `head.weight`
ma non viene mai usata durante l'inferenza.

Inoltre il modello ha una **MoE feed-forward** con `n_routed_experts` esperti
per layer (256 in V4-Flash production, 8 in V4-Flash test). Su un input
distribuzionale ristretto (es. solo italiano), gli esperti non si attivano
uniformemente: alcuni ricevono pochissimi token. Quelli sono candidati
naturali al pruning.

Questo modulo (`DeepSeekVocabPruner` + CLI `vocab_pruner`) implementa **due
fasi opt-in** che si compongono pulitamente:

| Fase | Trigger CLI | Cosa fa | Costo |
|------|-------------|---------|-------|
| **Vocab** | `--corpus` o `--keep-ids` | Riduce `embed.weight` / `head.weight` a ~32-50k righe + ricostruisce `tokenizer.json` | Solo tokenizzazione del corpus |
| **Expert** | `--prune-experts` + `--calib-corpus` o `--expert-stats` | Droppa esperti raramente attivati. Rimuove i tensor `experts.<E>.{w1,w2,w3}` per `E` droppato, imposta righe gate a sentinel negativo, rimappa `tid2eid` per hash layer, aggiunge `pruned_experts` in `config.json` | Forward del modello su corpus di calibrazione (heavy) |

Entrambe le fasi sono **senza fine-tuning** e **senza toccare i pesi del transformer
attivi**: cambiano solo le dimensioni delle matrici di embedding (vocab phase) o
la presenza degli esperti (expert phase).

Il loader di inferenza auto-rileva la nuova `vocab_size` dallo shape di
`embed.weight` (vedi `Sources/DeepSeekKit/Config.swift`), quindi il checkpoint
risultante è drop-in nel resto della pipeline.

## Quando ha senso

- ✅ **Italiano-only** (target principale).
- ✅ Multilingua latino (italiano + francese + spagnolo + tedesco + ecc.).
- ✅ Use case con corpus omogeneo (medical italiano, legal italiano, ecc.).
- ❌ Codebase CJK-heavy o multilingua globale: lo script-foreign filtering
  eliminerebbe troppo. Lascia il tokenizer com'è.
- ❌ Se hai bisogno di mantenere la fedeltà esatta del tokenizer originale (es.
  per riprodurre un benchmark pubblicato): la rimappatura degli ID cambia gli
  embedding rows; non puoi confrontare logits directamente con la versione full.

## Come funziona

Due fasi sequenziali. Le statistiche di copertura sono emesse via
`VocabPruneEvent` (analogo a `ConversionEvent`).

### Fase 1 — Analyzer

`VocabAnalyzer.analyze(...)` legge il `tokenizer.json` del checkpoint sorgente
e scansiona il corpus:

1. **Tokenizzazione**: per ogni linea del corpus, `BPETokenizer.encode(line)` →
   array di ID. Contatore di frequenza per token-ID.
2. **Cumulative coverage**: sort dei token per frequenza, accumulo, trova K
   tale che i top-K coprono la soglia (default 99.95%).
3. **Force-include** (a prescindere dal conteggio):
   - Tutti gli `added_tokens` del tokenizer originale (special token DeepSeek
     come `<｜begin▁of▁sentence｜>`, `<｜User｜>`, ecc. — il chat template li
     deferenzia per stringa, ma la riga di embedding deve esistere).
   - I 256 token byte-level base (mapping GPT-2 byte→unicode) — necessari per
     il fallback UTF-8 su qualunque input.
   - Tutti i token che decodificano in stringhe ASCII / Latin-1 / Latin
     Extended (utili per coprire input italiano anche se rari nel corpus).
4. **Force-drop**: token che decodificano in stringhe contenenti caratteri
   CJK / Hangul / Hiragana / Katakana / arabo / ebraico / devanagari / thai.
   Anche se per caso comparissero nel corpus, vengono tagliati.
5. **Remap ID**: gli `added_tokens` mantengono l'ID originale (così non c'è
   bisogno di toccare il chat template). Tutti gli altri vengono ricompattati
   a partire da 0, saltando gli slot occupati dagli `added_tokens`.

Output: `KeepDecision` con `keepIds`, `oldToNew`, `newVocabSize`. Serializzato
in `keep_ids.json` nell'output dir per ispezione e replay.

### Fase 2 — Rewriter

`VocabRewriter.rewrite(...)` produce il nuovo checkpoint:

1. **Slicing riga-wise di `embed.weight` e `head.weight`** (e dei loro alias
   MTP se sopravvivono nel checkpoint sorgente): per ogni `(oldId, newId)` in
   `oldToNew`, copia `bytesPerRow` byte dall'offset `oldId * bytesPerRow` alla
   riga `newId` del nuovo tensor.
2. **Pass-through zero-copy** per tutti gli altri tensor: il nuovo
   safetensors usa `SafeTensorsWriter.Source.file(url:offset:byteCount:)` che
   stream-copia dal file sorgente senza materializzare in RAM.
3. **`tokenizer.json` ricostruito**: vocab filtrato + rimappato, merges
   filtrate (eliminate quelle dove `a`, `b` o `ab` non sono più nel vocab),
   `added_tokens` preservati verbatim, `pre_tokenizer` / `decoder` /
   `normalizer` / `post_processor` copiati così come sono.
4. **`config.json`**: solo `vocab_size` aggiornato; tutto il resto invariato.
5. **`model.safetensors.index.json`**: ricostruito con il nuovo `total_size`;
   gli shard mantengono gli stessi nomi (non ri-shardiamo).

Output: directory pronta per essere usata da `deepseek` o dalla UI senza
modifiche.

## Comando end-to-end

```bash
swift build -c release
swift run -c release vocab_pruner \
  --input-dir ~/models/V4-Flash-converted \
  --output-dir ~/models/V4-Flash-it \
  --corpus ~/corpora/wikipedia-it \
  --coverage 0.9995
```

Il corpus può essere:
- un singolo file `.txt` (una linea per record);
- un singolo file `.jsonl` (HF dataset format, campo `text` per linea);
- una directory walkata ricorsivamente per `.txt` e `.jsonl`.

Flag opzionali:
- `--coverage 0.9999` per copertura quasi totale (vocab risultante più grande);
- `--keep-ids keep_ids.json` per saltare la Fase 1 e usare una decisione
  pre-computata (replay deterministico, ispezione manuale);
- `--dry-run` per stimare la dimensione del nuovo vocab senza scrivere
  l'output.

## Risparmio atteso

Le matrici `embed.weight` e `head.weight` sono `[vocab_size, hidden_dim]`. Per
V4-Flash (`hidden_dim = 4096`, dtype `bf16` = 2 byte):

| Coverage | Vocab atteso (corpus IT misto) | Risparmio per matrice | Risparmio totale (2 matrici) |
|----------|--------------------------------|-----------------------|------------------------------|
| 99.95%   | ~32k–40k                       | ~750 MB               | ~1.5 GB                       |
| 99.99%   | ~50k–60k                       | ~570 MB               | ~1.1 GB                       |
| 99.999%  | ~80k–90k                       | ~325 MB               | ~650 MB                       |

I numeri esatti dipendono dal corpus: un Wikipedia italiano completo dà cifre
diverse da un corpus tecnico specialistico.

## Limiti noti

1. **Fertility peggio su input multilingua**: se l'utente scrive in giapponese
   a un modello con vocab IT-only, il tokenizer cade sui byte-base e ogni
   character giapponese diventa 3 token. L'output del modello sarà comunque
   nonsense (perché non c'è semantica giapponese dietro), ma il decoder
   produrrà bytes UTF-8 validi.
2. **Non è fine-tuning**: il transformer continua a "pensare" multilingua
   internamente. Riduci la matrice di output ma le rappresentazioni nascoste
   restano quelle del modello originale.
3. **Drift di logits**: se confronti la distribuzione di output del modello
   pruned con quello full su un prompt, vedrai numeri diversi anche sui
   token comuni — l'embedding row di `t_i` ha NewID diverso da OldID, e la
   softmax normalizza su un universo più piccolo.
4. **Token added_tokens preservano ID alto**: se un special token aveva
   id=128815, il nuovo vocab avrà comunque slot 128815. Per i ~30k token
   "normali" vengono usati gli slot 0..30k-1, ma `newVocabSize` resta il
   max degli ID, non il count. È atteso e necessario per non rompere il
   chat template.

## API programmatica

```swift
import DeepSeekVocabPruner
import DeepSeekConverter  // per CancellationToken

let spec = VocabPruneSpec(
    inputDir: URL(fileURLWithPath: "/path/to/V4-Flash"),
    outputDir: URL(fileURLWithPath: "/path/to/V4-Flash-it"),
    corpus: URL(fileURLWithPath: "/path/to/corpus-it"),
    coverage: 0.9995)

let token = CancellationToken()
try await VocabPruner.run(spec: spec, cancellation: token) { event in
    print(event)
}
```

`VocabPruneEvent` casi: `.scanned`, `.coverage`, `.shardWritten`, `.log`,
`.finished`. Vedi `Sources/DeepSeekVocabPruner/VocabPruneEvent.swift` per i
campi esatti.

## Componenti riusati

Il pruner è puro Swift, in-process, no subprocess. Riusa:

| Bisogno                                       | Riuso                                           |
|-----------------------------------------------|-------------------------------------------------|
| Lettura `safetensors`                         | `SafeTensorsFile` (`Sources/DeepSeekKit/SafeTensors.swift`) |
| Scrittura `safetensors` zero-copy             | `SafeTensorsWriter` (idem)                       |
| Tokenizer BPE parser + vocab/merges/added     | `BPETokenizer` (`Sources/DeepSeekKit/BPETokenizer.swift`) |
| Pattern facade enum + `run(spec:...)` async   | `Converter` (`Sources/DeepSeekConverter/Converter.swift`) |
| `CancellationToken`                           | `Sources/DeepSeekConverter/ConversionProgress.swift` |
| CLI args parsing manuale                      | `Sources/converter/main.swift`                  |
| Schema `model.safetensors.index.json`         | `Sources/converter/main.swift`                  |

## UI (DeepSeekApp)

La UI espone il pruner via un bottone "Prune vocab…" (icona forbici)
nella toolbar accanto a "Convert model…" e "Fine-tune model…". Apre
un sheet modale (`VocabPrunerSheet`) con la stessa struttura di
`ConvertSheet`:

- **Source**: directory del checkpoint convertito.
- **Destination**: directory di output (deve essere diversa).
- **Corpus**: file `.txt`/`.jsonl` o directory.
- **Coverage**: slider 0.99..1.0, default 0.9995.
- **Options**: toggle `Dry-run`.

Mentre il job gira la form si blocca, una progress bar mostra
l'avanzamento per shard, e una banner sopra il log riporta in tempo
reale `vocab: N → K (coverage XX.XX%)`. Il bottone "Cancel" segnala
il `CancellationToken` interno.

Lo stato vive in un `@StateObject VocabPrunerViewModel` locale al
sheet — niente persistenza fra apertura e chiusura (stesso pattern
di Converter / FineTuner).

## Expert pruning

La seconda fase opt-in. Stesso CLI, flag `--prune-experts`.

### Razionale

Per layer la MoE ha `n_routed_experts` esperti (256 in V4-Flash production)
e attiva `n_activated_experts` (default 2) per token. La routing decision
arriva da un gate appreso (score-based, sqrtsoftplus) o da una lookup
table per token id (hash routing, primi `n_hash_layers` layer).

Su un input distribuzionale ristretto (es. solo italiano) la usage
histogram per layer è skewed: i top-K esperti ricevono il 99% delle
assignment, gli altri si attivano raramente o mai. Il pruner droppa
quelli sotto coverage e produce un checkpoint più piccolo che **continua
a runnare con lo stesso codice di inferenza** — il loader (`Assembly.swift`)
salta gli esperti marcati come `pruned_experts` in `config.json` e mette
`nil` in `MoEFFN.experts[]`. Il dispatch path ha già `guard let expert =
experts[e] else { continue }`.

### Come funziona

**Fase 1 — Analyzer** (`ExpertAnalyzer.analyze(...)`):

1. Carica il modello via `Transformer.load(...)` (mmap OK).
2. Wire `block.ffn.routingObserver` su ogni Block. L'hook viene chiamato
   dentro `MoEFFN.callAsFunction` subito dopo il read host-side di `idxArr`
   per la dispatch plan — costo aggiuntivo zero, niente kernel extra.
3. Tokenizza il `--calib-corpus` (file `.txt` / `.jsonl` o directory) e
   processa in chunk da `--max-tokens-per-batch` (default 1024).
4. Per ogni chunk: forward del modello, l'observer accumula counts per
   `(layerId, expertId)`. KV cache rilasciata fra chunk.
5. **Decisione coverage-based**: per ogni layer, ordina gli esperti per
   count decrescente, taglia al threshold (default `--expert-coverage 0.99`).
   Floor minimo: `max(--min-experts-floor, n_activated_experts)` (default 4).

**Fase 2 — Rewriter** (`ExpertRewriter.rewrite(...)`):

1. **Drop dei tensor degli esperti**: per ogni `(L, E)` droppato, i tensor
   `layers.<L>.ffn.experts.<E>.{w1,w2,w3}.{weight,scale}` vengono semplicemente
   **omessi** dal nuovo safetensors. Il loader li trova mancanti, riconosce
   il marker in `pruned_experts`, e salta la costruzione.
2. **Sentinel sul gate**: per ogni layer con esperti droppati, le righe
   `gate.weight[E, :]` corrispondenti vengono sovrascritte con `-1e9`
   (`ExpertRewriter.droppedGateLogit`). Stessa cosa per `gate.bias[E]` se
   presente. Risultato: `sqrt(softplus(-1e9)) ≈ 0`, il top-K kernel non
   sceglie mai quell'expert.
3. **Remap di `tid2eid`** per i primi `n_hash_layers` layer: per ogni entry
   che punta a un expert droppato, sostituisce con l'expert vivo più
   simile (cosine similarity sul `gate.weight` row del dropped vs ogni kept).
4. **`config.json`**: aggiunto `pruned_experts: [[Int]]` con la lista per
   layer. `n_routed_experts` resta invariato.
5. **Pass-through** zero-copy degli altri tensor (attention linears, norms,
   HC params, shared experts, embed, head, MTP).

### Comando end-to-end (solo expert, su vocab già fatto)

Esattamente lo use case che ti porta qui:

```bash
swift run -c release vocab_pruner \
    --input-dir ~/models/V4-Flash-it \         # output di un vocab_pruner precedente
    --output-dir ~/models/V4-Flash-it-lean \
    --prune-experts \
    --calib-corpus ~/corpora/calib-it-small \
    --expert-coverage 0.99 \
    --min-experts-floor 4 \
    --max-calib-tokens 200000
```

Il CLI rileva automaticamente che non hai passato `--corpus` né `--keep-ids`,
quindi salta la vocab phase e va dritto a quella expert. Il modello viene
caricato dal `--input-dir` già pruned (vocab più piccolo), la calibrazione
gira sul corpus italiano, e l'output va in `--output-dir`.

### Comando pipeline (entrambe le fasi in un'unica run)

```bash
swift run -c release vocab_pruner \
    --input-dir ~/models/V4-Flash-converted \
    --output-dir ~/models/V4-Flash-it-lean \
    --corpus ~/corpora/wikipedia-it --coverage 0.9995 \
    --prune-experts --calib-corpus ~/corpora/calib-it-small \
    --expert-coverage 0.99
```

Il CLI crea una directory intermedia `~/models/V4-Flash-it-lean.vocab-stage/`,
esegue la vocab phase scrivendoci dentro, poi esegue la expert phase leggendo
da lì e scrivendo nell'output finale. La intermediate dir viene cancellata
a fine job (`--expert-dry-run` la preserva per ispezione).

### Resume

Stesso pattern della vocab phase. Il checkpoint vive in
`<output>/checkpoint/expert_pruner.json`, sibling del `vocab_pruner.json`.
Lo specHash include `inputDir + calibCorpus + coverage + minKeptFloor`, quindi
cambiare uno di questi invalida il resume e riparte da zero.

Granularità:
- **Phase 1 (analyzer)**: per-file. Se il job crasha durante un file, al
  resume quel file viene re-processato dall'inizio. I file fully-processed
  prima del crash sono nel checkpoint e vengono skippati.
- **Phase 2 (rewriter)**: per-shard. Stesso pattern della vocab phase.

### Risparmio atteso

V4-Flash production (256 esperti × 7 layer, expert tensors in FP4 + scale
≈ 26 MB/expert/layer):

| coverage | esperti tenuti per layer (tipico) | spazio risparmiato |
|----------|------------------------------------|---------------------|
| 0.999    | ~200                              | ~5 GB              |
| 0.99     | ~120-150                          | ~18 GB             |
| 0.95     | ~60-80                            | ~30 GB             |

Numeri dipendono fortemente dalla distribuzione del corpus di calibrazione.
Il tool stampa il breakdown per layer prima della Phase 2.

### Quality drift

L'expert pruning **non è gratuito**:
- Esperti rari sono comunque imparati. Tagliarli rischia quality drop su
  input fuori-distribuzione (es. il modello pruned su corpus IT vede un
  prompt in francese e routes verso esperti vivi non ottimali).
- Il `tid2eid` remap usa cosine similarity sulla riga del gate, che è una
  approssimazione. Esperti molto specializzati potrebbero non avere un
  buon "vicino".
- Niente fine-tuning post-pruning: la rete continua a "pensare" in
  rappresentazioni che includevano gli esperti droppati.

Mitigazione:
- `--expert-coverage 0.999` lascia tantissimi esperti vivi (saving modesto
  ma rischio basso).
- `--min-experts-floor 8` (vs default 4) garantisce abbondanza di top-K.
- Smoke-test il modello pruned su un set di prompt rappresentativi.
  `expert_keep_ids.json` resta nell'output per audit / rollback.

## API programmatica

```swift
import DeepSeekVocabPruner
import DeepSeekConverter  // per CancellationToken

// Vocab + Expert in pipeline:
let vocabSpec = VocabPruneSpec(
    inputDir: URL(fileURLWithPath: "/path/to/V4-Flash"),
    outputDir: URL(fileURLWithPath: "/path/to/V4-Flash-stage"),
    corpus: URL(fileURLWithPath: "/path/to/corpus-it"),
    coverage: 0.9995)
try await VocabPruner.run(spec: vocabSpec) { event in print(event) }

let expertSpec = ExpertPruneSpec(
    inputDir: URL(fileURLWithPath: "/path/to/V4-Flash-stage"),
    outputDir: URL(fileURLWithPath: "/path/to/V4-Flash-lean"),
    calibCorpus: URL(fileURLWithPath: "/path/to/calib-it"),
    coverage: 0.99,
    minKeptFloor: 4)
try await ExpertPruner.run(spec: expertSpec) { event in print(event) }
```

I due facade emettono lo stesso `VocabPruneEvent` enum — la UI / CLI può
sottoscriversi una volta sola.

## Roadmap
- [ ] Support per `--keep-ids` con json prodotto da uno script Python
  esterno (es. analisi su corpus enorme HF dataset).
- [ ] Validazione che il pruned checkpoint round-trippi su una manciata di
  prompt italiani (preset di smoke-test).
- [ ] Quantizzazione W4A4 / W2A8 in tandem col pruning per ottenere il
  massimo della compressione.
- [ ] **Expert phase — analyzer chunked**: oggi se un file di
  `--calib-corpus` è enorme la calibrazione non checkpointa intra-file.
  Aggiungere granularità per-chunk come in `VocabAnalyzer`.
- [ ] **Expert phase — fine-tune di recovery**: opzionale, post-pruning,
  per recuperare quality drift sui dropped experts via short fine-tune
  dei kept (richiede `DeepSeekTraining` backward, oggi stub).
