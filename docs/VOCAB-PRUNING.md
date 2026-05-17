# Vocabulary pruning per DeepSeek-V4

DeepSeek-V4 ship con un tokenizer da **129'280 token** pensato per multilingua
(con copertura forte di CJK, hangul, arabo, devanagari, ecc.). Se il caso d'uso
reale è italiano-only (o più in generale multilingua latino), la maggior parte
di questi token resta "morta": occupa righe in `embed.weight` e in `head.weight`
ma non viene mai usata durante l'inferenza.

Questo modulo (`DeepSeekVocabPruner` + CLI `vocab_pruner`) produce un nuovo
checkpoint con il vocabolario ridotto a ~32-50k token, **senza fine-tuning** e
**senza toccare i pesi del transformer**: cambia solo la dimensione delle due
matrici di embedding.

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

## Roadmap

- [ ] Wire della UI in DeepSeekUI per lanciare il pruner via sheet (come
  `ConvertSheet`).
- [ ] Support per `--keep-ids` con json prodotto da uno script Python
  esterno (es. analisi su corpus enorme HF dataset).
- [ ] Validazione che il pruned checkpoint round-trippi su una manciata di
  prompt italiani (preset di smoke-test).
- [ ] Quantizzazione W4A4 / W2A8 in tandem col pruning per ottenere il
  massimo della compressione.
