# Istruzioni passo-passo

Guida operativa dal Mac vuoto al primo token generato. Tutto in
ordine, niente di saltato.

Se sei già dentro il progetto e cerchi solo i comandi sintetici,
guarda [`USAGE.md`](USAGE.md). Questa guida è la versione lunga, fatta
per chi parte da zero.

## Prima di iniziare

Cosa serve avere:

- **macOS 14 (Sonoma) o superiore** con processore Apple Silicon
  (M1, M2, M3, M4 e relative varianti Pro/Max/Ultra). Su Intel Mac il
  progetto non gira: serve la GPU Apple e il supporto BF16 di Metal 3+.
- **Xcode 15+** installato. Se non ce l'hai, scaricalo dall'App Store
  (è gratis), poi apri il Terminale e lancia:
  ```bash
  xcode-select --install
  ```
- **Spazio libero**: V4-Flash richiede ~140 GB per il release
  HuggingFace + 140–600 GB per la versione convertita (dipende dal
  flag `--target-dtype`). Pianifica almeno **300 GB liberi**
  realisticamente, preferibilmente su un SSD esterno se il tuo disco
  interno è piccolo.
- **RAM consigliata**:
  - 128 GB: V4-Flash gira con paging dal SSD, ~1–3 s/token a regime
  - 192 GB+: V4-Flash gira fluido, ~0.2–0.5 s/token
  - Sotto i 64 GB: V4-Flash è troppo grosso. V4-Pro non gira proprio
    su nessun Mac (è da ~900 GB).
- **Connessione di rete decente** per scaricare i pesi (~140 GB).

Strumenti aggiuntivi da installare con Homebrew:

```bash
# Installa Homebrew se non ce l'hai: https://brew.sh
brew install git-lfs huggingface-cli
git lfs install
```

`git-lfs` serve per scaricare i file grossi del repo HuggingFace.
`huggingface-cli` è il client ufficiale che gestisce il download dei
modelli.

## Step 1 — Clona il progetto

Scegli una cartella di lavoro (l'esempio assume `~/Documents`):

```bash
cd ~/Documents
git clone https://github.com/giorgiooppo/deepseek-v4-pro-macos.git
cd deepseek-v4-pro-macos
```

Spostati sul branch con tutte le modifiche più recenti:

```bash
git checkout claude/convert-to-swift-FJDJC
```

C'è un piccolo passo manuale: lo script che compila i kernel Metal
deve essere eseguibile.

```bash
chmod +x Plugins/MetalLibPlugin/build_metallib.sh
```

Senza questo, la build successiva fallisce con "Permission denied".

## Step 2 — Compila il progetto

La prima compilazione richiede 5–10 minuti perché Swift deve costruire
sia il codice Swift sia i 23 file Metal (`.metal`) che vengono
compilati in `default.metallib`.

```bash
swift build -c release
```

Se tutto va bene, vedrai alla fine:

```
Build complete!
```

Controllo che il file `.metallib` sia stato prodotto:

```bash
find .build -name "default.metallib"
```

Deve stamparti almeno un path tipo
`.build/release/DeepSeekKit_DeepSeekKit.bundle/default.metallib`.

Se viene fuori vuoto, vedi [§Troubleshooting](#troubleshooting) sotto.

A questo punto hai due eseguibili pronti dentro `.build/release/`:

- `converter` — converte il release HuggingFace in formato Mac-friendly
- `deepseek` — esegue l'inferenza

## Step 3 — Scegli dove salvare il modello

Importantissimo: il modello convertito può essere grande **fino a
600 GB**. Pianifica dove tenerlo. Tre scenari tipici:

**Scenario A — SSD esterno con tanto spazio**

```bash
# Esempio: SSD montato in /Volumes/DATA con 1 TB liberi
INPUT_DIR=/Volumes/DATA/checkpoints/V4-Flash-HF
OUTPUT_DIR=/Volumes/DATA/checkpoints/V4-Flash-bf16
```

Questo è lo scenario ideale: in più puoi sfruttare al massimo il flag
`--target-dtype bf16` di default, che è il più veloce in inferenza.

**Scenario B — Solo disco interno, poco spazio**

Stai sul flag `--target-dtype keep` al momento della conversione
(output ~140 GB, stessa size dell'input):

```bash
INPUT_DIR=~/checkpoints/V4-Flash-HF
OUTPUT_DIR=~/checkpoints/V4-Flash-keep
```

L'inferenza sarà più lenta (Metal deve dequantizzare FP4/FP8 ad ogni
forward), ma funziona.

**Scenario C — Hai un secondo SSD/Volume**

Tieni l'input su un volume e l'output su un altro. Riduci il
read/write thrashing durante la conversione.

```bash
INPUT_DIR=/Volumes/DRIVE_A/V4-Flash-HF
OUTPUT_DIR=/Volumes/DRIVE_B/V4-Flash-bf16
```

D'ora in avanti uso questi due nomi (`INPUT_DIR`, `OUTPUT_DIR`).
Sostituisci con i tuoi path concreti.

## Step 4 — Scarica i pesi da HuggingFace

DeepSeek-V4-Flash è ospitato su HuggingFace. Il download è ~140 GB
suddivisi in ~46 file `.safetensors` più i config e il tokenizer.

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir "$INPUT_DIR"
```

Il download impiega 30–60 minuti su una rete media. Puoi interromperlo
e ri-eseguirlo: `huggingface-cli` riprende automaticamente.

Verifica che sia andato a buon fine:

```bash
ls -lh "$INPUT_DIR" | head -20
```

Devi vedere file come:
- `config.json` — la configurazione del modello (parametri architetturali)
- `tokenizer.json` + `tokenizer_config.json` — il tokenizer
- `model-NNNNN-of-NNNNN.safetensors` — i 46 shard dei pesi (alcuni
  GB ciascuno)

Se vedi file da **3 byte o 132 byte** invece che da GB, vuol dire che
`git-lfs` non è configurato e hai scaricato solo i puntatori. Re-fai
`git lfs install` e ripeti il download.

### Leggi `n_experts` dal config

Il converter ha bisogno di sapere quanti esperti ha il modello (è
diverso tra V4-Flash e V4-Pro). Sta nel `config.json`:

```bash
grep -E "n_routed_experts|num_experts" "$INPUT_DIR/config.json"
```

Per V4-Flash di solito è `256`. Annota il valore — ti serve al passo
successivo.

## Step 5 — Converti i pesi

Ecco il momento centrale. Il `converter` legge i 46 shard HuggingFace,
applica i rename (`self_attn → attn` ecc.), fonde i pesi quantizzati
con le loro scale, e scrive il risultato in shard allineati per layer.

Comando base (default `--target-dtype bf16`, output ~600 GB):

```bash
.build/release/converter \
  --hf-ckpt-path "$INPUT_DIR" \
  --save-path   "$OUTPUT_DIR" \
  --n-experts   256
```

Variante per disco piccolo (output ~140 GB, inferenza più lenta):

```bash
.build/release/converter \
  --hf-ckpt-path "$INPUT_DIR" \
  --save-path   "$OUTPUT_DIR" \
  --n-experts   256 \
  --target-dtype keep
```

### Cosa aspettarti durante la conversione

Output tipo:

```
Converter: 46 input shard(s)
  target dtype:    bf16 (FP8/FP4+scale fused into native)
  shard size cap:  5.0 GB
  sharding:        layer-aligned (top-level + one bucket per layer)
  note:            FP8 → 2× size, FP4 → 4× size; expect ~3-4× the input footprint.
Indexing input …
Collected 69143 tensors plus 44 wo_a scales.
Discovered structure (from tensor names, no config.json read):
  top-level tensors:  6
  layers:             43 (indices 0…42)
  mtp blocks:         1
Packing 35020 tensors into 133 shard(s) (layer-aligned; max 5.0 GB/shard)
  [1/133] model-00001-of-00133.safetensors — 6 tensors, 2.12 GB
  [2/133] model-00002-of-00133.safetensors — 296 tensors, 5.00 GB
  …
Wrote 35020 tensors across 133 shard(s); 567.3 GB total.
Done.
```

Tempo di esecuzione:
- Con `--target-dtype bf16` su M3 Max + SSD esterno veloce: 15–30 minuti.
- Con `--target-dtype keep`: 5–15 minuti (più veloce, niente fusione).
- Limite: la velocità di scrittura del SSD di destinazione.

### Se viene interrotto a metà

Pace, il converter è **resume-safe**. Se cade per disk-full,
ctrl-C, o crash, basta rilanciare lo **stesso comando** con gli
stessi flag. Vedrai:

```
Resume: 14/133 shard(s) already on disk (66.0 GB skipped).
  [15/133] model-00015-of-00133.safetensors — …
```

Riparte dallo shard 15. **Attenzione**: se cambi `--target-dtype` o
`--shard-size-gb`, il numero totale di shard cambia e il resume non
trova match. In quel caso cancella la directory di output e ricomincia.

### Se finisce lo spazio

```
Swift/ErrorType.swift:254: Fatal error: ... No space left on device
```

Soluzione: cancella il parziale e ripeti con `--target-dtype keep` (output
4× più piccolo) o cambia volume di destinazione.

```bash
rm -rf "$OUTPUT_DIR"
# poi rilancia il converter con --target-dtype keep o un altro path
```

## Step 6 — Copia `config.json` nella directory convertita

Il converter copia automaticamente i due file del tokenizer
(`tokenizer.json`, `tokenizer_config.json`) nella directory di output,
ma **non** copia `config.json` (perché tecnicamente è opzionale per il
converter). Il CLI di inferenza invece lo legge, quindi va copiato a
mano:

```bash
cp "$INPUT_DIR/config.json" "$OUTPUT_DIR/config.json"
```

A questo punto `$OUTPUT_DIR/` contiene tutto il necessario:

```bash
ls "$OUTPUT_DIR"
```

Devi vedere:
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors.index.json`
- ~133 file `model-NNNNN-of-NNNNN.safetensors`

Se vuoi puoi cancellare `$INPUT_DIR/` ora — i pesi convertiti sono
autosufficienti.

## Step 7 — Primo run (test di sanità)

Genera un singolo token per verificare che la pipeline funzioni end-to-end:

```bash
.build/release/deepseek \
  "$OUTPUT_DIR" \
  "Ciao" \
  --mode chat --max-tokens 1
```

Cosa aspettarti:

```
Loading model … Indexed 35020 tensors across 133 shard(s).
 ready.
Prompt tokens: 6
---
<un singolo token, potrebbe essere quello vero o garbage al primo run>
```

**Il primo run è il più lento** (10 secondi – 1 minuto), perché l'OS
deve paginare un sacco di GB dal SSD. I run successivi sono molto più
veloci perché la page cache è già calda.

Se vedi un crash invece dell'output, salta a [§Troubleshooting](#troubleshooting).

## Step 8 — Genera testo vero

Adesso il run "serio". 100 token, modalità chat, temperatura 0.7:

```bash
.build/release/deepseek \
  "$OUTPUT_DIR" \
  "Scrivimi una poesia in tre versi sul mare" \
  --mode chat \
  --max-tokens 100 \
  --temperature 0.7
```

In `--mode raw` vedi i token uscire uno alla volta in streaming.
In `--mode chat` il CLI bufferizza l'intera generazione (per gestire
correttamente i blocchi `<think>…</think>` se presenti) e poi stampa
il risultato finale.

Flag che puoi tunare:

| Flag | Default | Cosa fa |
|---|---|---|
| `--max-tokens N` | 32 | Numero massimo di token generati (si ferma anche su EOS) |
| `--temperature T` | 1.0 | 0 = greedy (sceglie sempre il più probabile); 0.7–1.0 = sampling con creatività; >1.5 = caotico |
| `--mode raw\|chat` | chat | `chat` formatta il prompt con BOS + role markers; `raw` lo passa così com'è |

## Step 9 — Riusa il modello

Ogni volta che vuoi generare di nuovo, basta rilanciare:

```bash
.build/release/deepseek "$OUTPUT_DIR" "Una nuova domanda" --mode chat
```

Non serve ri-fare la conversione: i file in `$OUTPUT_DIR/` sono
permanenti.

Se vuoi rendere il comando più ergonomico, aggiungi `.build/release/`
al PATH della tua shell:

```bash
# Aggiungi a ~/.zshrc:
export DEEPSEEK_REPO="$HOME/Documents/deepseek-v4-pro-macos"
export PATH="$DEEPSEEK_REPO/.build/release:$PATH"
```

Poi `source ~/.zshrc` e puoi semplicemente fare:

```bash
deepseek "$OUTPUT_DIR" "Ciao" --mode chat --max-tokens 50
```

## Troubleshooting

### "no default library was found" / MTLLibraryErrorDomain 6

Il metallib non è stato compilato. Ricontrolla:

```bash
chmod +x Plugins/MetalLibPlugin/build_metallib.sh
swift build -c release
find .build -name "*.metallib"
```

Se il plugin script non si esegue, può essere un problema di
permessi macOS — guarda Privacy & Security → Developer Tools nelle
preferenze di sistema.

### "config.json not found"

Hai dimenticato il passo 6. Copia il file:

```bash
cp "$INPUT_DIR/config.json" "$OUTPUT_DIR/config.json"
```

### "all safetensors files … were LFS pointers"

Il `git lfs` non è configurato e hai scaricato solo i puntatori da 3
righe invece dei veri pesi. Fix:

```bash
git lfs install
rm -rf "$INPUT_DIR"
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir "$INPUT_DIR"
```

### "No space left on device"

Sei senza spazio su disco. Cancella il parziale e ripeti con un
target più compatto o un altro volume.

```bash
df -h        # vedi quanto spazio hai dove
rm -rf "$OUTPUT_DIR"
# Ripeti il converter con --target-dtype keep o cambia --save-path
```

### "N tensor name(s) were not found"

Alcuni tensor non hanno match nei nomi attesi. Se sono tanti, il
modello non riuscirà a generare bene (vengono inizializzati random).
Possibili cause:

- Hai mescolato V4-Flash e V4-Pro nella stessa directory
- Il release HuggingFace ha cambiato schema di naming

Manda il primo log significativo e si aggiusta `Assembly.swift` con
fallback aggiuntivi.

### Primo token impiega tantissimo (> 60 secondi)

Normale per la prima run, l'OS deve fault-in i pesi dal SSD. Se
persiste al secondo run, controlla con Activity Monitor:

- **Memory pressure** > 50% → la RAM è satura, il modello swap-pa di
  continuo. Soluzione: usa V4-Flash con `--target-dtype keep`
  invece di bf16 (meno memoria), o se non basta, V4-Flash è troppo
  grosso per il tuo Mac.
- **Disk I/O alto continuo** → il SSD è il bottleneck. Se è un SSD
  esterno via USB-C lento, prova a spostare i pesi su quello interno
  o uno Thunderbolt.

### Output garbage / loop di un token

Probabili cause:

- `--temperature 0` + prompt strano → il modello entra in loop greedy.
  Prova `--temperature 0.7`.
- La conversione è incompleta (vedi "N tensor name(s) were not
  found"). Le predizioni con pesi random sono sostanzialmente rumore.

### "MLA decode expects seqlen == 1" o altri precondition

Bug nel CLI loop. Manda l'output completo e il comando preciso che
hai eseguito.

## Cosa puoi fare dopo

- **Sperimenta con il sampling**: `--temperature` da 0 a 1.5,
  combinazioni di top-K/top-P sono nel codice di
  `Sources/DeepSeekKit/Sampling.swift`.
- **Modifica il prompt di sistema**: il CLI usa la modalità chat di
  default. Per istruzioni più strutturate vedi
  [`EXAMPLES.md` §7](EXAMPLES.md#7-encode-a-chat-message-with-tool-calls).
- **Capisci cosa c'è dentro**: parti da
  [`ARCHITECTURE.md`](ARCHITECTURE.md) e
  [`GLOSSARY.md`](GLOSSARY.md). Se vuoi modificare codice,
  [`DEVELOPING.md`](DEVELOPING.md) ha le ricette.
- **Performance**: il progetto è "correctness-first" — non è ancora
  ottimizzato. Vedi [`PERFORMANCE.md`](PERFORMANCE.md) per i
  bottleneck attuali e per dove c'è margine.

## Riferimenti rapidi

| Hai bisogno di… | Vai a |
|---|---|
| Tutti i flag del converter / deepseek | [`USAGE.md`](USAGE.md) |
| Capire un termine (MLA, FP4, RoPE, …) | [`GLOSSARY.md`](GLOSSARY.md) |
| Capire l'architettura del modello | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Esempi di codice Swift | [`EXAMPLES.md`](EXAMPLES.md) |
| Mappa file-per-file dei sorgenti | [`MODULES.md`](MODULES.md) |
| Tutti i passi commit-by-commit | `git log --oneline` |
