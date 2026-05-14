# DeepSeek V4 su macOS

Port Swift + Metal del transformer Mixture-of-Experts [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
per Apple Silicon. Include un binario da riga di comando (`deepseek`) e
un'app nativa SwiftUI per macOS, ed è in grado di caricare in streaming
i pesi di **V4-Flash** (~142 GB) anche su Mac da 16 GB grazie a un
rotating buffer per-layer.

> **Sperimentale.** V4-Pro (1.6T parametri, ~800 GB a FP4) non entra
> nella memoria unificata di nessun Mac; il target realistico
> on-device è **DeepSeek-V4-Flash** (284B / 13B attivati). Stesso
> codice, diversi pesi e `config.json`.

🇬🇧 [English version](README.md) · 🏗 [Architettura (dettagli)](docs/ARCHITECTURE.md)
· 🧪 [Test](docs/TESTING.md) · 🛠 [Sviluppo](docs/DEVELOPING.md)

---

## Requisiti di sistema

| Cosa | Minimo | Consigliato |
|---|---|---|
| **CPU/GPU** | Apple Silicon (M1, M2, M3, M4…) | M-Ultra / M-Max |
| **macOS** | 14.0 Sonoma | 15.x |
| **RAM (unified)** | 16 GB (V4-Flash, streaming) | 64+ GB |
| **Disco** | 150 GB liberi per i pesi V4-Flash | SSD NVMe |
| **Tool** | Swift 5.10 / Xcode 15+ | Xcode 16, Homebrew, Python 3 |

Il loader sceglie automaticamente la strategia in base alla RAM disponibile:

| RAM disponibile | Strategia | Comportamento |
|---|---|---|
| ≥ 192 GB | `preload` | Tutto residente in RAM, velocità massima |
| 32–192 GB | `mmap` | Il sistema pagina su richiesta, veloce dopo warm-up |
| 16–32 GB | `streaming` | Un layer alla volta in RAM, primo token più lento |

I Mac Intel non sono supportati (le pipeline Metal richiedono hardware
con tipo `bfloat`; `Sources/DeepSeekKit/Device.swift` rifiuta di
inizializzarsi).

---

## 1. Scaricare il progetto

```bash
git clone https://github.com/giorgiooppo/DeepSeek-V4-Pro-MacOS.git
cd DeepSeek-V4-Pro-MacOS
swift package resolve
```

Il repository **non** contiene i pesi del modello né il tokenizer —
quei file sono nel `.gitignore`. Il passo successivo li scarica
separatamente.

## 2. Scaricare i pesi

Il checkpoint consigliato è **DeepSeek-V4-Flash** nel layout nativo
HuggingFace (FP8 per l'attention + FP4 per gli expert). Il loader Swift
legge quel layout direttamente — non serve nessuna conversione.

```bash
pip install --upgrade huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
    --local-dir ~/Downloads/V4-Flash-HF
```

A download completato, in `~/Downloads/V4-Flash-HF/` devono esserci:

- `config.json`, `generation_config.json`
- `tokenizer.json`, `tokenizer_config.json`
- `model.safetensors.index.json`
- 46 shard `model-NNNNN-of-NNNNN.safetensors` (~142 GB totali)

Il binario `converter` serve **solo** se vuoi transcodificare il
checkpoint in BF16/INT8/INT4 per varianti più compatte. Vedi
[`docs/USAGE.md`](docs/USAGE.md) per quel flusso.

## 3. Build

### Solo CLI (più veloce)

```bash
swift build -c release
```

Produce:

- `.build/release/deepseek` — CLI di inference
- `.build/release/converter` — transcoder dei pesi offline

### App GUI (Xcode)

```bash
brew install xcodegen        # una volta sola
./Tools/generate-xcodeproj.sh
open DeepSeekV4Pro.xcodeproj
```

Seleziona lo scheme **`DeepSeekApp`** (non "DeepSeekUI", che è il target
SPM eseguibile — entrambi i nomi compaiono perché condividono i sorgenti)
e premi ⌘R.

---

## 4. Quick start

### CLI

```bash
./.build/release/deepseek ~/Downloads/V4-Flash-HF \
    "Qual è la capitale del Brasile?" \
    --mode chat --max-tokens 50 --temperature 0.7
```

I token vengono streammati su stdout man mano che sono campionati. Il
primo token può richiedere 30 s – 3 min su un Mac da 16 GB mentre il
loader streaming scalda la cache; quelli successivi sono molto più
rapidi.

### GUI

1. Avvia l'app da Xcode (⌘R) oppure aprendo l'`.app` prodotto.
2. Quando appare il picker, seleziona la cartella del modello
   (`~/Downloads/V4-Flash-HF`). Le cartelle recenti vengono ricordate.
3. Attendi che l'indicatore di prefill finisca — mostra i secondi
   trascorsi in tempo reale.
4. Scrivi il messaggio nel composer e premi Invia (o ⌘↩).

---

## 5. Riferimento CLI

```
deepseek <model-dir> "<prompt>" [opzioni]
```

Due argomenti posizionali; il secondo è opzionale solo in modalità
diagnostiche.

### Flag di generazione

| Flag | Tipo | Default | Cosa fa |
|---|---|---|---|
| `--mode` | `raw` \| `chat` | `chat` | `raw` antepone solo BOS; `chat` applica il template chat V4. |
| `--thinking` | `off` \| `high` \| `max` | `off` | Budget di ragionamento in chat. `off` appende `</think>` e il modello risponde direttamente; `high` appende `<think>` e il modello produce prima un blocco di ragionamento; `max` aggiunge anche il blocco di sistema REASONING_EFFORT_MAX. |
| `--temperature` | float | `1.0` | Temperatura di sampling. **Imposta `0.7`** — vedi "Valori consigliati" sotto. |
| `--max-tokens` | int | `32` | Numero massimo di token da generare. |

### Flag loader / memoria

| Flag | Tipo | Default | Cosa fa |
|---|---|---|---|
| `--load-strategy` | `auto` \| `preload` \| `mmap` \| `streaming` | `auto` | Forza una specifica strategia di caricamento pesi. |
| `--force-load` | flag | off | Bypassa i controlli di sicurezza RAM (shard > 70 % della RAM, totale > 25× RAM). Usalo solo se sai di poter tollerare il paging aggressivo. |
| `--max-seq-len` | int | da `config.json` | Override delle righe della KV cache per-layer. Più basso = meno RAM, contesto più corto. |
| `--max-batch-size` | int | da `config.json` | Override della dimensione batch della KV cache. V4-Flash di default è 1. |

### Modalità diagnostiche

| Flag | Cosa fa |
|---|---|
| `--print-config` | Carica `config.json`, stampa il `ModelConfig` risolto su stderr ed esce. Verifica che ogni chiave sia stata effettivamente letta invece di ricadere silenziosamente su un default. |
| `--trace-norms` | Stampa L2 norm + min/max/mean + contatori NaN/Inf del residual stream in punti chiave del forward. Utile per individuare il layer in cui le attivazioni divergono. |
| `--list-tensors [PREFIX]` | Elenca tutti i nomi dei tensor nel checkpoint, eventualmente filtrati per prefisso. Passa `""` come prompt. |
| `--dump-tensor NAME[:row=R][:cols=A..B]` | Dequantizza una riga di slice del tensor indicato e stampa i valori float uno per riga. Default `row=0`, `cols=0..32`. |

### Valori consigliati

- **`--temperature 0.7`**. Sotto greedy argmax (temperature = 0) il
  routing MoE di V4-Flash cade in punti fissi auto-rinforzanti e la LM
  head finisce in loop su un singolo token filler (`好的好的好的…`,
  `_type_type_type…`). Valori intorno a 0.6–0.9 producono i campioni più
  coerenti. La GUI vincola lo slider all'intervallo `[0.5, 1.0]` per
  questo motivo.
- **`--mode chat --thinking off`** per Q&A brevi; `--thinking high` per
  problemi in cui il modello deve "pensare ad alta voce".

### Esempio: inference completa

```bash
./.build/release/deepseek ~/Downloads/V4-Flash-HF \
    "Spiega la fusione nucleare in due frasi." \
    --mode chat \
    --thinking off \
    --temperature 0.7 \
    --max-tokens 256 \
    --max-seq-len 4096 \
    --max-batch-size 1
```

### Esempio: diagnostica

```bash
# Verifica cosa il loader ha letto da config.json
./.build/release/deepseek ~/Downloads/V4-Flash-HF "" --print-config

# Cerca il nome di un tensor
./.build/release/deepseek ~/Downloads/V4-Flash-HF "" \
    --list-tensors layers.0.

# Dequantizza la prima riga di un weight expert
./.build/release/deepseek ~/Downloads/V4-Flash-HF "" \
    --dump-tensor layers.6.ffn.experts.56.w1.weight:row=0:cols=0..64
```

---

## 6. L'app macOS

### Picker / loading del modello

Al primo avvio scegli la cartella del modello tramite un classico
`NSOpenPanel`. Le cartelle usate di recente vengono ricordate
(Preferenze → Loading). Durante il caricamento vedi il riepilogo
`LoadPlan` (numero di shard, RAM prevista, strategia scelta) e uno
spinner di progresso. Se qualcosa va storto il pannello offre
**Try again**, **Force load** o **Choose another folder**.

### Schermata chat

- **Sidebar** elenca ogni conversazione, con timestamp e un piccolo
  spinner accanto a quella che sta generando. Tasto destro → Delete.
  ⌘N crea una nuova chat.
- **Indicatore di prefill**: un contatore live
  (`Prefilling 256 tokens · 12.3s`) mostra che il primo forward sta
  procedendo.
- **Barra di throughput** sotto i messaggi mostra due righe monospaced
  appena inizia il decode:
  ```
  Prefill: 256 tok in 8.32s · 1850 tok/min
  Generation: 42 tok in 9.15s · 275 tok/min
  ```
  La riga di generation si aggiorna ogni ~0.5 s.
- **Streaming dei token**: i token compaiono nella bolla dell'assistant
  man mano che vengono campionati, come avviene nella CLI.
- **Blocchi di reasoning**: i contenuti `<think>…</think>` sono
  renderizzati come disclosure collassabile (icona cervello cliccabile).
- **Send/Stop**: il bottone Send diventa Stop mentre una generazione è
  in flight, in coerenza con il gating dentro `ChatStore`.

### Preferenze

Quattro tab. Le modifiche hanno effetto sul Send successivo (o, per
`Model Config`, sul prossimo caricamento del modello).

| Tab | Cosa controlla |
|---|---|
| **Generation** | Temperatura (slider 0.5–1.0, default 0.7), top-K (0 = disabilitato), top-P, max-tokens, thinking mode. |
| **Model Config** | Ogni campo di `ModelConfig`. Scrive in `~/Library/Application Support/<app>/config-overrides.json`; il loader ne onora `max_seq_len` e `max_batch_size` al prossimo load. |
| **Loading** | Override strategia loader, toggle force-load, ultima cartella caricata, cartelle recenti, percorso del binario converter. |
| **Storage** | Posizione history conversazioni, dimensione su disco, "Reveal in Finder", "Clear all". |

### Convert sheet (quantizzazione offline)

L'azione **Convert model…** nella toolbar apre uno sheet che pilota lo
stesso binario `converter` usato dalla CLI. Scegli una cartella sorgente
(formato HF nativo), una destinazione, un target dtype (BF16 / F16 /
INT8 / INT4 / INT2 / keep) e una shard size. Lo sheet mostra in tempo
reale progresso e log mentre la conversione gira.

---

## 7. Risoluzione problemi

**Il primo token impiega minuti.**
È atteso sotto la strategia `streaming` su un Mac da 16 GB. Il loader
deve leggere da disco lo shard di ~3 GB di ogni layer prima di
elaborarlo. I token successivi sono molto più rapidi — il rotating slot
resta caldo.

**Errore di build: `precompiled file '…/ModuleCache/…' was compiled with module
cache path '…'`.**
I percorsi intermedi cached si sono "stalli", tipicamente perché la
cartella di progetto è stata spostata (rinominata, sync iCloud,
Cestino). Pulisci e ricompila:

```bash
rm -rf .build
swift package clean
swift build
```

Per Xcode svuota anche `~/Library/Developer/Xcode/DerivedData/DeepSeekV4Pro-*`.

**Il modello continua a ripetere lo stesso token.**
Stai samplando con `--temperature 0`. V4-Flash richiede sampling
stocastico — passa `--temperature 0.7`. La GUI lo vincola via slider.

**"No Metal device" / Mac Intel.**
È richiesto Apple Silicon. Non c'è un fallback.

**Out of memory al caricamento.**
Prova `--load-strategy streaming` per forzare il rotating loader, o
`--max-seq-len 2048 --max-batch-size 1` per ridurre la KV cache.

---

## 8. Licenza e crediti

Il codice Swift in questo repository è MIT (vedi [`LICENSE`](LICENSE))
ed eredita la licenza del modello upstream. I pesi del modello e
l'implementazione Python di riferimento in `Reference/inference/`
appartengono a DeepSeek; vedi la loro
[scheda Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
per i termini di licenza.

Se vuoi capire come funziona il port sotto il cofano — mapping dei
kernel, amplificazione del residual stream, design dello streaming
pool, dispatch MoE — leggi [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Per il workflow contributore vedi [`docs/DEVELOPING.md`](docs/DEVELOPING.md),
per esempi e ricette pronte vedi [`docs/EXAMPLES.md`](docs/EXAMPLES.md).
