# DS4Demo

Eseguibile CLI di demo/diagnostica che pilota il motore **pure-Swift**
(`DS4Core` + `DS4Metal`) **senza la GUI** e senza link esterni (niente engine C,
niente static lib). Utile per: verificare il runtime Metal, fare l'audit dei
dtype di un GGUF, e generare token reali misurando prefill / decode / I/O.

- **`main.swift`** — bring-up del runtime Metal + self-test GPU; con un percorso
  GGUF apre il modello (mmap no-copy), esegue 1 forward di prova e — se richiesto
  — fa prefill **layer-major** del prompt e genera N token in streaming (greedy).

La finestra di contesto della demo è fissa a `maxKeys = 4096`. Il sampling è
**greedy** (temperature 0): la demo non espone i parametri di sampling, serve a
misurare il motore, non a chattare (per quello c'è l'app DwarfStar).

---

## Argomenti posizionali

```
swift run DS4Demo [gguf-path] [maxNew] [prompt]
```

| # | Argomento | Default | Significato |
|---|-----------|---------|-------------|
| — | *(nessuno)* | — | Solo bring-up Metal + self-test GPU, poi esce. Non serve il modello. |
| 1 | `gguf-path` | — | Percorso del file `.gguf`. Apre il modello, stampa il quant MoE rilevato, esegue 1 forward di prova. |
| 2 | `maxNew` | `4` | Quanti token generare. `0` = solo il forward di prova (nessuna generazione). |
| 3 | `prompt` | `"ciao come stai? rispondi in 1 parola"` | Testo dell'utente, tokenizzato col chat template del modello. |

> Gli argomenti sono **posizionali**: per passare `prompt` devi passare anche
> `maxNew`. Esempio: `swift run DS4Demo model.gguf 32 "Spiega la RoPE"`.

---

## Variabili d'ambiente

Tutte le opzioni avanzate passano da variabili d'ambiente, perché il motore le
legge a runtime (gli stessi knob valgono per l'app e per i test).

### Specifiche della demo

| Variabile | Valori | Default | Effetto |
|-----------|--------|---------|---------|
| `DS4_TYPES_ONLY` | presenza (`=1`) | off | **Modalità audit**: stampa il dtype GGUF dei pesi che il motore si aspetta (esperti, router, attn…), i token speciali del tokenizer e gli id del prompt, poi esce **senza** caricare il decoder. Diagnostica un GGUF "sbagliato" prima di sprecare I/O. |
| `DS4_ACTIVE_EXPERTS` | `1…6` | `6` | Riduce i top-k esperti attivi per token. Meno esperti = meno I/O (più veloce su macchine con poca RAM), qualità inferiore. Clampato a `[1, k]`. |

### Knob del motore (validi anche per la demo)

La demo costruisce uno `StreamingDecoder`, quindi eredita i knob del motore.
Sono **opt-in / sperimentali**: cambiano prestazioni o RAM, non i numeri (salvo
`DS4_ACTIVE_EXPERTS` e `DS4_FUSED_MOE=0`, che cambiano il risultato di proposito).

| Variabile | Valori | Default | Effetto |
|-----------|--------|---------|---------|
| `DS4_RAW_RING` | `=1` | off | Raw-KV come **ring buffer di `nSWA` (128) righe** invece dell'intero contesto → RAM della raw-KV **costante** (l'attention NSA legge solo le ultime `nSWA` righe). Riallinea il port all'upstream. |
| `DS4_PREFILL_UNION` | intero | `64` | Massimo numero di esperti per gruppo nell'I/O batchato del prefill (limita la memoria transitoria delle union, ~7 MB/esperto sul 2-bit). Mai sotto `k` (6). |
| `DS4_EXPERT_CACHE_SLOTS` | intero | `0` (off) | Pool LRU per layer che tiene gli esperti "caldi" residenti in GPU (solo i miss vengono copiati dall'mmap). Memoria **wired** ~6.9 MB/slot × layer: su poca RAM parti piccolo (8) e guarda l'hit-rate nel profilo. Minimo effettivo 8 quando attivo. |
| `DS4_PREFETCH` | `=1` | off | Read-ahead (`madvise`) dei pesi non-routed del **layer successivo** sovrapposto al compute corrente. Default off: su path I/O-bound può **rubare banda SSD** al gather reale. Da misurare per macchina. |
| `DS4_PREFETCH_EXPERTS` | intero | `0` | Con `DS4_PREFETCH=1`, prefetcha anche questo numero di esperti "probabili" (speculativo, dalla usage prior). |
| `DS4_FUSED_MOE` | `=0` per disattivare | on | Kernel MoE fusi (pair-SwiGLU / down-sum6). `=0` usa il path non fuso, utile per **confronto A/B** numerico. |

---

## Esempi

**1. Solo runtime Metal (nessun modello)** — verifica device, kernel compilati e self-test GPU:

```sh
swift run DS4Demo
# DS4Demo: Metal runtime up on Apple M1 Pro, N kernels compiled
# DS4Demo: GPU self-test PASSED
```

**2. Audit di un GGUF** — controlla dtype/tokenizer prima di generare (niente decode):

```sh
DS4_TYPES_ONLY=1 swift run DS4Demo /path/DeepSeek-V4-Flash-…-imatrix.gguf
#   TYPE blk.2.ffn_gate_exps.weight = iq2_xxs (code …)
#   SPECIAL bos=… eos=… user=… assistant=…
#   PROMPT ids = [ … ]
```

**3. Forward di prova soltanto** (`maxNew=0`) — 1 passata, controlla che i logit siano finiti:

```sh
swift run DS4Demo /path/model.gguf 0
# DS4Demo: 1 forward in 3.2s — logits[…] finite=YES argmax=… (logit …)
```

**4. Generazione reale** con prompt personalizzato (prefill + decode in streaming):

```sh
swift run DS4Demo /path/model.gguf 32 "Spiega in breve cos'è la RoPE."
# … prefill … / Risposta: … / [tok 1 … tok/s] … + report del profilo decode
```

**5. Macchina con poca RAM** — meno esperti attivi + ring KV + cache esperti:

```sh
DS4_ACTIVE_EXPERTS=4 DS4_RAW_RING=1 DS4_EXPERT_CACHE_SLOTS=8 \
  swift run DS4Demo /path/model.gguf 16
```

**6. Confronto A/B dei kernel MoE** (fusi vs non fusi) sullo stesso prompt:

```sh
                 swift run DS4Demo /path/model.gguf 8 "1+1?"   # fusi (default)
DS4_FUSED_MOE=0  swift run DS4Demo /path/model.gguf 8 "1+1?"   # non fusi
```

---

## Output

Su `stderr` la demo logga il quant rilevato, i tempi di prefill, e per ogni
token decodificato `[tok N  tempo  tok/s]`. Il testo generato va su `stdout`
(streaming, non bufferizzato). A fine generazione stampa il **report del
profilo** (`dec.profile.report()`): la ripartizione del tempo di decode (attn,
expert gather, FFN, sampling…) — la metrica chiave su macchine I/O-bound.

```sh
swift run DS4Demo /path/model.gguf 8 > risposta.txt   # solo la risposta su file,
                                                       # i log restano a schermo
```
