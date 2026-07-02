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
> Il prompt è letto da `CommandLine.arguments[3]`: se contiene spazi, mettilo
> tra virgolette; non ci sono flag tipo `--prompt`.

---

## Variabili d'ambiente

Tutte le opzioni avanzate passano da variabili d'ambiente, perché il motore le
legge a runtime (gli stessi knob valgono per l'app e per i test).

Come leggerle:

- **Diagnosi**: non cambia il modello, aggiunge misure/log per capire dove va il tempo.
- **Memoria/I/O**: cambia come i pesi vengono letti o tenuti residenti. Serve
  soprattutto quando il modello non entra in RAM e l'SSD domina il decode.
- **Cache esperti**: prova a tenere in GPU gli esperti MoE più usati. Costa RAM
  wired, ma può ridurre letture da SSD se il routing si concentra su pochi esperti.
- **Numerica/qualità**: alcuni knob cambiano davvero il calcolo o il numero di
  esperti usati. Sono utili per A/B, non per confronti di parità.

Regola pratica: prima fai un run con `DS4_DIAG=1`, poi cambia **un solo knob per
volta** e confronta token/s, `gather`, `route/attn`, hit-rate della cache e
throughput di regime.

### Specifiche della demo

| Variabile | Valori | Default | Effetto |
|-----------|--------|---------|---------|
| `DS4_TYPES_ONLY` | presenza (`=1`) | off | **Audit del file GGUF**. Stampa dtype dei tensori critici, token speciali e tokenizzazione del prompt, poi esce senza costruire il decoder. Usalo come primo controllo quando un modello produce output senza senso o quando vuoi verificare che la quantizzazione corrisponda a quella attesa dal motore. |
| `DS4_DIAG` | `=1` | off | **Run diagnostico completo**. Prima misura il disco con `F_NOCACHE`, stampa i knob attivi e controlla se nel GGUF ci sono pesi MTP. Dopo il decode stampa routing per layer, concentrazione degli esperti, slot assegnati e banda effettiva del gather. È il punto di partenza per decidere se lavorare su SSD/page-cache, cache esperti o profiling dei kernel. |
| `DS4_ACTIVE_EXPERTS` | `1…6` | `6` | Riduce quanti esperti MoE vengono realmente usati per token. Abbassa I/O e tempo di gather, ma cambia la qualità perché il modello Flash è tarato su 6 esperti. Utile come modalità degradata su poca RAM o per capire quanto del tempo è dovuto agli esperti. Clampato a `[1, k]`. |
| `DS4_USAGE_FILE` | path, oppure `off` | `<gguf-path>.usage.json` | File JSON della **usage imatrix**, cioè lo storico degli esperti scelti dal router. Se lo lasci attivo, il run successivo può pre-warmare la cache sugli esperti già risultati caldi. Usa un path dedicato per benchmark ripetibili; usa `off` quando vuoi misurare un run freddo senza storia. |
| `DS4_WARMUP` | intero `>=0` | `0` (`DS4_DIAG`: `min(4, maxNew-1)`) | Esclude i primi N token dal profilo di decode. I primi token spesso pagano costi una-tantum: cache fredde, wiring dei buffer, assestamento della memoria. Con `DS4_WARMUP=4`, per esempio, la demo resetta il profiler dopo il quarto token e riporta il throughput di regime separato dal totale. |

### Knob del motore (validi anche per la demo)

La demo costruisce uno `StreamingDecoder`, quindi eredita i knob del motore.
Sono **opt-in / sperimentali**: cambiano prestazioni o RAM, non i numeri (salvo
`DS4_ACTIVE_EXPERTS` e `DS4_FUSED_MOE=0`, che cambiano il risultato di proposito).

| Variabile | Valori | Default | Effetto |
|-----------|--------|---------|---------|
| `DS4_RAW_RING` | `=1` | off | Tiene la raw-KV in un **ring buffer di `nSWA` righe** invece che per tutto il contesto. La raw-KV è la cache usata dalla sliding-window attention; siccome quella attention guarda solo le ultime 128 righe, il ring rende quella parte della memoria costante. Utile su contesti lunghi e RAM stretta; non riduce tutte le cache KV compresse. |
| `DS4_PREFILL_UNION` | intero | `64` | Controlla quanti esperti possono essere raccolti insieme nel prefill layer-major. Valori alti fanno meno round di I/O ma usano più memoria transitoria; valori bassi riducono il picco ma possono rallentare il prefill. Il motore non scende mai sotto `k` (6), perché almeno gli esperti attivi devono essere disponibili. |
| `DS4_EXPERT_CACHE_SLOTS` | intero | `0` (off) | Abilita una cache LRU per layer degli esperti MoE in GPU. Ogni slot costa circa 6.9 MB wired per layer sul modello 2-bit; `8` è il minimo effettivo quando la cache è attiva. Se l'hit-rate sale, il gather da SSD scende. Se la RAM va in pressione, può peggiorare tutto. Su macchine piccole prova `8`, poi `12`, poi `16`. |
| `DS4_EXPERT_CACHE_UNIFORM` | `=1` | off | Disattiva la distribuzione usage-driven degli slot. Di default, a parità di budget totale, i layer con routing più concentrato ricevono più slot e quelli meno concentrati meno. Con `=1` ogni layer riceve lo stesso numero di slot: è utile per misurare se l'allocazione guidata dalla usage imatrix sta davvero aiutando. |
| `DS4_EXPERT_PREAD` | `=1` | off | Legge gli slab degli esperti con `pread` + `F_NOCACHE` direttamente nei buffer di destinazione. In pratica gli esperti non sporcano la page cache del sistema, quindi non cacciano via i pesi densi caldi. È spesso il knob più interessante su 16 GB quando il profilo mostra `route/attn`, `embed` o `head` lenti per continui re-fault da SSD. Numerica invariata; da confrontare A/B perché dipende molto da macchina e SSD. |
| `DS4_PREFETCH` | `=1` | off | Anticipa con `madvise` i pesi non-routed del layer successivo mentre il layer corrente sta calcolando. Può aiutare se il compute lascia tempo all'I/O; può peggiorare se il collo di bottiglia è già il gather degli esperti, perché consuma banda SSD in anticipo. |
| `DS4_PREFETCH_EXPERTS` | intero | `0` | Con `DS4_PREFETCH=1`, anticipa anche N esperti probabili secondo la usage prior. È speculativo: se il router poi sceglie altri esperti, hai letto byte inutili. Usalo solo dopo aver visto con `DS4_DIAG=1` che il routing è concentrato e prevedibile. |
| `DS4_WILLNEED_EXPERTS` | `=0` per disattivare | **on** | Hint non speculativo: appena il router ha scelto gli esperti reali del token, il motore fa `madvise(WILLNEED)` proprio su quegli slab. Non cambia i byte letti né la numerica; prova solo a far partire prima il read-ahead del sistema. È ON di default; `DS4_WILLNEED_EXPERTS=0` serve per misurare il vecchio comportamento on-demand. |
| `DS4_RESIDENT_DENSE` | `=1` | off | Copia i pesi non-esperti per layer, circa 5 GB, in buffer wired invece di lasciarli come mmap no-copy. Aiuta quando gli esperti streaming sfrattano dalla page cache i pesi densi e quindi `route/attn` li rilegge da SSD a ogni token. Costa molta RAM fissa: può migliorare su 24/32 GB, ma su 16 GB può peggiorare se toglie spazio alla cache degli esperti e al sistema. |
| `DS4_FUSED_MOE` | `=0` per disattivare | on | I kernel MoE fusi sono il path normale. Con `DS4_FUSED_MOE=0` usi il path non fuso per confronto A/B e debug numerico. Questo può cambiare arrotondamenti e output, quindi non mischiarlo con benchmark di prestazioni “normali”. |
| `DS4_PROFILE_ROUTE` | `=1` | off | Spezza `route/attn` in sottofasi nel profilo: compressor, proiezioni Q/KV, attention, ecc. È diagnostico: aggiunge sincronizzazioni e può gonfiare i tempi assoluti. Guardalo per capire le proporzioni interne, poi spegnilo per misurare tok/s reali. |
| `DS4_Q8_NSG` | `1…8` | `4` | Regola quanti simdgroup per threadgroup usa il matvec Q8_0 denso. Non cambia il risultato matematico: cambia solo come il lavoro viene partizionato sulla GPU, quindi occupancy e latency hiding. Il default `4` segue il riferimento C; per tuning prova `2/4/6/8` con stesso prompt, stesso `DS4_WARMUP` e stessa usage imatrix. |

### Quale parametro provo?

| Sintomo nel profilo | Primo knob da provare | Perché |
|---|---|---|
| `expert gather` domina e la banda effettiva è molto sotto il tetto SSD | `DS4_WILLNEED_EXPERTS=0` vs default, poi `DS4_EXPERT_PREAD=1` | Capisci se gli hint aiutano; se la page cache viene disturbata, bypassarla può stabilizzare il decode. |
| `route/attn`, `embed` o `head` restano lenti anche dopo i primi token | `DS4_EXPERT_PREAD=1`, poi `DS4_RESIDENT_DENSE=1` su RAM ampia | Probabile re-fault dei pesi densi causato dal churn degli esperti. |
| Hit-rate cache basso ma routing concentrato su pochi esperti | `DS4_EXPERT_CACHE_SLOTS=8/12/16` | Più slot possono trattenere gli esperti caldi e ridurre letture da SSD. |
| Cache esperti attiva ma non migliora | `DS4_EXPERT_CACHE_UNIFORM=1` A/B | Verifica se la redistribuzione usage-driven aiuta o se serve un budget diverso. |
| Run corti molto instabili | `DS4_WARMUP=4` e `DS4_USAGE_FILE=<path>` | Separi costi freddi da regime e rendi ripetibile lo storico del router. |
| Vuoi il massimo su una singola macchina | sweep `DS4_Q8_NSG=2/4/6/8` | L'ottimo del matvec Q8 dipende dal SoC e dalla pressione memoria. |

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

**7. Diagnosi completa delle ottimizzazioni di streaming** — banda del disco,
pesi MTP, routing per layer e verdetto finale (≥48 token per la tabella
allocazione; la cache slot va attivata per misurare gli hit):

```sh
DS4_DIAG=1 DS4_EXPERT_CACHE_SLOTS=8 \
  swift run DS4Demo /path/model.gguf 48 "Raccontami la storia di Roma."
# ── Diagnosi (DS4_DIAG=1) ──
#   knob: DS4_EXPERT_CACHE_SLOTS=8  DS4_WILLNEED_EXPERTS=·  …
#   SSD (F_NOCACHE, slab 2 MB):
#     sequenziale         5.10 GB/s   <- tetto teorico del gather
#     random coda 1       1.80 GB/s   <- gather senza hint/parallelismo
#     random parallelo    4.20 GB/s   <- gather con madvise + copie parallele
#   MTP: nessun peso nel GGUF -> decodifica speculativa NON possibile con questo file
# … generazione + profilo …
# ── Diagnosi cache esperti ──
#   layer   route  conc(top8)  conc(top16)  slot
#       2     288       0.41        0.55    14
#   …
#   gather effettivo 3.90 GB/s = 76% del tetto SSD (5.10 GB/s) -> vicino alla fisica del disco: puntare su hit-rate/MTP
```

Gli A/B utili con questa modalità: `DS4_WILLNEED_EXPERTS=0` vs default (quanto
rendono gli hint), `DS4_EXPERT_CACHE_UNIFORM=1` vs default (quanto rende
l'allocazione usage-driven), sweep di `DS4_EXPERT_CACHE_SLOTS` (8/12/16) e di
`DS4_Q8_NSG` (2/4/6/8).

**8. Profilo pulito del regime** — ignora i primi token e conserva/azzera la
usage imatrix in modo esplicito:

```sh
DS4_WARMUP=4 DS4_USAGE_FILE=/tmp/ds4-demo-usage.json \
  swift run DS4Demo /path/model.gguf 32 "Scrivi tre frasi su Metal."

DS4_USAGE_FILE=off swift run DS4Demo /path/model.gguf 8 "Run senza storia."
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
