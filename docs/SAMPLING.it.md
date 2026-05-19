# Il sampler

Come l'engine trasforma una riga di logit `[1, vocab]` nel prossimo
token id. Apri questo documento quando vuoi sapere cosa fa ogni flag
di sampling, in che ordine, e come aggiungerne uno nuovo.

I documenti complementari coprono i pezzi attorno:

- [`MODEL.md`](MODEL.md) — cosa produce i logit.
- [`MODULES.md`](MODULES.md) — indice per-file di `Sources/`.
- [`USAGE.md`](USAGE.md) — flag CLI che mappano a `SamplingOptions`.
- [`EXAMPLES.md`](EXAMPLES.md) — ricette per config sampler comuni.

> 🇬🇧 La versione inglese è [`SAMPLING.md`](SAMPLING.md).

---

## 1. Cos'è il sampler

Una pipeline di trasformazioni indipendenti su un array di logit
grande quanto il vocabolario, che termina con un token id. Riferimento:
`Sources/DeepSeekKit/Sampling.swift:35` (struct `SamplingOptions`) e
`Sources/DeepSeekKit/Sampling.swift:108` (`enum Sampler`).

Ogni stage ha un default "neutro" che lo disabilita. Quando *ogni*
stage è neutro e `temperature == 0`, la pipeline accorcia direttamente
a un argmax lato GPU. Altrimenti l'intero buffer di logit viene letto
in host una volta e il resto gira in Swift (con `Accelerate` SIMD per
i loop caldi).

Il reference Python implementa solo Gumbel-max + temperature
(`Reference/inference/generate.py:19-24`). Ogni altro stage in questo
file è stato aggiunto dal port Swift per allineare la toolbox de-facto
LLM (llama.cpp, vLLM, koboldcpp).

```
logits [1, vocab] f32 su GPU
        │
        ├─ allFiltersDisabled?  ── sì ──►  Sampler.argmax(logits) → id (early return)
        ▼
read in host come [Float]
        │
        ▼
1. temperature scaling          (vDSP_vsmul, vettorizzato)
        │
        ▼
2a. repetition penalty          (per-history-id)
2b. frequency + presence        (per-history-id con count)
        │
        ▼
options.mirostatTau > 0?  ── sì ──►  mirostatV2Sample(...)  → id (terminale)
        │ no
        ▼
3. top-K filter                 (quickselect nth-largest)
4. min-P filter                 (softmax vettorizzato + threshold)
5. tail-free                    (derivata seconda su prob ordinate)
6. locally-typical              (sort per |surprise - entropy|)
7. top-P (nucleus)              (cumulativo ordinato)
        │
        ▼
8. Gumbel-max multinomial       (argmax(log p + g) con g ~ Gumbel(0,1))
        │
        ▼
prossimo token id
```

`options.rngState` (seed LCG) e `options.mirostatMu` (stima rolling di
Mirostat) sono mutati in place tra una call e l'altra — ecco perché
`Sampler.sample(_:history:options:)` prende `options` come `inout`.

---

## 2. `SamplingOptions`

```swift
public struct SamplingOptions {
    public var temperature: Float = 1.0
    public var topK: Int = 0                    // 0 = disabilitato
    public var topP: Float = 1.0                // 1.0 = disabilitato
    public var minP: Float = 0.0                // 0 = disabilitato
    public var tailFree: Float = 1.0            // 1 = disabilitato
    public var typical: Float = 1.0             // 1 = disabilitato
    public var repetitionPenalty: Float = 1.0   // 1.0 = disabilitato
    public var frequencyPenalty: Float = 0.0    // 0 = disabilitato
    public var presencePenalty: Float = 0.0     // 0 = disabilitato
    public var mirostatTau: Float = 0.0         // 0 = disabilitato (usa step 3-8 invece)
    public var mirostatEta: Float = 0.1
    public var mirostatMu: Float = 10.0         // stima rolling, aggiornata in place
    public var rngState: UInt64 = defaultSamplerSeed()
}
```

Ogni campo è `var` così il caller può tweakare valori tra turn — il
tab Generation della GUI, i sampling default degli agent, e i flag
CLI condividono questa struct.

### Seed di default

`defaultSamplerSeed()` mixa nanosecondi wall-clock e process id tramite
un passo LCG. Due run consecutivi della CLI con stesso prompt e
`temperature > 0` quindi producono stream diversi. Per riproducibilità,
setta `rngState` esplicitamente.

### Cheap shortcut

`allFiltersDisabled` è true iff ogni filtro è al default neutro *e*
`temperature == 0`. La call sample lo rileva e fa short-circuit a
`argmax(logits)` su GPU, saltando il read-to-host interamente.
L'invocazione CLI con `--temperature 0` finisce qui quando anche gli
altri flag sono al default.

---

## 3. La pipeline, passo per passo

L'ordine degli stage è fisso e matcha quello che fanno llama.cpp e
koboldcpp. Riordinare cambierebbe la matematica (es. top-P dopo min-P
vede una distribuzione diversa da top-P prima).

### 3.1 Argmax (shortcut GPU-only)

`Sampler.argmax(_:)` — `Sources/DeepSeekKit/Sampling.swift:136`.

Due path, scelti dalla vocab size `V`:

- **Single threadgroup** (`V < 8192`): un dispatch `argmax_f32` con
  256 thread che fa una riduzione ad albero sul buffer di logit.
  Un commit di command buffer, un `UInt32` scritto in un shared
  buffer.
- **Multi-stage** (`V ≥ 8192`): stage 1 splitta il vocab in tile di
  `argmaxTileSize = 2048` e produce `M = ceil(V / tileSize)` coppie
  `(val, idx)` parziali in buffer private-storage. Stage 2 riduce
  gli `M` parziali nella risposta finale. Due invocazioni encoder,
  un commit. Per V=129 280 (V4-Flash), M = 64 threadgroup in
  parallelo — satura una Apple GPU da 10-40 core e taglia la
  latenza argmax di ~5-8× vs il path single-threadgroup.

La soglia (8192) e tile size (2048) sono tunate empiricamente; sotto
8192 l'overhead di kernel dispatch supera il guadagno di parallelismo.

### 3.2 Temperature scaling lato GPU

`Sampler.applyTemperature(_:_:)` esiste ma la pipeline completa
`Sampler.sample(...)` non lo usa — `T == 0` fa shortcut ad argmax, e
qualsiasi altra temperature è applicata host-side via `vDSP_vsmul`
dopo che i logit sono letti.

L'helper è tenuto per caller che vogliono scalare i logit in-place
senza poi campionare (es. visualizzatori).

### 3.3 Temperature (host-side, vettorizzato)

```
arr[i] *= 1 / max(T, 1e-5)
```

Fatto via `vDSP_vsmul` (Accelerate). ~3-5× più veloce di un loop
Swift per-elemento a V=130 k.

Saltato quando `T == 1.0`. Il floor `1e-5` impedisce a una richiesta
`T == 0` di esplodere (in pratica viene filtrata prima da
`allFiltersDisabled`).

### 3.4 Repetition penalty (stile HuggingFace)

Per ogni token id visto in `history`:

```
if arr[id] >= 0: arr[id] /= rep_penalty
else:            arr[id] *= rep_penalty
```

Moltiplicativa — sign-dependent così la penalty rende
consistentemente il logit più piccolo in magnitudine
indipendentemente dal segno. `rep_penalty == 1.0` disabilita.

### 3.5 Frequency + presence penalty (stile OpenAI)

Entrambe sono additive nello spazio logit; entrambe girano nello
stesso loop dopo aver contato `counts[id]` per ogni id in `history`:

```
if counts[id] > 0:
    arr[id] -= freq_pen * counts[id]
    arr[id] -= pres_pen
```

`freq_pen == 0 && pres_pen == 0` disabilita (l'intero branch è
saltato).

Si compongono con `repetitionPenalty` — NON sono mutuamente
esclusive. Consiglio pratico: usa una *o* l'altra nel preset di un
agente.

### 3.6 Mirostat v2 (terminale)

`mirostatV2Sample(...)` — `Sampler.swift:457`. Quando
`options.mirostatTau > 0`, l'intera catena `(top-K | min-P | tail-free
| typical | top-P | Gumbel-max)` è sostituita da:

1. Softmax per ottenere `probs[]` (stabile numericamente, via
   `softmaxDouble(...)`).
2. Sort dei token per probabilità decrescente.
3. Truncate: tieni il prefisso dove surprise `-log(p_i) < μ`.
   (`options.mirostatMu` è la stima rolling della surprise
   accettabile, inizializzata a `2τ` per convenzione.)
4. Sample Gumbel-max sul set tenuto.
5. Aggiorna μ in place: `μ ← μ − η · (S_t − τ)`, dove
   `S_t = -log(p_selected)`. Clampato a `≥ 0.01`.

L'update di Mirostat mantiene la surprise *media* dei token generati
vicina al target `τ`, che si traduce in perplessità ≈ `2^τ`. Un valore
τ tipico è 5.0 (perplessità target ≈ 32).

Riferimento: Basu et al. 2020, "Mirostat: A Neural Text Decoding
Algorithm that Directly Controls Perplexity".

### 3.7 Top-K filter

`nthLargest(arr, k: options.topK)` trova il K-esimo valore più grande
via quickselect (O(N) medio). Tutto strettamente più piccolo diventa
`-Float.infinity` — che Gumbel-max poi ignora, e che gli step
softmax-based downstream trattano come massa zero.

Saltato quando `topK == 0` (il default disabilitato) o `topK >= V`.

### 3.8 Min-P filter (vettorizzato)

`applyMinP(...)` — tiene solo i token la cui probabilità è almeno
`minP × max_prob`. Implementato interamente con primitive Accelerate:

1. `vDSP_maxv` → max logit.
2. `vDSP_vsadd` → shift di `-m`.
3. `vDSP_vspdp` → promuove Float a Double (la somma softmax è
   calcolata in double per stabilità numerica).
4. `vvexp` → esponenziale per-elemento.
5. `vDSP_sveD` → somma.
6. `vDSP_vsmulD` → divide per somma (prob normalizzate).
7. `vDSP_maxvD` → max prob.
8. Threshold = `pMax · minP`; entry con prob < threshold → `-inf`.

Valori `minP` tipici: 0.05 - 0.10 (il range llama.cpp).

### 3.9 Tail-free sampling

`applyTailFree(...)` — la "tail" è definita come la parte della curva
di probabilità ordinata dove la derivata seconda `|p_{i+2} - 2p_{i+1}
+ p_i|` ha accumulato oltre `z` del suo totale. Tutto oltre quel
cutoff è mascherato.

In pratica preserva la "testa" più una breve zona di transizione e
taglia la lunga coda piatta. Disabilitato a `z == 1`; valori utili:
0.95.

### 3.10 Locally-typical sampling

`applyTypical(...)` — tiene i token la cui surprise `-log(p_i)` è più
vicina all'entropia della distribuzione `H = -Σ p log p`, ordinati per
`|s_i − H|`, finché la massa cumulativa raggiunge `p`.

Disabilitato a `typical == 1`. Valori tipici: 0.95.

L'intuizione: un modello equo dovrebbe produrre token la cui
information content matcha l'entropia della distribuzione. I token alle
code (troppo comuni = noiosi; troppo rari = noisy) sono tagliati.

### 3.11 Top-P (nucleus) filter

`applyTopP(...)` — ordina le probabilità softmaxate, percorre la
somma cumulativa, e maschera ogni token sotto la cutoff probability
che prima fa raggiungere alla massa cumulativa `topP`.

Disabilitato a `topP == 1`.

### 3.12 Gumbel-max multinomial

`Sampler.sample(...)` linee 308-326. Dopo che ogni filtro è girato,
il sampler sceglie il token il cui `log(p_i) + g_i` è più grande,
dove `g_i ~ Gumbel(0, 1)`.

Gumbel-max è matematicamente equivalente a un sample categorico da
`softmax(logits)`, ma evita di costruire esplicitamente la
distribuzione: niente normalizzazione, niente CDF, solo un `argmax`
dopo aver aggiunto rumore. Riferimento:
`Reference/inference/generate.py:19-24`.

Calcolo per-step (nell'engine):

```
for i in 0..<V:
    if arr[i] == -inf: continue          # mascherato da filtri precedenti
    u ~ Uniform(0, 1)
    g = -log(-log(u))                    # Gumbel(0,1)
    key = (arr[i] - max_logit) + g
    track argmax(key)
```

Lo shift `max_logit` costante non cambia l'argmax — è solo per tenere
l'aritmetica lontana dall'overflow FP32. Il fallback difensivo
(`if !anyFinite`) cattura la combo rara dove ogni token è stato
mascherato: ritorna l'argmax GPU dei logit *originali*.

### 3.13 RNG

`nextUnit(_:)` è uno step LCG inline (`state = state *
6364136223846793005 + 1442695040888963407`, poi prendi i 53 bit più
alti). Uno state per `SamplingOptions`; il caller è responsabile di
tenere la stessa istanza di `SamplingOptions` tra step di decode se
vuole uno stream singolo.

---

## 4. State per-istanza

Due campi mutano tra le call:

- `rngState` — il seed LCG, avanzato da `nextUnit` su ogni token
  campionato. Aggiornato prima che `Sampler.sample` ritorni.
- `mirostatMu` — la stima rolling di surprise di Mirostat. Aggiornata
  solo dentro `mirostatV2Sample` dopo la pick Gumbel-max.

Tutto il resto (`temperature`, `topK`, …) sono "settings": puoi
modificarli tra turn senza rompere la continuità.

La chat surface tiene un `SamplingOptions` per conversazione; la CLI
ne costruisce uno dai flag command-line allo startup e lo riusa per
l'intero decode loop. Riproducibilità è quindi "fissa `rngState` a
una costante; tieni `temperature`, `topK`, etc. costanti; resetta
`mirostatMu` a `2 * mirostatTau` se Mirostat è attivo".

---

## 5. Setting raccomandati

Questi sono i valori che il README e la UI Settings suggeriscono di
default:

| Use case | T | top-K | top-P | min-P | rep_pen |
|---|---|---|---|---|---|
| Q&A brevi | 0.7 | 0 | 0.9 | 0 | 1.0 |
| Coding | 0.6 | 0 | 0.85 | 0 | 1.0 |
| Long-form / creativo | 0.85 | 0 | 0.95 | 0.05 | 1.05 |
| Determinismo (CI) | 0 (argmax) | 0 | 1 | 0 | 1 |

**Gotcha critico per DeepSeek-V4-Flash**: a `--temperature 0` il
router MoE cade in loop self-reinforcing (`好的好的好的…`). La GUI
clampa lo slider di temperature a `[0.5, 1.0]` per quel motivo; la
CLI accetta 0 ma il README raccomanda 0.7. Vedi sezione Recommended
values del README.

**Mirostat vs filtri statici**: non mixare. Il path Mirostat non
rispetta top-K / top-P / min-P / tail-free / typical — accenderlo
disabilita quell'intero branch. Se setti `mirostatTau > 0` insieme a
`topK == 50`, il valore topK è silenziosamente ignorato.

---

## 6. Performance

L'intero sampler gira in O(V) host-side dopo un singolo sync CPU-GPU
per leggere i logit. La vettorizzazione via Accelerate è
l'ottimizzazione portante:

- **softmax (precisione Double)**: `vDSP_maxv` + `vDSP_vsadd` +
  `vDSP_vspdp` + `vvexp` + `vDSP_sveD` + `vDSP_vsmulD`. ~3-5× più
  veloce del loop Swift equivalente a V=130 k.
- **temperature**: `vDSP_vsmul` (~3-5× speedup).
- **min-P**: stessa pipeline softmax + singolo threshold pass.

Gli stage rimanenti (`top-K nth-largest`, derivata seconda di
`tail-free`, surprise per-token di `typical`) sono O(V log V) per via
del sort. Per V=130 k è comunque sotto il millisecondo su Apple
Silicon, piccolo rispetto al forward pass del modello.

Il path shortcut argmax salta il read-to-host interamente — al
sampling greedy l'unico lavoro host è un singolo load buffer di 4
byte.

---

## 7. Mapping dei flag CLI

La CLI locale espone ogni parametro di sampling come flag. Vedi
[`USAGE.md`](USAGE.md) per il riferimento canonico; mappa rapida:

| Flag | Campo |
|---|---|
| `--temperature T` | `temperature` |
| `--top-k K` | `topK` |
| `--top-p P` | `topP` |
| `--min-p P` | `minP` |
| `--tfs Z` | `tailFree` |
| `--typical P` | `typical` |
| `--repetition-penalty P` | `repetitionPenalty` |
| `--frequency-penalty P` | `frequencyPenalty` |
| `--presence-penalty P` | `presencePenalty` |
| `--mirostat τ` | `mirostatTau` |
| `--mirostat-eta η` | `mirostatEta` |

Il path remoto OpenRouter passa qualunque subset OpenRouter accetta
nel JSON body (`temperature`, `top_p`, `frequency_penalty`,
`presence_penalty`); Mirostat / tail-free / typical / min-P sono
local-only.

---

## 8. Aggiungere un nuovo sampler

La pipeline vive in `Sampler.sample(_:history:options:)`. I nuovi
filtri si inseriscono tra una coppia esistente, dopo il blocco
temperature / penalty e prima di Gumbel-max.

Checklist:

1. Aggiungi un campo a `SamplingOptions` con un *default neutro* che
   lascia la distribuzione invariata. Aggiorna `allFiltersDisabled` così
   il valore neutro del nuovo campo è richiesto per lo shortcut
   argmax-GPU.
2. Scrivi il filtro come `private static func applyXxx(_ arr: inout
   [Float], vocabSize V: Int, ...)`. Rispetta la convenzione di
   masking in-place `arr[i] = -.infinity` così Gumbel-max ignora i
   token filtrati gratis.
3. Se il filtro ha bisogno di prob softmax, chiama `softmaxDouble`
   (già vettorizzata con Accelerate); non ri-implementare.
4. Aggiungi il flag CLI in `Sources/deepseek/main.swift` e lo slider
   GUI in
   `Sources/DeepSeekUI/Views/Settings/GenerationSettingsTab.swift`.
5. Aggiungi un XCTest in `Tests/DeepSeekKitTests/SamplerTests.swift`
   che verifichi che il default-disabilitato produce output identico
   a "nessun filtro" e che un valore estremo produce il comportamento
   degenerato atteso (es. top-K=1 forza argmax).

---

## 9. Source map

| Topic | File |
|---|---|
| Struct `SamplingOptions` | `Sources/DeepSeekKit/Sampling.swift:35` |
| `Sampler.argmax` (GPU multi-stage) | `Sources/DeepSeekKit/Sampling.swift:136` |
| `Sampler.applyTemperature` | `Sources/DeepSeekKit/Sampling.swift:204` |
| `Sampler.sample` (pipeline completa) | `Sources/DeepSeekKit/Sampling.swift:224` |
| `applyMinP` / `applyTailFree` / `applyTypical` / `applyTopP` | stesso file, §"Filter helpers" |
| `mirostatV2Sample` | `Sources/DeepSeekKit/Sampling.swift:457` |
| `softmaxDouble` (vettorizzato) | `Sources/DeepSeekKit/Sampling.swift:509` |
| `nthLargest` / `nextUnit` | `Sources/DeepSeekKit/Sampling.swift:562` / `:583` |
| Kernel Metal (`argmax_f32`, `apply_temperature`) | `Sources/DeepSeekKit/Kernels/sampling.metal` |
| Riferimento Python (solo `sample`) | `Reference/inference/generate.py:19-24` |
| Plumbing dei flag CLI | `Sources/deepseek/main.swift` |
| Slider GUI | `Sources/DeepSeekUI/Views/Settings/GenerationSettingsTab.swift` |
| XCTest | `Tests/DeepSeekKitTests/SamplerTests.swift` |
