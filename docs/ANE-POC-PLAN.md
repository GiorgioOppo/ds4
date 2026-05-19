# Piano di implementazione — 3 POC ANE/NPU

Piano operativo per i 3 POC deferred in [TODO §11 "Esperimenti ANE"](../TODO.md)
e citati in [`NPU-ANE-ANALYSIS.md`](NPU-ANE-ANALYSIS.md) §8.

Scopo: produrre numeri misurati che confermino (o ribaltino) le
conclusioni dell'analisi *prima* di archiviare definitivamente il
filone NPU. Ogni POC è progettato per stare in 2-3 giorni e dare
**un risultato comunicabile** (latency / acceptance rate / scheduling
trace).

## Prerequisiti trasversali

Vincoli hardware/software validi per tutti e 3 i POC:

- **Mac con Apple Silicon, macOS ≥ 14**, ideale M3 / M3 Max o M4. Su
  M5+ aggiungere una run per misurare l'effetto dei GPU Neural
  Accelerators (relevant per POC 1).
- **Xcode 15+** con Command Line Tools, Swift 5.9+.
- **Python 3.11 + `coremltools` ≥ 8.0 + `torch` ≥ 2.4** in un venv
  dedicato sotto `Tools/ane-bench/.venv/` per la fase di conversione
  modello (lato Mac, non containerizzabile).
- **Permessi**: `powermetrics` richiede sudo per ispezionare l'ANE.
  Documentare le invocazioni `sudo`-only nel README di ogni tool.
- **Instruments**: per i trace dell'ANE servono i template "Metal
  System Trace" + "CPU Counters" (l'ANE non ha un template proprio,
  si guarda l'attività via Activity Monitor → Energy → ANE column o
  via `powermetrics --samplers ane_power`).

**Importante**: nessuno di questi POC è eseguibile nel container di
sviluppo Linux remoto. Tutta la fase di run-and-measure deve avvenire
su una macchina Apple Silicon fisica.

## Output condiviso

Ogni POC produce:

1. Un **report markdown** sotto `docs/ANE-POC-<N>-RESULTS.md` con
   metriche grezze + analisi.
2. Uno **screenshot Instruments** (se applicabile) committato sotto
   `docs/assets/ane-poc-<N>/`.
3. Un **commit di chiusura** che aggiorna lo stato in TODO §11 da
   "deferred" a `[x]` con link al report.

## Sequenza consigliata


```
POC 3 (MPSGraph smoke)  →  POC 1 (microbench)  →  POC 2 (draft model)
   ~1 giorno                 ~2-3 giorni            ~3-5 giorni
   exit gate: SCRAP          exit gate: SCRAP       exit gate: PROCEED?
```



Razionale: POC 3 è il più economico e dà la risposta "MPSGraph mi
risparmia il lavoro?". Se la risposta è no (atteso), POC 1 misura
quanto un singolo offload chirurgico costa davvero. Se POC 1 mostra
che il sync ANE↔GPU domina (atteso), POC 2 è inutile come
"speculative decoding economico" e si chiude tutto.

Se a un qualunque gate il risultato sorprende positivamente, si
procede al successivo.

---

## POC 3 — MPSGraph attention smoke test

**Goal**: verificare se MPSGraph schedula `scaledDotProductAttention`
sull'ANE o ricade su GPU, con e senza mask sliding-window.

### Scope

In: una singola attention "decode-style" (`B=1, H=64, S=1,
T=2048, D=64`) eseguita 1000× e ispezionata con `powermetrics`.

Out: tutto il resto. Non riscriviamo la nostra attention, non
integriamo niente in `MLA.swift`. Sola misura osservativa.

### Step

1. **Setup**. Nuova directory `Tools/ane-poc-3-mpsgraph/`.
   Aggiunta target eseguibile in `Package.swift`:
   ```swift
   .executableTarget(
       name: "ane-poc-3-mpsgraph",
       path: "Tools/ane-poc-3-mpsgraph"
   )
   ```

2. **Swift binary** `Tools/ane-poc-3-mpsgraph/main.swift`:
   - Costruisce un `MPSGraph` con Q, K, V come `MPSGraphTensorData`
     in FP16, shape `[1, 64, 1, 64]`, `[1, 64, 2048, 64]`,
     `[1, 64, 2048, 64]`.
   - Nodo `scaledDotProductAttention(...)`.
   - Compila con `MPSGraphCompilationDescriptor` con
     `optimizationLevel = .level1`.
   - Loop 1000× su `executeAsync(...)` (per dare a `powermetrics`
     tempo di campionare).
   - Stampa la latency media + p95.

3. **Trace dell'ANE**. Mentre la binary gira (loop esteso a 30 s):
   ```sh
   sudo powermetrics --samplers ane_power,gpu_power -i 200 -n 150 \
       > docs/assets/ane-poc-3/powermetrics.log
   ```
   Cerchiamo se la colonna `ANE Power` è > 0 mentre la GPU è attiva.

4. **Variante con mask custom**. Aggiungere uno step (b) con
   sliding-window causal mask `[1, 1, 1, 2048]` passata come tensore
   addizionale + somma logits. Ripetere `powermetrics`.

5. **Report** `docs/ANE-POC-3-RESULTS.md`:
   - Tabella: `{configurazione → ANE Power on/off, latency, GPU%}`
   - Conclusione: "MPSGraph SDPA su questa hardware schedula su X".

### Files toccati

- `Package.swift` (+1 target)
- `Tools/ane-poc-3-mpsgraph/main.swift` (~120 LOC)
- `docs/ANE-POC-3-RESULTS.md` (nuovo)
- `docs/assets/ane-poc-3/` (log + screenshot)

### Acceptance criteria

- Binary compila e gira 30 s senza crash.
- `powermetrics` log committato.
- Report fornisce conclusione binaria documentata.

### Rischi

- `powermetrics` su macOS recenti potrebbe richiedere
  `--samplers ane_power` invertito o rinominato. Fallback: leggere
  Activity Monitor → Energy.
- `scaledDotProductAttention` MPSGraph è stato aggiunto in iOS 16.4 /
  macOS 13.3; su versioni più vecchie va espresso a mano come catena
  matmul + softmax + matmul.

### Stima: 1 giorno uomo

---

## POC 1 — Microbenchmark ANE vs Metal su `wq_b`

**Goal**: misurare la latency di una `Linear` dense (shape realistica
di `wq_b` in V4-Pro) eseguita su ANE via CoreML vs sui nostri kernel
`gemm_bf16` / `simdgroup_matrix`, **includendo** il costo di
sincronizzazione GPU↔ANE in uno scenario realistico (input vivente
sulla GPU, output consumato dalla GPU).

### Scope

In: una singola matmul `[1, 2048] × [2048, 32768] → [1, 32768]`
(shape default `q_lora_rank × (n_heads × head_dim)` da
`Reference/inference/model.py`).

Out: nessuna integrazione nel forward pass. Nessuna conversione di
altri layer. Una sola GEMM, isolata.

### Step

1. **Conversione modello** (Python, lato Mac).
   `Tools/ane-poc-1-microbench/build_mlpackage.py`:
   ```python
   import torch, coremltools as ct
   m = torch.nn.Linear(2048, 32768, bias=False).half().eval()
   ex = torch.jit.trace(m, torch.randn(1, 2048, dtype=torch.float16))
   mlmodel = ct.convert(
       ex,
       inputs=[ct.TensorType(shape=(1, 2048), dtype=ct.converters.mil.mil.types.fp16)],
       compute_units=ct.ComputeUnit.CPU_AND_NE,
       compute_precision=ct.precision.FLOAT16,
       minimum_deployment_target=ct.target.macOS14,
   )
   mlmodel.save("Tools/ane-poc-1-microbench/wq_b.mlpackage")
   ```

2. **Swift binary** `Tools/ane-poc-1-microbench/main.swift`:
   - Carica `wq_b.mlpackage` con `MLModelConfiguration.computeUnits =
     .cpuAndNeuralEngine`.
   - Verifica che il modello giri davvero su ANE controllando
     `MLComputeDevice` (`.neuralEngine` deve apparire nel device
     plan).
   - Pre-popola un MTLBuffer FP16 di shape `[1, 2048]` su GPU con
     random data.
   - **Scenario "ANE pure"**: copia GPU→CPU→ANE→CPU→GPU per ogni
     iterazione (worst case: simula tile che entra ed esce dalla
     GPU). 1000 iterazioni, misura p50/p95/p99.
   - **Scenario "ANE batched"**: 1000 iterazioni back-to-back senza
     uscita su GPU (best case ANE). Misura come sopra.
   - **Baseline Metal**: stesso input, lancia `gemm_bf16_to_f32`
     usando il wrapper esistente in `Layers/Linear.swift`. 1000
     iterazioni con `MTLCommandBuffer.commit + waitUntilCompleted`.
     Misura come sopra.
   - **Baseline Metal MPS**: stessa GEMM via `MPSMatrixMultiplication`
     (per avere un secondo punto di riferimento "Apple-blessed").

3. **Output CSV** `docs/assets/ane-poc-1/latencies.csv` con colonne
   `{scenario, p50_us, p95_us, p99_us, mean_us, throughput_gflops}`.

4. **Report** `docs/ANE-POC-1-RESULTS.md`:
   - Tabella latency + conclusione: "ANE wins / loses by X on
     this shape".
   - Calcolo break-even: a quale shape (M, N, K) ANE pareggia? È
     una shape che compare nel forward V4-Pro?

### Files toccati

- `Tools/ane-poc-1-microbench/build_mlpackage.py` (~30 LOC)
- `Tools/ane-poc-1-microbench/main.swift` (~250 LOC)
- `Tools/ane-poc-1-microbench/README.md` (istruzioni run)
- `Package.swift` (+1 target)
- `docs/ANE-POC-1-RESULTS.md` (nuovo)
- `docs/assets/ane-poc-1/` (CSV + Instruments screenshot)

### Acceptance criteria

- Conversion script produce `.mlpackage` deterministicamente.
- Binary gira tutti e 4 gli scenari su 1000 iter senza crash.
- Report contiene tabella latency + break-even analysis.
- Verifica esplicita (con log) che il modello CoreML stia girando
  su ANE e non su CPU fallback.

### Rischi

- **CoreML potrebbe ricadere su CPU**. Mitigazione: stampare
  `model.modelDescription.computeDeviceUsage` (API recente) per
  confermare. Se cade su CPU, segnalare nel report e tentare con
  `compute_units=ct.ComputeUnit.CPU_AND_NE` vs `ALL`.
- **Shape `(1, 2048) × (2048, 32768)`** potrebbe eccedere lo SRAM
  cliff ANE (`32768 × 2 bytes = 64 KB` per output row, plus weight
  tile). Aspettarsi un cliff nel risultato. Aggiungere una run con
  shape ridotte `(1, 2048) × (2048, 2048)` per confronto.
- **Sync cost dominante**. Atteso. È esattamente il dato che vogliamo
  misurare.

### Stima: 2-3 giorni uomo

---

## POC 2 — Draft model standalone su ANE

**Goal**: caricare un modello dense 1-3B come `.mlpackage` su ANE,
misurarne tok/s standalone, e quantificare l'acceptance rate
*teorico* con V4-Pro su un corpus fisso. Niente integrazione
end-to-end in `Generation.swift` — quella è una fase 2 da aprire
solo se i numeri qui sono buoni.

### Scope

In: misure isolate del draft model + acceptance rate offline su un
corpus di ~100 prompt.

Out: integrazione speculative decoding live nel forward pass, gestione
KV cache draft, retry su mismatch — tutta roba per il follow-up.

### Step

1. **Scelta del draft model**. Default: `meta-llama/Llama-3.2-1B-Instruct`
   (1.2 B dense, FP16, contesto 4096). Alternativa:
   `apple/OpenELM-1_1B-Instruct` (più piccolo, FP16-friendly). Vincolo:
   deve esistere già una conversione CoreML pubblica oppure essere
   convertibile via `coremltools` senza patch.

   ⚠️ **Issue tokenizer**: i tokenizer di V4 e Llama sono *diversi*.
   La forma più semplice di speculative decoding richiede che draft e
   target condividano il vocabolario. Strategie:
   - **(a) Re-tokenize per prefix-match**: il draft genera testo, il
     prefisso comune viene rilevato a livello stringhe e re-tokenizzato
     col tokenizer V4. Funziona ma riduce acceptance rate.
   - **(b) Restringere a un draft model che usa lo stesso tokenizer
     V4**. Non esiste oggi pubblicamente. Out of scope per POC.

   Scegliamo (a). Documentare nel report che acceptance rate misurato
   è un *lower bound* — un draft custom con vocab condiviso farebbe meglio.

2. **Conversione + caricamento**. Riusare la pipeline `coremltools`
   già impostata in POC 1. Output: `Tools/ane-poc-2-draft/draft.mlpackage`.

3. **Swift binary** `Tools/ane-poc-2-draft/main.swift`:
   - Carica `draft.mlpackage` su ANE.
   - Carica un tokenizer Llama-3.2 via `swift-transformers` (dipendenza
     SPM da `huggingface/swift-transformers`) — *l'unica nuova
     dipendenza esterna del piano*.
   - Loop di benchmark "draft standalone": per 100 prompt da un corpus
     fisso (committato sotto `docs/assets/ane-poc-2/corpus.txt`),
     genera 64 token autoregressive, misura tok/s p50/p95.

4. **Acceptance rate offline**. Stesso corpus, per ogni prompt:
   - Il draft propone N=8 token successivi.
   - Tokenizziamo il risultato col tokenizer V4 (path attuale in
     `DeepSeekKit`).
   - Confrontiamo i primi K token che V4 *avrebbe* generato
     (usando il path locale esistente con greedy decoding) con quelli
     proposti dal draft.
   - Misuriamo: numero medio di token accettati per chiamata draft
     (= acceptance length).

5. **Modello di speedup**. Formula classica speculative decoding:
   `speedup = (1 - α^(N+1)) / ((1-α) * (1 + c))`
   dove `α` = acceptance prob per-token, `N` = draft length, `c` =
   costo draft / costo target. Inseriamo i nostri numeri.

6. **Report** `docs/ANE-POC-2-RESULTS.md`:
   - Tabella: draft tok/s su ANE vs lo stesso modello su Metal
     (tramite la nostra Llama path che già esiste in
     `LlamaModel.swift`).
   - Acceptance rate distribution.
   - Speedup predetto end-to-end.
   - Verdetto: "vale la pena costruire la fase 2?".

### Files toccati

- `Tools/ane-poc-2-draft/build_mlpackage.py` (~50 LOC, riusa pattern
  POC 1)
- `Tools/ane-poc-2-draft/main.swift` (~400 LOC)
- `Tools/ane-poc-2-draft/README.md`
- `Package.swift` (+1 target, +1 SPM dependency
  `huggingface/swift-transformers`)
- `docs/ANE-POC-2-RESULTS.md` (nuovo)
- `docs/assets/ane-poc-2/corpus.txt` (~100 prompt sample)

### Acceptance criteria

- Draft `.mlpackage` carica e gira su ANE (verifica via
  `computeDeviceUsage`).
- Benchmark tok/s standalone su 100 prompt completo.
- Acceptance rate misurato e committato come CSV grezzo.
- Speedup predetto calcolato esplicitamente nel report.

### Rischi

- **Conversione Llama-3.2-1B → CoreML**. Apple ha guide ufficiali
  (`huggingface/swift-transformers` ha esempi), ma non è
  garantito che la conversione preservi quality. Mitigazione:
  validare con un prompt fisso che draft FP16 CoreML e draft FP16
  PyTorch producano gli stessi token (top-1 match).
- **`swift-transformers` non in repo**. Aggiungere come SPM
  dependency è un cambio non banale (toglie il claim "DeepSeekKit
  ha 0 dipendenze runtime non-Apple"). Confinare la dipendenza al
  solo target `ane-poc-2-draft` lascia DeepSeekKit pulito.
- **Acceptance rate troppo basso con re-tokenize**. Atteso. Il
  report deve dire chiaramente "lower bound" e proporre il follow-up
  (draft con vocab V4).
- **Memoria**. Llama-3.2-1B FP16 = ~2.4 GB. KV cache ~64 MB.
  Sta tutto in ANE working set di 3 GB, ma giusto.

### Stima: 3-5 giorni uomo

---

## Exit gates e decision tree


```
POC 3 (MPSGraph smoke)
  │
  ├── ANE scheduling osservato? ───── No (atteso)
  │                              │
  │   Sì (sorprendente)          ▼
  │      │              ┌──────────────────┐
  │      ▼              │  Procedi a POC 1 │
  │  Apri filone        └────────┬─────────┘
  │  "MPSGraph                   │
  │   attention" come    POC 1 (microbench)
  │   path principale,     │
  │   chiudi POC 2.        ├── ANE faster (incl. sync)? ─── No (atteso)
  │                        │                              │
  │                        Sì                             ▼
  │                         │                  ┌──────────────────────┐
  │                         ▼                  │ Chiudi filone NPU.    │
  │                  Procedi a POC 2           │ Update ROADMAP §6.    │
  │                         │                  │ TODO §5 → [x] done.   │
  │                         ▼                  └──────────────────────┘
  │                  POC 2 (draft model)
  │                         │
  │                         ├── Speedup > 1.3×?
  │                         │
  │                         Sì ──► Apri progetto "ANE draft integration" (4-8 sett)
  │                         │
  │                         No ──► Chiudi filone NPU (come sopra)
  │
  ▼
```


## Effort totale

| POC | Stima | Cumulativo |
|---|---:|---:|
| POC 3 | 1 giorno | 1 g |
| POC 1 | 2-3 giorni | 3-4 g |
| POC 2 | 3-5 giorni | 6-9 g |

**~6-9 giorni uomo totali** se si fanno tutti e 3 fino in fondo.
**~1 giorno** se POC 3 chiarisce subito che MPSGraph non aiuta e
si vuole chiudere senza misure ulteriori (rischioso: non si hanno
ancora numeri sui sync ANE↔GPU).

Raccomandazione: fare almeno POC 3 + POC 1. POC 2 solo se POC 1 dà
risultati ambigui e c'è interesse genuino in speculative decoding
esterno a MTP.

## Cosa NON è in questo piano

Tutto questo materiale resta intenzionalmente fuori dai POC. Aprire
issue separate se serve:

- **Integrazione full speculative decoding** in `Generation.swift`
  con gestione KV cache + retry su mismatch. È la fase 2 e dipende
  dall'esito di POC 2.
- **Conversione del modello principale V4-Pro a CoreML** —
  l'analisi mostra che non è esprimibile senza perdere il routing
  MoE e i kernel FP4/FP8. Non si tenta.
- **Esplorazione MPSGraph come backend GEMM** — diverso da POC 3
  che testa solo l'attention. Se interessa, apre un POC 4 separato
  (non in questo piano).
- **Confronto con ANEMLL**. ANEMLL ha già pubblicato numeri (~9
  tok/s su 8B); riprodurli non aggiunge informazione vs i nostri
  POC 2.

## Tracking

Quando si parte:

1. Aprire un branch `claude/ane-poc-N-<short-name>` per ciascun POC.
2. Commit incrementali, push a fine giornata.
3. Quando il report markdown è pronto, aggiornare la voce in
   TODO §11 da bullet a `[x]` con link al report e PR mergeata.
4. La voce TODO §5 "ANE / NPU come backend: valutazione" rimane `[x]`
   perché l'**analisi** è completa; i POC sono *verifiche
   empiriche* dell'analisi, tracciate in §11.

## Riferimenti

- [`NPU-ANE-ANALYSIS.md`](NPU-ANE-ANALYSIS.md) — l'analisi che ha
  generato questo piano.
- [`KERNELS.md`](KERNELS.md) — pattern dei kernel Metal esistenti
  (utile per POC 1 baseline).
- [`PERFORMANCE.md`](PERFORMANCE.md) — metodologia di benchmark
  esistente.
- Apple — *Core ML Performance Best Practices* (developer.apple.com).
- HuggingFace — [`swift-transformers`](https://github.com/huggingface/swift-transformers)
  (dipendenza POC 2).
- ANEMLL — [`anemll/anemll`](https://github.com/anemll/anemll)
  (reference esterna per ordini di grandezza ANE).
