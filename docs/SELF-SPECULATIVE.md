# Decode self-speculative CLI sperimentale

> **Stato al 13 luglio 2026:** le fasi A-C sono implementate nella demo CLI e
> abilitate soltanto con `DS4_SPEC_K`; non sono integrate nel percorso GUI o
> nell'`InferenceService`. Le prove del 2026-07-08 su M1 Pro 16 GB e GGUF Flash
> IQ2_XXS hanno verificato la parità greedy sui prompt provati, ma il throughput
> è risultato inferiore al decode normale. La funzione è quindi opt-in e
> parcheggiata, non un profilo consigliato. Tutti i numeri sotto sono snapshot di
> quella configurazione e non una promessa per altri modelli o Mac.

Obiettivo dell'esperimento: rompere la struttura "un giro completo per token"
del decode — route → readback selezione → gather → FFN, ×43 layer — pagandola
**una volta ogni N token**. Nello snapshot a 3.33 tok/s del 2026-07-08 le leve
locali apparivano esaurite: gather all'87-94% del tetto SSD, sync 3 ms, FFN
routed nascosta dall'ASYNC_FFN.
Il GGUF provato non aveva pesi MTP; la variante qui sotto non li richiede.

## Schema

```
stato S a posizione P (ultimo token accettato t0)
1. SNAPSHOT   s = specSnapshot()                  (~1-2 MB, solo stato ricorrente)
2. DRAFT      K forward ECONOMICI (activeExperts=2) da P: candidati c1..cK
              (scrivono KV raw alle posizioni P..P+K-1 e avanzano i compressori
               con valori "draft" — verranno RISCRITTI dalla verifica)
3. RESTORE    specRestore(s)
4. VERIFY     un passo BATCH full-config sui token [t0? no: c1..cK] con logit
              PER POSIZIONE: riusa la fase A batchata del prefill (route di K
              token in UN command buffer) + dedup dell'unione esperti
              (batchedExpertLayer). Riscrive KV raw e stato compressori
              full-config per P..P+K-1.
5. ACCEPT     greedy: j = lunghezza del prefisso ci == argmax(logits[i-1]);
              il logit dell'ULTIMA posizione verificata dà GRATIS il token
              successivo (bonus token): avanzamento = j+1 token per round.
6. ROLLBACK   se j < K: specRestore(s) e RIVERIFICA batch dei soli j token
              accettati (mini-passo, ricostruisce lo stato pulito a P+j).
              Se j == K: lo stato della verifica È già quello giusto.
```

Stima iniziale del costo per round: 1 verify batch (≈ 1 giro pieno,
ammortizzato su j+1 token) + K draft a ~1/3 dell'I/O esperti + (solo su
rifiuto) un mini-verify dei token
accettati. L'attesa era **1.5-2.5×** con accettazione media 2-3; le misure
successive hanno smentito l'ipotesi di un draft abbastanza economico.

## Vincoli verificati sul codice

- **Ricorrenza NSA** (il vincolo che rende impossibile "riavvolgere" e la
  ragione del `kvDirty` dell'engine): per layer, `CompressorState.stateKv` +
  `stateScore` ([coff·ratio × width] f32, ~32 KB sul ratio-4) + `count`,
  doppio per i layer con indexer (`indexStates`). Il rollback NON richiede le
  righe di cache: l'emissione indicizza per `count`, quindi le righe scritte
  oltre il count ripristinato vengono riscritte (dalla verifica full-config)
  prima di qualunque lettura. → snapshot LEGGERO (`SpecRecurrentState`),
  ~1-2 MB, non i ~22 KB/token del `KVSnapshot` dei checkpoint.
- **KV raw**: posizionale (ring `pos % rawRows`): le righe delle posizioni
  rifiutate vengono sovrascritte dai token successivi; nessun rollback attivo.
- **activeExperts**: già runtime nel dispatch (`min(d.activeExperts, d.k)`
  in DecodeLayer, route weights rinormalizzati); serviva solo renderlo
  mutabile sul decoder (`setActiveExperts`, Fase A).
- **Indexer top-K**: si attiva per soglia deterministica di righe compresse —
  con contesti in cui il DIAG prova che non può attivarsi (`DS4_LAZY_IDX`),
  draft e verify restano sul percorso denso e coerente. Con indexer attivo la
  verifica batch ricade sul percorso per-token della fase A (già gestito).

## Correttezza

- Greedy: i token accettati sono ESATTAMENTE quelli che il modello pieno
  avrebbe generato (confronto argmax posizione per posizione) — parità
  bit-per-bit con il decode normale, per costruzione.
- Sampling: richiede rejection sampling (accetta ci con prob min(1,
  p_full(ci)/p_draft(ci)), altrimenti ricampiona dalla distribuzione
  residua). FASE SUCCESSIVA: si parte greedy-only.
- Il repetition penalty e il tool-parsing lavorano su token ACCETTATI — la
  finestra speculativa resta interna al motore.

## Piano incrementale

1. **Fase A (primitivi) — FATTA**: `setActiveExperts(_:)`/`activeExpertsNow`
   runtime; snapshot/restore leggero dello stato ricorrente
   (`SpecRecurrentState`, `SpecDecode.swift`). Nessun cambiamento di
   comportamento finché inutilizzati.
2. **Fase B (verify) — FATTA**: `specVerifyStep(tokens:startPos:) ->
   [[Float]]` (`Sources/DS4Metal/Decode/Prefill/StreamingDecoder+Prefill.swift`,
   accanto a `prefillRange`, di cui
   riusa stage batchato e unione esperti): passo batch full-config con
   logit per posizione (output head su ogni hidden finale, ~8 ms/token).
3. **Fase C (loop, demo) — FATTA e misurata su copertura limitata**:
   `DS4_SPEC_K=N`
   (+`DS4_SPEC_DRAFT_EXPERTS`, default 2) nella demo: loop
   draft/verify/accept greedy con bonus token e ricostruzione su rifiuto.
   La validazione permanente deve confrontare lo stesso testo del decode
   normale su più prompt e fare sweep K=2..6 leggendo token/round e accettazione.
4. **Fase D (GUI/engine) — non avviata, parcheggiata**: integrazione in
   InferenceService (generate greedy → speculativo quando temperature==0 o
   dietro toggle), telemetria
   accettazione nel profilo (token/round), poi rejection sampling.
   Nota per l'integrazione: ripristinare SEMPRE activeExperts sugli error
   path (defer), e il draft inquina marginalmente la usage imatrix (route
   registrate con selezione draft) — accettabile, eventualmente gate.

## Prima misura sul campo

Snapshot 2026-07-08: M1 Pro 16 GB, Flash IQ2_XXS, K=4, draft 2 esperti.

- **Parità: PERFETTA** — testo identico carattere per carattere al decode
  normale su 48 token. Snapshot/rollback della ricorrenza NSA, verifica e
  accettazione funzionano (accettazione draft 49%, 2.53 token/round).
- **Economia: oggi PERDE** (0.79 vs 3.13 tok/s). Due cause misurate:
  1. la verifica passava da `batchedExpertLayer`, che a finestre piccole
     rilegge l'UNIONE dal disco ignorando la slot-cache (692 vs 477
     MB/token) → CORRETTO: `specVerifyStep` ora usa il percorso per-token
     layer-major (hit dal pool, densi amortizzati una volta per layer);
  2. il DRAFT non è ~1/3 del costo come stimato: con 2 esperti si riduce
     solo il gather routed, ma ogni forward paga per intero il passaggio
     denso in streaming (~0.9 GB/pass con QKV_Q4, senza SHARED_Q4) e
     l'overhead per-layer → un draft forward ≈ 0.8× un forward pieno, e
     con K=4 sono 3 passaggi extra per round.
- **Perché vinca serve un draft da ≤0.2× forward**: (a) `DS4_SHARED_Q4=1`
  (toglie le shared FFN dallo stream: beneficio anche al decode normale,
  da rimisurare ora che q_a/kv/trio sono residenti); (b) draft che SALTA
  la shared FFN (è un'approssimazione comunque: tocca solo l'accettazione);
  (c) K piccolo (2) per ridurre i passaggi draft per round; (d) in
  prospettiva, layer-skip nel draft o una futura integrazione MTP con draft head
  dedicata. Il DIAG oggi ne rileva soltanto la presenza: il percorso corrente
  non carica né usa quei pesi.

## Seconda misura (stesso snapshot, verify via slot-cache + SHARED_Q4)

- Parità di nuovo perfetta; K=2: **accettazione 78%**, 1.78 token/round;
  K=4: 50%, 2.40 token/round. Forward medio sceso a 224 ms (dal fix
  della verifica: 308 vs 692 MB/token dal disco).
- **Ancora in perdita** (K=2: 2.38 vs 3.36 tok/s del baseline pari-knob).
  Causa STRUTTURALE, non di implementazione: dopo le ottimizzazioni della
  sessione il costo per token è dominato da route/attn SERIALE (~60%),
  che la verifica batch non ammortizza — ogni posizione paga la sua
  attention; si ammortizzano solo densi (ormai piccoli) e gather (ormai
  in cache). Lo speculativo paga quando il token è dominato da I/O pesi
  ammortizzabile: era vero a 0.47 tok/s, non lo è più a 3.4.
- CONCLUSIONE: parcheggiato come opt-in sperimentale (DS4_SPEC_K).
  Torna conveniente se/quando: (a) la verifica ottiene una route/attn
  MULTI-TOKEN vera (flash-attn causale sulla finestra in un dispatch,
  matvec→matmul per K righe); oppure (b) route/attn per token scende
  molto (fusione micro-catene). Da rivalutare dopo quei cantieri.

## Fase M — draft con la testa MTP (in corso)

La seconda misura ha chiuso il draft a esperti ridotti: costa ~0.8× un forward
pieno (paga per intero densi in streaming e overhead per-layer) e il tetto
strutturale resta la route/attn seriale che la verifica non ammortizza. La
testa MTP di DeepSeek è il draft giusto: UN blocco transformer + head
(~1/43 dei layer per candidato) addestrato esattamente per il token
successivo — accettazione attesa ben sopra il 49-78% del draft ridotto. Il
sidecar esiste nel catalogo download (id `mtp`,
`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`, ~4 GB, solo Flash); interfaccia
DeepSeek: `h' = blocco(eh_proj([enorm(emb(t)); hnorm(hidden)]))` →
`shared_head(norm(h'))`, iterata K-1 volte riusando `h'`.

1. **M1 (FATTA)**: `MTPSidecar` (`Sources/DS4Metal/Model/MTP/`) — apertura,
   classificazione dei ruoli (eh_proj, embed_tokens, enorm, hnorm,
   shared_head.*, blocco) e report di validazione contro vocab/nEmbd del
   modello principale. Demo: `DS4_MTP_GGUF=<path>` (o `=1` per cercare
   `*MTP*.gguf` accanto al modello). Nessun effetto sul decode: produce il
   ground truth (nomi/forme/quant REALI del sidecar) per cablare M2 senza
   indovinare la conversione.
2. **M2**: caricamento residente (~4 GB, RAM-gated) + forward del draft.
   Richiede: l'hidden finale pre-head del modello principale esposto dal
   decoder (l'`oembd` di outputHead, prima di `out.norm`); il KV proprio del
   blocco MTP (la sua attention vede le posizioni della finestra); il wiring
   esatto del blocco deciso dal report M1 (se il blocco è un layer DSV4
   completo si riusa `decodeLayer` con `LayerWeights` dedicati, altrimenti
   kernel dedicati).
3. **M3**: aggancio al loop demo (il draft chain MTP sostituisce la catena a
   esperti ridotti dentro `DS4_SPEC_K`), stessa verifica/accettazione; sweep
   K=2..4, parità greedy carattere-per-carattere, telemetria accettazione.

Aspettativa onesta (dalla seconda misura): draft MTP da solo ≈ 1.2-1.3×
perché la verifica resta per-token sulla route/attn; il pacchetto da 1.5-2×
richiede anche la verifica multi-token vera (flash-attn causale sulla
finestra + matvec→matmul per K righe).

## Rischi e mitigazioni

- Stato sporco dopo rifiuto → mini-verify di ricostruzione (passo 6);
  parità K=1 come test permanente.
- Draft KV "economico" letto dalla verifica? NO: la verifica ricalcola e
  riscrive TUTTE le posizioni della finestra partendo dallo snapshot; il
  draft legge il proprio KV (self-attention interna alla finestra draft) —
  coerente con lo schema self-speculative (il draft è un modello "diverso").
- Accettazione bassa su testo difficile → K adattivo (ridurre K quando
  l'accettazione media scende; già previsto nel loop di Fase C).
- Banda: il draft aggiunge ~K/3 di I/O esperti per round; se l'accettazione
  è ≥1.5 il bilancio byte/token accettato resta favorevole. Il DIAG del
  round (token/round, byte/token) lo misura.
