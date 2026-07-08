# Decode self-speculative (design)

Obiettivo: rompere la struttura "un giro completo per token" del decode —
route → readback selezione → gather → FFN, ×43 layer — pagandola **una volta
ogni N token**. A 3.33 tok/s (misure 2026-07-08) le leve locali sono esaurite:
gather all'87-94% del tetto SSD, sync 3 ms, FFN routed nascosta dall'ASYNC_FFN.
Questa è l'unica leva strutturale rimasta con il GGUF corrente (che NON ha
pesi MTP — il DIAG lo verifica: la variante qui sotto non li richiede).

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

Costo per round: 1 verify batch (≈ 1 giro pieno, ammortizzato su j+1 token) +
K draft a ~1/3 dell'I/O esperti + (solo su rifiuto) un mini-verify dei token
accettati. Con accettazione media 2-3: **1.5-2.5×** attesi.

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
   [[Float]]` (StreamingDecoder.swift, accanto a prefillRange di cui
   riusa stage batchato e unione esperti): passo batch full-config con
   logit per posizione (output head su ogni hidden finale, ~8 ms/token).
3. **Fase C (loop, demo) — FATTA, da validare on-device**: `DS4_SPEC_K=N`
   (+`DS4_SPEC_DRAFT_EXPERTS`, default 2) nella demo: loop
   draft/verify/accept greedy con bonus token e ricostruzione su rifiuto.
   VALIDAZIONE prima di misurare: stesso testo del decode normale (stessa
   parità greedy) su più prompt, poi sweep K=2..6 e lettura di
   token/round + accettazione dal log.
4. **Fase D (GUI/engine)** — integrazione in InferenceService (generate
   greedy → speculativo quando temperature==0 o dietro toggle), telemetria
   accettazione nel profilo (token/round), poi rejection sampling.
   Nota per l'integrazione: ripristinare SEMPRE activeExperts sugli error
   path (defer), e il draft inquina marginalmente la usage imatrix (route
   registrate con selezione draft) — accettabile, eventualmente gate.

## Prima misura sul campo (2026-07-08, M1 Pro, K=4, draft 2 esperti)

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
  prospettiva, layer-skip nel draft o un GGUF con pesi MTP (draft head
  dedicata, il DIAG già li rileva).

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
