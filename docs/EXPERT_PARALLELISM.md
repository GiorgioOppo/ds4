# Expert Parallelism — scissione VERTICALE del modello (design)

Stato: **Fase A** (protocollo scaffolded, non attivo). Prerequisito di rete:
RTT < 1 ms tra i nodi (Thunderbolt bridge o Ethernet diretta). Su Wi-Fi
(RTT ~7 ms misurato) questa architettura è matematicamente perdente:
~41 layer routed × RTT ≈ 300 ms/token di sola latenza.

## Perché verticale, e perché SOLO per la MoE

La pipeline orizzontale (layer a fette) usa gli SSD dei worker IN SERIE:
worker1 gathera gli esperti dei suoi layer (~150 ms), POI worker2 i suoi.
Token = somma. Il taglio verticale giusto per una MoE non è il tensor
parallelism classico (matmul spezzati: 2 all-reduce per layer per
accelerare la parte già veloce) ma l'**expert parallelism**: i 256 esperti
per layer si dividono tra i worker, e per ogni layer TUTTI gli SSD
gatherano in parallelo la loro quota dei 6 esperti selezionati.
Token = max invece di somma sul costo dominante (~2× su 2 worker).

## Architettura

- **Coordinatore = backbone denso completo**: embed, route/attention di
  TUTTI i layer (stream denso, KV, compressori NSA — come un motore locale),
  selezione del router, FFN condivisa (densa), somma finale, head.
  Il coordinatore NON tiene esperti routed.
- **Worker = shard di esperti**: possiede un sottoinsieme dei 256 esperti
  (lo STESSO sottoinsieme per tutti i layer — il bundle sidecar è già
  ordinato per (layer, esperto), lo shard legge solo i suoi record).
  Niente KV, niente attention, niente stato di sequenza: un worker
  verticale è puro (gather dall'SSD + FFN esperti + somma parziale pesata).

### Flusso per token, per layer routed

1. Coordinatore: route/attn → 6 id + pesi (`routerFinalizeTop6`).
2. Partiziona gli id per proprietario; a ogni worker coinvolto invia
   `expertWork`: layer, seq, i SUOI id+pesi, l'attivazione `s.cur`
   (4096 × f32/f16 secondo `activationBits`, come la pipeline).
3. Ogni worker: gather dei suoi id (slot-cache + bundle, invariati) →
   gate/up/down → SOMMA PARZIALE pesata (4096) → `expertSum` (seq).
4. Coordinatore: somma le parziali + output FFN condivisa → hcExpand →
   layer successivo.

Overlap gratuiti: la FFN condivisa del layer i gira sul coordinatore
MENTRE i worker computano gli esperti; lo stream denso di i+1 è già in
background; per i layer hash (0-2) gli id dipendono solo dal token → le
`expertWork` dei 3 layer partono in anticipo appena campionato il token.

### Costi di comunicazione (2 worker, 41 layer routed)

- ~64 KB/layer/token andata+ritorno a f32 (32 KB a f16) ≈ 2.6 MB/token:
  ~3 ms su bridge Thunderbolt (~10 Gbit), ~25 ms su gigabit.
- Latenza: 1 RTT per layer (scatter e gather in un solo round):
  41 × 0.3 ms ≈ 12 ms su cavo. Su Wi-Fi ≈ 300 ms → vietato.

### Assegnazione degli esperti

`expertMask` esplicita (32 byte, bit e = posseduto) nell'assign — non un
range: la usage imatrix bilancia il CARICO (route effettive), non il
conteggio. Costruzione: greedy sui conteggi della imatrix (hot expert
alternati tra i worker), fallback pari/dispari senza storia. Ogni id deve
essere posseduto da ESATTAMENTE un worker (validato al connect).

### Trasferimento file

Un worker verticale non ha bisogno del GGUF intero: gli bastano i record
del bundle sidecar dei SUOI esperti (~72 GB / n worker) + il manifest.
Fase B parte pragmatica: si riusa la distribuzione v8 del bundle completo
(ripresa a checkpoint inclusa); l'offerta "solo shard" è un'ottimizzazione
successiva (nuovo kind nel manifest con la stessa catena di hash).

## Fasi

- **A (fatta)**: design + frame `expertWork`/`expertSum`/`expertAssign` +
  payload con encode/decode bound-checked + test round-trip. Nessun
  percorso attivo li usa; i tipi sconosciuti sono ignorati dai loop v9.
- **B**: worker shard — modalità del DistWorker che carica SOLO la
  macchineria esperti (bundle + slot-cache filtrata sulla mask, tutte le
  layer) e serve `expertWork` → `expertSum`. Testabile da sola (loopback).
- **C**: coordinatore verticale — variante del motore locale in cui
  `decodeRoutedExperts` diventa scatter/gather remoto; toggle
  "Verticale (expert parallelism)" nel tab Distribuito; v10.
- **D**: overlap (hash layers in anticipo, FFN condivisa concorrente),
  bilanciamento dalla imatrix, benchmark A/B contro pipeline e locale.

## Criteri di successo / abbandono

- Prerequisito misurato PRIMA di attivare: `ping` sul link diretto < 1 ms.
- Obiettivo: ≥ 1.5× il locale a parità di qualità (6 esperti).
- Se il gather parallelo non batte la pipeline configurata (v9, slot
  raddoppiati sui worker), il verticale resta una modalità opzionale.
