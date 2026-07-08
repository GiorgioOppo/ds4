# Valutazione delle prestazioni della demo (DS4Demo)

Dove va il tempo e dove va la memoria quando la demo genera token, e quali
ottimizzazioni restano sul tavolo — in ordine di resa attesa. I numeri citati
vengono dalle misure già registrate nel repository (commenti del motore,
`Sources/DwarfStar/Bench/README.md`, `Sources/DS4Demo/README.md`); macchina di
riferimento: M1 Pro 16 GB, modello DeepSeek V4 Flash (43 layer, esperti
IQ2_XXS/Q2_K, 256 esperti, top-6).

## 1. Il quadro in una riga

Il decode è **I/O-bound sull'SSD**: ogni token muove ~0.6–1.8 GB di esperti
(gather) più — con `DS4_DENSE_STREAM` e senza Q4 — fino a ~6.2 GB di pesi densi
in streaming. Il compute GPU e la sincronizzazione dei command buffer sono
costi secondari. Il prefill invece è già ammortizzato layer-major e migliora
con la lunghezza del prompt (4.6 → ~8 tok/s da 64 a 3k token).

## 2. Decode: budget per token

Percorso: `StreamingDecoder.forward()` → per ognuno dei 43 layer
`runLayer()` (route+attn → readback selezione → shared FFN async → gather →
routed FFN) → `outputHead()`. Riferimenti: `StreamingDecoder.swift:267`
(forward), `:874` (runLayer), `:1090` (outputHead).

| Fase (profilo) | Cosa muove/fa | Costo tipico | Vincolo |
|---|---|---|---|
| `gather IO` | 6 esperti × ~6.9 MB × 43 layer = **~1.78 GB/token** a freddo; ~0.6 GB/token con slot-cache calda | 200–600 ms | banda random-parallela SSD (~tetto misurato dal DIAG) |
| dense stream (dentro `route/attn`) | ~145 MB/layer × 43 ≈ **6.2 GB/token** senza `DS4_DENSE_Q4`; ~1.6 GB/token con Q4+shared Q4 residenti | sovrapposto un layer avanti; emerge quando l'SSD è conteso col gather | banda sequenziale SSD |
| `route/attn` (compute) | proiezioni dense + flash-attn + router | decine di ms a densi residenti; **2.4 s/token misurati** quando i densi rileggono da page cache degradata (la patologia che `DENSE_STREAM`/`MLOCK` curano) | RAM/compressore |
| sincronizzazione | ~3 command buffer per layer ≈ **~130 commit+wait/token** (misurabile col probe del DIAG, `DS4Demo/main.swift:167`) | ~10–25 ms/token a ~100 µs/cb | latenza round-trip CPU↔GPU |
| `output head` | matvec Q8 da ~560 MB | ~10–20 ms residente+`MLOCK`; **235–260 ms misurati** se compresso/rimappato | RAM vs memory compressor |
| `embed` | 1 riga da ~8 KB (staging) | trascurabile | — |

Risultato di regime documentato: **~2.5 tok/s** a contesto 4–8k; ~1.3 tok/s
con finestra configurata a 104k (pressione del KV pre-allocato sui 16 GB).

Due proprietà strutturali da tenere a mente:

- **La selezione del router torna sulla CPU a ogni layer** (readback dopo il
  commit della route): serve alla CPU per emettere i `pread` degli esperti.
  Finché il gather è fatto dalla CPU, il layer non può essere interamente
  GPU-driven — il limite dei ~130 sync/token è architetturale, non un bug.
- **L'unico overlap I/O–compute nel decode è la shared FFN** (commit async
  prima del gather, `StreamingDecoder.swift:892`). Il gather del layer i non
  può sovrapporsi al compute del layer i−1 perché dipende dalla route dello
  stesso layer i; l'unica alternativa è la predizione speculativa dal prior
  d'uso (`DS4_PREFETCH_EXPERTS`, oggi opt-in e dichiaratamente rischiosa).

## 3. Prefill: budget per chunk

Percorso layer-major (`prefill()` → `prefillRange()` → `batchedExpertLayer()`):
i pesi di ogni layer si caricano **una volta per chunk** (default 512 token) e
si applicano a tutti i token del chunk.

- Costo fisso per chunk: rilettura dei densi di TUTTI i layer (~6 GB con
  `DENSE_STREAM`) → **~12 MB/token a chunk 512**, dimezzabile con
  `DS4_PREFILL_CHUNK=1024` (+~160 KB/token di attivazioni transienti).
- Gather: unione degli esperti per gruppo (cap `DS4_PREFILL_UNION`, default
  192 — a 64 leggeva ~1.7 GB/token, misurato). L'I/O del gruppo g+1 gira in
  background durante le FFN del gruppo g (`PrefillGather`).
- Sync: fase A batchata (32 route per command buffer) e fase B a un command
  buffer per gruppo hanno già tolto i ~22k sync/chunk storici.
- Misurato: 4.6 tok/s a 64 token → ~8 tok/s a 3k token; il costo fisso si
  ammortizza col prompt.

## 4. Memoria: dove vanno i GB

| Voce | Dimensione | Note |
|---|---|---|
| Slot-cache esperti | 6.9 MB/slot/layer wired → **S=16 ≈ 4.7 GB** su 43 layer | il knob più costoso; usage-driven redistribution a budget fisso |
| Densi Q4 residenti (`DS4_DENSE_Q4`) | ~1.4 GB (+ shared con `DS4_SHARED_Q4`) | toglie ~4.6 GB/token dallo stream SSD |
| Output head residente (con `DENSE_STREAM`) | ~560 MB | necessario `DS4_MLOCK` per non finire nel compressore |
| Staging ring densi | ~300 MB (2 slot) / +150 MB con `DS4_DENSE_AHEAD=2` | al posto di ~6 GB residenti |
| `DS4_MLOCK` totale | ~3.3 GB pinnati ai default | il compressore macOS rilegge a ~2.4 GB/s i buffer non pinnati |
| Prefill transiente | unione 192 × ~7 MB × 2 (pipeline) ≈ **~2.7 GB** + ~80 MB (`PREFILL_MM`) | abbassare `DS4_PREFILL_UNION` su macchine strette |
| KV raw | lazy (zero-fill-on-demand); `DS4_RAW_RING` lo rende costante | il footprint segue i token realmente generati |

Su 16 GB il budget va conteso: slot-cache 16 + Q4 residenti + mlock ≈ 7 GB
wired prima ancora del KV — è il motivo per cui il bench a contesto 104k
scende a 1.3 tok/s.

## 5. Runbook di misura (sul Mac)

La demo è già strumentata; la matrice minima per aggiornare i numeri:

```sh
# base + diagnosi completa (tetto SSD, probe command buffer, profilo per fase)
DS4_DIAG=1 DS4_EXPERT_CACHE_SLOTS=16 DS4_DENSE_STREAM=1 DS4_MLOCK=1 \
  swift run -c release DS4Demo model.gguf 48 "Tell me the history of Rome."

# A/B che decidono le prossime mosse (un knob alla volta, stesso usage file):
#  1. DS4_DENSE_Q4=1 [+DS4_SHARED_Q4=1]  -> quanto scende route/attn e il totale
#  2. DS4_EXPERT_BUNDLE=1                -> banda gather vs tetto (verdetto nel DIAG)
#  3. DS4_EXPERT_CACHE_SLOTS=8/12/16     -> hit-rate vs RAM
#  4. DS4_PREFILL_MM=1 e DS4_PREFILL_CHUNK=1024 con prompt @file da ~3k token
scripts/bench.sh   # matrice automatica con report unico
```

Leggere: `gather IO` MB/token e banda effettiva vs tetto (verdetto stampato dal
DIAG), hit-rate della cache, `route/attn` prima/dopo Q4, tok/s di REGIME (i
primi 4 token sono esclusi dal profilo con `DS4_DIAG`).

## 6. Ottimizzazioni possibili, in ordine di resa

1. **Decodifica speculativa MTP** (resa potenziale: 2–4×, sforzo alto).
   Il decode paga GB/token *indipendentemente* da quanti token verifica: con i
   pesi MTP (il DIAG già controlla se sono nel GGUF, `mtpReport`) un draft di
   N token verificato in un passo batch ammortizza l'intero stream
   densi+esperti su N token. I mattoni esistono già: la fase A batchata del
   prefill è di fatto un passo di verifica multi-token, e `batchedExpertLayer`
   sa deduplicare l'unione degli esperti di più token. È l'unica leva che
   attacca il vincolo fondamentale (byte/token dall'SSD) invece di limarlo.

2. **Difendere i default "vincenti" anche nella demo CLI** (resa: 2× vs run
   ingenuo, sforzo minimo). La GUI ha come default la config misurata (16
   slot, `DENSE_STREAM`, `MLOCK`, Q4); la demo parte con tutto OFF e un run
   senza env riproduce la patologia dei 2.4 s/token. Allineare i default della
   demo (o stampare un suggerimento quando `DS4_DIAG` rileva la config
   debole) renderebbe ogni valutazione ripetibile senza incantesimi d'ambiente.

3. **Promuovere `DS4_PREFILL_MM` a default dopo l'A/B** (resa: prefill,
   sforzo basso). Il percorso matrix-matrix legge i pesi una volta per tile
   invece che una per token; è opt-in solo perché non ancora validato A/B
   on-device. Idem `DS4_PREFILL_CHUNK=1024` dove la RAM lo consente (dimezza
   il costo fisso di rilettura densi per chunk).

4. **Ridurre ancora i byte del dense stream** (resa: media, sforzo medio).
   Dopo Q4 su q_b/output_a/output_b e shared FFN restano ~1.6 GB/token di
   densi Q8 in streaming (kv, q_a, compressori, router…). Candidati: requant
   Q4 dei tensori medi rimanenti o residenza selettiva dei piccoli (norm,
   router: pochi MB, zero rischio). Ogni GB tolto dallo stream è banda
   restituita al gather, che è il collo.

5. **Fusione dei command buffer del decode** (resa: ~10–25 ms/token, sforzo
   medio-alto). Dei ~3 cb/layer, la coppia "routed FFN del layer i" + "route
   del layer i+1" è fondibile (nessun readback CPU in mezzo quando la
   slot-cache serve tutti e 6 gli esperti). Ne toglierebbe fino a ~43
   round-trip/token nei layer a hit pieno. Il probe del DIAG dice se il gioco
   vale la candela sulla macchina target: sotto i 100 µs/cb probabilmente no.

6. **Prefetch speculativo degli esperti guidato dal prior** (resa: incerta,
   sforzo basso — esiste già). `DS4_PREFETCH_EXPERTS=N` è spento perché ruba
   banda al gather reale quando il prior è freddo; con la usage imatrix
   persistita (`<gguf>.usage.json`) e concentrazione di routing alta nel DIAG,
   vale un A/B mirato sui layer più concentrati.

Non-leve (già chiuse o non paganti): l'embed è già a staging di riga (~8 KB);
il pool interleaved ha già portato il miss a 1 pread da ~7 MB; il bundle
sidecar copre il caso "gather < 60% del tetto"; l'output head residente +
mlock ha già eliminato i 235 ms del compressore.

## 7. Misure reali (2026-07-05, M1 Pro 16 GB, Flash IQ2XXS)

Runbook §5 eseguito su prompt da 13 token, 48 generati, regime = 44 token.
Tetto SSD misurato: **5.25–5.69 GB/s** (random parallelo); command buffer
vuoto **~21 µs** → la sincronizzazione vale ~3 ms/token (non è una leva).
**Il GGUF in uso NON contiene pesi MTP** → la leva 1 (§6) richiede un file
convertito con i tensori `nextn`/`mtp`.

| Config | Decode regime | gather IO | route/attn | experts | head |
|---|---|---|---|---|---|
| base (slots 16, stream, mlock) | **2110 ms/tok (0.47 tok/s)** | 1685 ms (80%), 617 MB/tok @ **0.38 GB/s = 7% del tetto** | 216 ms | 190 ms | 19 ms |
| + `DENSE_Q4` + `SHARED_Q4` | **880 ms/tok (1.14 tok/s)** | 662 ms (75%), 636 MB/tok @ **1.01 GB/s = 18% del tetto** | 116 ms | 94 ms | 7 ms |
| + `DS4_EXPERT_PREAD` | **455 ms/tok (2.20 tok/s)** | 225 ms (49%), 635 MB/tok @ **2.97 GB/s = 53% del tetto** | 126 ms | 96 ms | 8 ms |
| + `EXPERT_BUNDLE` | *non testato*: build saltata per spazio disco (~72 GB richiesti, 46 liberi) — run ≈ identico al precedente (443 ms, 3.10 GB/s = 57%) | | | | |

Cache esperti: 63–65% hit (94 miss/token ≈ 650 MB letti); concentrazione
top-16 ~0.3–0.5 per layer, allocazione usage-driven attiva.

Lettura delle misure:

- **Q4 residente vale 2.4×** da solo (0.47 → 1.14 tok/s). Non solo per i
  ~90+95 ms tolti a route/attn+experts: il gather — a byte QUASI IDENTICI
  (617 vs 636 MB/token) — è passato da 1685 a 662 ms/token. Il dense stream
  contendeva il disco al gather; toglierlo ha quasi triplicato la banda
  effettiva del gather. Conferma sperimentale della leva 4 (§6).
- **`EXPERT_PREAD` vale un altro 1.9×** (1.14 → 2.20 tok/s): il miss non è
  più un memcpy dal mmap a colpi di page fault da 16 KB, ma una pread
  F_NOCACHE dell'intero slab. La banda del gather è salita da 1.01 a
  2.97 GB/s, e il prefill da 1085 a 480 ms/token (il gather dell'unione era
  la stessa patologia). La proiezione di §6 (~2.3 tok/s) è stata centrata.
- Il profilo route/attn split (run separato, senza cache/Q4 — leggere i
  rapporti): `q` 27% e `out` 24% dominano, flash-attn 9% ⇒ sono proprio i
  tensori che `DENSE_Q4` rende residenti; sul percorso Q4 non c'è più molto
  da spremere lì.
- **Cumulato: 0.47 → 2.20 tok/s (4.7×) con soli knob.** Il gather resta il
  49% del token a ~53-57% del tetto: coda NVMe ~6-9 richieste (2-3 miss ×
  3 slab per layer) contro le ~24 a cui il disco rende il tetto.

## 8. Misure 2026-07-08 (stesso M1 Pro 16 GB, bundle ATTIVO, prompt 17 tok, 48 generati)

Il bundle sidecar è entrato in funzione (costruito dall'app, riusato dalla
demo via `DS4_BUNDLE_DIR`) e la matrice del giorno ha chiuso quattro domande:

| Config (sopra la base Q4+PREAD+bundle) | Decode regime | gather IO | hit |
|---|---|---|---|
| slots 16 | 2.78 tok/s | 627 MB/tok, 158 ms | 64% |
| + `DS4_QKV_Q4` | **3.06 tok/s (+10%)** | 606 MB/tok, 141 ms | 65% |
| + slots 20 | 3.20 tok/s | 550 MB/tok, 128 ms | 68% |
| + slots 24 | **3.33 tok/s** | 479 MB/tok, 115 ms | **73%** (nessun collasso) |

- **`DS4_QKV_Q4` promosso** (toggle in Settings, opt-in): q_a+kv residenti
  Q4 valgono +10% da soli — meno byte nello stream E matvec dimezzato.
  La cache `.q4dense` si estende in modo incrementale (86 tensori, ~30 s).
- **`DS4_PREAD_SPLIT=2`: nel rumore** (2.84 vs 2.78) — il gather gira già
  all'89-94% del tetto misurato; la coda non è più il collo.
- **Allineamento pread: NON è una leva.** Il probe "random DISALLINEATO"
  (DIAG) oscilla sopra e sotto l'allineato tra i run: varianza termica del
  disco, nessuna penalità sistematica su questo SSD/OS.
- **`DS4_MOE_NSG`: il default 4 resta il migliore su M1 Pro** (A/B sulla
  config QKV+24 slot: nsg=2 → 3.10, nsg=8 → 3.25, nsg=4 → 3.33 tok/s di
  regime; nel prefill `experts` peggiora del ~20% con 2/8). Coerente col
  fatto che in decode l'ASYNC_FFN nasconde gran parte della FFN routed
  dal critical path. Il knob resta (auto-tune 2/8) per GPU più larghe.
- Split `DS4_PROFILE_ROUTE` sul percorso Q4+QKV (decode, sincrono):
  out 63 ms (output proj+HC+router), q 38, comp 28, attn 24, kv 16, più
  experts 100 ms che l'ASYNC_FFN sovrappone quasi per intero. `q` e `out`
  restano grandi CON i pesi già Q4 residenti ⇒ non è banda pesi: sono le
  CATENE di micro-dispatch (proiezione→norm→rope, HC, router+readback),
  ~140 ms/token di piccole operazioni in sequenza.

### Aggiornamento serale (stesso giorno)

- **`DS4_SHARED_Q4` oggi PAGA: +7%** (3.13 → 3.36 tok/s totali, regime
  3.45) — era neutro nel tuning del 07-06, ma ora che trio+q_a+kv sono
  residenti le shared FFN erano l'ultima voce grossa dello stream.
  Lossy (la continuazione greedy cambia restando coerente): toggle GUI.
- **Self-speculative (DS4_SPEC_K): parità perfetta, economia negativa**
  (K=2: 78% accettazione ma 2.38 vs 3.36). Ragione strutturale: il token
  è ormai dominato da route/attn seriale, che la verifica batch non
  ammortizza. Parcheggiato come opt-in; dettagli in SELF-SPECULATIVE.md.

- **MTLIO (Metal fast resource loading): leva CHIUSA.** Probe A/B nel
  DIAG, rapporto MTLIO/pread nello stesso run: 1.11 poi 0.81 — non
  ripetibile, varianza termica. La pread F_NOCACHE scrive già in DMA
  nella memoria unificata e il gather è al 100-101% del tetto misurato:
  su questo hardware non c'è canale più veloce.

## 9. Prossimo passo

1. **Decode multi-token self-speculative** (il GGUF non ha pesi MTP): draft
   con `DS4_ACTIVE_EXPERTS=2` + verifica batch stile fase A del prefill —
   l'unica leva rimasta che ammortizza I/O e compute su N token. A 3.3
   tok/s il critical path è route (compute) ⊕ FFN assorbita ⊕ gather al
   ~90% del tetto: le leve locali sono quasi esaurite.
2. Fusione delle micro-catene di route/attn: resa incerta (gli assoluti
   dello split sono gonfiati dai sync del profilo, ~0.5 ms × fase ×
   layer) — prima di investirci serve una misura GPU-side pulita
   (Instruments, i debug group ci sono già).
3. Hit-rate oltre 24 slot: provare 28 in demo (guardare il collasso
   progressivo); l'auto-tune della GUI ora include 24 sui 16 GB.
4. Prefill: `experts` è il 36-38% — vale l'A/B `DS4_PREFILL_MM=1` su un
   prompt @file da ~3k token.
