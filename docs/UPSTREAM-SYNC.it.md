[English](UPSTREAM-SYNC.md) | **Italiano**

# Sincronizzazione con l'upstream (`antirez/ds4`)

L'engine di DwarfStar è una **riscrittura in puro Swift** del progetto C
upstream [`antirez/ds4`](https://github.com/antirez/ds4.git). Questo file
registra l'ultimo confronto con l'upstream, in modo che la prossima revisione
parta da una baseline nota invece di riscoprire la stessa cronologia di commit.

Per un piano di implementazione turnkey dei gap runtime ancora aperti (routed
Q8_K, Pro Q4 split-load, batch misto prefill+decode nel server) vedi
[`PORTING-GAPS.it.md`](PORTING-GAPS.it.md). I gap di tooling offline (GGUF
writer, requantizzatore offline) sono già implementati in puro Swift.

Questo è uno snapshot di sincronizzazione datato, non un'affermazione
sull'attuale HEAD upstream. La ristrutturazione della documentazione del
2026-07-13 non ha eseguito un nuovo confronto in rete; usare i comandi nella
sezione finale prima di far avanzare la baseline.

## Baseline attuale

| Campo | Valore |
|---|---|
| HEAD upstream esaminato | `80ebbc3` |
| Data | 2026-06-17 |
| Repository | `https://github.com/antirez/ds4.git` |
| Risultato | **Nessuna modifica urgente richiesta** per il percorso del modello standard. |

## Aggiornamento 2026-07-28 (rivisto il branch `laguna-s2.1`, head `448d569`)

Rivisto il **feature branch upstream `laguna-s2.1`** (17 commit sopra la merge
base `efdadd4`, non ancora su `main` upstream): supporto nativo a **Laguna
S 2.1** (modello GQA+MoE di Poolside — chat, reasoning interlacciato,
tool-call taggati, due ricette quant pubblicate più un file misto Q2_K/Q3_K)
con decoder Metal/CUDA/ROCm e decoding speculativo **DFlash** opzionale.

Decisione di port: portare subito il **frontend completo, decoder come gap**,
seguendo lo schema a stadi di GLM. Portato in questo passaggio, ciascun pezzo
con unit test deterministici: registrazione/rilevamento dell'architettura,
`LagunaConfiguration` (`config_validate_laguna_model`), `LagunaTokenizer`
(pre-split sui newline di `bpe_tokenize_text_laguna` + gruppi a una cifra
nella forma GLM4), `LagunaChatRenderer`/`LagunaToolCodec`
(`render_laguna_chat_prompt_text`, `laguna_chat_append_*`,
`append_laguna_tool_calls_text`), i default di sampling di riferimento
(`ds4_engine_sampling_defaults`), `LagunaTensorSchema`
(`weights_validate_laguna_layout`) e il catalogo download (target
`laguna-q4`, `laguna-q2-q3`, `laguna-dflash` di `download_model.sh`).
Fuori da questo passaggio: `metal/laguna.metal` + driver e DFlash — tracciati
come **Gap 4** in [`PORTING-GAPS.it.md`](PORTING-GAPS.it.md) perché il loro
gate di validazione richiede hardware Apple e i GGUF reali.
`LagunaRuntimeGate.enabled` resta spento finché quel gate non passa. Il codice
Laguna CUDA/ROCm resta fuori scope.

## Aggiornamento 2026-07-27 (rivisto `80ebbc3..0a7ad77`)

Confronto rieseguito. Dalla baseline precedente ci sono **15 commit upstream**
(fino a `0a7ad77`, 2026-07-23), **12 sullo scope condiviso**. Rivisti a livello
di messaggio + statistiche file; le voci *valutare* richiedono ancora una
lettura del codice su Mac.

| Commit | Area | Decisione port |
|---|---|---|
| `519c4d8` Fix PRO streaming and sampling correctness | engine/Metal/server | **NUOVO GAP — valutare.** Aggiunge un guard sul service-thread della cache esperti e rende i logit calcolati **indipendenti dal budget della cache esperti in streaming SSD**, più un warning cache troppo piccola e un fix di sampling. Verificare che la cache esperti Swift (`ExpertSlotCache`, streaming) mantenga lo stesso invariante "logit ⟂ budget". |
| `427e281` Optimize Metal prefill and decode kernels | Metal | **NUOVO GAP — resync.** Grande pass sui kernel (ds4_metal.m +2572; dense/rope/misc/cpy `.metal`). I kernel del port derivano da questi file e sono ora indietro — valutare il port delle ottimizzazioni/fix. |
| `36cd0ca` Add native Metal session batching | server/engine | **Riferimento per il Gap 3.** L'implementazione Metal del batch misto prefill+decode + slot server — il gap di continuous batching del port ha ora un riferimento Metal (non solo CUDA). |
| `a185c36` Stabilize CUDA and Metal session batching | Metal/CUDA | **Rimandato** — si accompagna al Gap 3 (batching Metal non ancora nel port); CUDA fuori scope. |
| `0a7ad77` Fix batched server session recovery race | server | **Rimandato** — dipende dal session batching (Gap 3), assente nel port (`RequestGate` serializzato). |
| `005afed` Add GLM 5.2 inference and quality fixtures | engine/Metal | **Rilevante per parità GLM.** Landing GLM upstream + fixture di qualità; il port ha GLM (sperimentale). Usare le fixture per certificare la parità dei logit GLM. |
| `e0824dd` Make release builds warning-free | C | **N/A** (warning solo-C). |
| `ef8d923` Add ROCm GLM 5.2 support | ROCm | **N/A** (fuori scope). |
| `66cbce5` Unify distributed inference CLI and documentation | CLI/docs | **N/A** (CLI/docs distribuito; il port ha il proprio). |
| `36ee8c1` Add CUDA tensor parallelism and session batching | CUDA/TP | **N/A** (fuori scope). |
| `63d9874` Add two-machine Metal tensor parallelism | TP | **N/A** (tensor-parallel + RDMA fuori scope). |
| `fc9efd1` Add DSpark speculative decoding | speculative | **N/A** (famiglia MTP/speculativo fuori scope; il port ha il proprio self-speculative). |

Voci nuove azionabili: `519c4d8` (correttezza cache esperti/sampling) e
`427e281` (resync kernel Metal) — entrambe da valutare on-device. Avanzare la
baseline rivista a `0a7ad77` dopo che le due sono state triageate su Mac.

## Che cosa rientra nell'ambito

DwarfStar condivide con l'upstream solo la superficie dell'engine di
inferenza. Sono rilevanti questi file upstream:

- `ds4.c` — decoder, MoE/NSA, streaming SSD, cache degli expert. Equivalente in
  DwarfStar: `DS4Core` + `DS4Metal`.
- `ds4_metal.m` — runtime e kernel Metal. Equivalente in DwarfStar: `DS4Metal`.
- `ds4_server.c` — comportamento del server HTTP. Equivalente in DwarfStar:
  `Sources/DwarfStar/Features/Server`.

Fuori dall'ambito di questo port: i backend CUDA/ROCm (`ds4_cuda.cu`,
`ds4_rocm.cu`, `rocm/`), il percorso speculativo upstream basato su MTP,
l'agente da terminale (`ds4_agent.c`, comportamento raw-mode/TTY), `ds4-eval` e
`ds4_cli.c`. L'esperimento CLI self-speculative di DwarfStar, separato e privo
di MTP, è documentato in [`SELF-SPECULATIVE.md`](SELF-SPECULATIVE.it.md) e non è
una prova che la superficie MTP upstream sia stata portata.

## Revisione dei commit recenti (fino a `80ebbc3`)

| Commit | Area | Decisione di port |
|---|---|---|
| `d75e23d` Protezione delle free dei tensori Metal | Metal | **N/A.** Protegge dal double-free di handle Objective-C bridged in C. Il `GPUTensor` Swift è gestito da ARC; il bug non esiste in questo port. |
| `8384adf` Correzione del riuso della cache dello streaming SSD Metal | Metal/streaming | **N/A.** Evita di sfrattare uno slot di expert mentre un command buffer è ancora in volo. `GraphContext.commit()` di DwarfStar attende il completamento, quindi non esiste alcuna race tra command buffer di gather/evict. |
| `91bafb5` Recupero delle tool call dentro `<think>` non chiuso | server/generazione | **N/A.** DwarfStar entra in modalità tool sul token DSML indipendentemente da `inReasoning`; se il parsing fallisce, il markup viene mostrato come testo invece di essere scartato. |
| `fd2d173` Irrobustimento del parsing JSON del server | server | **N/A.** Il parser C era scritto a mano; DwarfStar usa `JSONSerialization` di Foundation. |
| `cafc134` Correzione di un warning const del server | server | **N/A.** Warning solo C. |
| `1cfa5cc` Refactoring dell'API della cache degli expert in streaming | streaming | **N/A.** Refactoring multi-backend senza modifiche comportamentali da portare. |
| `7a77a28` Rilascio del margine di cache in caso di fallimento di mlock / `cd57428` Limite alle cache sovradimensionate | streaming | **Parzialmente coperto.** DwarfStar ora ha un `DS4_MLOCK` best-effort per i buffer hot e una cache degli expert limitata a slot, ma non usa lo stesso allocatore slab C. Tenere a mente questa classe di fallimenti quando si modifica il dimensionamento della cache. |
| `f2d701a` Correzione degli slice di layer dello streaming SSD distribuito | distribuito | **Rinviato.** La modalità distribuita di DwarfStar è implementata ma necessita ancora di validazione numerica. Da rivedere insieme a quel lavoro di validazione. |
| `81f35e7` + `b548d86` expert routed a precisione mista | streaming/quant | **Portato.** La quantizzazione per-layer degli expert routed è supportata. Vedere sotto. |

I commit non elencati qui, come i lavori su ROCm/CUDA, il percorso MTP
upstream, l'agente da terminale e `ds4-eval`, sono attualmente al di fuori del
perimetro del port di DwarfStar.

## Portato: expert routed a precisione mista per layer

Alcuni GGUF usano una quantizzazione degli expert routed non uniforme tra i
layer, ad esempio una base IQ2_XXS/Q2_K con alcuni layer selezionati promossi
(upcast) a Q4_K tramite `--tensor-type`. DwarfStar la supporta senza cambiare
il comportamento dei modelli uniformi:

- `LayerWeights` memorizza `gateQuant`, `upQuant` e `downQuant` per layer.
- `GGUFWeights.layer` rileva i tipi reali dei tensori anche quando gli expert
  non sono completamente caricati (`loadExperts == false`).
- `decodeExperts` seleziona i kernel dai campi quant locali al layer invece che
  dai campi quant globali di `DSV4Dims`, coprendo sia il decode sia il prefill
  batched.
- `GGUFWeights.gatherExperts` calcola già i byte per expert dai `blockBytes`
  del tensore, quindi la dimensione della copia è corretta per ogni layer.
- La slot-cache degli expert resta una cache a singola classe di dimensione. I
  layer misti fuori classe la bypassano e usano il gather diretto, il che è
  corretto.
- `InferenceService` registra nel log il numero di layer fuori classe
  all'avvio.

Questo richiede ancora una validazione on-device con una fixture GGUF mista.

## Lacuna aperta: correzione distribuita

`f2d701a` va riesaminato quando l'inferenza distribuita riceverà la
validazione di parità numerica. La priorità attuale è validare la pipeline
distribuita nel suo complesso prima di portare in isolamento le correzioni
upstream a livello di slice.

## Come ripetere il confronto

```sh
git clone --depth 60 https://github.com/antirez/ds4.git /tmp/ds4-upstream
git -C /tmp/ds4-upstream log --oneline --since=2026-06-17
git -C /tmp/ds4-upstream log --oneline 80ebbc3..HEAD -- ds4.c ds4_metal.m ds4_server.c
```

Per ogni nuovo commit, chiedersi:

- Tocca un'area che DwarfStar condivide con l'upstream: engine, Metal o server?
- È una modifica comportamentale o di correttezza, piuttosto che una modifica
  specifica del C su memoria o warning, o riservata a un solo backend?
- È al di fuori delle superfici escluse: CUDA/ROCm, esecuzione MTP upstream,
  agente TTY, strumenti di eval?

Se la risposta è sì, valutare un port in Swift. Altrimenti, registrare il
commit come N/A e far avanzare la baseline dopo la revisione.
