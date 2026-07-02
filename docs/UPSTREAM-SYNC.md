# Sincronizzazione con l'upstream (`antirez/ds4`)

Il motore di DwarfStar è una **riscrittura pure-Swift** del C upstream
[`antirez/ds4`](https://github.com/antirez/ds4.git). Questo file tiene traccia
dell'ultima volta che abbiamo confrontato i commit upstream con il nostro port,
così la prossima revisione riparte da un punto noto invece che da zero.

## Baseline corrente

| | |
|---|---|
| Upstream HEAD valutato | `80ebbc3` |
| Data | 2026-06-17 |
| Repo | `https://github.com/antirez/ds4.git` |
| Esito | **Nessun cambiamento urgente da portare** per il percorso del modello standard. |

## Cosa confrontare (e cosa ignorare)

Condividiamo con l'upstream solo il **motore di inferenza**. I file rilevanti:

- `ds4.c` — decoder, MoE/NSA, streaming, cache esperti → noi: `DS4Core` + `DS4Metal`.
- `ds4_metal.m` — runtime e kernel Metal → noi: `DS4Metal`.
- `ds4_server.c` — server HTTP → noi: `Sources/DwarfStar/Server`.

**Da ignorare** (non li portiamo): backend `ds4_cuda.cu` / `ds4_rocm.cu` / `rocm/`,
MTP / speculative decoding, l'agente da terminale (`ds4_agent.c`, raw mode/TTY),
il grader `ds4-eval`, la CLI `ds4_cli.c`.

## Verdetto sui commit recenti (≤ `80ebbc3`)

| Commit | Area | Verdetto per il nostro port |
|---|---|---|
| `d75e23d` Guard Metal tensor frees | Metal | **N/A** — guard contro il double-free di handle ObjC *bridged* in C. In Swift `GPUTensor` è ARC, niente free manuale: il bug non esiste. |
| `8384adf` Fix Metal SSD streaming cache reuse | Metal/streaming | **N/A** — evita di evictare uno slot esperto mentre un command buffer è ancora *in volo*. Da noi `GraphContext.commit()` fa sempre `waitUntilCompleted` → nessun CB in volo al gather/evict successivo. Race impossibile per costruzione (niente pipelining). |
| `91bafb5` Recover tool calls inside unclosed `<think>` | server/gen | **N/A** — il loop C ignora i marker tool dentro `<think>`. Il nostro `InferenceService.generate` entra in modalità tool sul token DSML *a prescindere* da `inReasoning`, e se non parsabile la mostra come testo invece di scartarla. |
| `fd2d173` Harden server JSON parsing | server | **N/A** — irrobustisce un parser JSON scritto a mano in C. Noi usiamo `JSONSerialization` (Foundation). |
| `cafc134` Fix server const warning | server | **N/A** — warning C. |
| `1cfa5cc` Refactor streaming expert cache API | streaming | **N/A** — refactoring multi-backend, nessun cambio di comportamento. |
| `7a77a28` Release cache margin on mlock failure · `cd57428` Cap oversized caches | streaming | **N/A / marginale** — legati a `mlock` e allo slab allocator C. Noi non facciamo `mlock`; la slot-cache è opt-in e già limitata per slot. |
| `f2d701a` Fix distributed SSD streaming layer slices | distribuito | **Differito** — il nostro distribuito è "implementato, non ancora validato numericamente". Da rivedere *insieme* alla validazione del distribuito, non prima. |
| **`81f35e7` (+`b548d86`) mixed-precision routed experts** | streaming/quant | **Portato** — quant degli esperti **per-layer** (decode + cache). Vedi sotto. |

Commit non elencati (ROCm/CUDA, MTP, agente TTY, `ds4-eval`): fuori perimetro.

## Risolti

### Esperti a precisione mista per-layer (`81f35e7`) — portato

Supporto ai GGUF con esperti routed **non uniformi tra i layer** (es. base
IQ2_XXS/Q2_K con alcuni layer upcastati a Q4_K via `--tensor-type`). Implementazione
(no-op byte-identico sui modelli uniformi):

- **Quant per-layer su `LayerWeights`** (`gateQuant/upQuant/downQuant`), rilevato dai
  tipi reali dei tensori in `GGUFWeights.layer` (i tensori esperti esistono nel GGUF
  anche in streaming, `loadExperts==false`).
- **Decode per-layer**: `decodeExperts` sceglie i kernel su `w.*Quant` invece del
  globale `d.*Quant`. Copre sia il decode sia il prefill batched (stessa funzione).
- **Gather già corretto**: `GGUFWeights.gatherExperts` calcola i byte/expert dal
  `blockBytes` del tensore → copia il numero giusto di byte per ogni layer.
- **Slot-cache** (opt-in): è a singola size-class (quant globale); i layer fuori-classe
  saltano la cache e usano il gather (corretto). `fill`/`warm` partono solo da
  `acquire`, quindi basta il gate in `runLayer` (nessuna modifica a `ExpertSlotCache`).
- **Log** all'avvio (`InferenceService`) della quota di layer fuori-classe.

Da validare on-device con un GGUF misto (qui non si compila e non c'è un fixture misto).

## Gap aperti

### Fix distribuito (`f2d701a`) — differito

Da valutare quando si affronta la validazione numerica del distribuito.

## Come rifare il confronto

```sh
git clone --depth 60 https://github.com/antirez/ds4.git /tmp/ds4-upstream
git -C /tmp/ds4-upstream log --oneline --since=2026-06-17   # nuovi commit dopo la baseline
# per i soli file che portiamo:
git -C /tmp/ds4-upstream log --oneline 80ebbc3..HEAD -- ds4.c ds4_metal.m ds4_server.c
```

Per ogni nuovo commit, chiedersi: tocca un'area che condividiamo (motore/Metal/server)
ed è un **cambio di comportamento/correttezza** (non un fix C-specifico di memoria,
non ROCm/CUDA/MTP/TTY)? Se sì → valutare il port; altrimenti annotarlo qui come N/A.
Aggiornare poi la baseline al nuovo HEAD.
