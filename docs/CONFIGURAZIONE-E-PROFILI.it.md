[English](CONFIGURAZIONE-E-PROFILI.md) | **Italiano**

# Configurazione e profili di esecuzione

Questo documento spiega come leggere e applicare la configurazione. La tabella
completa di ogni chiave, valore e default resta la
[Configuration Reference](../README.it.md#riferimento-di-configurazione), che è la
fonte autorevole per i singoli parametri.

I knob storici `DS4_*` descritti qui appartengono al backend DeepSeek V4 salvo
indicazione contraria. La GUI li espone soltanto quando il descrittore runtime
dichiara la capability corrispondente; il futuro backend Qwen avrà un profilo
proprio e non erediterà automaticamente cache esperti, NSA o geometrie Q4.

## Livelli di configurazione

La configurazione arriva da tre livelli:

1. impostazioni GUI persistite in `UserDefaults`;
2. variabili d'ambiente lette dal motore e dalla demo;
3. parametri locali di server, distribuzione, MCP, agenti e download.

Nell'app la GUI esporta i principali `DS4_*` all'avvio e quando cambiano. Per
questi knob il valore persistito dall'app prevale sull'ambiente del processo.
Nella demo, nei test e negli strumenti da terminale prevale invece l'ambiente.

## Quando viene letto un valore

La maggioranza delle opzioni di memoria, formato e pipeline viene fissata al
caricamento del modello. Cambiarla richiede unload/reload. Le opzioni di prefill
indicate come aggiornabili vengono rilette a ogni chiamata. Alcune impostazioni
di sampling o UI si applicano al turno successivo senza ricaricare i pesi.

Prima di un benchmark annotare sempre:

- GGUF e hash/size;
- RAM e modello del Mac;
- contesto e prompt;
- cache calda o fredda;
- tutti i knob non predefiniti;
- warm-up escluso dalla misura.

## Categorie di knob

| Categoria | Esempi | Rischio principale |
|---|---|---|
| memoria/I/O | dense stream, pread, mlock, bundle, MetalIO | pressione RAM o contesa SSD |
| cache | slot esperti, Q4 cache, KV su disco | spazio e invalidazione |
| prefill | chunk, union, route batch, FFN batch | picco di memoria transitoria |
| kernel | fusioni e numero di simdgroup | compatibilità GPU e parità |
| diagnostica | profilo route, DIAG, warm-up | overhead che altera la misura |
| qualità | Q4/Q8 derivati, active experts | divergenza numerica o qualitativa |

## Lossless e lossy

Non tutti i toggle hanno lo stesso significato:

- streaming, layout, prefetch, cache e fusioni dichiarate bit-identiche non
  riducono l'informazione del modello;
- alcune fusioni o matmul cambiano l'ordine di accumulo e possono differire di
  pochi ulp;
- requantizzazione Q4/Q8 e riduzione degli esperti attivi sono lossy.

Ogni profilo condiviso deve indicare esplicitamente quali opzioni sono lossy.
Disattivare contemporaneamente `DENSE_Q4`, `QKV_Q4` e `SHARED_Q4` non garantisce
da solo qualità perfetta se restano altri knob lossy o se il GGUF di partenza è
fortemente quantizzato.

## Profilo misurato a bassa RAM

Il profilo usato come base sui Mac da 16 GB privilegia:

- streaming dei pesi densi;
- letture esperti dirette o bundle;
- pinning best-effort dei buffer caldi;
- cache esperti dimensionata senza saturare la memoria;
- Q4 delle grandi proiezioni, quando il compromesso qualitativo è accettato;
- chunk prefill abbastanza grande da ammortizzare i pesi;
- fusioni e pipeline asincrona già validate.

Il valore migliore dipende dal margine di RAM reale. Applicazioni aperte in
parallelo possono spingere macOS in compressione/swap e ridurre il decode fino a
ordini di grandezza; prima di attribuire il rallentamento a MetalIO o a un
kernel controllare pressione memoria e swap.

## Profilo orientato alla qualità

Per isolare la qualità:

1. usare il GGUF di qualità desiderata;
2. tenere `DS4_ACTIVE_EXPERTS=6`;
3. disattivare cache derivate lossy (`DENSE_Q4`, `QKV_Q4`, `SHARED_Q4`,
   `COMP_Q8`);
4. usare sampling deterministico o seed fissato;
5. conservare streaming, bundle e cache lossless se aiutano le prestazioni;
6. confrontare logit o testo contro lo stesso prompt e template.

Le opzioni lossless non devono essere disattivate per principio: cambiare il
percorso I/O non cambia i pesi letti.

## Profilo di diagnosi

`DS4_DIAG=1` abilita il report ampio della demo. `DS4_PROFILE_ROUTE=1` aggiunge
sincronizzazioni per separare sottostadi e non va usato per il numero finale di
tok/s. Usare un knob per volta e tenere invariati usage file, prompt e cache.

Esempio di forma del comando:

```sh
DS4_DIAG=1 \
DS4_USAGE_FILE=/percorso/profilo.usage.json \
DS4_WARMUP=4 \
swift run -c release DS4Demo /percorso/modello.gguf 32 "Prompt di controllo"
```

I comandi operativi completi della demo sono in
[`Sources/DS4Demo/README.md`](../Sources/DS4Demo/README.it.md).

## Cache e file derivati

- `.usage.json` contiene frequenze di routing, non pesi modificati;
- `.q4dense` e cache compressori contengono rappresentazioni derivate;
- `.expbundle` riordina gli stessi byte degli esperti per letture contigue;
- i checkpoint KV dipendono da modello, token e configurazione compatibile;
- i worker distribuiti conservano file verificati nel proprio store gestito.

Una cache deve essere validata per dimensione, versione e identità del modello.
Se si sospetta corruzione, rimuovere soltanto il derivato interessato, non il
GGUF originale.

## Precedenza nella distribuzione

Il coordinator invia ai worker una whitelist di knob prestazionali. Un worker
non deve sostituirli con i propri default dopo l'assegnazione. Opzioni lossy e
sidecar viaggiano in campi tipizzati per impedire configurazioni implicite.

Vedere [INFERENZA-DISTRIBUITA.md](INFERENZA-DISTRIBUITA.it.md).

## Regola di documentazione

Quando viene introdotto un nuovo parametro:

1. documentare default, momento di lettura e dipendenze;
2. indicare se è bit-identico, equivalente o lossy;
3. aggiungerlo alla Configuration Reference;
4. aggiornare il README della cartella proprietaria;
5. se viene propagato ai worker, aggiornare e testare la whitelist;
6. aggiungere un A/B riproducibile quando è un knob prestazionale.

## Knob GLM 5.2

I knob con lo stesso significato tra backend usano il namespace condiviso
`DS4_*`; solo i dettagli esclusivi di GLM mantengono il prefisso
`DS4_GLM_*`. I vecchi nomi prefissati sono accettati soltanto come alias di
fallback e il nome canonico condiviso ha sempre precedenza. Le ottimizzazioni
del motore sono tutte default; i knob esistono per A/B e diagnosi, sono
speculari nella UI delle impostazioni GLM, e ogni valore viene stampato a
init del motore nel Log motore (`DS4 glm: knob …`). I verdetti misurati sono
su M1 Pro 16 GB con l'IQ2_XXS pubblicato.

| Knob | Default | Note |
|---|---|---|
| `DS4_MTLIO` | 1 | 0 misurato +18% di decode su 16 GB (MetalIO contende i commit del decode); il preset GUI imposta 0 |
| `DS4_ACTIVE_EXPERTS` | 8 | esperti instradati per token; il condiviso è sempre attivo. 6 = −25% di I/O esperti, lieve trade-off qualità |
| `DS4_RESIDENT_LAYERS` | adattivo | floor di 3 layer dense sotto pressione RAM (residenti extra vengono paginati: ~+750 ms/token misurati) |
| `DS4_GLM_FUSE` | 1 | fusione dei commit (metà delle attese sincrone) |
| `DS4_GLM_MOE_BATCH` | 1 | MoE batched (tutti gli esperti instradati in due dispatch) |
| `DS4_PREFILL_MOE_BATCH` | 1 | fase B del prefill multi-token (pesi esperti letti una volta per tile di 4 token; 3 dispatch per wave, bit-esatto vs il percorso per-applicazione) |
| `DS4_GLM_GPU_ROUTER` | 1 | router fuso su GPU (−18% di prefill) |
| `DS4_MLOCK` | 1 | wiring dei pesi residenti (head da 433 a 39 ms/token) |
| `DS4_PREAD_SPLIT` | 4 | letture layer parallele nel prefill (solo prefill; il decode resta seriale per scelta misurata) |
| `DS4_GLM_SG` / `DS4_NSG` | 1 / 4 | dispatch cooperativo dei kernel e simdgroup per threadgroup |
| `DS4_STREAM_SLOTS` | 3 (4 con fusione) | slot di staging dei layer (~250 MiB l'uno) |
| `DS4_GLM_EXPERT_ARENA` | 24 | slot dell'arena condivisa degli esperti staged (~10 MiB l'uno) |
| `DS4_GLM_SPEC_EXPERTS` | off | speculazione esperti; N ≥ 2 = solo top-N. Misurata net-zero su SSD saturo |
| `DS4_SPEC_K` | off | decode speculativo prompt-lookup (demo) |
| `DS4_GLM_LAYERQ4` / `DS4_GLM_LAYERQ4_DIR` | 1 / sibling→gestita | pack sidecar Q4 dei layer (`<gguf>.q4dense`) |
| `DS4_BUNDLE_DIR` | sibling→gestita | pack del bundle esperti legacy (`<gguf>.expbundle`) |
| `DS4_GLM_NOCACHE` | off | letture F_NOCACHE; misurate controproducenti come default |
| `DS4_USAGE_FILE` | `<gguf>.glm-usage.json` | persistenza della usage imatrix ("off" disattiva) |
| `DS4_GLM_AUTOTUNE` / `DS4_GLM_BUILD_LAYERQ4` | off | prepass del demo: auto-tune / build del pack sidecar |

I knob di dispatch vengono riletti a ogni init del motore
(`GLM52DispatchKnobs`): un toggle della GUI ha effetto al prossimo
caricamento del modello, senza riavviare l'app.
