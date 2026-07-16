# DS4Metal

Runtime Metal multi-backend. Le primitive comuni gestiscono device, tensori,
pipeline e operazioni GPU; ogni famiglia di modello possiede architettura,
pesi, stato ricorrente, streaming e orchestrazione del decode. Il target dipende
da `DS4Core` e da `Metal.framework`.

## Struttura

- [`Runtime/`](Runtime/README.md): device, command queue, pipeline e tensori GPU.
- [`Backends/`](Backends/README.md): implementazioni specifiche per famiglia di modello.
- [`Model/`](Model/README.md): tipi di modello realmente condivisi tra backend.
- [`Kernels/`](Kernels/README.md): wrapper Swift dei kernel `.metal`.
- [`Graph/`](Graph/README.md): operazioni che compongono il grafo di inferenza.

## Flusso

`MetalRuntime` compila le sorgenti incorporate. Il backend selezionato converte
i tensor descriptor di `DS4Core.GGUFModel` in `GPUTensor`, dimensiona scratch e
KV cache e orchestra prefill/forward tramite `GraphContext`. Il backend
DeepSeek-V4 è operativo; la cartella Qwen documenta il confine preparato ma non
fornisce ancora un decoder.

## Regole di modifica

- Correttezza numerica e parità con il percorso di riferimento precedono le
  ottimizzazioni.
- Dichiarare esplicitamente layout, tipo quantizzato, offset e sincronizzazione.
- Non aggiungere campi di famiglie diverse a un unico contenitore di pesi o scratch.
- La selezione del backend avviene fuori dal ciclo per-layer: il percorso caldo
  resta concreto e privo di dispatch dinamico per operazione.
- Non modificare direttamente codice generato; rigenerarlo dalla sorgente.
- Documentare nuovi knob `DS4_*` nella configurazione principale e nel dominio.
- Aggiungere test CPU o Metal mirati per ogni nuovo percorso di dispatch.
