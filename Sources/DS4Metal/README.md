# DS4Metal

Backend Metal di DeepSeek-V4: carica i pesi, compila i kernel incorporati,
costruisce il grafo e gestisce prefill e generazione token-per-token. Il target
dipende da `DS4Core` e da `Metal.framework`.

## Struttura

- [`Runtime/`](Runtime/README.md): device, command queue, pipeline e tensori GPU.
- [`Model/`](Model/README.md): architettura, pesi GGUF, expert e streaming SSD.
- [`Kernels/`](Kernels/README.md): wrapper Swift dei kernel `.metal`.
- [`Graph/`](Graph/README.md): operazioni che compongono il grafo di inferenza.
- [`Decode/`](Decode/README.md): stato ricorrente, prefill e generazione.

## Flusso

`MetalRuntime` compila le sorgenti incorporate; il loader converte i tensor
descriptor di `DS4Core.GGUFModel` in `GPUTensor`; `StreamingDecoder` dimensiona
scratch e KV cache. Ogni forward usa `GraphContext` e i wrapper kernel per
codificare command buffer, quindi l'output head restituisce i logits al sampler
di `DS4Core`.

## Regole di modifica

- Correttezza numerica e parità con il percorso di riferimento precedono le
  ottimizzazioni.
- Dichiarare esplicitamente layout, tipo quantizzato, offset e sincronizzazione.
- Non modificare direttamente codice generato; rigenerarlo dalla sorgente.
- Documentare nuovi knob `DS4_*` nella configurazione principale e nel dominio.
- Aggiungere test CPU o Metal mirati per ogni nuovo percorso di dispatch.
