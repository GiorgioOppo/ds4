# DeepSeekV4/Decode/Execution

Implementazione principale del decoder streaming e del forward per layer.

## File principali

- [`StreamingDecoder.swift`](StreamingDecoder.swift): ownership di runtime,
  pesi, cache, stato KV e knob letti all'inizializzazione.
- [`StreamingDecoder+Factories.swift`](StreamingDecoder+Factories.swift): factory
  resident/streaming, expert bundle, MetalIO e cache.
- [`StreamingDecoder+Forward.swift`](StreamingDecoder+Forward.swift): forward di
  un token e coordinamento delle fasi.
- [`StreamingDecoder+LayerExecution.swift`](StreamingDecoder+LayerExecution.swift):
  command buffer per attention, routing e FFN, inclusa asincronia.
- [`StreamingDecoder+TensorIO.swift`](StreamingDecoder+TensorIO.swift): upload,
  readback e accesso controllato ai tensor.
- [`DecodeLayer.swift`](DecodeLayer.swift): sequenza matematica di un layer.
- [`SpecDecode.swift`](SpecDecode.swift): stato e percorso per decode speculativo.

## Flusso

Il factory costruisce le strategie di peso e cache. `forward` prepara input e
attraversa i layer; `DecodeLayer` usa le operazioni di [`Graph`](../../../../Graph/README.md),
scrive KV e produce lo stato successivo. Il percorso asincrono sovrappone gather
e FFN rispettando le dipendenze tra command buffer.

## Regole di modifica

Non leggere un buffer prima del completamento dell'operazione che lo scrive né riusare staging
in-flight. I knob vengono acquisiti una volta salvo quelli dichiarati live.
Separare cambi numerici da cambi di scheduling e offrire un percorso A/B per le
ottimizzazioni sperimentali.
