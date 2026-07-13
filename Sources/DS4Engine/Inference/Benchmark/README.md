# Inference/Benchmark

Misura il backend già caricato senza dipendere dalla GUI.

## Componente

`InferenceService+Benchmark.swift` definisce `BenchPoint`, `benchmark` e
`warmup`. Il benchmark usa un prompt sintetico, misura prefill e decode e
registra sia throughput medio sia velocità per-token; il warm-up inizializza
kernel e cache degli esperti una sola volta.

## Flusso e dipendenze

L'estensione opera sull'actor di [`Service`](../Service/README.md) e usa sampler
e decoder di `DS4Core`/`DS4Metal`. Poiché altera la KV con dati sintetici,
invalida esplicitamente la conversazione reale.

## Estensione

Una nuova metrica deve separare chiaramente tempo di preparazione, prefill,
sampling CPU e decode GPU. Non cambiare parametri di qualità dell'utente durante
un benchmark e mantenere cancellabili i loop lunghi.
