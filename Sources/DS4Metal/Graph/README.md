# Graph

Livello di composizione tra decoder e wrapper dei kernel Metal. Traduce le fasi
matematiche del modello in dispatch su tensori e command buffer.

## Struttura

- [`Core/`](Core/README.md): contesto condiviso, pipeline e configurazione.
- [`Operations/`](Operations/README.md): operazioni attention, compressor, MoE,
  router, output e trasformazioni element-wise.

## Flusso e dipendenze

[`Decode`](../Decode/README.md) crea o riusa un `GraphContext`; le estensioni in
`Operations` scelgono il wrapper di [`Kernels`](../Kernels/README.md), impostano
buffer/offset e codificano il dispatch. I tensor sono forniti da
[`Runtime`](../Runtime/README.md) e i pesi da [`Model`](../Model/README.md).

## Regole di modifica

Il grafo orchestra ma non deve duplicare il codice Metal. Ogni operazione deve
esplicitare forma, quantizzazione, ownership del command buffer e requisiti di
sincronizzazione. Mantenere le estensioni suddivise per fase matematica.
