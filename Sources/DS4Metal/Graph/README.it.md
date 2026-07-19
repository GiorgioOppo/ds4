[English](README.md) | **Italiano**

# Graph

Livello di composizione tra decoder e wrapper dei kernel Metal. Traduce le fasi
matematiche del modello in dispatch su tensori e command buffer.

## Struttura

- [`Core/`](Core/README.it.md): contesto condiviso, pipeline e configurazione.
- [`Operations/`](Operations/README.it.md): operazioni attention, compressor, MoE,
  router, output e trasformazioni element-wise.

## Flusso e dipendenze

Il [`decode DeepSeek-V4`](../Backends/DeepSeekV4/Decode/README.it.md) crea o riusa un `GraphContext`; le estensioni in
`Operations` scelgono il wrapper di [`Kernels`](../Kernels/README.it.md), impostano
buffer/offset e codificano il dispatch. I tensor sono forniti da
[`Runtime`](../Runtime/README.it.md) e i pesi dal backend selezionato.

## Regole di modifica

Il grafo orchestra ma non deve duplicare il codice Metal. Ogni operazione deve
esplicitare forma, quantizzazione, ownership del command buffer e requisiti di
sincronizzazione. Mantenere le estensioni suddivise per fase matematica.
