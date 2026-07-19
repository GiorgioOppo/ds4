[English](README.md) | **Italiano**

# Metal Graph Tests

Questa cartella conserva le operazioni di grafo realmente comuni: gestione del
`GraphContext` e blocco FFN denso. Le composizioni MLA, HyperConnections, NSA,
router e decode layer sono testate nel
[`backend DeepSeek-V4`](../Backends/DeepSeekV4/Graph/README.it.md).

I test di grafo stanno sopra i singoli kernel. Devono rilevare errori di wiring,
shape, durata dei buffer e ordine dei comandi usando componenti CPU/reference
per calcolare i risultati attesi. I casi limite dei singoli kernel restano in
`Kernels/`.
