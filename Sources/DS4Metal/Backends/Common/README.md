# Backends/Common

Confine previsto per contratti di alto livello realmente indipendenti dal
modello, come identificazione della famiglia, capacità dichiarate e operazioni
token/chunk. Al momento non contiene sorgenti Swift: le API pubbliche esistenti
rimangono invariate durante il puro spostamento preparatorio.

Runtime e operazioni condivise continuano a vivere in
[`Runtime`](../../Runtime/README.md), [`Graph`](../../Graph/README.md),
[`Kernels`](../../Kernels/README.md) e [`Model`](../../Model/README.md).

Un eventuale protocollo comune può essere invocato al confine della sessione,
ma non deve introdurre type erasure, lookup per stringa o dispatch dinamico nel
percorso caldo dei layer.
