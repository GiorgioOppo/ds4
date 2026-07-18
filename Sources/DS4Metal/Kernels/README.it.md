# Kernels

Wrapper Swift delle compute pipeline Metal. Ogni file associa una famiglia di
funzioni `.metal`, valida gli argomenti essenziali e codifica il dispatch.

## Struttura

- [`Attention/`](Attention/README.md): flash attention, RoPE e indexer sparso.
- [`Compression/`](Compression/README.md): compressor KV e HyperConnections.
- [`Dense/`](Dense/README.md): matvec/matmul densi e quantizzati.
- [`MoE/`](MoE/README.md): router ed expert FFN.
- [`Tensor/`](Tensor/README.md): trasformazioni tensor generiche.

Le sorgenti autorevoli sono in `metal/*.metal`; la copia incorporata è descritta
in [`Runtime/Generated`](../Runtime/Generated/README.md).

## Flusso e regole

Il [`Graph`](../Graph/README.md) chiama questi wrapper con `GPUTensor` e command
buffer. Un wrapper non decide la sequenza del modello e non deve effettuare
attese CPU nascoste. Dopo un cambio kernel: modificare la sorgente `.metal`,
eseguire `make embed-kernels`, aggiornare firma/dispatch Swift e aggiungere test
su forma, dtype, offset non nullo e limiti delle threadgroup.
