# Backend DeepSeek-V4

Implementazione Metal per DeepSeek V4 Flash e Pro. Il GGUF validato produce una
geometria runtime instance-based, così decoder, scratch, KV, pesi e router usano
le dimensioni del profilo senza dispatch dinamico nel loop per-layer. Il runtime
locale supporta il Pro Q2 a file singolo; il caricamento Pro Q4 split e la
distribuzione Pro non sono ancora dichiarati operativi.

## Struttura

- [`Architecture/`](Architecture/README.md): shape, dimensioni e RoPE.
- [`Weights/`](Weights/README.md): schema tensor e caricamento GGUF.
- [`Streaming/`](Streaming/README.md): staging dei pesi densi da SSD.
- [`Experts/`](Experts/README.md): expert bundle, cache e MetalIO.
- [`MTP/`](MTP/README.md): sidecar Multi-Token Prediction.
- [`Decode/`](Decode/README.md): stato KV/NSA, prefill e generazione.

## Confine architetturale

HyperConnections, MLA con KV latente, compressori NSA, indexer DSA e router
top-6 sono semantica DeepSeek-V4. Non devono essere usati come fallback per
altre famiglie: un backend incompatibile deve fallire esplicitamente al load.
Il router supporta 256 esperti/scala 1,5 per Flash e 384 esperti/scala 2,5 per
Pro; la rete bitonica Pro usa 512 lane e maschera il padding 384...511.
