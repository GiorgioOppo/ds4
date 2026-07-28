[English](README.md) | **Italiano**

# Riferimenti CPU Laguna

Oracoli scalari senza Metal per il percorso di decode di Laguna S 2.1,
portati da `laguna_graph_forward_token` in `ds4.c` e da `metal/laguna.metal`
del branch di riferimento `laguna-s2.1`. Fissano la semantica della famiglia —
RMSNorm per-testa + coppie rotanti NeoX con YaRN solo sui blocchi
full-attention, il gate softplus per-testa sull'output dell'attention, la KV
cache ad anello F16, SwiGLU senza clamp e il router sigmoide condiviso con GLM
a 10 esperti attivi — così grafo e kernel Metal hanno un confine di
correttezza stabile che gira in normali unit test, esattamente come gli
oracoli `Reference/` di GLM.
