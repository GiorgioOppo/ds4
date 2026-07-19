[English](README.md) | **Italiano**

# Test della DSA compatta GLM

Test in pura CPU coprono i byte della cache compatta, la crescita lazy degli
slab, lo schedule IndexShare a 21 layer, lo scoring ReLU pesato, il top-k
causale deterministico e il riuso della selezione. Non servono un device
Metal né un GGUF del modello.

La suite di riferimento dell'attention dimostra che gli ordini di valutazione
espanso e assorbito di `GLM52AttentionCPUReference` concordano entro la
tolleranza float, che l'ordine di selezione è irrilevante, che le selezioni
degeneri si riducono a semplici proiezioni dei value e che gli input
malformati (selezioni vuote/duplicate/fuori intervallo, dimensioni errate,
valori non finiti) vengono rifiutati prima di qualsiasi calcolo.
