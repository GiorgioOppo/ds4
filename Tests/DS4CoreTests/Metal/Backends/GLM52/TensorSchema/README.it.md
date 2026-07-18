# Test dello schema dei tensori GLM 5.2

Record sintetici della directory dei tensori esercitano tutti i confini dei
layer e le quantizzazioni routed supportate senza creare o mappare un payload
GGUF a dimensione piena.

I test dello stream-planner coprono le dimensioni esatte dei blocchi gate/up
IQ2_XXS, Q2_K e Q4_K, le proiezioni down Q6_K, la selezione top-8 ordinata per
rank, gli ID duplicati e fuori intervallo, i layer MoE non validi, la geometria
non corrispondente, le dimensioni di directory troncate e l'overflow degli
offset. I test della weight-map fissano anche i nomi GGUF canonici tipizzati e
il contratto dei descrittori privi di payload.

`GLM52RealHeaderIntegrationTests` è un controllo di contratto opzionale contro
una copia sparsa a dimensione esatta di un file reale. Imposta
`DS4_GLM52_SPARSE_GGUF` per validare i metadati, il tokenizer, tutte le
ricerche tipizzate dei pesi e un piano di lettura degli esperti senza leggere i
payload dei pesi.
