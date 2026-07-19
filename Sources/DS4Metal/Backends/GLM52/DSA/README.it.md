[English](README.md) | **Italiano**

# Riferimento DSA compatto GLM 5.2

Questa directory contiene politiche non eseguibili e oracoli di correttezza su
CPU per l'attenzione DSA compatta di GLM 5.2.

La cache compatta memorizza, per ogni token vivo, 512 valori KV-LoRA e 64
valori di coda RoPE per ciascuno dei 78 layer normali. Una chiave dell'indexer
da 128 valori è memorizzata solo per i 21 layer full-indexer: 0, 1, 2, poi da
6 a 74 ogni quattro layer. In F16 sono 95,232 byte per token.

`GLM52CompactDSACapacityPolicy` fa crescere slab impaccati append-only su
richiesta invece di allocare l'intero contesto logico alla creazione della
sessione. Una futura implementazione GPU deve preservare le righe popolate
durante l'aggiunta di slab e deve applicare il controllo del budget di memoria
prima dell'allocazione.

`GLM52IndexerCPUReference` implementa il punteggio ReLU pesato 32 per 128, il
top-k causale deterministico e il riuso IndexShare. È un oracolo, non un
percorso di decode ottimizzato e non una dichiarazione che il runtime GLM sia
pronto. È tenuto separato dal contratto fratello di score batch-Metal: questo
oracolo F32 a token singolo possiede la validazione dei punteggi finiti, la
selezione deterministica e la politica di riuso, mentre il contratto batch
modella l'input di cache F16 e il mascheramento causale a `-inf`.

`GLM52AttentionCPUReference` è l'oracolo per lo stadio successivo alla
selezione: il nucleo di attenzione sulle righe della cache compatta. Mantiene
due ordini di valutazione — `expanded` (il riferimento F32 da manuale di
upstream: materializzare chiavi e valori per testa da k_b/v_b, poi eseguire
l'attenzione) e `absorbed` (l'ordine del kernel: assorbire la query in k_b,
valutare le righe di cache grezze larghe 512, accumulare la softmax nel
dominio KV-LoRA, proiettare attraverso v_b una sola volta). Il loro accordo
verificato entro tolleranza è la fixture che i futuri kernel Metal
`qk_lowrank`/`attention_indexed` dovranno rispettare.
