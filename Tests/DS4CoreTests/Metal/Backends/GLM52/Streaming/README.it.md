[English](README.md) | **Italiano**

# Test dello streaming GLM 5.2

Fedeltà al byte di `GLM52PayloadReader` contro file sintetici a pattern: una
lettura a descrittore singolo deve restituire esattamente la slice di payload
del descrittore, e un piano di stream di esperti eseguito deve depositare
ogni intervallo pianificato all'offset del proprio record packed
gate|up|down, in modo identico sul percorso concorrente e su quello seriale.

La suite di rifiuto dimostra che gli errori tipizzati scattano prima che un
byte si muova: dimensioni di destinazione errate, descrittori o intervalli
pianificati oltre la fine reale del file (download troncato), piani costruiti
a mano che mescolano dimensioni in byte per esperto, piani vuoti e percorsi
non apribili.

La suite della slot cache dimostra hit identici al byte rispetto a letture
fresche, l'eviction LRU con sopravvivenza in caso di sovrapposizione
parziale, il pinning del batch e la soglia minima di budget alla creazione
(la selezione di un token deve poterci stare).

Tutti i fixture sono piccole geometrie legali per i blocchi (Q4_K/Q6_K, 16
esperti, ~2 MiB) scritte su file temporanei — nessun device Metal e nessun
download di modelli reali coinvolti. Il passaggio end-to-end open+plan+read
su un GGUF sparso reale vive in
`../TensorSchema/GLM52RealHeaderIntegrationTests.swift` dietro
`DS4_GLM52_SPARSE_GGUF`.
