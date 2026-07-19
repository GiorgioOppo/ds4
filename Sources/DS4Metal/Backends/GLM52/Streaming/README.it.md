[English](README.md) | **Italiano**

# Streaming GLM 5.2

Prima tranche del passo 1 della roadmap GLM: dal piano di lettura validato ai
byte reali del payload GGUF.

`GLM52PayloadReader` è accesso `pread` con limiti a un singolo payload
GLM 5.2. Legge per intero un descrittore validato (`read(_:into:)`,
`bytes(of:)`) oppure esegue un `GLM52ExpertStreamPlan`, impacchettando gli
otto esperti selezionati come record gate|up|down contigui in ordine di rank
del router (`read(plan:into:)`, layout esposto da `packedLayout(of:)`).

Ogni lettura è dimostrata due volte prima che un byte si muova: il
planner/schema ha già dimostrato che l'intervallo giace dentro il suo
tensore, e il reader lo ridimostra contro la dimensione reale del file
(`fstat`). Il costruttore di produzione (`init(path:weightMap:)`) rifiuta
all'apertura un GGUF troncato o sostituito verificando il descrittore più
lontano della mappa dei pesi rispetto alla dimensione del file. Le 24 letture
di un piano riempiono destinazioni disgiunte in modo concorrente
(`DispatchQueue.concurrentPerform`, offset espliciti — il pattern di
`GGUFWeights.gatherExperts`); il percorso seriale è identico al byte.

Il record gate|up|down rispecchia il layout a slot interlacciato del pool di
esperti DeepSeek. `GLM52ExpertSlotCache` vi si appoggia: una slot cache LRU
con chiave (layer, esperto) i cui hit servono record identici al byte a una
lettura fresca — la lezione a monte sull'invarianza dei logits vale per
costruzione. Il batch in corso di servizio è pinnato contro sé stesso, e i
budget inferiori al working set top-8 di un token vengono rifiutati alla
creazione. I riempimenti MetalIO arriveranno più avanti al suo fianco, con
questo percorso pread come fallback di correttezza permanente.

Il reader non interpreta mai i byte quantizzati che sposta e non possiede
alcuna risorsa GPU: resta testabile contro file sintetici senza un device
Metal. Ogni nuova primitiva di lettura deve mantenere la doppia dimostrazione
dei limiti (piano + file) e gli errori tipizzati che precedono qualsiasi
riempimento parziale.
