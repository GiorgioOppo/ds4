# Storage

Pianificazione portabile del working set quando il modello è più grande della
RAM disponibile.

## File principali

- [`SSDCachePlan.swift`](SSDCachePlan.swift): calcola budget e numero di expert
  memorizzabili e interpreta gli argomenti relativi allo streaming SSD.
- [`SimulatedMemoryLock.swift`](SimulatedMemoryLock.swift): riserva e blocca
  memoria anonima per testare scenari con RAM ridotta.

## Flusso e dipendenze

Il piano viene costruito prima del decoder e guida la dimensione delle cache
concrete di `DS4Metal`. La simulazione è uno strumento diagnostico: non contiene
policy del decoder e non legge pesi. Le opzioni runtime sono raccolte nella
[Configuration Reference](../../../README.md#configuration-reference).

## Regole di modifica

Usare aritmetica controllata per byte e GiB, distinguere chiaramente stime da
allocazioni reali e rilasciare sempre le risorse bloccate. Mantenere questo
livello privo di dipendenze Metal.
