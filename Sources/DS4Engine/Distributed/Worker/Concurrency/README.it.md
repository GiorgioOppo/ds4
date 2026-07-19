[English](README.md) | **Italiano**

# Worker/Concurrency

`DistGate.swift` definisce l'actor che serializza il calcolo sul decoder del
worker, anche quando più connessioni TCP vengono servite in parallelo.

## Flusso e dipendenze

[`Serving`](../Serving/README.it.md) esegue le closure Metal attraverso il gate.
La proprietà della sessione viene controllata separatamente dal lifecycle; il
gate protegge l'esecuzione, non la semantica dei frame.

## Estensione

Non inserire attese di rete dentro `run`. Un futuro scheduler concorrente deve
dimostrare che decoder, cache KV e buffer scratch sono indipendenti prima di
consentire più lavori simultanei.
