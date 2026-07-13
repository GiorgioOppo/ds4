# Persistence

Contiene persistenza applicativa indipendente dalla GUI. Attualmente ospita la
cache dei checkpoint KV in [`KV`](KV/README.md).

## Flusso e dipendenze

`InferenceService` e i worker distribuiti richiedono lookup, restore e store;
la cartella `Persistence` definisce il ciclo di vita su disco, mentre snapshot
e import layer sono forniti da `DS4Metal`.

## Estensione

Ogni nuovo archivio deve dichiarare chiave, formato/versione, budget, strategia
di eviction, atomicità e comportamento su file corrotti. Evitare dipendenze da
SwiftUI e mantenere RAM limitata durante I/O di grandi dati.
