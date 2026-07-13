# Projects

`ProjectCache` indicizza un progetto importato separatamente dalla memoria della
chat. Il modello esplora soltanto le parti richieste tramite i tool `project_*`
e `file_*`.

## File

- `ProjectCache.swift`: singleton thread-safe, limiti e stato dell'indice.
- `+Indexing`: import, filtri, traversal e reload.
- `+Queries`: lista, albero, ricerca e letture limitate.
- `+Editing`: write/edit con rilettura del contenuto corrente.
- `+Files`: accesso raw, range di linee e operazioni confinate.

Le invarianti di sicurezza sono in
[`SICUREZZA-PERCORSI.md`](SICUREZZA-PERCORSI.md).

## Flusso e dipendenze

L'import registra percorsi relativi testuali entro limiti di quantità e
dimensione. I contenuti sono caricati pigramente sotto un budget LRU-like; una
ricerca su file freddi non deve espellere inutilmente la cache. I built-in in
[`Tools/Builtins/Projects`](../Tools/Builtins/Projects/README.md) sono il
principale consumer.

## Estensione

Limitare sempre output e memoria, preservare thread safety e riconvalidare il
percorso reale immediatamente prima di ogni I/O. Le operazioni distruttive
richiedono un contratto tool esplicito e non devono operare su directory.
