# Distributed/Worker

`DistWorker` è un nodo inizialmente inattivo che ascolta una porta, riceve file
e carica il lavoro assegnato dal coordinator.

## Struttura

- `DistWorker.swift`: configurazione e stato condiviso protetto da lock.
- [`Lifecycle`](Lifecycle/README.md): listener, start/stop e `HELLO`.
- [`Assignments`](Assignments/README.md): caricamento slice o shard esperti.
- [`Files`](Files/README.md): ricezione resumable e verifica.
- [`KV`](KV/README.md): comandi di checkpoint.
- [`Serving`](Serving/README.md): dispatch dei frame e pipeline.
- [`Concurrency`](Concurrency/README.md): serializzazione del lavoro Metal.

## Flusso

Ogni connessione è servita in una task, ma `DistGate` consente un solo lavoro
di calcolo alla volta. L'assegnazione può essere riusata se coincide; altrimenti
il worker carica il nuovo motore fuori dal lock e pubblica lo stato solo a
caricamento completato.

## Estensione

Validare frame e sessione prima di accedere all'engine. Non tenere lock durante
I/O o caricamenti lunghi. Ogni risorsa persistente deve essere identificata da
modello e responsabilità del nodo per evitare riuso fra shard incompatibili.
