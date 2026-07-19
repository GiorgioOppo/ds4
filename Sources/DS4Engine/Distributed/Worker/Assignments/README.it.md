[English](README.md) | **Italiano**

# Worker/Assignments

Carica e pubblica le responsabilità assegnate al worker.

## File

- `DistWorker+Assignment.swift`: slice di layer, output head, cache KV e knob.
- `DistWorker+ExpertAssignment.swift`: maschera verticale di esperti e relativo
  `ExpertShardEngine`.

## Flusso e dipendenze

L'handler valida il payload, risolve i file trasferiti, applica solo knob
whitelisted e costruisce il motore fuori dal lock. Lo stato precedente resta
utilizzabile fino al commit atomico del nuovo motore; al termine viene inviato
`READY`. Dipende da [`Execution`](../../Execution/README.it.md) e
[`Protocol/Handshake`](../../Protocol/Handshake/README.it.md).

## Estensione

Separare fase di claim, caricamento e commit. Non pubblicare assegnazioni
parziali e non riusare un motore se modello, slice, contesto o opzioni numeriche
non coincidono.
