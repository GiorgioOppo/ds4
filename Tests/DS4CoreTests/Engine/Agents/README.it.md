# Agent tests

Questa cartella verifica i profili predefiniti di `DS4Engine/Agents` senza
avviare inferenza o tool reali.

## Copertura

- contratto operativo condiviso: lingua dell'utente, dati tool non fidati,
  round multipli ed effetti collaterali limitati;
- privilegio minimo del Reviewer, che deve esporre soltanto tool di lettura;
- assenza del tool `git` mutante dai ruoli che non ne hanno bisogno.
- scope di delega esplicito dell'Orchestrator, sempre sottoinsieme dei tool
  concedibili e senza `git`, cancellazione o orchestrazione annidata.

Quando si aggiunge un agente o si cambia il contratto comune, aggiornare questi
test insieme a `Sources/DS4Engine/Agents/README.md`.
