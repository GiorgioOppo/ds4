# Agents

Questa cartella definisce i profili degli agenti disponibili all'orchestratore.
Non esegue inferenza e non contiene logica UI.

## Componenti

- `AgentProfile.swift`: `AgentProfile`, prompt di sistema, strumenti concessi,
  profilo esperti e valori predefiniti.
- `AgentRegistry`: registro thread-safe usato da chat, strumenti e sub-agent.

## Flusso e dipendenze

La GUI o `InferenceService` seleziona un profilo dal registro; il prompt e la
lista dei nomi tool vengono poi risolti tramite [`Tools`](../Tools/README.md).
L'area dipende soltanto da Foundation, mentre l'esecuzione resta in
[`Inference`](../Inference/README.md).

## Estensione

Per aggiungere un ruolo, definire un profilo stabile e un identificatore unico,
concedere solo i tool necessari e aggiornare i test del registro. Non inserire
qui stato di conversazione, accesso a file o chiamate di rete.
