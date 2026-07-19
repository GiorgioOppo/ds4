[English](README.md) | **Italiano**

# Tools/Builtins

Contiene gli strumenti incorporati. Ogni file aggiunge una proprietà
`BuiltinTool` tramite `extension ToolRegistry`.

## Gruppi

- [`Arithmetic`](Arithmetic/README.it.md): tempo e calcolo deterministico.
- [`Files`](Files/README.it.md): accesso raw confinato al progetto.
- [`Projects`](Projects/README.it.md): navigazione dell'indice `ProjectCache`.
- [`Web`](Web/README.it.md): ricerca e fetch protetti.
- [`Agents`](Agents/README.it.md): elenco e delega a sub-agent.
- `Git.swift`: operazioni git locali whitelisted.
- `GitHubClone.swift`: import di un repository pubblico controllato.

## Registrazione

Un nuovo tool va creato nella cartella del dominio, aggiunto a
`ToolRegistry.builtins` e classificato come `projectScoped` quando necessario.
Aggiornare i profili in [`../../Agents`](../../Agents/README.it.md) che devono
esporlo. `subAgentGrantable` include soltanto operazioni ammesse nei contesti
delegati.

## Regole

Validare sempre il JSON e restituire messaggi brevi. Non implementare client di
rete o invocazioni di processo direttamente nel file del tool: collocare quella
logica in [`../Integrations`](../Integrations/README.it.md). Vedi anche
[`../SICUREZZA.md`](../SICUREZZA.it.md).
