# Builtins/Agents

Espone il catalogo degli agenti e la delega di lavori isolati.

## Tool

- `agents_list`: ruoli disponibili e tool associati.
- `subagent_search`: orientamento/ricerca delegata.
- `subagent_run`: richiesta completa con target, ruolo e limiti.

## Flusso e dipendenze

Le specifiche leggono [`AgentRegistry`](../../../Agents/README.md); l'esecuzione
effettiva del sub-agent è gestita da
[`Inference/Subagents`](../../../Inference/Subagents/README.md), così il tool non
possiede direttamente il decoder.
Il ruolo e la lista `tools` sono input del modello e possono soltanto restringere
lo scope `delegatedToolNames` del profilo padre, mai ampliarlo.

## Estensione

Non rendere grantable a un sub-agent gli strumenti di orchestrazione stessa.
Limitare profondità, round, token e tool concessi per impedire ricorsione o
amplificazione non controllata.
