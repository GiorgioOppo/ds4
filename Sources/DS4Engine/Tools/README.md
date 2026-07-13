# Tools

Implementa il function calling del modello: contratti, strumenti incorporati,
integrazioni condivise e server MCP esterni.

## Struttura

- [`Core`](Core/README.md): `BuiltinTool`, `ToolOutput` e `ToolRegistry`.
- [`Builtins`](Builtins/README.md): un tool per file, raggruppato per dominio.
- [`Integrations`](Integrations/README.md): client web, git e GitHub riutilizzabili.
- [`MCP`](MCP/README.md): configurazione, protocollo, trasporti e manager MCP.

Le proprietà di sicurezza e le superfici autorizzate sono riassunte in
[`SICUREZZA.md`](SICUREZZA.md).

## Flusso

1. Il client chiede a `ToolRegistry` le `ToolSpec` abilitate.
2. Il modello emette nome e JSON degli argomenti.
3. `executeAuto` risolve prima i built-in e poi l'indice MCP.
4. `ToolOutput` torna al ciclo di inferenza come risultato del tool.

I tool di progetto usano [`ProjectCache`](../Projects/README.md); i profili in
[`Agents`](../Agents/README.md) decidono quali nomi sono dichiarati.

## Estensione

Un nuovo tool deve avere schema ristretto, output limitato, errori leggibili e
una politica chiara per progetto/sub-agent. La logica condivisa o con side
effect esterni va in `Integrations`, non duplicata nella closure del built-in.
