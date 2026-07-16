# Tools/Core

`ToolRegistry.swift` definisce il contratto comune e il catalogo dei built-in.

## Tipi e responsabilità

- `BuiltinTool`: specifica `DS4Core.ToolSpec` più closure di esecuzione.
- `ToolOutput`: testo e metadati restituiti al ciclo del modello.
- `ToolExecutionPolicy`: allow-list esplicita e deny-by-default per ogni ciclo.
- `constrainSubAgentTools`: interseca le scelte del modello con lo scope di
  delega fidato del profilo padre; `subagent_run` non amplia i permessi.
- `ToolRegistry`: liste `builtins`, `projectScoped`, `subAgentGrantable`, lookup,
  composizione spec, dispatch e helper per argomenti.
- `ArithmeticEvaluator`: parser condiviso dal calcolatore.

## Flusso e dipendenze

Dipende da Foundation e `DS4Core`. Le definizioni concrete sono estensioni del
registro in [`Builtins`](../Builtins/README.md); gli strumenti MCP sono aggiunti
da [`MCPManager`](../MCP/README.md).

## Estensione

Registrare ogni tool una sola volta e mantenere stabili i nomi pubblici. Se
richiede un progetto, inserirlo in `projectScoped`; se non è sicuro per un
sub-agent, escluderlo esplicitamente. Gli helper qui devono essere generici.

Ogni esecuzione deve passare la stessa allow-list dichiarata al modello:
`execute(_:policy:)` per i built-in o `executeAuto(_:policy:)` per built-in e
MCP. Una chiamata negata restituisce `tool_not_allowed`; soltanto una chiamata
consentita ma sconosciuta restituisce `nil` e può quindi usare il flusso manuale.
