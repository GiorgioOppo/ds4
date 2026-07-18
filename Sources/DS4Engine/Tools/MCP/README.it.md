# Tools/MCP

Collega server Model Context Protocol e presenta i loro strumenti accanto ai
built-in.

## File

- `MCPConfig.swift`: configurazioni stdio/HTTP e import/export `mcpServers`.
- `MCPProtocol.swift`: JSON-RPC 2.0, initialize, list/call e parsing risultati.
- `MCPTransport.swift`: processi stdio e Streamable HTTP/JSON/SSE.
- `MCPClient.swift`: actor per handshake, request ID, timeout, ping e disconnect.
- `MCPManager.swift`: registro thread-safe di client, stati e tool namespaced.

## Flusso

Il manager applica la configurazione, crea un client per server, completa
`initialize` e mantiene una snapshot sincrona delle `ToolSpec`. Un tool remoto
`read_file` del server `fs` viene esposto come `mcp_fs_read_file`; collisioni
ricevono un suffisso e la mappatura inversa resta nell'indice. Il registry
instrada la call al client corretto e converte il risultato in testo.

## Dipendenze e ciclo di vita

Dipende da Foundation e `DS4Core`; è consumato da [`Core`](../Core/README.md).
Timeout e cancellazione devono risolvere immediatamente la richiesta pendente.
I change handler notificano i consumer quando cambia la lista dichiarabile.

## Estensione

Preservare compatibilità JSON-RPC, correlazione per ID e gestione delle
notification. Non ricavare server/tool analizzando il nome pubblico. Un nuovo
trasporto implementa `MCPTransport` e deve definire framing, sessione,
cancellazione e diagnostica stderr/rete.

In un'app sandboxed i processi stdio ereditano il sandbox; per server che
richiedono accessi più ampi preferire un processo esterno raggiunto via HTTP.
