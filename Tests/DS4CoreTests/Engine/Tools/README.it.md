[English](README.md) | **Italiano**

# Test del motore dei tool

- `ToolRegistryTests.swift` valida registrazione, concessioni, schemi e
  dispatch.
- `MCPTests.swift` copre messaggi/configurazione MCP con fixture isolati.
- `GitHubToolTests.swift` valida il parsing e la costruzione sicura dei
  comandi.

Non invocare servizi remoti reali, non mutare i repository dello sviluppatore
e non dipendere da credenziali installate. I test dei tool dovrebbero
iniettare i confini di esecuzione e asserire i fallimenti di autorizzazione
con la stessa cura delle chiamate riuscite.
