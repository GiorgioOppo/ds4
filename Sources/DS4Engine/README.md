# DS4Engine

Lo strato di servizio che alimenta la GUI: l'attore di inferenza, il function-calling, gli agenti, la persistenza KV e il runtime distribuito. Dipende da `DS4Core` + `DS4Metal`. Nessun link esterno.

- **`Service/`** — `InferenceService` (attore: prompt → stream di eventi, riuso KV multi-turno, loop dei tool, sub-agent isolati), `DiskKVStore` (checkpoint KV su disco), `Diagnostics`.
- **`Tools/`** — `ToolRegistry` + un file per tool in `Builtins/`; `ProjectCache` (libreria progetti), `GitTool`, `Agents` (`AgentProfile` + `AgentRegistry`).
- **`Download/`** — `ModelDownloader` (download GGUF resumibile da Hugging Face + verifica SHA-256).
- **`Distributed/`** — inferenza distribuita (coordinatore/worker, protocollo, transport).
