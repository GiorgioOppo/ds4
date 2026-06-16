# DS4Engine/Tools

Il function-calling: i tool integrati che il modello può invocare, più la libreria progetti su cui operano.

- **`ToolRegistry.swift`** — superficie del registry (`builtins`, `projectScoped`, `subAgentGrantable`, `execute`, `specs`) + gli helper condivisi (parsing argomenti, tool aritmetici, valutatore di espressioni).
- **`Builtins/`** — **un file per tool** (`extension ToolRegistry { static let X }`). Aggiungere un tool = nuovo file qui + voce in `builtins[]`.
- **`ProjectCache.swift`** — indice di un progetto importato + i tool `project_*`/`file_*` (read/list/search/write/edit/add/modify per riga). Non tocca la memoria della chat.
- **`GitTool.swift`** — esecuzione di sottocomandi git locali (whitelist, no rete) nella radice del progetto.
- **`Agents.swift`** — `AgentProfile` (ruolo = system prompt + tool + profilo esperti) e `AgentRegistry` (roster condiviso, letto da `agents_list`).
