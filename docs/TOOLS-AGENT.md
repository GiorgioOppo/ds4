# DeepSeekTools — Toolbox per Agent

Toolbox puramente Swift, model-agnostico, per agenti software: operazioni su file, shell, ricerca, Git, percorsi, Xcode, sistema. Non ha dipendenze da MLX/Metal — può essere usato da CLI, UI o server headless.

## Architettura

### Tool (Protocollo)

```swift
public protocol Tool: Sendable {
    var schema: ToolSchema { get }
    func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput
    func permissionSummary(input: [String: Any]) -> String
}
```

### ToolRegistry (Actor)

Registro centrale dei tool disponibili. Gestisce:

- **Registrazione** e lookup per nome
- **Filtro modalità**: in `.plan`, solo tool `.readOnly` e `.planning` sono visibili
- **Cache sessionale** delle decisioni `alwaysAllow`

```swift
let registry = ToolRegistry()
registry.registerAll(DefaultTools.standard(planStore: store))
let schemas = registry.availableSchemas(mode: .build)
let output = try await registry.run(named: "read", input: ["path": "foo.swift"], context: ctx)
```

### ToolSchema

Descrizione JSON-Schema di un tool, compatibile con MCP e OpenAI.

```swift
public struct ToolSchema: Sendable {
    public let name: String
    public let description: String
    public let category: ToolCategory
    public let inputSchema: JSONValue
}
```

### ToolCategory

Categorie di tool per il sistema di permessi:

| Categoria | Comportamento |
|-----------|---------------|
| `.readOnly` | Lettura, sempre permesso in `.plan` |
| `.mutating` | Modifica file, passa dalla policy permessi |
| `.dangerous` | Operazioni rischiose (shell, delete), richiedono conferma |
| `.planning` | Pianificazione (plan, task, todo) |
| `.shell` | Esecuzione shell, richiede conferma esplicita |
| `.network` | Accesso rete, policy permessi |

### ToolContext

Contesto per-esecuzione passato a ogni tool:

```swift
public struct ToolContext: Sendable {
    public let rootDirectory: URL
    public let allowEscapingRoot: Bool
    public let additionalReadRoots: [URL]
    public let mode: AgentMode
    public let permission: PermissionDelegate
    public let environment: [String: String]?
    public let cancellation: Cancellation?
}
```

### PermissionDelegate

Politica di permessi per azioni potate. Interfacce fornite:

- **AutoPermissionDelegate**: per test/CLI (allow/dangerous sempre, dangerous negato)
- **InteractivePermissionDelegate**: per UI con modale di conferma

## Tool nativi

### File System
| Tool | Categoria | Descrizione |
|------|-----------|-------------|
| `ReadTool` | `.readOnly` | Legge file di testo |
| `WriteTool` | `.mutating` | Scrive file |
| `EditTool` | `.mutating` | Modifica file (sostituzione testo esatto) |
| `GlobTool` | `.readOnly` | Pattern matching su file |
| `GrepTool` | `.readOnly` | Ricerca regex nei file |
| `ApplyPatchTool` | `.mutating` | Applica unified diff |

### Repository
| Tool | Categoria | Descrizione |
|------|-----------|-------------|
| `RepoOverviewTool` | `.readOnly` | Riepilogo albero repository |
| `RepoCloneTool` | `.network` | Git clone remoto |

### Pianificazione
| Tool | Categoria | Descrizione |
|------|-----------|-------------|
| `PlanTool` | `.planning` | Legge/sostituisce il piano corrente |
| `TaskTool` | `.planning` | Gestisce lista task |
| `TodoTool` | `.planning` | Lista TODO persistente |

### Rete
| Tool | Categoria | Descrizione |
|------|-----------|-------------|
| `WebFetchTool` | `.network` | GET HTTP |
| `WebSearchTool` | `.network` | Ricerca web (DuckDuckGo, Tavily, Brave, Serper) |

### Shell
| Tool | Categoria | Descrizione |
|------|-----------|-------------|
| `ShellTool` | `.shell` | Esecuzione comandi shell (con sandbox macOS opzionale) |

### Unix Tools (opzionali)
Strumenti POSIX aggiuntivi: `Archive`, `Files`, `Git`, `Hash`, `Json`, `Mutate`, `Process`, `System`, `Text`, `TextBin`.

### Xcode Tools (opzionali)
Strumenti per sviluppo Xcode: `Build`, `Device`, `Inspect`, `Plist`, `SPM`, `Signing`, `Simulator`.

### Altri
| Tool | Categoria | Descrizione |
|------|-----------|-------------|
| `LSPTool` | `.readOnly` | Stub LSP (non ancora implementato) |

## DefaultTools

Helper per popolare un `ToolRegistry` con i tool standard:

```swift
DefaultTools.standard(
    planStore: store,
    includeShell: true,
    includeNetwork: true,
    includeRepoClone: true,
    includeStubs: false,
    includeUnixTools: false,
    includeXcodeTools: false,
    shellUsesSandbox: true,
    webSearchProvider: nil
)
```

## Agent Mode

Due modalità operative:

| Modalità | Comportamento |
|----------|---------------|
| `.build` | Accesso completo. Mutazioni e shell passano dalla policy permessi |
| `.plan` | Sola lettura. Mutazioni negate, shell richiede conferma |

## Plugin System

### Plugin (Protocollo)

```swift
public protocol Plugin: Sendable {
    var name: String { get }
    var version: String { get }
    func bootstrap(host: PluginHost) async throws
    func shutdown() async
    func observe(envelope: any MessageEnvelope) async
}
```

### PluginBase

Classe astratta con default sovrascrivibili per lifecycle e osservazione eventi.

### PluginRegistry

Registro centralizzato di plugin con:

- `register(_:)` / `unregister(_:)`
- `bootstrapAll()` / `shutdownAll()`
- Distribuzione envelope a tutti i plugin
- Lookup per nome

## MessageEnvelope

Protocollo per eventi di trasporto nel sistema:

```swift
public protocol MessageEnvelope: Sendable {
    var kind: String { get }
    var id: UUID { get }
}
```

Implementazioni concrete:
- `Question` / `Answer` — turni di chat
- `AgentChatMessage` — messaggi agente
- `AgentEvent` — eventi stream agente

## Chat e Agent

### ChatBase

Classe astratta per orchestrazione chat:

- Template method `ask(_:)`: Question → Answer
- Mantiene stato `Chat` (turni Q/A)
- Notifica observer via `publish(envelope:)`

### AgentBase

Classe astratta per agent runtime:

- `step(input: AgentInput) → AgentEventStream` — stream di eventi
- `reset()` / `cancel()` — lifecycle
- Modalità `.build` / `.plan`

### ToolProvider

Sorgente di tool registrabili (per scoperta dinamica, bundle plugin, API remote).

## MCP Transport

### MCPTransport (Protocollo)

Trasporto JSON-RPC astratto per MCP:

```swift
public protocol MCPTransport: Sendable {
    func connect() async throws
    func send(_ jsonRPCMessage: Data) async throws
    func receive() -> AsyncThrowingStream<Data, Error>
    func disconnect() async
}
```

Supporta stdio (Process), HTTP/SSE, WebSocket, in-memory loopback.

## JSONValue

Tipo Sendable + Codable per grafi JSON, duplicato in DeepSeekKit e DeepSeekTools per evitare dipendenza incrociata.