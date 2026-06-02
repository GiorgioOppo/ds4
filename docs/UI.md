# DeepSeekUI — App Desktop SwiftUI

Applicazione macOS completa per chat con modelli locali e remoti, con supporto per agenti, MCP (Model Context Protocol), progetti, documenti vettorizzati, server locale OpenAI-compatibile e molto altro.

## Architettura

```
DeepSeekUI/
├── DeepSeekUIApp.swift          # Entry point @main
├── State/                       # Modelli e servizi (36 file)
│   ├── InferenceService.swift   # Wrapper seriale per Transformer + Tokenizer
│   ├── ChatStore.swift          # Store multi-chat con persistenza
│   ├── Conversation.swift       # Modello conversazione Codable
│   ├── ModelState.swift         # Lifecycle caricamento modello
│   ├── ModelLibrary.swift       # Registry endpoint modello
│   ├── AgentLibrary.swift       # Registry agenti configurabili
│   ├── MCPServerLibrary.swift   # Registry server MCP
│   ├── MCPClient.swift          # Client MCP individuale
│   ├── DocumentLibrary.swift    # Documenti vettorizzati
│   ├── ProjectLibrary.swift     # Progetti con indicizzazione
│   ├── AppSettings.swift        # Preferenze utente @AppStorage
│   ├── LocalServer.swift        # Server HTTP/1.1 locale actor
│   ├── LocalServerController.swift # Controller server locale
│   ├── LocalServerRoutes.swift  # Route server locale
│   ├── ThemeStore.swift         # Tema UI
│   ├── KeybindingStore.swift    # Shortcut tastiera
│   ├── SkillLibrary.swift       # Skill registry
│   ├── SlashCommandLibrary.swift# Comandi slash
│   ├── NativeToolHost.swift     # Host tool nativi
│   ├── PermissionStore.swift    # Decisioni permessi persistenti
│   ├── KeychainStore.swift      # Credenziali API keychain
│   ├── OpenRouterAPI.swift     # Client API OpenRouter
│   ├── OpenRouterCatalog.swift  # Catalogo modelli OpenRouter
│   ├── AnthropicAPI.swift      # Client API Anthropic
│   └── Altri store...
├── Views/                       # Viste SwiftUI (30+ file)
│   ├── ContentView.swift        # Vista principale
│   ├── ChatView.swift           # Chat completa
│   ├── SettingsScene.swift      # Finestra impostazioni
│   └── Sottocartelle: Chat, Settings, Agents, MCP, Projects, ...
├── Utility/                     # Utility (MarkdownText, ProjectIndexer, ...)
└── Resources/                   # Info.plist, Assets.xcassets, entitlements
```

## Features Principali

### Chat
- Streaming token-by-token con buffer live
- Prefill trace (mostra cosa vede il modello)
- Reasoning disclosure (pensiero ` thinking... response`)
- Throughput metrics (tok/min prefill e generazione)
- Tool calling integrato con MCP
- Persistenza conversazioni in JSON

### Modelli
- **Locali**: DeepSeek-V4 da checkpoint safetensors/GGUF/MLX-native
- **Remoti**: OpenRouter (qualsiasi modello), Anthropic (Claude)
- Picker con recents e caricamento asincrono
- LoadPlan con strategia automatica (MLX_NATIVE vs MEMORY_MAPPED)
- Impostazioni di quantizzazione, W8A8, warmup

### Agenti
- Presets configurabili con system prompt, tool allowlist, thinking mode
- Modalità Build/Plan
- Delegazione tra agenti
- Skill registry

### MCP (Model Context Protocol)
- Connessione a server MCP esterni (tools, resources, prompts)
- Piscina di client MCP con pooling
- Tool calling integrato nel flusso chat
- Server MCP locali via stdio

### Progetti
- Raggruppamento file/cartelle in progetti
- Indicizzazione con vettorizzazione (tokens → KV cache)
- Inventario file con albero
- Security bookmarks per accesso persistente

### Documenti
- Import file per vettorizzazione
- Model fingerprint per sicurezza riutilizzo
- Integrazione progetti

### Server Locale
- Server HTTP/1.1 locale su `127.0.0.1:8080`
- API OpenAI-compatible (chat completions, streaming SSE)
- Bearer token auth opzionale
- Richiede local endpoint per sicurezza

### Impostazioni
- 13 tab: API Keys, Generation, Loading, Model Config, Quantization, Server, Permissions, Skills, Tools, Themes, Keybindings, Storage, Loading

## Ciclo di Vita di una Generazione

1. User digita messaggio in `ComposerView`
2. `ChatStore.send()` costruisce il prompt con chat template
3. `InferenceService.generate()` esegue prefill + decode
4. Eventi streaming: `.prefillStart` → `.prefillToken` → `.prefillDone` → `.token` → `.done`
5. UI renderizza in `StreamingAssistantTurnView` con metriche live
6. Tool calls vengono eseguiti via MCP pool
7. Risultati tool reinseriti nel prompt per turno successivo

## LocalServer

```swift
actor LocalServer {
    var port: UInt16
    var running: Bool
    
    func start() async throws
    func stop()
    func register(method: String, path: String, handler: LocalServerHandler)
}
```

Supporta:
- `POST /v1/chat/completions` — streaming SSE o JSON
- `GET /v1/models` — lista modelli
- Bearer token opzionale
- `requireLocalEndpoint` per sicurezza