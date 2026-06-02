# DeepSeek-V4-Pro-MacOS — Documentazione

Porting nativo in Swift/MLX del modello DeepSeek-V4 per Apple Silicon, con un client desktop SwiftUI completo che supporta anche modelli remoti via OpenRouter e Anthropic.

## Architettura del Progetto

```
DeepSeek-V4-Pro-MacOS/
├── Package.swift                          # SwiftPM manifest (10 target, MLX dep)
├── Sources/
│   ├── DeepSeekKit/                       # Motore di inferenza core (46 file)
│   ├── DeepSeekConverter/                 # Conversione checkpoint (7 file)
│   ├── DeepSeekTraining/                  # Fine-tuning scaffold (4 file)
│   ├── DeepSeekTools/                     # Toolbox agnostico (60+ file)
│   ├── DeepSeekIntegrations/              # Adapter esterni (HTTP, Sandbox, Slack)
│   ├── DeepSeekVocabPruner/               # Pruning vocabolario/esperti (12 file)
│   ├── DeepSeekUI/                        # App SwiftUI completa (70+ file)
│   └── deepseek/ converter/ vocab_pruner/ # CLI executables
├── Tests/
│   ├── DeepSeekKitTests/                  # 34 test suite
│   ├── DeepSeekToolsTests/
│   ├── DeepSeekUITests/
│   └── DeepSeekVocabPrunerTests/
├── References/                            # Riferimenti upstream Python
├── Tools/                                 # Script Python/sh di supporto
└── docs/                                  # Documentazione
```

## Moduli

| Modulo | Descrizione | Dipendenze |
|--------|-------------|------------|
| **DeepSeekKit** | Motore Transformer: MLA, MoE, HyperConnections, kernel Metal, tokenizer, sampling, quantizzazione, caricamento pesi | MLX, MLXNN, MLXFast, MLXOptimizers, Metal |
| **DeepSeekConverter** | Conversione checkpoint tra formati (safetensors, GGUF, MLX-native) | DeepSeekKit |
| **DeepSeekTraining** | Fine-tuning scaffold (stub — nessun backward kernel ancora) | — |
| **DeepSeekTools** | Toolbox agnostico: read/write/edit/grep/glob/shell/apply_patch, agent mode, permission policy, plugin system | Nessuna dip. MLX |
| **DeepSeekIntegrations** | Bridge per sistemi esterni: HTTPRecorder, Sandbox, Slack | DeepSeekTools |
| **DeepSeekVocabPruner** | Pruning vocabolario ed esperti per ridurre dimensione checkpoint | DeepSeekKit, DeepSeekConverter |
| **DeepSeekUI** | App desktop SwiftUI completa: chat, agenti, MCP, progetti, documenti, server locale, settings | Tutti i moduli sopra |
| **deepseek** (CLI) | Interfaccia a riga di comando per inferenza | DeepSeekKit |
| **converter** (CLI) | Tool CLI per conversione checkpoint | DeepSeekKit, DeepSeekConverter |
| **vocab_pruner** (CLI) | Tool CLI per pruning vocabolario | DeepSeekKit, DeepSeekConverter, DeepSeekVocabPruner |

## Dipendenze esterne

- **MLX** v0.22+ — array framework Apple Silicon (MLX, MLXNN, MLXFast, MLXOptimizers)
- **Metal** — GPU compute shaders per kernel custom
- **Network.framework** — server locale HTTP/1.1
- **Accelerate** (vDSP, vForce) — sampling ottimizzato

## Piattaforma

- macOS 14+ (Sonoma)
- Apple Silicon (M1+, GPU unificata)

## Licenza

Vedi [LICENSE](../LICENSE)