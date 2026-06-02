# DeepSeekTraining — Fine-Tuning

Modulo scaffold per il fine-tuning del modello DeepSeek-V4. Attualmente in fase di implementazione — il runner è uno stub che valida la spec e lancia `FineTuneNotImplemented` perché il motore Metal non ha ancora kernel backward.

## Architettura

```
DeepSeekTraining/
├── FineTuner.swift          # Facade pubblico
├── FineTuneSpec.swift       # Specifica di fine-tuning
├── FineTuneRunner.swift     # Runner asincrono con progress
└── FineTuneProgress.swift   # Progresso e metriche
```

## FineTuneSpec

```swift
public struct FineTuneSpec: Sendable {
    public var inputDir: URL            // Checkpoint base
    public var outputDir: URL           // Checkpoint fine-tunato
    public var trainDataDir: URL        // Dataset training
    public var valDataDir: URL?         // Dataset validazione
    public var learningRate: Float      // Learning rate
    public var batchSize: Int           // Batch size
    public var epochs: Int              // Epoche
    public var loraRank: Int?           // LoRA rank (opzionale, nil = full fine-tune)
    public var loraTargets: [String]?   // Layer target per LoRA
    public var quantizationBits: Int?   // Bits per quantizzazione (opzionale)
}
```

## Stato Implementazione

| Componente | Stato |
|------------|-------|
| Validazione spec | ✅ Implementato |
| Pianificazione | ✅ Implementato |
| Backward pass | ❌ Stub (richiede kernel Metal backward) |
| LoRA | ❌ Stub |
| Full fine-tune | ❌ Stub |
| Progress reporting | ✅ Scaffold |
| Dataset loading | ❌ Stub |

Il modulo ha la stessa forma di DeepSeekConverter (Spec + Progress + Runner + Facade) in modo che quando un backend reale arriverà, il cambio sarà locale a `FineTuneRunner`.