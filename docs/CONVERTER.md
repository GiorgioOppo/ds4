# DeepSeekConverter — Conversione Checkpoint

Modulo per la conversione di checkpoint tra formati: safetensors, GGUF, MLX-native quantizzati.

## Architettura

```
DeepSeekConverter/
├── Converter.swift          # Facade pubblico (enum con metodi statici)
├── ConverterRunner.swift    # Runner asincrono con progress reporting
├── ConversionSpec.swift     # Specifica di conversione (input, output, formato, opzioni)
├── ConversionProgress.swift # Progresso conversione + CancellationToken
├── Rename.swift             # Mappatura nomi tensori tra formati
├── DTypePacking.swift       # Packing/unpacking formati numerici
└── NativeFusion.swift       # Fusione pesi in formato MLX-native
```

## Formati Supportati

| Formato Input | Formato Output | Descrizione |
|---------------|----------------|-------------|
| Safetensors (FP8/FP4) | Safetensors (INT8/INT4) | Quantizzazione pesi |
| Safetensors (FP8/FP4) | MLX-native (int4) | Fusione in formato MLX quantizzato |
| Safetensors | GGUF | Conversione formato llama.cpp |
| GGUF | Safetensors | Deconversione |
| MLX-native | Safetensors | Deconversione |

## ConversionSpec

```swift
public struct ConversionSpec: Sendable {
    public var inputURL: URL
    public var outputURL: URL
    public var format: ConversionFormat  // .safetensors, .gguf, .mlxNative
    public var quantMethod: QuantMethod? // .rtn, .awq (opzionale)
    public var bits: Int?               // 8, 4, 2 (opzionale)
    public var groupSize: Int?          // 128, 64, 32 (opzionale)
    public var calibrationData: URL?    // Dataset per AWQ/GPTQ
}
```

## ConverterRunner

Runner asincrono con callback di progresso:

```swift
public final class ConverterRunner {
    public func run(spec: ConversionSpec,
                    progress: @escaping (ConversionProgress) -> Void) async throws
}
```

## Rename

Mappatura tra convenzioni di naming dei tensori:

| DeepSeek V4 | HuggingFace | MLX-native |
|-------------|-------------|------------|
| `embed.weight` | `model.embed_tokens.weight` | `model.embed_tokens.weight` |
| `layers.0.attn.wq_a.weight` | `model.layers.0.self_attn.q_a_proj.weight` | `model.layers.0.self_attn.q_proj.weight` |
| `layers.0.ffn.moe.gate.weight` | `model.layers.0.mlp.gate.weight` | `model.layers.0.mlp.gate_proj.weight` |

## NativeFusion

Fusione di pesi FP8/FP4 nel formato MLX-native quantizzato (int4 con groupSize=32):

1. Dequantizza peso FP8/FP4 a bf16
2. Quantizza a int4 con MLX.quantized()
3. Salva triplet (weight, scales, biases) come safetensors
4. Aggiorna config.json con `quantization` dict

## DTypePacking

Utility per packing/unpacking di formati numerici custom:

- FP4-E2M1 ↔ 4-bit packed
- FP8-E4M3 ↔ 8-bit
- E8M0 scale ↔ 8-bit
- INT8/INT4/INT2 packed