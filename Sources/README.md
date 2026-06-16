# Sources

Tutto il codice Swift, diviso per **target** (moduli). Pipeline: `DS4Core → DS4Metal → DS4Engine → DwarfStar`.

| Target | Tipo | Ruolo |
|---|---|---|
| `DS4Core/` | libreria | core puro: GGUF mmap, tokenizer, sampler, shape, formato chat/tool (no Metal) |
| `DS4Metal/` | libreria | runtime Metal + grafo di decode + kernel (port di `ds4_metal.m`) |
| `DS4Engine/` | libreria | `InferenceService` (attore), tool/agenti, KV su disco, sub-agent, distribuito |
| `DwarfStar/` | app | GUI SwiftUI (chat, agenti, progetti, server, benchmark, diagnostica) |
| `DS4Demo/` | CLI | demo: bring-up Metal + streaming GGUF |

Swift non usa le cartelle per i moduli: dentro un target le sottocartelle sono solo organizzazione (gli `import` non cambiano). I confini tra target invece sono reali (vedi le `dependencies` in `Package.swift`).
