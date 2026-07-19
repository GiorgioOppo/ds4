[English](README.md) | **Italiano**

# Model/Backends/DeepSeekV4

Forma e validazione metadata di DeepSeek V4 Flash/Pro.

- `DeepSeekV4Configuration.swift` contiene `DeepSeekV4Shape`,
  `DeepSeekV4Configuration`, default ed errori specifici.
- Gli alias pubblici storici (`ModelShape`, `ModelConfig`, `ModelDefaults`,
  `ModelVariant`, `ModelConfigError`) restano disponibili senza cambiare il
  comportamento dei consumer esistenti.

La configurazione verifica prima `general.architecture`; un GGUF Qwen viene
riconosciuto come famiglia ma rifiutato perché il relativo backend non è ancora
implementato. I vecchi GGUF DeepSeek senza la chiave generale continuano a
essere riconosciuti tramite i metadata `deepseek4.*`.

