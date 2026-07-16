# Tokenization/Backends/DeepSeekV4

Tokenizer byte-level BPE con pre-tokenizer JoyAI/DeepSeek, token speciali di
DeepSeek V4 e modalità reasoning.

- `DeepSeekV4Tokenizer.swift` contiene l'implementazione concreta e conserva
  l'alias pubblico storico `Tokenizer`.
- `ThinkMode.swift` contiene modalità e prefisso di reasoning DeepSeek.

Il comportamento byte-per-byte e le API esistenti restano invariati; la nuova
collocazione impedisce di riutilizzare accidentalmente questo tokenizer per un
GGUF Qwen.

