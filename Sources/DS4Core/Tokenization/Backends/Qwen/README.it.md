# Tokenization/Backends/Qwen

Punto di estensione riservato al futuro tokenizer Qwen.

Non riutilizzare automaticamente il pre-tokenizer JoyAI/DeepSeek: vocabolario,
merge, regex/pre-tokenizer, token speciali e detokenizzazione dovranno essere
letti e verificati sul GGUF Qwen scelto. Nessuna implementazione viene esposta
finché tali invarianti non sono coperte da test di parità.

