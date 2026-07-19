[English](README.md) | **Italiano**

# Test dell'API di tokenizzazione

I test della factory verificano la politica basata sulla sola architettura e
la costruzione a partire da minuscole tabelle tokenizer GGUF sintetiche. Le
architetture Qwen esplicite e quelle sconosciute devono fallire invece di
ripiegare su DeepSeek; il fallback legacy su DeepSeek si applica solo quando
`general.architecture` è assente.

La selezione del protocollo di conversazione è testata allo stesso confine di
frontend e non cambia la disponibilità a runtime.
