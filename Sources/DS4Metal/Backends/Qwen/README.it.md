[English](README.md) | **Italiano**

# Backend Qwen

Placeholder per il futuro supporto Qwen. La famiglia è riconosciuta nella
struttura del progetto, ma non esistono ancora loader, decoder o kernel Qwen e
nessun modello Qwen deve essere accettato dal runtime attuale.

Prima dell'implementazione occorre scegliere esplicitamente la variante GGUF,
validarne metadata e tensor naming e definire tipi concreti per pesi, scratch,
KV cache, attention e output head. Il backend non deve riutilizzare per
fallback l'attention MLA, le HyperConnections, i compressori NSA o il router
DeepSeek-V4.
