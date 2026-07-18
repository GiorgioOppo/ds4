# Backend Qwen

Stato: famiglia GGUF riconosciuta, backend non ancora implementato.

L'ispezione restituisce nome, famiglia e capability del modello, ma le capability
runtime restano vuote e `BackendSelector` termina con
`backend <architettura> non ancora implementato`. Questo avviene prima che il
codice tenti di leggere metadati, token speciali o tensori DeepSeek.

Per abilitare Qwen serviranno decoder Metal, schema pesi, KV, chat template/tool
codec, test numerici e smoke test GGUF dedicati; non è sufficiente rimuovere il
controllo dalla factory.
