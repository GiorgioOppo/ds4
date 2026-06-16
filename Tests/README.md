# Tests

Test unitari dell'engine puro-Swift. La **correttezza è la regola #1** del progetto: questi test validano il port contro il riferimento C.

- **`DS4CoreTests/`** — kernel (matvec MoE, flash-attn, norm, RoPE…), grafo di decode, GGUF, tokenizer, sampler, serializzazione KV, downloader.

```sh
make test        # oppure: swift test
```
