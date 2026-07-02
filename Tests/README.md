# Tests

Test unitari dell'engine puro-Swift. La **correttezza è la regola #1** del progetto: questi test validano il port contro il riferimento C.

- **`DS4CoreTests/`** — kernel (matvec MoE, flash-attn, norm, RoPE…), grafo di decode, GGUF, tokenizer, sampler, serializzazione KV, downloader.

```sh
make test        # oppure: swift test
```

## Esecuzione da Xcode

I test sono anche un target del `.xcodeproj` generato (`DS4CoreTests`, un
*logic-test bundle* senza app host). Dopo `make xcodeproj`, apri `DwarfStar.xcodeproj`
e premi **⌘U**: lo schema `DwarfStar` ha la Test action collegata a `DS4CoreTests`.

```sh
make xcodeproj
xcodebuild test -project DwarfStar.xcodeproj -scheme DwarfStar -destination 'platform=macOS'
```

> I test dei kernel Metal si auto-saltano (`XCTSkipUnless`) finché il loro
> `metalDir` — attualmente un percorso assoluto fisso in cima a ogni file di test
> — non punta alla cartella `metal/` reale; vanno aggiornati per usare i kernel
> embedded (`MetalRuntime()`) prima che girino in CI/Xcode.
