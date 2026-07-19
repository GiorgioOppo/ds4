[English](README.md) | **Italiano**

# Test

Test unitari per gli strati del motore in puro Swift. **La correttezza è la
regola n. 1** di questo progetto: questi test validano il port Swift rispetto
al riferimento C originale e, dove utile, rispetto a implementazioni CPU
fedeli.

- **`DS4CoreTests/Core/`** copre i formati deterministici, la tokenizzazione,
  il rendering delle conversazioni, il sampling, le forme dei modelli e la
  pianificazione dello storage.
- **`DS4CoreTests/Metal/`** copre i singoli kernel GPU, la composizione dei
  grafi, il comportamento di decode/cache, il caricamento dei modelli e la
  creazione del runtime.
- **`DS4CoreTests/Engine/`** copre il protocollo distribuito, la persistenza,
  la sicurezza dei progetti, la gestione dei modelli, la diagnostica e i tool.

```sh
make test        # oppure: swift test
```

## Esecuzione da Xcode

I test sono anche collegati al `.xcodeproj` generato come bundle di logic test
`DS4CoreTests`, senza app host. Dopo aver generato il progetto, apri
`DwarfStar.xcodeproj` e premi **Cmd+U**. Lo schema `DwarfStar` ha l'azione
Test collegata a `DS4CoreTests`.

```sh
make xcodeproj
xcodebuild test -project DwarfStar.xcodeproj -scheme DwarfStar -destination 'platform=macOS'
```

## Prerequisiti Metal e skip

I test Metal possono saltare quando l'host non espone alcun device Metal.
Alcuni test legacy usano inoltre ancora una `metalDir` specifica dello
sviluppatore o il percorso di un GGUF di produzione; questi saltano quando la
fixture è assente e andrebbero migrati a kernel incorporati o a fixture
compatte. Uno skip non è un pass e va riportato separatamente.

Vedi [`METAL-TESTS.md`](METAL-TESTS.it.md) per le convenzioni complete su skip,
parità e confronto numerico. Ogni sottodirectory di test ha un README locale
che ne descrive l'ambito e le regole per le fixture.
