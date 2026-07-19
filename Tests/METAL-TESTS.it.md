[English](METAL-TESTS.md) | **Italiano**

# Esecuzione dei test Metal

I test Metal confrontano i risultati della GPU con riferimenti compatti fedeli
alla CPU. Fanno parte di `DS4CoreTests`, ma richiedono capacità che possono
mancare in CI, nei processi sandbox o in ambienti non Apple.

## Politica di skip

- Usare `XCTSkipUnless` o lanciare `XCTSkip` con una motivazione precisa quando
  non è disponibile alcun device Metal o quando manca una fixture esterna
  esplicitamente opzionale.
- Non catturare i fallimenti di compilazione delle pipeline o delle asserzioni
  numeriche trasformandoli in skip su un host con supporto Metal.
- Non fare `return` silenziosamente da un test: XCTest deve riportare perché
  non è stato eseguito.
- Un test saltato non fornisce alcun segnale di correttezza. Le note di
  rilascio e i riepiloghi devono elencare separatamente i test passati, i
  fallimenti e gli skip.

Il runtime preferito è `MetalRuntime()` con i kernel incorporati. I percorsi
assoluti hard-coded verso un checkout di sviluppo sono un comportamento legacy
e non devono essere introdotti nei nuovi test.

## Confronti numerici

1. Costruire buffer di input piccoli e deterministici.
2. Calcolare un riferimento CPU indipendente usando l'ordine di accumulazione
   previsto.
3. Eseguire il kernel GPU o l'operazione del grafo.
4. Verificare la shape dell'output e la finitezza dei valori prima del
   confronto dei valori.
5. Usare l'uguaglianza esatta per i contratti documentati come bit-identici;
   altrimenti dichiarare una tolleranza assoluta/relativa appropriata al tipo
   di dato.

I parametri di scheduling come il numero di simdgroup dovrebbero essere testati
con più di un valore quando promettono un output identico. I percorsi fusi
dovrebbero essere confrontati con il loro riferimento non fuso o CPU prima di
dare fiducia alle misure di performance.

## Comandi

```sh
swift test
```

Dal progetto Xcode generato:

```sh
xcodebuild test \
  -project DwarfStar.xcodeproj \
  -scheme DwarfStar \
  -destination 'platform=macOS'
```

Usare il filtro di XCTest per un'area specifica, ad esempio
`swift test --filter MetalRoPETests`. Eseguire l'intera suite dopo aver
modificato codice condiviso di runtime, buffer, grafo o kernel.
