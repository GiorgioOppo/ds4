# Formato dei checkpoint KV

Il file combina header portabili di `DS4Core` e un corpo Swift esplicito:

```text
KVC header (48 byte)
u32 modelNameLength + UTF-8 modelName
u32 tokenCount + tokenCount × u32 tokenId
DSV4 payload header (52 byte)
per ogni layer:
  u32 rawStart
  u32 rawFloatCount + raw Float32
  u8 hasCompressor
  se presente:
    u32 stateCount
    u32 stateLength + stateKV Float32
    stateScore Float32
    u32 cacheFloatCount + cache Float32
```

Tutti gli interi sono little-endian. L'header identifica quantizzazione,
contesto, numero token, hit e timestamp; il nome modello e la sequenza completa
impediscono il riuso fra modelli o prefissi diversi.

## Lettura

Il reader valida header, conteggi e frontiera token, quindi importa un batch di
layer per volta e libera i buffer intermedi. Dati troncati, conteggi incoerenti
o modello diverso invalidano l'intera entry.

## Scrittura

Lo snapshot viene trasferito in un contenitore a proprietà unica; ogni layer è
scritto e rilasciato. Il file diventa visibile come entry soltanto a conclusione
della scrittura. Successivi restore aggiornano in-place soltanto i metadati
dell'header.

## Compatibilità

Non aggiungere campi senza aggiornare versione/header e test fixture. Un lettore
nuovo non deve tentare di interpretare euristicamente un corpo incompatibile.
