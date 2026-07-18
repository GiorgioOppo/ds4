# DeepSeekV4/Streaming

Streaming dei pesi densi per-layer da SSD tramite ring di staging, con cache
residenti opzionali per ridurre la banda per token.

## File principali

- [`DenseStreamer.swift`](DenseStreamer.swift): stato, piano per layer, file
  descriptor e inizializzazione del ring.
- `DenseStreamer+Types.swift`: entry, campi e tipi interni del piano.
- `DenseStreamer+Pipeline.swift`: read-ahead e consegna ordinata degli slot.
- `DenseStreamer+Q4Cache.swift`: requant Q8_0 -> Q4_K e cache persistente.
- `DenseStreamer+CompressorQ8.swift`: conversione opzionale dei compressor F16.

## Flusso

All'avvio viene costruito un piano di regioni allineate. Durante il decode,
mentre la GPU elabora il layer corrente, il ring legge i layer futuri con
`pread/F_NOCACHE`; i pesi residenti Q4 o compressor vengono esclusi dallo stream.
Con `DS4_LAZY_IDX`, anche le proiezioni di scoring dell'indexer restano fuori
dal piano: vengono lette una sola volta in buffer residenti quando il contesto
effettivamente usato raggiunge la soglia sparse. Al consumo, le subview dello
slot e gli eventuali buffer residenti popolano un `LayerWeights` temporaneo.

## Regole di modifica

Uno slot non può essere sovrascritto finché la GPU lo usa. Mantenere limitato il
numero di richieste in-flight e gestire tutti i worker nei percorsi di errore.
Separare ottimizzazioni senza perdita da quantizzazioni con perdita e invalidare la cache
quando cambiano modello, formato o parametri di conversione.
