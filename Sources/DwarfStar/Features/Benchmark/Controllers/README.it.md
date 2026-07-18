# Controller dei benchmark

`BenchController.swift` possiede il tipo di benchmark, gli input, il ciclo di
vita, i risultati e lo stato di errore sul main actor. **Speed** dispaccia
verso l'`InferenceService` locale condiviso oppure verso il coordinatore
distribuito connesso. **Correctness** chiama
`InferenceService.accuracyBenchmark` solo sull'engine locale condiviso; una
selezione Distribuito + Correctness è visibile ma non può essere avviata.

Il controller è l'unico punto in cui una run di benchmark della UI può mutare
lo stato KV dell'engine. Mantieni intatto il gate di chat inattiva, pubblica
valori `BenchRow` pronti per la view e non caricare mai una copia privata del
modello da questo livello.

Le osservazioni di accuratezza e il risultato finale attraversano il confine
del task detached tramite continuation tipizzate di `AsyncStream`. I loro task
consumatori ereditano `MainActor` e sono l'unico codice che muta
`accuracyObservations` e `accuracyResult`. Mantieni questo confine quando
aggiungi funzionalità di avanzamento, esportazione o confronto.

Ogni osservazione porta i tre candidati ordinati per il token successivo.
L'aggregazione pubblica conteggi e accuratezze annidati top-1, top-2 e top-3
sia a livello globale sia per ogni brano campionato. Il valore globale è
derivato dai conteggi totali corretti/valutati, non da una media non pesata
delle percentuali dei brani. I nomi legacy `correctTokens`, `accuracy`,
`correct`, `localAccuracy` e `cumulativeAccuracy` restano alias top-1, così i
chiamanti scritti prima dell'estensione top-k conservano il loro significato.

Gli input di correttezza includono un seed riproducibile, il numero di brani,
l'intervallo di lunghezza del contesto e il massimo di token valutati per
brano. Il piano li limita alla capacità del corpus e della KV. I suoi punti di
partenza target distinti prevengono campioni duplicati, ma i brani possono
sovrapporsi; il controller deve presentare il piano/risultato effettivo invece
di assumere che ogni brano richiesto abbia la lunghezza massima.

Le run grandi mantengono contatori live esatti ma conservano solo un insieme
decimato di osservazioni per il rendering. Quando necessario il controller alza
anche la dimensione effettiva dei bucket del grafico, così sia i grafici live
sia quelli finali restano vicini a 4,096 punti; questo influisce solo sulla
densità di visualizzazione, mai sui totali di correttezza.

Il corpus di correttezza predefinito è italiano. Il backend può limitare o
troncare il numero di brani, l'intervallo di contesto e il limite di token
valutati al corpus caricato e al contesto; il risultato finale è autorevole e
guida tutti i KPI e i grafici.
