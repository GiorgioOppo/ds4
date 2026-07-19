[English](README.md) | **Italiano**

# Test del benchmark di accuratezza

Questi test validano il contratto di aggregazione puro usato dal benchmark di
correttezza next-token. Coprono l'input vuoto, esecuzioni completamente
corrette e completamente errate, rapporti misti noti, l'accuratezza cumulativa
rispetto a quella dei bucket locali e un bucket finale più corto della
dimensione di bucket configurata. I casi top-k verificano inoltre che top-1,
top-2 e top-3 restino annidati a livello di osservazione, bucket e risultato;
l'ordine dei candidati, la de-duplicazione e il troncamento a tre; liste di
candidati corte e vuote; e l'esatta compatibilità degli alias legacy top-1.

I test del piano di sampling coprono il determinismo del seed, la variazione
tra seed, inizi di target distinti, limiti di contesto uniformi, inizi di
segmento non negativi, limiti di valutazione per pezzo, la preferenza per
pezzi a lunghezza piena e il fallback a una coda corta del corpus solo quando
necessario. Proteggono inoltre i clamp corpus/KV, le richieste eccessive di
pezzi e la normalizzazione di un intervallo di contesto invertito.

Gli indici di piano/risultato dei pezzi sono zero-based e devono coprire
`0..<pieces.count`; l'aggiunta di uno è solo presentazione.

Il caso limite dell'esecuzione massimale è `N` token di corpus che producono
`N - 1` osservazioni quando il prefisso non valutato contiene un token. Con un
prefisso di `C` token ci sono `N - C` target eleggibili prima del troncamento.
Mantieni un test esplicito di off-by-one ogni volta che il loop di scoring o
il modello dei risultati cambia.

Nessun test in questa directory carica un GGUF, accede alla rete o richiede un
device Metal. Le misure di qualità end-to-end dipendono da un modello reale e
da un corpus fisso e vengono eseguite dal pannello Benchmark; questi unit test
proteggono solo la logica deterministica di aggregazione dei risultati e dei
grafici.

L'overload legacy a pezzo singolo resta parte del contratto di compatibilità;
il pianificatore di sampling puro può essere testato senza caricare un GGUF né
un device Metal.

I tre valori classificati sono candidati next-token dal vocabolario, non
esperti MoE. Un token di riferimento trovato al rango uno è necessariamente
corretto anche a top-2 e top-3; un hit di rango due contribuisce solo a
top-2/top-3, e un hit di rango tre solo a top-3. Ogni bucket parziale finale
deve dividere il conteggio di ciascun rango per il proprio numero effettivo di
osservazioni valutate.
