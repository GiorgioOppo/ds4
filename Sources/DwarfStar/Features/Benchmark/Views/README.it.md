[English](README.md) | **Italiano**

# View dei benchmark

`BenchView.swift` renderizza il picker Speed/Correctness, la configurazione,
l'avanzamento e i risultati con Swift Charts. Osserva `BenchController` e non
contiene alcuna implementazione di inferenza.

Speed preserva il grafico del throughput di prefill/generazione. Correctness
fornisce un editor per testo incollato, controlli per seed, numero di brani,
intervallo di contesto e token per brano, schede KPI e un grafico per brano che
confronta top-1, top-2 e top-3. I valori di accuratezza provenienti dall'engine
sono frazioni in `0...1` e vengono convertiti in percentuali solo per la
presentazione. I KPI globali usano conteggi aggregati pesati per token. Un
grafico cumulativo mostra l'accuratezza aggregata progressiva, mentre il
grafico per brano tiene i campioni separati così la variabilità legata alla
posizione nel corpus resta visibile.

Gli indici dei brani nei risultati e nelle osservazioni dell'engine partono da
zero. Le etichette mostrate alle persone usano `index + 1`; non reimmettere
quel valore di visualizzazione nella ricerca dei risultati o nell'identità del
grafico.

Usa un colore e uno stile di linea distinti e stabili per ogni rank. La legenda
deve nominare tutti e tre i rank; il grafico non deve far pensare che questi
candidati siano esperti MoE.

Il testo esplicativo deve continuare a dire che l'accuratezza esatta sul token
successivo non è correttezza fattuale. Quando per Correctness è selezionato
Distribuito, mostra la limitazione e disabilita Start invece di cambiare engine
in modo silenzioso.

Mantieni qui la formattazione e la presentazione dei grafici. Nuove modalità o
metriche di benchmark devono prima essere rappresentate dallo stato del
controller, così l'esecuzione dei comandi resta testabile fuori dal corpo della
view.
