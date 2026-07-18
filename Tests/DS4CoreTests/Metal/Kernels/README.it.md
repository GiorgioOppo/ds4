# Test dei kernel Metal

Test di parità mirati per i kernel incorporati da `DS4Metal`. Ogni test
dovrebbe costruire il più piccolo buffer utile, eseguire una singola
primitiva e confrontare il suo output con un riferimento fedele su CPU.

Le directory figlie raggruppano le operazioni di attenzione, compressione,
dense, MoE e tensoriali generiche. Dichiara esplicitamente le tolleranze e la
precisione di accumulo attesa. Usa il runtime incorporato condiviso invece di
un percorso dei kernel specifico dello sviluppatore.
