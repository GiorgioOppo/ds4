# Analisi della comprimibilita dei GGUF

Questa guida descrive i due strumenti esplorativi usati per misurare la
ridondanza dei pesi prima di investire in nuovi kernel, rappresentazioni
fattorizzate o fine-tuning. Entrambi leggono il GGUF senza modificarlo e non
producono un modello immediatamente eseguibile da DwarfStar.

Tornare all'[indice degli script](README.md).

## Requisiti

```sh
python3 -m pip install -U gguf numpy
```

Il pacchetto `gguf` fornisce la dequantizzazione dei formati supportati da
`gguf.quants`. I comandi seguenti presuppongono di essere eseguiti dalla radice
del repository.

## `gguf_spectrum.py`: spettro e ridondanza

Per i tensori selezionati lo script calcola lo spettro dei valori singolari e
riporta il rango effettivo al 90%, 95% e 99% dell'energia. Per gli esperti
routed stima inoltre, tramite proiezione e PCA, quante basi condivise servono a
spiegare la variabilita fra esperti.

```sh
python3 scripts/gguf_spectrum.py /percorso/modello.gguf
python3 scripts/gguf_spectrum.py /percorso/modello.gguf --layers 0,2,20 --experts
python3 scripts/gguf_spectrum.py /percorso/modello.gguf --full
python3 scripts/gguf_spectrum.py /percorso/modello.gguf --layers 2 --json risultati.json
```

`--full` include anche embedding e testa di output e puo richiedere molto piu
tempo e memoria. Le stime sugli esperti dipendono da `--proj`: sono uno
strumento decisionale, non una prova di equivalenza del modello.

## `gguf_to_graph.py`: fattorizzazione su grafo

Ogni matrice `W[out x in]` selezionata viene approssimata con una SVD troncata:

```text
input -- B[r x in] --> bottleneck r -- A[out x r] --> output
```

I parametri risiedono sugli archi `A` e `B`; i nodi descrivono soltanto gli
spazi di attivazione. Lo script puo produrre JSON, Graphviz DOT e, su richiesta,
un archivio NumPy con i fattori.

```sh
python3 scripts/gguf_to_graph.py /percorso/modello.gguf \
  --layers 2 --energy 0.95 --dot grafo.dot --json grafo.json

dot -Tsvg grafo.dot -o grafo.svg

python3 scripts/gguf_to_graph.py /percorso/modello.gguf \
  --layers 0,2 --experts --rank 64 --npz fattori.npz
```

La trasformazione e **lossy**. Il riepilogo riporta frazione di parametri
conservata ed errore relativo di ricostruzione; recuperare la qualita richiede
normalmente addestramento o fine-tuning e una successiva validazione end-to-end.

## Scelta del GGUF

Il motore gestisce tensori esperti in `Q4_K`, `Q2_K` e `IQ2_XXS`, tensori densi
in `Q8_0` e valori `F16`/`F32` dove necessari. Per un'analisi dettagliata degli
esperti conviene partire da un GGUF meno aggressivamente quantizzato: le
conclusioni tratte da un modello a 2 bit non si trasferiscono automaticamente
ad altre varianti.

## Interpretazione corretta

- Un rango numericamente basso non garantisce che la qualita linguistica resti
  invariata.
- I risultati dipendono dal layer, dal tensore, dalla quantizzazione e dai
  parametri di campionamento dell'analisi.
- Prima di implementare un nuovo formato, confrontare memoria, banda, errore
  sui logit e qualita su un insieme di prompt riproducibile.
- Questi script sono strumenti offline: non fanno parte del percorso di
  caricamento o inferenza della GUI e della demo.
