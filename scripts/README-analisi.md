# Analisi di compressione del GGUF

Due strumenti Python per ragionare *sui numeri* prima di investire in nuovi
kernel o fine-tuning (vedi la discussione su quantizzazione / rappresentazione
"a grafo" coi pesi sugli archi).

```sh
pip install -U gguf numpy        # gguf di llama.cpp = dequantizzazione corretta di tutti i formati
```

## 1. `gguf_spectrum.py` — quanto è comprimibile?

Per i tensori principali calcola lo **spettro dei valori singolari** e riporta il
rank effettivo (90/95/99% dell'energia) e il guadagno se fattorizzati low-rank.
Per gli **esperti** stima la **ridondanza fra i 256** (PCA fra esperti) → quante
"basi" condivise basterebbero.

```sh
python3 scripts/gguf_spectrum.py model.gguf                      # layer 2, dense
python3 scripts/gguf_spectrum.py model.gguf --layers 0,2,20 --experts
python3 scripts/gguf_spectrum.py model.gguf --full              # +output/embeddings (lento)
```

Legge **misura soltanto**, non modifica nulla. Serve a decidere *dove* conviene
fattorizzare.

## 2. `gguf_to_graph.py` — i pesi dai nodi agli archi

Trasforma le matrici in un **grafo fattorizzato**: ogni `W[out×in]` diventa un
cammino `in ──B[r×in]──► (bottleneck r) ──A[out×r]──► out` (SVD troncata
`W≈A·B`). I **parametri vivono sugli archi** A, B; i nodi sono spazi di
attivazione senza parametri. Emette il grafo (JSON / Graphviz DOT), opzionalmente
i fattori (`.npz`), e il riepilogo compressione + **errore di ricostruzione**.

```sh
python3 scripts/gguf_to_graph.py model.gguf --layers 2 --energy 0.95 --dot graph.dot
dot -Tsvg graph.dot -o graph.svg                                # visualizza il grafo
python3 scripts/gguf_to_graph.py model.gguf --layers 0,2 --experts --rank 64 --npz factors.npz
```

⚠️ La fattorizzazione è **lossy** (lo script stampa l'errore relativo per matrice):
per recuperare la qualità servirebbe un fine-tuning. Lo strumento costruisce e
misura il grafo, non produce un modello pronto all'uso.

## Nota sui formati

L'engine Swift esegue solo: esperti **Q4_K / Q2_K / IQ2_XXS**, densi **Q8_0**,
più F16/F32. Gli script dequantizzano qualunque formato (via `gguf.quants`), ma
per ANALIZZARE gli esperti col massimo dettaglio conviene la GGUF **q4**
(esperti Q4_K) — il modello a 2 bit è già al limite dei formati supportati.
