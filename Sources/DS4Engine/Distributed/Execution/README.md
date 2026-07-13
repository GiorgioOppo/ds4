# Distributed/Execution

Adatta `DS4Metal` alle unità di lavoro distribuite senza includere networking.

## Componenti

- `DistEngine.swift`: embedding, forward di slice singole o batch, output head,
  tokenizer, sampling, KV shard e percorso verticale.
- `ExpertShard.swift`: `ExpertShardEngine`, che carica una maschera di esperti e
  restituisce la somma parziale richiesta da `DistExpertWork`.

## Dipendenze e flusso

Dipende da `DS4Core`, `DS4Metal` e dai tipi in [`Protocol`](../Protocol/README.md).
Coordinator e worker trasformano i frame in chiamate a questi motori; nessun
tipo di questa cartella apre socket.

## Estensione

Conservare qui la semantica numerica dell'esecuzione distribuita. Validare shape,
layer, quantizzazione e maschere prima del dispatch GPU. Nuove strategie di
trasporto appartengono a [`Transport`](../Transport/README.md).
