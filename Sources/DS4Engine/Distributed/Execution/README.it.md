[English](README.md) | **Italiano**

# Distributed/Execution

## Confine architetturale

`DistEngine` ed `ExpertShard` restano fisicamente in questa cartella per non
rompere le API pubbliche e il protocollo distribuito v11, ma l'implementazione è
DeepSeek-V4-specifica. La geometria non è fissa: Flash usa 43 layer e 256
esperti, Pro 61 layer e 384 esperti. Entrambi passano ora da
`RuntimeBackendFactory` prima di costruire tokenizer o decoder: una famiglia
Qwen riconosciuta viene rifiutata come backend non ancora implementato.

Non spostare semplicemente questi tipi sotto un backend Qwen: prima occorre una
nuova capability distribuita e un handshake che includa architettura, fingerprint
del modello e geometria delle attivazioni. Ogni modifica wire incompatibile
richiede un nuovo protocollo; non viene negoziata fra build diverse.

Il GGUF Pro Q2 a file singolo usa questi percorsi. Il package Pro Q4 a due file
resta download-only: i nomi `Layers00-30` e `Layers31-output` non sostituiscono
un loader capace di assemblare e validare più shard.

Adatta `DS4Metal` alle unità di lavoro distribuite senza includere networking.

## Componenti

- `DistEngine.swift`: embedding, forward di slice singole o batch, output head,
  tokenizer, sampling, KV shard e percorso verticale.
- `ExpertShard.swift`: `ExpertShardEngine`, che carica una maschera di esperti e
  restituisce la somma parziale richiesta da `DistExpertWork`.

## Dipendenze e flusso

Dipende da `DS4Core`, `DS4Metal` e dai tipi in [`Protocol`](../Protocol/README.it.md).
Coordinator e worker trasformano i frame in chiamate a questi motori; nessun
tipo di questa cartella apre socket.

`DistEngine.chatPromptIds` applica la stessa neutralizzazione dei token di
controllo usata dall'inferenza locale a turni, storico e schemi tool, poi lascia
al renderer l'aggiunta esclusiva dei delimitatori strutturali fidati.

## Estensione

Conservare qui la semantica numerica dell'esecuzione distribuita. Validare shape,
layer, quantizzazione e maschere prima del dispatch GPU. Nuove strategie di
trasporto appartengono a [`Transport`](../Transport/README.it.md).
