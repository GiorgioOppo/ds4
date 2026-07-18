# Protocol/Experts

Definisce i frame del parallelismo MoE verticale, esteso dal protocollo v11
per la geometria Pro.

## Tipi

- `DistExpertAssign`: modello, maschera degli esperti a lunghezza fissa,
  cache, uso e manopole. La maschera è di 32 byte per Flash e 48 per Pro.
- `DistExpertWork`: sequenza, layer, esperti selezionati, pesi e attivazione.
- `DistExpertSum`: sequenza, layer e la somma parziale dello shard.

## Flusso e dipendenze

Il coordinator partiziona gli esperti, assegna le maschere e invia una
richiesta per ogni layer routed.
[`ExpertShardEngine`](../../Execution/README.md) calcola il contributo
locale; il coordinator valida sequenza/layer e aggrega.

## Estensione

Le maschere devono essere disgiunte o avere una politica di aggregazione
esplicita. Lunghezza esatta, bit di padding, conteggi di ID/pesi e precisione
e dimensione dell'attivazione devono essere validati prima di eseguire il
kernel.
