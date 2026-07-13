# Protocol/Experts

Definisce i frame del parallelismo verticale MoE introdotti dal protocollo v10.

## Tipi

- `DistExpertAssign`: modello, maschera esperti, cache, usage e knob.
- `DistExpertWork`: sequenza, layer, esperti selezionati, pesi e attivazione.
- `DistExpertSum`: sequenza, layer e somma parziale dello shard.

## Flusso e dipendenze

Il coordinator partiziona gli esperti, assegna le maschere e invia una richiesta
per ogni layer instradato. [`ExpertShardEngine`](../../Execution/README.md)
calcola il contributo locale; il coordinator valida sequenza/layer e aggrega.

## Estensione

Le maschere devono essere disgiunte o avere una politica di aggregazione
esplicita. Validare conteggi di ID/pesi, precisione e dimensione dell'attivazione
prima di eseguire il kernel.
