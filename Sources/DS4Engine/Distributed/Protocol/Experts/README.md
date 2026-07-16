# Protocol/Experts

Definisce i frame del parallelismo verticale MoE, estesi dal protocollo v11 per
la geometria Pro.

## Tipi

- `DistExpertAssign`: modello, maschera esperti a lunghezza prefissata, cache,
  usage e knob. La maschera è di 32 byte per Flash e 48 per Pro.
- `DistExpertWork`: sequenza, layer, esperti selezionati, pesi e attivazione.
- `DistExpertSum`: sequenza, layer e somma parziale dello shard.

## Flusso e dipendenze

Il coordinator partiziona gli esperti, assegna le maschere e invia una richiesta
per ogni layer instradato. [`ExpertShardEngine`](../../Execution/README.md)
calcola il contributo locale; il coordinator valida sequenza/layer e aggrega.

## Estensione

Le maschere devono essere disgiunte o avere una politica di aggregazione
esplicita. Lunghezza esatta, bit di padding, conteggi di ID/pesi, precisione e
dimensione dell'attivazione devono essere validati prima di eseguire il kernel.
