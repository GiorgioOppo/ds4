# Protocol/Work

Definisce i dati della pipeline orizzontale.

## Tipi

- `DistWork`: sessione, posizione, token, slice, flag, route, return endpoint e
  stati hidden quantizzati.
- `DistResult`: sessione, tipo del risultato, precisione e valori restituiti.

## Flusso e dipendenze

Il coordinator crea un lavoro per chunk; i worker lo decodificano, verificano
che shape e slice coincidano con l'assegnazione, eseguono la propria parte e lo
inoltrano. `ActivationCodec` in [`Codec`](../Codec/README.md) compatta i buffer.

## Estensione

Conservare l'ID sessione in ogni risposta, validare `nTokens × hcStateCount`
prima di allocare o copiare e limitare la route. Un nuovo flag non deve cambiare
silenziosamente il significato dei bit esistenti.
