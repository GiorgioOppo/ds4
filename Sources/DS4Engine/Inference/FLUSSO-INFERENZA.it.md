[English](FLUSSO-INFERENZA.md) | **Italiano**

# Flusso dell'inferenza locale

## 1. Inizializzazione

`InferenceService.init` apre il GGUF, costruisce tokenizer e runtime Metal,
valida il profilo del modello e crea lo `StreamingDecoder`. Le variabili
`DS4_*` vengono lette prima del caricamento; cambiare un knob dopo la creazione
del servizio non riconfigura il decoder esistente.

## 2. Preparazione della conversazione

`ChatRenderer` produce il suffisso per il nuovo turno. `committedIds` descrive
esattamente i token già presenti nella KV: una conversazione append-only
prefilla soltanto il nuovo suffisso. Se il prefisso non è più affidabile
(`kvDirty`) il servizio ricostruisce prima la KV dai token confermati.

## 3. Ripristino e prefill

Quando la cache su disco è attiva, `DiskKVStore` cerca il prefisso più lungo
compatibile. Il ripristino avviene un layer alla volta. I token non coperti dal
checkpoint sono inviati al decoder in chunk; gli eventi `.progress` rendono
visibile l'avanzamento.

## 4. Generazione

Il decoder restituisce i logits, `Sampler` seleziona il token successivo e il
tokenizer lo converte in byte. Il servizio separa testo, reasoning e markup dei
tool emettendo `GenEvent`. Il contesto avanza solo con token realmente accettati.

## 5. Tool e conclusione

Una tool call completa sospende la risposta con `.toolCall`. Dopo l'esecuzione,
`provideToolResults` aggiunge al contesto il risultato e riapre l'assistente.
Una generazione pulita salva profilo esperti e, se configurato, checkpoint KV;
cancellazioni o errori marcano invece la cache come sporca.

## Invarianti

- `committedIds.count` deve coincidere con la frontiera KV valida.
- Il decoder è usato esclusivamente dall'executor seriale dell'actor.
- Reasoning e markup tool devono essere preservati nel contesto anche quando
  non sono mostrati come testo normale.
- Benchmark e sub-agent ripristinano o invalidano esplicitamente lo stato della
  conversazione principale.

Vedi anche [`Service`](Service/README.it.md), [`Persistence/KV`](../Persistence/KV/README.it.md)
e [`Tools`](../Tools/README.it.md).
