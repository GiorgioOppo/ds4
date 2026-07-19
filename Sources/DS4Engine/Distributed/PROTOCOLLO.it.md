# Protocollo distribuito v11

## Frame

Ogni frame contiene `DistFrameHeader` seguito da un payload: magic `DS4D`, tipo
di messaggio e lunghezza, tutti little-endian. Il decoder rifiuta magic errato,
tipi sconosciuti nel percorso attivo, payload troncati e conteggi oltre i limiti.

## Connessione e assegnazione

```text
coordinator                 worker
     | ------- TCP ----------> |
     | <------ HELLO ---------- |
     | ------ FILE_OFFER -----> |
     | <----- FILE_NEED ------- |
     | -- FILE_CHUNK/DONE ----> |
     | <------ FILE_ACK -------- |
     | ------- ASSIGN --------> |
     | <------- READY ---------- |
```

`HELLO` identifica versione e assegnazione corrente. `FILE_NEED` include offset
di ripresa validati tramite una catena SHA-256 a blocchi; il file completo viene
promosso solo dopo verifica dell'hash finale. `ASSIGN` definisce modello,
contesto, slice, cache, sidecar e knob prestazionali consentiti.

## Pipeline orizzontale

Il coordinator crea una sessione per turno, produce lo stato hidden e invia un
`DistWork`. Il messaggio contiene posizione assoluta, token del chunk, slice,
route e bit delle attivazioni. Ogni worker applica i propri layer e inoltra lo
stato; il nodo terminale restituisce hidden state o logits al return listener.
Un risultato con sessione obsoleta viene scartato.

## Parallelismo verticale

`EXPERT_ASSIGN` distribuisce maschere disgiunte di esperti. Il payload codifica
prima la lunghezza `UInt32` e poi i byte della maschera: 32 byte per i 256
esperti Flash, 48 byte per i 384 esperti Pro. Lunghezza, bit di padding e
copertura vengono validati rispetto alla geometria del GGUF. Per ciascun layer
MoE il coordinator invia `EXPERT_WORK` con ID, pesi e attivazione; ogni shard
risponde con `EXPERT_SUM`. Le somme parziali validate vengono aggregate nel
backbone locale.

Le assegnazioni orizzontali sono validate contro il numero di layer realmente
ispezionato: 43 per Flash o 61 per Pro. Un worker inattivo annuncia zero layer
fino al completamento di `ASSIGN`; `READY` deve poi riportare la geometria
caricata.

## Continuità KV

`KV_QUERY`/`KV_LENGTHS` cercano lunghezze comuni a tutti gli shard;
`KV_RESTORE` ripristina esattamente il prefisso concordato;
`KV_SAVE`/`KV_ACK` salvano il turno pulito. Se anche un solo worker non può
ripristinare la lunghezza scelta, il coordinator torna al prefill freddo.

## Compatibilità

Il protocollo non negozia feature fra versioni diverse: un mismatch interrompe
il setup. Ogni nuovo campo deve avere limiti espliciti, codifica deterministica,
decode che fallisce in modo atomico e test round-trip più casi troncati.

Vedi [`Protocol`](Protocol/README.md), [`Coordinator`](Coordinator/README.md) e
[`Worker`](Worker/README.md).
