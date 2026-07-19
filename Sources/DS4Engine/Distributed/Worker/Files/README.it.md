[English](README.md) | **Italiano**

# Worker/Files

Riceve file grandi in modo incrementale e riprendibile.

## Componente e flusso

`DistWorker+Files.swift` gestisce il file `.part`, ricostruisce l'hash fino
all'offset concordato, verifica i checkpoint della catena, appende chunk
sequenziali e promuove il file nel `DistFileStore` soltanto dopo SHA-256 finale.
Una disconnessione sospende il file senza cancellare il prefisso valido.

## Dipendenze

Usa CryptoKit, [`Distributed/Files`](../../Files/README.it.md) e i messaggi in
[`Protocol/Files`](../../Protocol/Files/README.it.md).

## Estensione

Rifiutare offset non monotoni, nomi non sanitizzati, chunk fuori indice e
dimensioni eccedenti il manifest. Non caricare l'intero artefatto in RAM e non
registrare nel manifest un file non verificato.
