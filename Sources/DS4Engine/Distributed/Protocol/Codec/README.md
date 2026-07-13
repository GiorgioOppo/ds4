# Protocol/Codec

`ActivationCodec.swift` converte vettori `Float` in payload a 32, 16 o 8 bit e
li ricostruisce alla ricezione. È il percorso caldo del traffico di inferenza.

## Flusso e dipendenze

I messaggi [`Work`](../Work/README.md) e [`Experts`](../Experts/README.md) usano
il codec per ridurre banda e copie. Il formato a 8 bit include i dati necessari
alla dequantizzazione; il decoder riceve sempre il conteggio atteso.

## Estensione

Ottimizzare con copie bulk e buffer contigui, mantenendo round-trip e controllo
della lunghezza. Una nuova precisione cambia il formato wire e richiede test
numerici con tolleranza dichiarata oltre al bump di protocollo.
