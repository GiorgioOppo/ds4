[English](README.md) | **Italiano**

# Protocol/Serialization

`Data+LittleEndian.swift` contiene le primitive interne per aggiungere e leggere
interi little-endian con un cursore esplicito.

## Flusso e dipendenze

È lo strato più basso del protocollo e dipende solo da Foundation/DS4Core. Tutti
i codec di messaggio lo usano per ottenere lo stesso layout su ogni Mac.

## Estensione

Aggiungere solo primitive wire generiche. Ogni lettura deve avanzare il cursore
in modo prevedibile ed essere preceduta dal controllo dei byte disponibili nel
decoder chiamante; non inserire qui logica di uno specifico messaggio.
