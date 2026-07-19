[English](README.md) | **Italiano**

# Protocol/Core

`Dist.swift` è la fonte autorevole del contratto globale: magic, versione wire,
dimensioni massime, tipi di frame, flag di lavoro e whitelist dei knob.

## Flusso e dipendenze

Tutti gli altri codec dipendono da `Dist`; coordinator, worker e trasporto ne
usano i valori senza ridefinirli. Dipende da Foundation e dalle utility di
`DS4Core`.

## Estensione

Assegnare valori numerici stabili ai nuovi `MsgType`, documentare la semantica
in [`../../PROTOCOLLO.md`](../../PROTOCOLLO.it.md) e incrementare
`protocolVersion` se un nodo precedente non può interpretare correttamente il
nuovo flusso. I limiti per collezioni provenienti dalla rete sono obbligatori.
