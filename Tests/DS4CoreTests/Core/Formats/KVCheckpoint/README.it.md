[English](README.md) | **Italiano**

# Test dei checkpoint KV

`KVCFileTests.swift` copre gli header dei checkpoint, i metadati, i payload
dei tensori, i round-trip e la validazione per `KVCFile`.

Usa file temporanei di proprietà del test e rimuovili nel teardown. Le
modifiche al layout serializzato richiedono sia un nuovo test di round-trip
sia un test di compatibilità, o di chiaro rifiuto, per la rappresentazione
precedente.
