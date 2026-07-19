[English](README.md) | **Italiano**

# Test di persistenza del motore

`DiskKVStoreTests.swift` valida lookup KV, serializzazione, indicizzazione,
streaming, invalidazione e comportamento del budget.

Usa una directory temporanea nuova per ogni test e puliscila. Includi casi
corrotti, parziali, incompatibili e di eviction quando cambiano la
rappresentazione su disco o la politica dell'indice.
