# Test del runtime Metal

`MetalRuntimeTests.swift` verifica la scoperta del device, la compilazione
della libreria di kernel incorporata e la creazione delle pipeline.

Salta con una motivazione esplicita se l'host non ha un device Metal. Il
mancato successo nella compilazione dei kernel incorporati su un host con
capacità Metal è un fallimento del test, non uno skip.
