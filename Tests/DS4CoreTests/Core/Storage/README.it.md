[English](README.md) | **Italiano**

# Test dello Storage di Core

`SSDCachePlanTests.swift` verifica la pianificazione deterministica di
storage/cache, i budget, l'allineamento e il comportamento ai confini senza
eseguire I/O di dimensioni di produzione.

Usa dimensioni sintetiche e rendi esplicite le assunzioni sulle unità in byte.
I benchmark di performance appartengono alla diagnostica, non a questi unit
test.
