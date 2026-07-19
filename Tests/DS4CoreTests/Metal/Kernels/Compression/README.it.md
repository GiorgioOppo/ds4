[English](README.md) | **Italiano**

# Test dei kernel di compressione

Test per le proiezioni del compressore e le primitive split/reduce delle
Hyper-Connection: `MetalCompressorTests`, `MetalHCSplitTests` e
`MetalHyperConnectionsTests`.

Convalida sia le forme sia l'output numerico, inclusi i percorsi fusi rispetto
a quelli di riferimento dove disponibili. Mantieni i tensori di test
abbastanza piccoli per esecuzioni locali veloci.
