[English](README.md) | **Italiano**

# Test dei kernel di attention

Test per RoPE, compressione KV, scoring/pooling dell'indexer, selezione
sparsa, flash attention e kernel di output dell'attention.

Copri lunghezze di sequenza corte e non allineate ai blocchi, maschere,
confini del top-k e il confronto numerico con l'attention su CPU. Gli
ambienti senza GPU devono usare `XCTSkip`, mai ritornare in silenzio.
