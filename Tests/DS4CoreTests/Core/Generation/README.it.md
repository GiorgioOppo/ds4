[English](README.md) | **Italiano**

# Test di generazione

`SamplerTests.swift` convalida i controlli di sampling greedy e stocastico,
la temperatura, il filtraggio top-k/top-p/min-p e il comportamento della
penalità di ripetizione. Verifica inoltre in modo incrociato il percorso
full-vocabulary a raccolta con soglia di DS4_FAST_SAMPLER rispetto alla build
storica completa (stesso token, stesso stream RNG) su una griglia di parametri
e forme di logit.

Assegna un seed ai test randomizzati così che i fallimenti siano
riproducibili. Testa esplicitamente la composizione dei filtri e le
distribuzioni degenerate; non dipendere dalla fortuna statistica.
