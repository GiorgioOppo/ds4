[English](README.md) | **Italiano**

# Test dei kernel densi

`MetalDenseTests.swift` e `MetalMatmulMMTests.swift` validano i percorsi
matvec densi e matrice-matrice sulle forme e sui formati di storage
supportati.

Esercita le righe/colonne di coda e le varianti di scheduling. Il tuning delle
prestazioni non deve indebolire le asserzioni di parità; misura il throughput
in benchmark separati dalla correttezza.
