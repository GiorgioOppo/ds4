[English](README.md) | **Italiano**

# Test Core — Encoder di quantizzazione

Fissaggio byte-esatto degli encoder per il writer GGUF di
`Formats/Quantization`.

- `QuantEncodeTests.swift` confronta ogni formato implementato (q8_0 con la
  matematica dell'offset di riga, q8_K, q4_K e q2_K nelle varianti reference
  e pesata imatrix, iq2_xxs) contro i byte delle fixture.
- `QuantEncodeFixtures.swift` è GENERATO: gli input sono un blocco di casi
  limite (zeri, costanti, spike, segni alternati) seguito da uno stream
  xorshift32(0x12345678), e i byte attesi vengono dal riferimento
  `gguf-tools/quants.c` di ds4 compilato con `clang -O2 -ffp-contract=off`
  (operazioni float32 semplici su entrambi i lati). Rigenerare con
  `scripts/quant-fixtures/fixture_gen.c`, non modificare a mano.

Un diff di byte qui significa che il port Swift è divergente dal riferimento
C — arrotondamento, ordine di ricerca o packing — non una questione di
tolleranza.
