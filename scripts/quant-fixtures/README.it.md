[English](README.md) | **Italiano**

# Generatore di fixture per gli encoder di quantizzazione

`fixture_gen.c` rigenera
`Tests/DS4CoreTests/Core/Quantization/QuantEncodeFixtures.swift`: codifica
uno stream di input deterministico (blocchi limite costruiti +
xorshift32(0x12345678)) con i quantizzatori C di riferimento di ds4 ed
emette i byte attesi su cui è fissato il port Swift `QuantEncode`.

```sh
git clone --depth 1 https://github.com/antirez/ds4.git /tmp/ds4
clang -O2 -ffp-contract=off -I /tmp/ds4/gguf-tools \
  scripts/quant-fixtures/fixture_gen.c /tmp/ds4/gguf-tools/quants.c \
  -o /tmp/fixture_gen -lm
(cd /tmp && ./fixture_gen)
cp /tmp/QuantEncodeFixtures.swift Tests/DS4CoreTests/Core/Quantization/
```

`-ffp-contract=off` è essenziale: Swift non fonde le multiply-add, quindi
nemmeno il riferimento C deve farlo, o gli arrotondamenti nei bit bassi
divergono e il confronto byte-per-byte perde significato.
