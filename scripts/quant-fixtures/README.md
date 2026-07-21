**English** | [Italiano](README.it.md)

# Quantization-encoder fixture generator

`fixture_gen.c` regenerates
`Tests/DS4CoreTests/Core/Quantization/QuantEncodeFixtures.swift`: it encodes
a deterministic input stream (crafted edge blocks + xorshift32(0x12345678))
with the ds4 C reference quantizers and emits the expected bytes the Swift
`QuantEncode` port is pinned against.

```sh
git clone --depth 1 https://github.com/antirez/ds4.git /tmp/ds4
clang -O2 -ffp-contract=off -I /tmp/ds4/gguf-tools \
  scripts/quant-fixtures/fixture_gen.c /tmp/ds4/gguf-tools/quants.c \
  -o /tmp/fixture_gen -lm
(cd /tmp && ./fixture_gen)
cp /tmp/QuantEncodeFixtures.swift Tests/DS4CoreTests/Core/Quantization/
```

`-ffp-contract=off` matters: Swift does not fuse multiply-adds, so the C
reference must not either, or low-bit rounding diverges and the byte
comparison becomes meaningless.
