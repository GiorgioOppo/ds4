**English** | [Italiano](README.it.md)

# Laguna engine gate

`LagunaRuntimeGate.enabled` is the single Laguna S 2.1 enablement switch:
backend selection, catalog availability and demo dispatch all key off this one
constant. It stays `false` until the Metal decoder from the reference
`laguna-s2.1` branch is ported and the end-to-end logits-parity gate passes on
real weights on hardware.
