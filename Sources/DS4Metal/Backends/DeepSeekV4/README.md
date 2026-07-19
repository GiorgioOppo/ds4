**English** | [Italiano](README.it.md)

# DeepSeek-V4 backend

Metal implementation for DeepSeek V4 Flash and Pro. The validated GGUF
produces an instance-based runtime geometry, so decoder, scratch, KV, weights
and router use the profile's dimensions without dynamic dispatch in the
per-layer loop. The local runtime supports the single-file Pro Q2; Pro Q4
split loading and Pro distribution are not yet declared operational.

## Structure

- [`Architecture/`](Architecture/README.md): shapes, dimensions and RoPE.
- [`Model/`](Model/README.md): tensor schema and GGUF loading.
- [`Streaming/`](Streaming/README.md): staging of dense weights from SSD.
- [`Experts/`](Experts/README.md): expert bundle, cache and MetalIO.
- [`MTP/`](MTP/README.md): Multi-Token Prediction sidecar.
- [`Decode/`](Decode/README.md): KV/NSA state, prefill and generation.

## Architectural boundary

HyperConnections, MLA with latent KV, NSA compressors, DSA indexer and top-6
router are DeepSeek-V4 semantics. They must not be used as fallbacks for other
families: an incompatible backend must fail explicitly at load time.
The router supports 256 experts/scale 1.5 for Flash and 384 experts/scale 2.5
for Pro; the Pro bitonic network uses 512 lanes and masks padding 384...511.
