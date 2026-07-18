# DeepSeekV4/Decode/Generation

Operations that turn tokens into decoder input and final state into logits.

## Main files

- [`StreamingDecoder+Generation.swift`](StreamingDecoder+Generation.swift):
  embedding, normalization/output head and primitives used by the generation
  loop.

## Flow and dependencies

The token id selects the embedding row; the result enters the forward. At the
last layer, output norm and output matrix produce logits readable by the
[`Sampler`](../../../../../DS4Core/Generation/README.md). The loop and the stop
policies are orchestrated by `DS4Engine`.

## Modification rules

Keep vocabulary size, dtype and output scale consistent. Avoid intermediate
readbacks: only the necessary logits should cross the GPU/CPU boundary.
