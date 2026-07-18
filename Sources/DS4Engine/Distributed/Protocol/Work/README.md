# Protocol/Work

Defines the data of the horizontal pipeline.

## Types

- `DistWork`: session, position, tokens, slice, flags, route, return endpoint
  and quantized hidden states.
- `DistResult`: session, result type, precision and returned values.

## Flow and dependencies

The coordinator creates one work item per chunk; workers decode it, verify
that shape and slice match their assignment, execute their own part and
forward it. `ActivationCodec` in [`Codec`](../Codec/README.md) compacts the
buffers.

## Extension

Preserve the session ID in every response, validate `nTokens × hcStateCount`
before allocating or copying, and bound the route. A new flag must not
silently change the meaning of existing bits.
