**English** | [Italiano](FLUSSO-INFERENZA.it.md)

# Local inference flow

## 1. Initialization

`InferenceService.init` opens the GGUF, builds the tokenizer and Metal
runtime, validates the model profile and creates the `StreamingDecoder`.
`DS4_*` variables are read before loading; changing a knob after the service
has been created does not reconfigure the existing decoder.

## 2. Conversation preparation

`ChatRenderer` produces the suffix for the new turn. `committedIds` describes
exactly the tokens already present in the KV: an append-only conversation
prefills only the new suffix. If the prefix is no longer trustworthy
(`kvDirty`), the service first rebuilds the KV from the confirmed tokens.

## 3. Restore and prefill

When the on-disk cache is enabled, `DiskKVStore` looks for the longest
compatible prefix. The restore happens one layer at a time. Tokens not covered
by the checkpoint are sent to the decoder in chunks; `.progress` events make
the progress visible.

## 4. Generation

The decoder returns the logits, `Sampler` selects the next token and the
tokenizer converts it to bytes. The service separates text, reasoning and tool
markup by emitting `GenEvent`. The context advances only with actually
accepted tokens.

## 5. Tools and completion

A complete tool call suspends the response with `.toolCall`. After execution,
`provideToolResults` appends the result to the context and reopens the
assistant. A clean generation saves the expert profile and, if configured, a
KV checkpoint; cancellations or errors instead mark the cache as dirty.

## Invariants

- `committedIds.count` must match the valid KV frontier.
- The decoder is used exclusively by the actor's serial executor.
- Reasoning and tool markup must be preserved in the context even when they
  are not shown as normal text.
- Benchmarks and sub-agents explicitly restore or invalidate the state of the
  main conversation.

See also [`Service`](Service/README.md), [`Persistence/KV`](../Persistence/KV/README.md)
and [`Tools`](../Tools/README.md).
