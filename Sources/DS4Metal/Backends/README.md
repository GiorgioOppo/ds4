**English** | [Italiano](README.it.md)

# Backends

Metal implementations specific to each model family. Runtime, tensors and
shared GPU primitives stay in the top-level folders of `DS4Metal`;
architecture, GGUF schema, weights, KV state, prefill and decode belong to
the respective backend instead.

## Backends

- [`Common/`](Common/README.md): rules of the shared boundary and future
  token/chunk-level selection APIs.
- [`DeepSeekV4/`](DeepSeekV4/README.md): the currently operational backend.
- [`GLM52/`](GLM52/README.md): schema, DSA references and Metal primitives
  under construction; it does not yet contain a runnable decoder.
- [`Qwen/`](Qwen/README.md): reserved space; Qwen is not supported yet.

## Modification rules

The family is chosen at model load time, never inside the per-layer or
per-kernel loop. Do not build universal containers full of optional fields:
each backend keeps weights, scratch and snapshots with concrete types.
Promote a function into the common layer only when semantics, layout and
synchronization constraints genuinely coincide across multiple backends.
