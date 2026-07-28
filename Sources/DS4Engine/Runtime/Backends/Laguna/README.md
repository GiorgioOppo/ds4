**English** | [Italiano](README.it.md)

# Laguna runtime registration

`LagunaBackendDefinition` is the registration record for Laguna S 2.1: the
frontend capabilities it advertises (chat, tools, reasoning, MoE) and the
runtime gate forwarded from `LagunaRuntimeGate` in DS4Metal, currently off.
While the gate is off, the backend selector refuses a Laguna GGUF with an
explicit not-implemented error instead of routing it to another decoder.
