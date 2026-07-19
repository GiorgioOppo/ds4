**English** | [Italiano](README.it.md)

# Protocol/Handshake

Defines the messages that turn an idle worker into an assigned node.

## Types

- `DistHello`: version, current state, model and slice already loaded.
- `DistAssign`: model, context, slice, output head, KV budget, expert cache,
  sidecar/Q4, usage profile and whitelisted performance knobs.

## Flow and dependencies

The worker sends `HELLO`; the coordinator checks compatibility and, after the
files, sends `ASSIGN`. The `READY` response reuses the `DistHello` payload.
Files are negotiated by the types in [`Files`](../Files/README.md).

## Extension

Every field must have a default or an explicit incompatibility. Received
environment variables must also be filtered by the worker using the whitelist
in [`Core`](../Core/README.md); do not carry secrets or arbitrary settings.
