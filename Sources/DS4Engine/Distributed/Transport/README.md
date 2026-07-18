# Distributed/Transport

Encapsulates the asynchronous TCP transport based on Network.framework.

## Components

- `DistError`: frame, network, version and transfer errors.
- `DistConnection`: framed connection with timeouts and exact reads.
- `DistRouteEntry`: address and slice of one hop of the route.
- `DistReturnListener`: coordinator listener for the terminal results.

## Flow and dependencies

`DistConnection` only adds/removes the header defined in
[`Protocol/Framing`](../Protocol/Framing/README.md); the payload semantics
stay in the protocol codecs. Coordinator and worker own the connection
lifecycle.

## Extension

Keep connect, send and receive cancellable; enforce timeouts and exact-length
reads. TLS, authentication or an alternative transport must preserve the
framed interface and have an explicit configuration.
