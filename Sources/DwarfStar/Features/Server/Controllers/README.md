**English** | [Italiano](README.it.md)

# Server Controllers

`ServerController.swift` owns user-facing host, port, API-key, CORS, start/stop,
and status state. It creates `LocalServer` only after the shared Chat engine is
ready and maps server callbacks into main-actor UI updates.

Stopping is asynchronous at the service boundary. Keep the server
`EngineActivityGate` lease until `LocalServer.stop()` has drained accepted
requests and engine background work; only then transition the UI out of its
stopping state and release the lease.

Keep listener and endpoint implementation in `Services`, `Networking`, and
`API`. Never load a model from this controller.
