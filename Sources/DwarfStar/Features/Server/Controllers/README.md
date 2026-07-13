# Server Controllers

`ServerController.swift` owns user-facing host, port, API-key, CORS, start/stop,
and status state. It creates `LocalServer` only after the shared Chat engine is
ready and maps server callbacks into main-actor UI updates.

Keep listener and endpoint implementation in `Services`, `Networking`, and
`API`. Never load a model from this controller.

