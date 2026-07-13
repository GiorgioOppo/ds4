# Server Services

`LocalServer.swift` owns the `Network.framework` listener, shared engine
reference, server configuration, and common routing state. Focused extensions
in sibling `Networking/` and `API/` directories implement transport and
endpoint behavior.

The service is `@unchecked Sendable`; synchronization therefore must remain
explicit. Preserve listener shutdown, connection cleanup, API-key validation,
and single-engine serialization when changing shared state.

