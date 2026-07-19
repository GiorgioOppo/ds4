**English** | [Italiano](README.it.md)

# Server Services

`LocalServer.swift` owns the `Network.framework` listener, shared engine
reference, server configuration, and common routing state. Focused extensions
in sibling `Networking/` and `API/` directories implement transport and
endpoint behavior.

The service is `@unchecked Sendable`; synchronization therefore must remain
explicit. Listener state and every accepted connection/request task are owned
under one lifecycle lock. `stop()` is an async barrier: it closes the listener,
cancels and awaits the accepted requests, then quiesces engine GPU/I/O work.
Preserve that ordering, API-key validation, and single-engine serialization
when changing shared state.
