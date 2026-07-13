# Server Concurrency

`RequestGate.swift` is the actor that serializes access to the single shared
inference engine. The network layer fully receives and parses a request before
acquiring the gate, preventing slow clients from blocking generation.

Do not replace this with an unlocked task or acquire it while reading a request
body. Cancellation and every error path must release the active slot.

