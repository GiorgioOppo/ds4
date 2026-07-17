# Server Concurrency

`RequestGate.swift` is a cancellation-aware FIFO mutex that serializes access
to the single shared inference engine. It uses a small explicit lock so gate
release is synchronous in each request's `defer`; shutdown never leaves an
untracked release task. The network layer fully receives and parses a request
before acquiring the gate, preventing slow clients from blocking generation.

Do not replace this with an unlocked task or acquire it while reading a request
body. Cancellation removes queued waiters and every error path must release the
active slot.
