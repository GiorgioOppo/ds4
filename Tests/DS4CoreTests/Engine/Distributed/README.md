# Distributed Engine Tests

`DistProtocolTests.swift` covers message encoding/decoding, framing, handshake,
file/KV/work messages, expert assignments, and malformed-input rejection.

Protocol additions require round-trip, boundary, and invalid-frame tests.
Avoid live multi-host assumptions; transport integration should use loopback or
injected streams with deterministic timeouts.

