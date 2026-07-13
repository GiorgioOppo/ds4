# Server Networking

This directory implements protocol-independent HTTP transport details:

- `HTTPRequest.swift` is the parsed request value.
- `LocalServer+Networking.swift` accepts connections and reads bounded requests.
- `LocalServer+JSON.swift` performs JSON encoding helpers.
- `LocalServer+HTTPResponses.swift` writes status lines, headers, bodies, and
  streaming frames.

Networking feeds parsed requests to the API adapters. Preserve the body-size
and read-time limits, avoid acquiring the inference gate during network reads,
and keep all responses valid HTTP even on parsing or cancellation failures.

