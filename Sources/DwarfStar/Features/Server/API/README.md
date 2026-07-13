# Server API Adapters

Endpoint-specific adapters translate HTTP JSON into engine requests and stream
engine output back in the requested wire format.

- `ChatRequestParser.swift`: shared message and generation-parameter parsing.
- `LocalServer+OpenAIChat.swift`: `/v1/chat/completions`.
- `LocalServer+Responses.swift`: `/v1/responses`.
- `LocalServer+LegacyCompletions.swift`: `/v1/completions`.
- `LocalServer+Anthropic.swift`: `/v1/messages`.

Adapters may depend on `LocalServer`, `DS4Core`, and `DS4Engine`; they must not
open sockets directly. Keep compatibility defaults explicit, report the actual
loaded model, and route all generation through the request gate.

