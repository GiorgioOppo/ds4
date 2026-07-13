# GGUF Format Tests

`GGUFTests.swift` exercises GGUF header, metadata, arrays, tensor descriptors,
alignment, and rejection of invalid or truncated data using compact fixtures.

Keep parsing tests independent from Metal weight loading. Add an explicit
fixture case whenever supporting a new GGUF metadata type or validation rule.

