# Engine Tests

Tests for orchestration and integration services in `DS4Engine`: diagnostics,
distributed protocol, model management, persistence, projects, and tools.

Prefer injected transports, temporary directories, and local fixtures. Unit
tests must not require credentials, internet access, the user's projects, or a
loaded production model.

