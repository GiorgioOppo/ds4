# Conversation Tests

[`Backends/DeepSeekV4/DSML/`](Backends/DeepSeekV4/DSML/README.md) validates chat
rendering, DSML/tool markup, tool-call parsing, malformed input handling and
conversation-role behavior.

[`Backends/GLM52/`](Backends/GLM52/README.md) verifica ruoli GLM, reasoning,
tool XML tipizzati, parsing incrementale e contenimento dei marker.

Add regression cases for boundary tokens and invalid model output. Assertions
should cover the parsed structure, not only the final display string.
