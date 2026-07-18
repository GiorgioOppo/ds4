# Tokenization/API

The minimal contract shared by the tokenizers of the different backends.

`TokenizerProtocol` exposes text tokenization, tokenization of already
rendered prompts and decoding of a token. It does not impose IDs for roles,
reasoning or tools: those delimiters belong to the conversation backend and
may not be atomic tokens in every family.

`TokenizerFactory` selects exclusively from the detected architecture:

- `deepseek4` → `DeepSeekV4Tokenizer`;
- `glm-dsa` → `GLM52Tokenizer`;
- recognized Qwen → explicit `tokenizerNotImplemented` error;
- unknown or missing architectures → explicit error.

The historical DeepSeek fallback applies only when
`general.architecture` is missing and `deepseek4.*` metadata is present. It is
never applied to a different explicit architecture.

Tokenizer availability is separate from runtime availability:
GLM 5.2 can be inspected and tokenized even while its Metal backend
remains `recognizedButNotImplemented`. `ConversationBackendPolicy` applies the
same separation by choosing `deepSeekDSML` or `glm52Native`, without building
an inference engine.
