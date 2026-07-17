# GLM 5.2 tokenizer

`GLM52Tokenizer` implements GPT-2 byte-level BPE with the ChatGLM4 `glm4`
pre-tokenizer, atomic native role/tool controls, detokenization, reasoning-aware
stop tokens, and `[gMASK]<sop>` prompt encoding.

The model initializer validates `general.architecture`, tokenizer tables,
pre-tokenizer type, and all required protocol tokens. A model-free internal
initializer and the pure `GLM4Pretokenizer` splitter support deterministic unit
tests without downloading or mapping a GLM GGUF.

This layer does not select an inference backend. Model detection may recognize
GLM while the Metal graph remains explicitly unavailable.
