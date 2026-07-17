# GLM 5.2 conversation protocol

This directory owns GLM 5.2 chat framing and native tool syntax. The wire
format starts with `[gMASK]<sop>`, uses `<|role|>` control tokens, and encodes
tool calls as `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value>`.

The renderer and parser are model-independent and testable without a GGUF.
Untrusted system, user, argument, and tool-result text is neutralized before a
rendered transcript is scanned for special tokens. Runtime registration stays
outside this directory; having a tokenizer and renderer does not imply that the
Metal inference graph is complete.

`GLM52ChatRenderer` follows the message loop in the real
`tokenizer.chat_template`: system messages remain at their original transcript
positions rather than being collected into a prologue. A run of consecutive
tool results opens `<|observation|>` once and then emits one
`<tool_response>…</tool_response>` wrapper per result; any intervening role
starts a new observation run.
