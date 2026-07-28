**English** | [Italiano](README.it.md)

# Laguna S 2.1 conversation protocol

This directory owns Laguna S 2.1 chat framing and native tool syntax. The wire
format starts with the `〈|EOS|〉` sequence marker, frames roles with
XML-looking text tags (`<system>`, `<user>`, `<tool_response>`), and reserves
dedicated control tokens only for `<assistant>`, `</assistant>`, `<think>`,
`</think>`, `<tool_call>` and `</tool_call>`.  Tool calls use the flat
`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value>` grammar that
upstream shares with GLM, declared in an `### Tools` system section wrapped by
`<available_tools>`.  Assistant turns carry interleaved reasoning between
`<think>` and `</think>` and close with `</assistant>`, the family's
end-of-turn token; reasoning content is preserved between tool calls.

The renderer and parser are model-independent and testable without a GGUF.
Untrusted system, user, argument, and tool-result text is neutralized before a
rendered transcript is scanned for special tokens; the textual role tags are
included in the neutralization set because they steer the model even though
they are not vocabulary tokens.  `LagunaConversationProtocol.SamplingDefaults`
records the reference sampling defaults (temperature 0.7, top-k 20, top-p
0.95, min-p 0.05).  Runtime registration stays outside this directory; having
a tokenizer and renderer does not imply that a Metal inference graph exists.
