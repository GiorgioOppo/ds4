**English** | [Italiano](README.it.md)

# Laguna conversation tests

Golden tests against the reference `laguna-s2.1` transcript framing: the
`〈|EOS|〉` opener, the hoisted system block with the `### Tools` /
`<available_tools>` section, text-tagged user/tool-response turns, assistant
turns with interleaved reasoning and `</assistant>` closes, separator-free
tagged tool calls, the strict/streamed tool parser, untrusted-content
containment, and the reference sampling defaults.
