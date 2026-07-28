**English** | [Italiano](README.it.md)

# Laguna tokenizer tests

Deterministic fixtures (control tokens plus the 256 byte-level singles) cover
the Laguna newline pre-split and its observable CRLF difference from the plain
GLM4 splitter, single-digit number groups, BPE merges, byte-level round trips,
atomic rendered-chat controls versus plain-text role tags, the EOS/EOT stop
policy and control-token neutralization — all without downloading a GGUF.
