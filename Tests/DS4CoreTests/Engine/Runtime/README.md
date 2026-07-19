**English** | [Italiano](README.it.md)

# Multi-backend Runtime tests

These tests exercise the selection boundary without loading a real model:
DeepSeek V4 Flash and Pro must select the concrete backend, unknown DeepSeek
profiles must be rejected, Qwen must be recognized and rejected as not yet
implemented, and an unknown architecture must produce a distinct error.

The numerical tests of the DeepSeek decoder remain in the existing Metal
folders.
