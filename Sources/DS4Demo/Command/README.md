**English** | [Italiano](README.it.md)

# DS4Demo Command

`main.swift` is the CLI entry point. It parses positional arguments and
environment knobs, initializes Metal, optionally audits a GGUF, performs a
forward smoke test, then runs prompt prefill and streaming decode.

Dependencies flow directly to `DS4Core`, `DS4Metal`, and Apple's `Metal`
framework; the demo intentionally does not depend on `DS4Engine` or SwiftUI.
It uses DS4Core's canonical `ModelArchitectureDetector` before constructing the
DeepSeek tokenizer or decoder. This keeps the target graph unchanged while
producing the same early Qwen/unknown rejection as the engine factory.

Both Flash and the single-file Pro Q2 profile are accepted. Their validated
metadata produces a profile-specific `DSV4RuntimeGeometry`; the Pro Q4 catalog
package is not accepted because the CLI does not assemble its two shards.

Keep argument compatibility documented in the parent README. Move reusable
logging, model inspection, or disk measurement helpers into `Diagnostics/`
instead of expanding the command file. New numerical defaults must be explicit
because this executable is used for performance and parity comparisons.
