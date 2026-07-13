# DS4Demo Command

`main.swift` is the CLI entry point. It parses positional arguments and
environment knobs, initializes Metal, optionally audits a GGUF, performs a
forward smoke test, then runs prompt prefill and streaming decode.

Dependencies flow directly to `DS4Core`, `DS4Metal`, and Apple's `Metal`
framework; the demo intentionally does not depend on `DS4Engine` or SwiftUI.

Keep argument compatibility documented in the parent README. Move reusable
logging, model inspection, or disk measurement helpers into `Diagnostics/`
instead of expanding the command file. New numerical defaults must be explicit
because this executable is used for performance and parity comparisons.

