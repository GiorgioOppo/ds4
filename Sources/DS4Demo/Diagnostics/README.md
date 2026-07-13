# DS4Demo Diagnostics

Focused helpers used by the command-line executable:

- `DemoLog.swift` keeps diagnostic logging on `stderr`, separate from generated
  text written to `stdout`.
- `DiskBenchmark.swift` measures storage behavior relevant to streamed weights.
- `ModelDiagnostics.swift` reports GGUF tensor and tokenizer information.

Diagnostics may observe and report runtime behavior but should not silently
change generation settings. Keep output stable enough for benchmark comparison
and shell capture.

