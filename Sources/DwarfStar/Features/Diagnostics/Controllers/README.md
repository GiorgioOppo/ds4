# Diagnostics Controllers

`DiagnosticsController.swift` converts model path and input text into tokenizer
and chat-template diagnostics through `DS4Engine`. It owns observable loading,
output, and error state on the main actor.

Keep filesystem and engine calls out of the SwiftUI view. Diagnostic operations
must not create an inference engine or modify the active chat KV state.

