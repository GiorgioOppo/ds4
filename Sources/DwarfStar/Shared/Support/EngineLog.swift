import Foundation
import Darwin

/// Captures the C engine's stderr (where Metal/kernel diagnostics are printed)
/// into an observable buffer, while still teeing it to the real stderr so a
/// terminal run keeps showing it. The wrapped chat error is generic
/// ("metal prefill failed"); the precise cause lands here.
@MainActor
@Observable
final class EngineLog {
    static let shared = EngineLog()

    private(set) var text = ""
    private var installed = false
    private var originalStderr: Int32 = -1
    /// Holds the redirect pipe and the file handles for the lifetime of the
    /// process. If we let ARC drop the Pipe, the kernel closes its read end and
    /// the first write to (redirected) stderr terminates the process with
    /// SIGPIPE — that's what "Terminated due to signal 13" was.
    private var pipe: Pipe?

    /// Tail of the captured log, useful to attach to a surfaced error.
    func tail(_ maxChars: Int = 1200) -> String {
        text.count <= maxChars ? text : String(text.suffix(maxChars))
    }

    /// Redirect fd 2 (stderr) to a pipe we drain. Call once at launch.
    func install() {
        guard !installed else { return }
        installed = true

        // Belt and suspenders: any broken pipe (subprocesses, sockets, here)
        // should not signal-kill the GUI. We surface errors via write() rcs.
        signal(SIGPIPE, SIG_IGN)

        let pipe = Pipe()
        self.pipe = pipe                          // keep both fds alive
        originalStderr = dup(2)
        setvbuf(stderr, nil, _IONBF, 0)           // unbuffered: see errors immediately
        dup2(pipe.fileHandleForWriting.fileDescriptor, 2)

        let original = originalStderr
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            // Tee back to the original stderr (terminal). write() can short-
            // write or fail; we don't retry — the GUI buffer below is the
            // authoritative copy.
            data.withUnsafeBytes { raw in
                if let base = raw.baseAddress { _ = write(original, base, data.count) }
            }
            if let s = String(data: data, encoding: .utf8) {
                Task { @MainActor in EngineLog.shared.text += s }
            }
        }
    }
}
