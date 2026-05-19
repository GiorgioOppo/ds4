import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Safe subprocess runner shared by every Unix-toolbox wrapper. Unlike
/// `ShellTool`, the binary path is always fixed by the caller and the
/// arguments are an already-split `[String]` — never `/bin/sh -c
/// "<concat>"`. That removes the only injection surface and lets the
/// permission story stay coarse (`.readOnly` for `ls`, `.mutating` for
/// `cp`, …) instead of "anything could happen".
///
/// Fixes three latent bugs in `ShellTool`'s spawn pattern:
///   1. `terminationHandler` is set BEFORE `process.run()` — otherwise
///      a child that exits faster than the assignment can fire on
///      no-op and the awaiting continuation never resumes.
///   2. Both stdout/stderr pipes drain concurrently via
///      `readabilityHandler`, capped at `outputCap`. Without this a
///      tar/find that emits >64 KB blocks on its write side and the
///      watchdog kills it as if it had timed out.
///   3. `SIGTERM` is followed by a 2 s grace period and then `SIGKILL`.
///      A `sed`/`awk` with a custom signal handler would otherwise
///      survive the terminate() call indefinitely.
///
/// Deadline arithmetic uses a single `DispatchTime` (monotonic) so a
/// wall-clock adjustment in the middle of a run can't push the deadline
/// backwards.
public enum UnixBinary {

    /// Default output cap, matches `ShellTool.swift`. Keep both stdout
    /// and stderr combined inside this budget so the model's context
    /// window can't be poisoned by a runaway log.
    public static let defaultOutputCap = 32 * 1024

    /// Time SIGTERM is given to land before the watchdog escalates to
    /// SIGKILL. Two seconds matches systemd's stop-timeout default and
    /// is long enough that a well-behaved cleanup runs.
    private static let sigtermGracePeriod: TimeInterval = 2.0

    /// Resolve a binary that may live in different places on different
    /// hosts (Homebrew arm64 vs Intel vs MacPorts). Returns the first
    /// candidate that exists and is executable, or nil. Callers should
    /// throw `ToolError.notFound` with an install hint when the result
    /// is nil — this helper deliberately stays silent so the caller can
    /// produce a tool-specific message.
    public static func resolveBinary(candidates: [String]) -> String? {
        for path in candidates {
            if FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }
        return nil
    }

    /// Spawn `launchPath` with `arguments`, drain stdout/stderr, enforce
    /// a wall-clock timeout, and return the combined output. The output
    /// is capped at `outputCap` bytes — when truncation happens, a
    /// trailing `\n[output truncated at N bytes]` line is appended so
    /// the model knows it didn't see everything.
    ///
    /// `separateStreams`: when true, stdout and stderr are returned as
    /// separate sections in the body (prefixed with `stdout:` and
    /// `stderr:`). Useful for tools like `diff` where the model needs
    /// to distinguish the two channels. Default false → behaviour
    /// matches `ShellTool`.
    public static func runBinary(launchPath: String,
                                 arguments: [String],
                                 context: ToolContext,
                                 cwd: URL? = nil,
                                 timeout: TimeInterval = 60,
                                 outputCap: Int = defaultOutputCap,
                                 separateStreams: Bool = false) async throws
        -> ToolOutput
    {
        guard FileManager.default.isExecutableFile(atPath: launchPath) else {
            throw ToolError.notFound("binary not found: \(launchPath)")
        }

        let process = Process()
        process.launchPath = launchPath
        process.arguments = arguments
        process.currentDirectoryURL = cwd ?? context.rootDirectory
        if let env = context.environment { process.environment = env }

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        // Drain accumulators. NSLock keeps the readabilityHandler
        // callbacks (Foundation thread) safe against the main task.
        let lock = NSLock()
        var outData = Data()
        var errData = Data()
        let cap = outputCap

        stdout.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            lock.lock()
            if outData.count < cap {
                outData.append(chunk.prefix(cap - outData.count))
            }
            lock.unlock()
        }
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            lock.lock()
            if errData.count < cap {
                errData.append(chunk.prefix(cap - errData.count))
            }
            lock.unlock()
        }

        // CRITICAL: set the termination handler BEFORE run(). A child
        // that exits between run() and setting the handler would
        // otherwise leak the continuation forever.
        let exitSignal = ExitSignal()
        process.terminationHandler = { _ in exitSignal.signal() }

        do {
            try process.run()
        } catch {
            stdout.fileHandleForReading.readabilityHandler = nil
            stderr.fileHandleForReading.readabilityHandler = nil
            throw ToolError.spawnFailed(error.localizedDescription)
        }

        // Monotonic deadline — DispatchTime is unaffected by NTP /
        // user clock adjustments mid-run.
        let deadline = DispatchTime.now() + .milliseconds(Int(timeout * 1000))
        var timedOut = false
        let watchdog = Task {
            while !exitSignal.fired {
                if DispatchTime.now() >= deadline || context.isCancelled() {
                    timedOut = true
                    process.terminate()
                    // Grace then SIGKILL.
                    try? await Task.sleep(nanoseconds: UInt64(sigtermGracePeriod * 1_000_000_000))
                    if process.isRunning {
                        kill(process.processIdentifier, SIGKILL)
                    }
                    return
                }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }

        await exitSignal.wait()
        watchdog.cancel()

        // Drain any final bytes the readabilityHandler missed (the
        // closing read after EOF is delivered before terminationHandler
        // in most cases, but not all).
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil
        let tailOut = stdout.fileHandleForReading.availableData
        let tailErr = stderr.fileHandleForReading.availableData
        lock.lock()
        if outData.count < cap {
            outData.append(tailOut.prefix(cap - outData.count))
        }
        if errData.count < cap {
            errData.append(tailErr.prefix(cap - errData.count))
        }
        let outSnap = outData
        let errSnap = errData
        lock.unlock()

        if timedOut {
            throw ToolError.timeout(after: timeout)
        }

        let outString = String(data: outSnap, encoding: .utf8) ?? ""
        let errString = String(data: errSnap, encoding: .utf8) ?? ""
        let body: String
        if separateStreams {
            var sections: [String] = []
            if !outString.isEmpty { sections.append("stdout:\n\(outString)") }
            if !errString.isEmpty { sections.append("stderr:\n\(errString)") }
            body = sections.joined(separator: "\n")
        } else {
            var combined = outString
            if !errString.isEmpty {
                if !combined.isEmpty && !combined.hasSuffix("\n") { combined += "\n" }
                combined += errString
            }
            body = combined
        }
        let capped = capOutput(body, max: cap)
        let status = process.terminationStatus
        return ToolOutput(
            output: "exit \(status)\n\(capped)",
            isError: status != 0,
            metadata: ["exit": "\(status)"]
        )
    }

    /// Append the truncation marker if the body exceeds `max` bytes.
    /// Counted by UTF-8 byte length so a 4-byte emoji doesn't sneak
    /// past a character-count limit.
    public static func capOutput(_ body: String, max: Int) -> String {
        if body.utf8.count <= max { return body }
        let prefix = String(body.prefix(max))
        return prefix + "\n[output truncated at \(max) bytes]"
    }
}

/// One-shot signal used to bridge `Process.terminationHandler` (which
/// is invoked from a Foundation queue) to async/await. `signal()` is
/// idempotent so the readabilityHandler tail or a double-fire from
/// Foundation doesn't desync the await.
private final class ExitSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var continuations: [CheckedContinuation<Void, Never>] = []
    private(set) var fired = false

    func signal() {
        lock.lock()
        if fired {
            lock.unlock()
            return
        }
        fired = true
        let pending = continuations
        continuations.removeAll()
        lock.unlock()
        for cont in pending { cont.resume() }
    }

    func wait() async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            lock.lock()
            if fired {
                lock.unlock()
                cont.resume()
            } else {
                continuations.append(cont)
                lock.unlock()
            }
        }
    }
}
