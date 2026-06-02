import Foundation

/// Apply a unified-diff patch. This implementation is deliberately
/// minimal: it supports the common subset (`---`/`+++` headers,
/// `@@` hunks, `+`/`-`/' ' line prefixes, new-file with `--- /dev/null`,
/// delete with `+++ /dev/null`). It does *not* do binary patches,
/// rename detection, or fuzzy matching — if the patch doesn't apply
/// cleanly we surface that and let the model retry.
///
/// For complex changes the model should prefer chaining `edit` calls;
/// `apply_patch` is the right escape hatch when there are many small
/// changes across one file or when the model wants to emit a single
/// reviewable artefact.
public struct ApplyPatchTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "apply_patch",
            description:
                "Applica una patch in formato unified-diff a uno o più file. Supporta " +
                "la creazione (--- /dev/null) e la cancellazione (+++ /dev/null). " +
                "Rifiuta su qualsiasi mismatch di hunk — non c'è fuzzy matching.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "patch": SchemaBuilder.string(description: "Corpo completo dello unified diff."),
                ],
                required: ["patch"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        let patch = input["patch"] as? String ?? ""
        let files = patch
            .components(separatedBy: "\n")
            .filter { $0.hasPrefix("+++ ") }
            .count
        return "apply_patch (\(files) file\(files == 1 ? "" : "s"))"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let patch = try input.string("patch")
        let hunks = try parsePatch(patch)
        var touched: [String] = []
        for fileHunks in hunks {
            try applyFile(fileHunks, context: context)
            touched.append(fileHunks.target)
        }
        return ToolOutput(
            output: "applied patch to \(touched.count) file(s): \(touched.joined(separator: ", "))",
            metadata: ["files": "\(touched.count)"]
        )
    }

    // MARK: - parsing

    /// One file's worth of hunks. Either the file is being created
    /// (source == "/dev/null"), deleted (target == "/dev/null"), or
    /// patched in place.
    private struct FileHunks {
        var source: String
        var target: String
        var hunks: [Hunk]
        /// True when a `\ No newline at end of file` marker applied to
        /// the NEW side's last line — i.e. the resulting file must NOT
        /// end with a trailing newline. Standard diffs (no marker) do.
        var newHasNoFinalNewline: Bool = false
    }

    private struct Hunk {
        /// 1-based start line in the source file, or 0 for new files.
        var oldStart: Int
        var oldLines: [String]
        var newLines: [String]
    }

    private func parsePatch(_ body: String) throws -> [FileHunks] {
        let lines = body.components(separatedBy: "\n")
        var result: [FileHunks] = []
        var current: FileHunks?
        var inHunk = false
        var currentHunk: Hunk?
        // Side of the most recent content line, so a following
        // `\ No newline at end of file` marker can be attributed to it.
        var lastSide: Character? = nil
        var i = 0
        while i < lines.count {
            let line = lines[i]
            if line.hasPrefix("--- ") {
                if let h = currentHunk, var c = current {
                    c.hunks.append(h); current = c; currentHunk = nil
                }
                if let c = current { result.append(c) }
                let source = String(line.dropFirst(4)).trimmingCharacters(in: .whitespaces)
                current = FileHunks(source: source, target: "", hunks: [])
                inHunk = false
                lastSide = nil
            } else if line.hasPrefix("+++ ") {
                let target = String(line.dropFirst(4)).trimmingCharacters(in: .whitespaces)
                current?.target = target
            } else if line.hasPrefix("@@") {
                if let h = currentHunk, var c = current {
                    c.hunks.append(h); current = c
                }
                let parsed = try parseHunkHeader(line)
                currentHunk = Hunk(oldStart: parsed, oldLines: [], newLines: [])
                inHunk = true
            } else if inHunk, var h = currentHunk {
                guard let first = line.first else {
                    // blank line inside a hunk → context blank
                    h.oldLines.append("")
                    h.newLines.append("")
                    currentHunk = h
                    i += 1
                    continue
                }
                let body = String(line.dropFirst())
                switch first {
                case " ":
                    h.oldLines.append(body)
                    h.newLines.append(body)
                    lastSide = " "
                case "+":
                    h.newLines.append(body)
                    lastSide = "+"
                case "-":
                    h.oldLines.append(body)
                    lastSide = "-"
                case "\\":
                    // "\ No newline at end of file" applies to the
                    // immediately preceding line's side. Record it for
                    // the NEW side so a created file's trailing newline
                    // is suppressed to match the diff.
                    if lastSide == "+" || lastSide == " " {
                        current?.newHasNoFinalNewline = true
                    }
                default:
                    // Out-of-hunk content marks hunk end.
                    inHunk = false
                    if var c = current { c.hunks.append(h); current = c }
                    currentHunk = nil
                    continue
                }
                currentHunk = h
            }
            i += 1
        }
        if let h = currentHunk, var c = current {
            c.hunks.append(h); current = c
        }
        if let c = current { result.append(c) }
        return result
    }

    private func parseHunkHeader(_ header: String) throws -> Int {
        // Format: @@ -A,B +C,D @@ optional context
        let scanner = Scanner(string: header)
        guard scanner.scanString("@@") != nil,
              scanner.scanString("-") != nil,
              let oldStart = scanner.scanInt() else {
            throw ToolError.invalidInput("malformed hunk header: \(header)")
        }
        return oldStart
    }

    // MARK: - apply

    private func applyFile(_ file: FileHunks, context: ToolContext) throws {
        let rawTarget = file.target == "/dev/null" ? file.source : file.target
        let path = stripPathPrefix(rawTarget)
        let url = try resolveInsideRoot(path, context: context)

        if file.source == "/dev/null" {
            // Creation. A unified diff's content lines each denote a
            // newline-terminated line, so the created file ends with a
            // trailing newline UNLESS a `\ No newline at end of file`
            // marker said otherwise.
            var created: [String] = []
            for h in file.hunks {
                created.append(contentsOf: h.newLines)
            }
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            var text = created.joined(separator: "\n")
            if !file.newHasNoFinalNewline && !created.isEmpty {
                text += "\n"
            }
            try text.write(to: url, atomically: true, encoding: .utf8)
            return
        }

        guard let data = try? Data(contentsOf: url),
              let original = String(data: data, encoding: .utf8) else {
            throw ToolError.notFound(path)
        }
        let lines = original.components(separatedBy: "\n")
        // Verify every hunk's `-`/context lines against the actual file
        // content (in reverse so earlier edits don't shift later
        // indexes). This runs for BOTH in-place patching and `/dev/null`
        // deletion, so a delete won't remove a file whose contents have
        // drifted from the patch.
        let patched = try applyHunks(file.hunks, to: lines, path: path)

        if file.target == "/dev/null" {
            // Deletion — context already verified above.
            try FileManager.default.removeItem(at: url)
            return
        }
        try patched.joined(separator: "\n").write(
            to: url, atomically: true, encoding: .utf8)
    }

    /// Apply `hunks` to `lines` in reverse order (so earlier edits don't
    /// shift later indexes), verifying each hunk's `-`/context lines
    /// against the actual content. Throws on any mismatch — no fuzzy
    /// matching. Returns the patched lines.
    private func applyHunks(_ hunks: [Hunk], to lines: [String],
                            path: String) throws -> [String] {
        var lines = lines
        for h in hunks.reversed() {
            let start = max(0, h.oldStart - 1)
            let end = start + h.oldLines.count
            guard end <= lines.count else {
                throw ToolError.invalidInput("hunk extends past end of \(path)")
            }
            let actual = Array(lines[start..<end])
            if actual != h.oldLines {
                throw ToolError.invalidInput(
                    "hunk context mismatch in \(path) at line \(h.oldStart)")
            }
            lines.replaceSubrange(start..<end, with: h.newLines)
        }
        return lines
    }

    /// Drop `a/` / `b/` prefixes the way `git diff` emits them.
    private func stripPathPrefix(_ path: String) -> String {
        if path.hasPrefix("a/") || path.hasPrefix("b/") {
            return String(path.dropFirst(2))
        }
        return path
    }
}
