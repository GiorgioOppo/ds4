import Foundation
import DS4Core

// `DS4Demo requantize` — an OFFLINE GGUF -> GGUF requantizer. Pure Swift, no
// Metal/GPU: it must run on machines without an Apple GPU (and is intercepted in
// main.swift BEFORE the Metal runtime is brought up). This is the in-process
// counterpart to the upstream C `gguf-tools/deepseek4-quantize.c` selective
// `--tensor-type` pass, reusing the byte-exact QuantEncode encoders.
//
// Usage:
//   DS4Demo requantize <in.gguf> <out.gguf> RULE [RULE ...]
//
// RULE = SRC>DST[@NAME]
//   SRC, DST : GGUF type names (f32, f16, q8_0, q2_k, q4_k, q8_k, iq2_xxs, ...)
//   @NAME    : optional — only tensors whose name contains NAME are remapped
//
// Examples:
//   DS4Demo requantize in.gguf out.gguf f16>q4_k q8_0>q4_k
//   DS4Demo requantize in.gguf out.gguf q8_0>q4_k@ffn_down.weight
//
// Tensors not matched by any rule (or whose source type cannot be dequantized)
// are copied through unchanged; the run prints a per-tensor report.

/// Reverse map GGUF type name -> code, from the reader's canonical table.
private func ggufTypeCode(_ name: String) -> UInt32? {
    let n = name.lowercased()
    for (code, info) in GGUF.typeTable where info.name == n { return code }
    return nil
}

private struct RequantRule {
    let src: UInt32
    let dst: UInt32
    let nameContains: String?
}

private func parseRule(_ s: String) -> RequantRule? {
    // SRC>DST[@NAME]
    var body = s
    var nameContains: String?
    if let at = body.firstIndex(of: "@") {
        nameContains = String(body[body.index(after: at)...])
        body = String(body[..<at])
    }
    let parts = body.split(separator: ">", maxSplits: 1).map(String.init)
    guard parts.count == 2, let src = ggufTypeCode(parts[0]), let dst = ggufTypeCode(parts[1]) else {
        return nil
    }
    return RequantRule(src: src, dst: dst, nameContains: nameContains)
}

private func requantizeUsage() {
    FileHandle.standardError.write(Data("""
    Usage: DS4Demo requantize <in.gguf> <out.gguf> RULE [RULE ...]
      RULE = SRC>DST[@NAME]   (types: f32 f16 q8_0 q2_k q4_k q8_k iq2_xxs ...)
      e.g. DS4Demo requantize in.gguf out.gguf f16>q4_k q8_0>q4_k@ffn_down.weight

    """.utf8))
}

/// Runs the offline requantizer. Returns a process exit code.
func runRequantizeCLI(_ args: [String]) -> Int32 {
    guard args.count >= 3 else { requantizeUsage(); return 2 }
    let inPath = args[0], outPath = args[1]
    let ruleArgs = Array(args.dropFirst(2))

    var rules: [RequantRule] = []
    for r in ruleArgs {
        guard let rule = parseRule(r) else {
            FileHandle.standardError.write(Data("requantize: invalid rule '\(r)'\n".utf8))
            requantizeUsage()
            return 2
        }
        rules.append(rule)
    }

    let source: GGUFModel
    do {
        source = try GGUFModel(path: inPath, metalMapping: false)
    } catch {
        FileHandle.standardError.write(Data("requantize: cannot open \(inPath): \(error)\n".utf8))
        return 1
    }

    // First matching rule (by source type + optional name filter) wins.
    let options = GGUFRequantizer.Options(
        targetType: { t in
            for rule in rules where rule.src == t.type {
                if let sub = rule.nameContains, !t.name.contains(sub) { continue }
                return rule.dst
            }
            return nil
        },
        imatrix: { _ in nil },   // offline imatrix input not wired yet (iq2 targets are skipped)
        log: { print("  \($0)") })

    print("requantize: \(inPath) -> \(outPath)")
    print("  \(source.n_tensors) tensors, \(source.n_kv) metadata keys, alignment \(source.alignment)")
    for rule in rules {
        let filter = rule.nameContains.map { " @\($0)" } ?? ""
        print("  rule: \(GGUF.typeName(rule.src)) > \(GGUF.typeName(rule.dst))\(filter)")
    }

    do {
        let report = try GGUFRequantizer.requantize(source: source, options: options, to: outPath)
        print("requantize: done — \(report.requantized) requantized, "
              + "\(report.passthrough) passthrough, \(report.skipped) skipped")
        if !report.skips.isEmpty {
            print("  skipped tensors:")
            for s in report.skips.prefix(20) { print("    \(s.name): \(s.reason)") }
            if report.skips.count > 20 { print("    … and \(report.skips.count - 20) more") }
        }
        return 0
    } catch {
        FileHandle.standardError.write(Data("requantize: failed: \(error)\n".utf8))
        return 1
    }
}
