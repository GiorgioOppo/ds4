import Foundation
import DeepSeekKit

// CLI: deepseek <model-dir> "<prompt>"
//                   [--max-tokens N]
//                   [--temperature T]
//                   [--mode raw|chat]
//                   [--load-strategy auto|preload|mmap]
//                   [--force-load]
//
// `<model-dir>` should contain config.json (optional) and tokenizer.json.
// safetensors weights are loaded if present, otherwise the model is
// initialised with random f32 weights so the smoke flow still runs.
//
// `--force-load` bypasses the conservative RAM-safety refusals
// (shard > 70% of available, total > 25× available). Use only if you
// know you can tolerate aggressive paging — on small-RAM Macs the
// system can lock up under sustained mmap thrash.

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: deepseek <model-dir> "<prompt>" \
        [--max-tokens N] [--temperature T] [--mode raw|chat] \
        [--load-strategy auto|preload|mmap] [--force-load] \
        [--dump-tensor NAME[:row=R][:cols=A..B]]

    --dump-tensor NAME[:row=R][:cols=A..B]
        Diagnostic mode. Dequantizes a single row slice of the named
        tensor and prints the float values one per line, then exits.
        The prompt argument is ignored in this mode (use "" as
        placeholder). Default row=0, default cols=0..32.
        Example:
            deepseek /path/to/model "" --load-strategy streaming \\
                --dump-tensor layers.0.attn.wq.weight:row=0:cols=0..32

    --list-tensors [PREFIX]
        Diagnostic mode. Prints every tensor name in the checkpoint
        (optionally filtered by prefix) to stdout, one per line, then
        exits. Useful when --dump-tensor reports "tensor not found"
        and you need to discover the actual naming convention.
        Example:
            deepseek /path/to/model "" --load-strategy streaming \\
                --list-tensors layers.0.

    --trace-norms
        Diagnostic mode. Prints L2 norm + min/max/mean + NaN/Inf
        counters of the residual stream at key points in the forward
        pass (after embed, hc-expand, selected layers, logits).
        Useful for finding the layer at which activations diverge,
        collapse to zero, or go NaN. Output goes to stderr; the
        usual token stream still goes to stdout. Off by default.

    """.utf8))
    exit(2)
}

let args = CommandLine.arguments
guard args.count >= 2 else { usage() }

let modelDir = URL(fileURLWithPath: args[1])
// In normal inference mode the prompt is positional (args[2]); in
// `--dump-tensor` mode the user can pass `""` (or omit it entirely
// when no other positional args remain). Treat a leading `--` token
// as "no prompt given".
var prompt = ""
var nextArg = 2
if nextArg < args.count, !args[nextArg].hasPrefix("--") {
    prompt = args[nextArg]
    nextArg += 1
}
var maxTokens = 32
var temperature: Float = 1.0
var mode = "chat"
var loadStrategy: String? = nil
var forceLoad = false
var dumpSpec: String? = nil
var listPrefix: String? = nil
var listEnabled = false

var i = nextArg
while i < args.count {
    switch args[i] {
    case "--max-tokens":
        guard i + 1 < args.count, let n = Int(args[i + 1]) else { usage() }
        maxTokens = n; i += 2
    case "--temperature":
        guard i + 1 < args.count, let t = Float(args[i + 1]) else { usage() }
        temperature = t; i += 2
    case "--mode":
        guard i + 1 < args.count, ["raw", "chat"].contains(args[i + 1]) else { usage() }
        mode = args[i + 1]; i += 2
    case "--load-strategy":
        guard i + 1 < args.count,
              ["auto", "preload", "mmap", "streaming"].contains(args[i + 1]) else { usage() }
        loadStrategy = args[i + 1]; i += 2
    case "--force-load":
        forceLoad = true; i += 1
    case "--dump-tensor":
        guard i + 1 < args.count else { usage() }
        dumpSpec = args[i + 1]; i += 2
    case "--list-tensors":
        listEnabled = true
        // Optional prefix argument. Treat the next token as a
        // prefix only if it doesn't start with `--`.
        if i + 1 < args.count, !args[i + 1].hasPrefix("--") {
            listPrefix = args[i + 1]; i += 2
        } else {
            i += 1
        }
    case "--trace-norms":
        TraceFlags.normTrace = true; i += 1
    default: usage()
    }
}

// ---------- Diagnostic: --list-tensors ----------
// Prints every tensor name in the checkpoint (optionally filtered
// by prefix) and exits. Runs before any model build so it works on
// directories whose downstream load would refuse.
if listEnabled {
    let prevLog = MemoryLogger.enabled
    MemoryLogger.enabled = true
    defer { MemoryLogger.enabled = prevLog }
    do {
        let plan = try LoadPlan.decide(modelDir: modelDir,
                                        override: loadStrategy,
                                        forceLoad: forceLoad)
        FileHandle.standardError.write(Data(plan.summary().utf8))
        let loader = try WeightLoader(plan: plan)
        var names = loader.allKnownNames
        if let pfx = listPrefix {
            names = names.filter { $0.hasPrefix(pfx) }
        }
        names.sort()
        FileHandle.standardError.write(Data(
            "\(names.count) tensor(s)\(listPrefix.map { " matching prefix \($0)" } ?? "")\n".utf8))
        for n in names { print(n) }
        exit(0)
    } catch {
        FileHandle.standardError.write(Data(
            "--list-tensors failed: \(error)\n".utf8))
        exit(1)
    }
}

// ---------- Diagnostic: --dump-tensor ----------
// Runs before config/tokenizer/model build so we can inspect tensors
// even on a checkpoint whose downstream load would fail. Builds only
// `WeightLoader` (so streaming-pool buffers are allocated, but no
// Transformer is constructed), prints the requested row slice, exits.
if let spec = dumpSpec {
    let parsed: TensorDumpSpec.Parsed
    do {
        parsed = try TensorDumpSpec.parse(spec)
    } catch {
        FileHandle.standardError.write(Data(
            "--dump-tensor: \(error)\n".utf8))
        exit(2)
    }
    // Force-enable MemoryLogger during the dump so any pread / pool
    // failures inside ensureLayer print to stderr instead of being
    // swallowed — a diagnostic mode that silently returns garbage
    // would defeat the whole point.
    let prevLog = MemoryLogger.enabled
    MemoryLogger.enabled = true
    defer { MemoryLogger.enabled = prevLog }
    do {
        let plan = try LoadPlan.decide(modelDir: modelDir,
                                        override: loadStrategy,
                                        forceLoad: forceLoad)
        FileHandle.standardError.write(Data(plan.summary().utf8))
        let loader = try WeightLoader(plan: plan)
        let r = try TensorDump.dumpRow(parsed.name,
                                        row: parsed.row,
                                        cols: parsed.cols,
                                        loader: loader)
        // Header lines on stderr so they don't pollute the numeric
        // dump on stdout — caller can pipe `2>/dev/null` to get just
        // the floats, or `1>vals.txt 2>info.txt` to keep both.
        FileHandle.standardError.write(Data("""
        tensor: \(r.name)
        shape:  \(r.shape)
        dtype:  \(r.dtype)
        row:    \(r.row)
        cols:   \(r.cols.lowerBound)..\(r.cols.upperBound) (\(r.cols.count) values)
        scale:  \(r.scaleName ?? "—")

        """.utf8))
        for v in r.values {
            print(String(format: "%.8e", v))
        }
        exit(0)
    } catch {
        FileHandle.standardError.write(Data(
            "--dump-tensor failed: \(error)\n".utf8))
        exit(1)
    }
}

if prompt.isEmpty {
    FileHandle.standardError.write(Data(
        "missing prompt (and no --dump-tensor given)\n".utf8))
    usage()
}

// ---------- Config ----------
let configURL = modelDir.appendingPathComponent("config.json")
let config: ModelConfig
if FileManager.default.fileExists(atPath: configURL.path) {
    do {
        config = try ModelConfig.load(from: configURL)
    } catch {
        FileHandle.standardError.write(Data("failed to parse config.json: \(error)\n".utf8))
        exit(1)
    }
} else {
    FileHandle.standardError.write(Data("""
    config.json not found at \(configURL.path) — using ModelArgs defaults.
    Note: defaults are toy-sized (n_layers=7, n_routed_experts=8) and not
    suitable for real inference.

    """.utf8))
    config = ModelConfig()
}

// ---------- Tokenizer ----------
let tokenizerURL = modelDir.appendingPathComponent("tokenizer.json")
let tokenizer: Tokenizer
do {
    tokenizer = try TokenizerLoader.load(from: tokenizerURL)
} catch {
    FileHandle.standardError.write(Data("\(error.localizedDescription)\n".utf8))
    exit(1)
}

// ---------- Model ----------
print("Loading model …", terminator: "")
fflush(stdout)
let model: Transformer
do {
    model = try Transformer.load(config: config, from: modelDir,
                                  strategyOverride: loadStrategy,
                                  forceLoad: forceLoad)
} catch {
    FileHandle.standardError.write(Data("model load failed: \(error.localizedDescription)\n".utf8))
    exit(1)
}
print(" ready.")
MemoryLogger.snapshot("model-ready", force: true)

// ---------- Prompt formatting ----------
let promptText: String
switch mode {
case "raw":
    promptText = prompt
case "chat":
    let msg = Message(role: .user, content: prompt)
    promptText = EncodingDSV4.encodeMessages([msg], mode: .chat)
default: usage()
}

let promptIds = tokenizer.encode(promptText)
print("Prompt tokens: \(promptIds.count)")
if promptIds.isEmpty {
    FileHandle.standardError.write(Data("""
    tokenizer produced 0 tokens for the prompt. Check tokenizer.json's
    pre_tokenizer regex — see `jq '.pre_tokenizer' \(tokenizerURL.path)`.

    """.utf8))
    exit(1)
}

// ---------- Generation loop ----------
// Prefill: feed the entire prompt at start_pos = 0. Then decode one token
// at a time, feeding the previous token at start_pos = (prompt_len + i).
// Each decode call updates the sliding-window KV cache + compressor state
// and produces logits for one new token.

print("---")
fflush(stdout)

let eosId: Int = tokenizer.eosId ?? -1
// Generation runs on a background DispatchQueue so the main thread
// is free to poll memory metrics at fixed intervals. Without this
// the only memory snapshots we get during a forward are the ones
// `MemoryLogger.snapshot(...)` is called at explicitly — between
// `cmd.commit()` and `waitUntilCompleted()` the calling thread is
// blocked for tens of seconds on V4-Flash. A separate polling
// thread on stdout/stderr captures the entire memory trace,
// including the spike that precedes a kernel panic.
//
// `generatedIds` is mutated only by the inference closure; the
// main thread reads it only after `done.wait()` returns (no race).
// `print()` to stdout is internally synchronized by libc, so
// streamed tokens and monitor lines interleave cleanly.
let inferenceQueue = DispatchQueue(label: "deepseek.inference",
                                    qos: .userInitiated)
let done = DispatchSemaphore(value: 0)
var generatedIds: [Int] = []
var inferenceError: Error? = nil

inferenceQueue.async {
    defer { done.signal() }
    var samplingOpts = SamplingOptions(temperature: temperature,
                                        topK: 0, topP: 1.0,
                                        repetitionPenalty: 1.0)

    // Prefill.
    var logits = model.forward(inputIds: [promptIds], startPos: 0)
    MemoryLogger.snapshot("prefill-complete", force: true)

    for step in 0..<maxTokens {
        let nextId = Sampler.sample(logits, history: generatedIds, options: &samplingOpts)
        if nextId == eosId { break }
        generatedIds.append(nextId)
        MemoryLogger.snapshot("decode:token-\(String(format: "%03d", step))",
                              force: true)

        // Stream the new token to stdout immediately. In chat mode, we buffer
        // the whole output and parse <think>...</think> at the end so the
        // reasoning block doesn't leak before its closing tag.
        if mode == "raw" {
            print(tokenizer.decode([nextId]), terminator: "")
            fflush(stdout)
        }

        // Stop after sampling the requested count without doing one more
        // unnecessary forward.
        if step == maxTokens - 1 { break }

        // Decode step: feed only the just-sampled token at startPos = promptLen + step.
        let startPos = promptIds.count + step
        logits = model.forward(inputIds: [[nextId]], startPos: startPos)
    }
}

// Main thread: poll memory metrics every 250 ms while the
// inference thread runs. Each tick is `force: true` so the line
// always emits regardless of `thresholdBytes`. The `tick:NNNN`
// label runs through 1 ≤ tick ≤ 9999; at 250 ms per tick that's
// ~40 minutes of trace before the counter wraps.
var monitorTick = 0
let pollInterval: DispatchTimeInterval = .milliseconds(250)
while done.wait(timeout: .now() + pollInterval) == .timedOut {
    monitorTick &+= 1
    MemoryLogger.snapshot(
        "monitor:tick-\(String(format: "%04d", monitorTick))",
        force: true)
}
if let err = inferenceError {
    FileHandle.standardError.write(Data(
        "inference failed: \(err.localizedDescription)\n".utf8))
    exit(1)
}

print("")    // newline after streaming
if mode == "chat" {
    let combined = tokenizer.decode(generatedIds)
    let msg = EncodingDSV4.parseCompletion(combined, mode: .chat)
    if let r = msg.reasoningContent, !r.isEmpty {
        print("[reasoning]\n\(r)\n[/reasoning]")
    }
    print(msg.content)
}
