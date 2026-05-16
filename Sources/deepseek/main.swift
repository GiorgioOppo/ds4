import Foundation
import DeepSeekKit

// CLI: deepseek <model-dir> "<prompt>"
//                   [--max-tokens N]
//                   [--temperature T]
//                   [--top-k K] [--top-p P]
//                   [--min-p P] [--tfs Z] [--typical P]
//                   [--repetition-penalty R]
//                   [--frequency-penalty F] [--presence-penalty P]
//                   [--mirostat TAU] [--mirostat-eta ETA]
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
        [--thinking off|high|max] \
        [--top-k K] [--top-p P] [--min-p P] \
        [--tfs Z] [--typical P] \
        [--repetition-penalty R] \
        [--frequency-penalty F] [--presence-penalty P] \
        [--mirostat TAU] [--mirostat-eta ETA] \
        [--load-strategy auto|preload|mmap] [--force-load] \
        [--max-seq-len N] [--max-batch-size N] \
        [--dump-tensor NAME[:row=R][:cols=A..B]]

    Sampling flags (all optional, see Sampling.swift for semantics):
        --top-k K               keep only the top-K logits (0 = disabled)
        --top-p P               nucleus mass (1.0 = disabled)
        --min-p P               filter < P × max_prob (0 = disabled)
        --tfs Z                 tail-free z (1.0 = disabled)
        --typical P             locally-typical mass (1.0 = disabled)
        --repetition-penalty R  HuggingFace-style penalty (1.0 = disabled)
        --frequency-penalty F   OpenAI-style, scales with count
        --presence-penalty P    OpenAI-style, binary in presence
        --mirostat TAU          enable mirostat v2 with target surprise TAU
        --mirostat-eta ETA      mirostat learning rate (default 0.1)

    --thinking off|high|max
        Chat-mode thinking budget. Only meaningful with --mode chat.
            off  (default) — no thinking: appends </think> after
                              <｜Assistant｜>, model answers directly.
            high           — appends <think>, model emits reasoning,
                              then </think>, then the answer.
            max            — same as high plus the REASONING_EFFORT_MAX
                              system-prompt block prepended.

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

    --print-config
        Diagnostic mode. Loads config.json (or the defaults if
        missing), prints every field of the resulting ModelConfig to
        stderr, and exits. Use to verify that all keys actually
        round-trip from JSON instead of silently falling back to
        hard-coded defaults.

    --max-seq-len N
        Override config.json's `max_seq_len`. Caps the KV cache /
        compressor size: cache row count per layer is
        `windowSize + N / compress_ratio`. Lower = less RAM, shorter
        context. Must be > 0. The model still accepts prompts up to
        N tokens; longer ones will overflow the cache.

    --max-batch-size N
        Override config.json's `max_batch_size`. Multiplies the KV
        cache footprint. Default in V4-Flash config is 1; raise only
        if you actually need parallel batched decode.

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
var topK: Int = 0
var topP: Float = 1.0
var minP: Float = 0.0
var tfsZ: Float = 1.0
var typicalP: Float = 1.0
var repetitionPenalty: Float = 1.0
var frequencyPenalty: Float = 0.0
var presencePenalty: Float = 0.0
var mirostatTau: Float = 0.0
var mirostatEta: Float = 0.1
var mode = "chat"
var thinking = "off"   // off | high | max — picks the trailing think marker in chat mode
var loadStrategy: String? = nil
var forceLoad = false
var dumpSpec: String? = nil
var listPrefix: String? = nil
var listEnabled = false
var printConfigAndExit = false
// Overrides for KV-cache-sizing fields in config.json. Nil = keep the
// loaded value. Smaller numbers trade context length / batch size for
// less RAM; larger numbers do the opposite. Applied AFTER
// ModelConfig.load and BEFORE Transformer.load so every per-layer
// allocation picks up the override.
var maxSeqLenOverride: Int? = nil
var maxBatchSizeOverride: Int? = nil

var i = nextArg
while i < args.count {
    switch args[i] {
    case "--max-tokens":
        guard i + 1 < args.count, let n = Int(args[i + 1]) else { usage() }
        maxTokens = n; i += 2
    case "--temperature":
        guard i + 1 < args.count, let t = Float(args[i + 1]) else { usage() }
        temperature = t; i += 2
    case "--top-k":
        guard i + 1 < args.count, let n = Int(args[i + 1]), n >= 0 else { usage() }
        topK = n; i += 2
    case "--top-p":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        topP = v; i += 2
    case "--min-p":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        minP = v; i += 2
    case "--tfs":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        tfsZ = v; i += 2
    case "--typical":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        typicalP = v; i += 2
    case "--repetition-penalty":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        repetitionPenalty = v; i += 2
    case "--frequency-penalty":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        frequencyPenalty = v; i += 2
    case "--presence-penalty":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        presencePenalty = v; i += 2
    case "--mirostat":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        mirostatTau = v; i += 2
    case "--mirostat-eta":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        mirostatEta = v; i += 2
    case "--mode":
        guard i + 1 < args.count, ["raw", "chat"].contains(args[i + 1]) else { usage() }
        mode = args[i + 1]; i += 2
    case "--thinking":
        guard i + 1 < args.count,
              ["off", "high", "max"].contains(args[i + 1]) else { usage() }
        thinking = args[i + 1]; i += 2
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
    case "--print-config":
        printConfigAndExit = true; i += 1
    case "--max-seq-len":
        guard i + 1 < args.count, let n = Int(args[i + 1]), n > 0 else { usage() }
        maxSeqLenOverride = n; i += 2
    case "--max-batch-size":
        guard i + 1 < args.count, let n = Int(args[i + 1]), n > 0 else { usage() }
        maxBatchSizeOverride = n; i += 2
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

if prompt.isEmpty && !printConfigAndExit {
    FileHandle.standardError.write(Data(
        "missing prompt (and no --dump-tensor / --print-config given)\n".utf8))
    usage()
}

// ---------- Config ----------
let configURL = modelDir.appendingPathComponent("config.json")
// `var` so the --max-seq-len / --max-batch-size CLI overrides below
// can replace the loaded values before Transformer.load picks them up.
var config: ModelConfig
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

// Apply CLI overrides for KV-cache sizing. Done HERE (after load,
// before --print-config and Transformer.load) so the override flows
// through to every per-layer allocation and the print-config dump
// reflects the effective values.
if let n = maxSeqLenOverride {
    let prev = config.maxSeqLen
    config.maxSeqLen = n
    FileHandle.standardError.write(Data(
        "max_seq_len: \(prev) → \(n) (--max-seq-len override)\n".utf8))
}
if let n = maxBatchSizeOverride {
    let prev = config.maxBatchSize
    config.maxBatchSize = n
    FileHandle.standardError.write(Data(
        "max_batch_size: \(prev) → \(n) (--max-batch-size override)\n".utf8))
}

// ---------- Diagnostic: --print-config ----------
// Exits after printing every field of the loaded ModelConfig, so the
// caller can diff against config.json line-by-line and spot any keys
// that silently fell back to a hard-coded default.
if printConfigAndExit {
    FileHandle.standardError.write(Data(config.summary.utf8))
    exit(0)
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
    // V4 is trained with `<｜begin▁of▁sentence｜>` as the unconditional
    // prefix on every sample. Feeding a raw prompt without it puts the
    // model in an out-of-distribution state and the first generated
    // token tends to be BOS itself (the model "recovers" by emitting
    // what it expected to see). Prepend the BOS string here; the BPE
    // tokenizer recognises it as a single added-token id, not a byte
    // sequence.
    promptText = EncodingDSV4.bosToken + prompt
case "chat":
    let msg = Message(role: .user, content: prompt)
    let thinkMode: ThinkingMode
    switch thinking {
    case "off":  thinkMode = .chat
    case "high": thinkMode = .high
    case "max":  thinkMode = .max
    default:     thinkMode = .chat
    }
    promptText = EncodingDSV4.encodeMessages([msg], mode: thinkMode)
default: usage()
}

let promptIds = tokenizer.encode(promptText)
print("Prompt tokens: \(promptIds.count) — ids=\(promptIds)")
// Echo the decoded form so we can spot tokenizer drift: if `decode(encode(s))`
// doesn't round-trip to the input text, the tokenizer is splitting or
// merging tokens wrong and the model sees something different from what
// the user typed.
let roundTripped = tokenizer.decode(promptIds)
print("Prompt round-trip: \"\(roundTripped)\"")
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
// V4-Flash has TWO stop tokens: `<｜end▁of▁sentence｜>` (eosId, ends
// the whole conversation) and `<|EOT|>` (ends just the assistant
// turn). The decode loop must break on either, otherwise EOT gets
// consumed as a regular token and the model loops on filler.
let stopTokens: Set<Int> = tokenizer.stopTokenIds
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
    var samplingOpts = SamplingOptions(
        temperature: temperature,
        topK: topK, topP: topP,
        minP: minP, tailFree: tfsZ, typical: typicalP,
        repetitionPenalty: repetitionPenalty,
        frequencyPenalty: frequencyPenalty, presencePenalty: presencePenalty,
        mirostatTau: mirostatTau, mirostatEta: mirostatEta,
        mirostatMu: 2.0 * mirostatTau)

    // Prefill.
    var logits = model.forward(inputIds: [promptIds], startPos: 0)
    MemoryLogger.snapshot("prefill-complete", force: true)

    for step in 0..<maxTokens {
        let nextId = Sampler.sample(logits, history: generatedIds, options: &samplingOpts)
        if stopTokens.contains(nextId) {
            // Mirrors generate.py:67 — Python appends the EOS token to
            // the completion stream so downstream decoders see a well-
            // terminated sequence. We also keep `<|EOT|>` (and any
            // future stop ids) on the stream for the same reason.
            generatedIds.append(nextId)
            break
        }
        generatedIds.append(nextId)
        MemoryLogger.snapshot("decode:token-\(String(format: "%03d", step))",
                              force: true)

        // Stream the new token to stdout immediately, in both modes. Chat
        // mode still re-parses the accumulated text at the end (extracting
        // the `<think>...</think>` block as reasoning), so this only
        // changes what the user *sees* during decode — the final structured
        // Message is unchanged.
        print(tokenizer.decode([nextId]), terminator: "")
        fflush(stdout)

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
    // The raw stream above includes the `<think>...</think>` block
    // inline; re-parse here only to surface a tidy reasoning summary
    // at the end. The content has already been printed so don't
    // re-print it.
    let combined = tokenizer.decode(generatedIds)
    let msg = EncodingDSV4.parseCompletion(combined, mode: .chat)
    if let r = msg.reasoningContent, !r.isEmpty {
        print("---")
        print("[reasoning summary]")
        print(r)
    }
}
