import Foundation

/// Tokenizer interface. Conforming type encodes UTF-8 strings to int token ids
/// and back. DeepSeek-V4 uses a byte-level BPE; the concrete implementations
/// are `BPETokenizer`, `SentencePieceTokenizer`, `WordPieceTokenizer`.
public protocol Tokenizer {
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
    var bosId: Int? { get }
    var eosId: Int? { get }
    /// All token ids whose sampling should terminate the generation loop.
    /// Always includes `eosId` (end-of-sentence = end-of-conversation) and
    /// `<|EOT|>` (end-of-turn) when present in the vocab. The decode loop
    /// must check this set, not just `eosId`, otherwise V4 chat
    /// generations run past the assistant's end-of-turn marker and the
    /// model — sampling whatever the LM head still ranks highest after
    /// EOT was already emitted — collapses into looped filler tokens.
    var stopTokenIds: Set<Int> { get }
}

/// Errors raised by the tokenizer dispatcher.
public enum TokenizerError: Error, CustomStringConvertible {
    case unsupportedType(String)
    case missingFile(String)
    case malformed(String)

    public var description: String {
        switch self {
        case .unsupportedType(let t): return "Tokenizer: unsupported type — \(t)"
        case .missingFile(let f):     return "Tokenizer: missing file — \(f)"
        case .malformed(let m):       return "Tokenizer: malformed — \(m)"
        }
    }
}

/// Bundle returned by the dispatcher: the tokenizer plus the chat
/// template both inferred from the same model directory.
public struct LoadedTokenizer {
    public let tokenizer: Tokenizer
    public let chatTemplate: ChatTemplate
    /// True iff the resolved template is `DSV4Template` (i.e. the model
    /// is DeepSeek-V4 and the legacy `EncodingDSV4` path is in use).
    /// Callers that want to reuse `EncodingDSV4.parseCompletion` /
    /// `encodeToolOutputs` / `encodeToolCalls` check this flag.
    public let isDSV4: Bool

    public init(tokenizer: Tokenizer, chatTemplate: ChatTemplate, isDSV4: Bool) {
        self.tokenizer = tokenizer
        self.chatTemplate = chatTemplate
        self.isDSV4 = isDSV4
    }
}

/// Dispatcher for HuggingFace-style tokenizer directories. Inspects
/// `tokenizer.json` (and optionally `tokenizer_config.json`, `*.model`)
/// and instantiates the right `Tokenizer` + `ChatTemplate`.
public enum TokenizerLoader {

    /// Backward-compatible shim: takes a path to a `tokenizer.json`
    /// directly and returns only the `Tokenizer`. Prefer
    /// `load(tokenizerDir:)` in new code so the chat template can be
    /// inferred too.
    public static func load(from url: URL) throws -> Tokenizer {
        let data = try Data(contentsOf: url)
        return try BPETokenizer(jsonData: data)
    }

    /// Full dispatch: reads `tokenizer.json` (required) and
    /// `tokenizer_config.json` (optional) from `directory`, then picks
    /// the matching tokenizer + chat template.
    public static func load(tokenizerDir directory: URL) throws -> LoadedTokenizer {
        let fm = FileManager.default
        let tokJSON = directory.appendingPathComponent("tokenizer.json")
        let cfgJSON = directory.appendingPathComponent("tokenizer_config.json")

        // 1. Pick the tokenizer.
        var tokenizer: Tokenizer
        var modelType = ""
        if fm.fileExists(atPath: tokJSON.path) {
            let data = try Data(contentsOf: tokJSON)
            modelType = (try? readTokenizerType(data: data)) ?? ""
            switch modelType {
            case "BPE":
                tokenizer = try BPETokenizer(jsonData: data)
            case "WordPiece":
                tokenizer = try WordPieceTokenizer(jsonData: data)
            case "Unigram", "":
                // Either explicitly Unigram (= SentencePiece) or
                // missing model.type. Fall back to a sibling .model file.
                if let modelFile = try findSentencePieceModel(in: directory) {
                    tokenizer = try SentencePieceTokenizer(modelBytes: try Data(contentsOf: modelFile))
                } else if modelType == "Unigram" {
                    throw TokenizerError.missingFile("Unigram tokenizer but no .model file")
                } else {
                    // No model.type and no .model file → assume BPE-like.
                    tokenizer = try BPETokenizer(jsonData: data)
                }
            default:
                throw TokenizerError.unsupportedType(modelType)
            }
        } else if let modelFile = try findSentencePieceModel(in: directory) {
            // .model only (legacy Mistral-style packs).
            tokenizer = try SentencePieceTokenizer(modelBytes: try Data(contentsOf: modelFile))
        } else {
            throw TokenizerError.missingFile("no tokenizer.json or .model in \(directory.path)")
        }

        // 2. Pick the chat template.
        var jinjaTemplateSource: String? = nil
        var configModelType: String? = nil
        if fm.fileExists(atPath: cfgJSON.path) {
            let cfgData = try Data(contentsOf: cfgJSON)
            if let cfg = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any] {
                jinjaTemplateSource = cfg["chat_template"] as? String
                configModelType = cfg["model_type"] as? String
            }
        }
        // DSV4 wins over a generic Jinja template even when one is
        // present: EncodingDSV4 wraps additional logic beyond message
        // rendering (REASONING_EFFORT_MAX prompt, tools block,
        // `<｜tool▁outputs｜>` post-processing) that the Jinja-only
        // path does not reproduce. Future non-V4 models still pick up
        // their own chat_template.
        let isV4 = isDSV4(modelType: configModelType, tokenizerData: try? Data(contentsOf: tokJSON))
        let template: ChatTemplate
        let templateIsDSV4: Bool
        if isV4 {
            template = DSV4Template()
            templateIsDSV4 = true
        } else if let src = jinjaTemplateSource, !src.isEmpty {
            template = try JinjaChatTemplate(src)
            templateIsDSV4 = false
        } else {
            throw TokenizerError.unsupportedType(
                "no chat_template in tokenizer_config.json and model is not DSV4"
            )
        }
        return LoadedTokenizer(tokenizer: tokenizer,
                               chatTemplate: template,
                               isDSV4: templateIsDSV4)
    }

    // MARK: - GGUF auto-detect (T2)

    /// Construct a `Tokenizer` straight from a GGUF file's embedded
    /// vocabulary, mirroring `llama.cpp`'s `gguf_init_from_file` →
    /// `LLM_TOKENIZER_BPE` path. The GGUF format stores both the
    /// tokenizer kind (`tokenizer.ggml.model`) and the entire vocab
    /// (`tokenizer.ggml.tokens` + `tokenizer.ggml.merges` for BPE) in
    /// metadata; we read those and synthesize the equivalent
    /// `tokenizer.json` shape so the existing `BPETokenizer` decoder
    /// can ingest it without a separate code path.
    ///
    /// Today's coverage:
    ///   - "gpt2" / "llama" → BPE (byte-level merges) — works for
    ///     every Llama 1/2/3, Mistral, Qwen, CodeLlama GGUF.
    ///   - "bert"           → WordPiece. Not yet wired (T2
    ///     follow-up) — throws `unsupportedType`.
    ///
    /// Returns a `LoadedTokenizer` with a Jinja-resolved chat
    /// template if the GGUF carries `tokenizer.chat_template`, else
    /// a placeholder template that emits plain `User: …\nAssistant:`
    /// turns (good enough for one-off greedy tests, not for
    /// agent flows).
    public static func loadFromGGUF(_ gguf: GGUFFile) throws -> LoadedTokenizer {
        let meta = gguf.header.metadata
        guard case .string(let kind)? = meta["tokenizer.ggml.model"] else {
            throw TokenizerError.unsupportedType(
                "GGUF: missing tokenizer.ggml.model metadata")
        }
        let lowerKind = kind.lowercased()
        let tokenizer: Tokenizer
        switch lowerKind {
        case "gpt2", "llama":
            tokenizer = try buildBPEFromGGUF(meta: meta)
        case "bert":
            throw TokenizerError.unsupportedType(
                "GGUF: WordPiece (bert) tokenizer reconstruction not yet implemented")
        default:
            throw TokenizerError.unsupportedType(
                "GGUF: unknown tokenizer.ggml.model '\(kind)'")
        }

        // Chat template: prefer the embedded `tokenizer.chat_template`
        // if present, else fall back to a minimal Jinja shim that
        // produces the bare-bones role-prefixed format. Production
        // chat flows should always ship a real chat_template.
        let template: ChatTemplate
        if case .string(let src)? = meta["tokenizer.chat_template"],
           !src.isEmpty
        {
            template = try JinjaChatTemplate(src)
        } else {
            template = try JinjaChatTemplate(
                "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}Assistant: ")
        }

        return LoadedTokenizer(tokenizer: tokenizer,
                                chatTemplate: template,
                                isDSV4: false)
    }

    /// Helper: build a BPE tokenizer from the GGUF vocab/merges
    /// metadata. Synthesizes a HuggingFace-shaped `tokenizer.json`
    /// blob and reuses `BPETokenizer(jsonData:)` so we don't drift
    /// from the parser used by the safetensors path.
    private static func buildBPEFromGGUF(
        meta: [String: GGUFValue]) throws -> Tokenizer
    {
        guard case .array(let tokValues)? = meta["tokenizer.ggml.tokens"]
        else {
            throw TokenizerError.malformed(
                "GGUF: tokenizer.ggml.tokens array missing")
        }
        var vocabDict: [String: Int] = [:]
        vocabDict.reserveCapacity(tokValues.count)
        for (id, value) in tokValues.enumerated() {
            if case .string(let s) = value {
                vocabDict[s] = id
            }
        }

        var merges: [String] = []
        if case .array(let mergeValues)? = meta["tokenizer.ggml.merges"] {
            merges.reserveCapacity(mergeValues.count)
            for v in mergeValues {
                if case .string(let s) = v { merges.append(s) }
            }
        }

        // Added tokens (special / control). GGUF stores per-id flags
        // in parallel arrays under `tokenizer.ggml.token_type` —
        // 1=normal, 2=unknown, 3=control, 4=user-defined, 5=unused,
        // 6=byte. We surface only the control + user-defined ones as
        // added_tokens so `BPETokenizer` recognises them as atomic.
        var addedTokens: [[String: Any]] = []
        if case .array(let typeValues)? = meta["tokenizer.ggml.token_type"] {
            for (id, tval) in typeValues.enumerated() where id < tokValues.count {
                guard case .int64(let ttype) = tval else { continue }
                let isAdded = (ttype == 3) || (ttype == 4)
                guard isAdded else { continue }
                if case .string(let s) = tokValues[id] {
                    addedTokens.append([
                        "id": id,
                        "content": s,
                        "special": ttype == 3,
                    ])
                }
            }
        }

        let json: [String: Any] = [
            "model": [
                "type": "BPE",
                "vocab": vocabDict,
                "merges": merges,
            ],
            "added_tokens": addedTokens,
        ]
        let data = try JSONSerialization.data(withJSONObject: json,
                                                options: [])
        return try BPETokenizer(jsonData: data)
    }

    // MARK: - Helpers

    /// Returns the `model.type` field of a tokenizer.json, or "" if
    /// the file is shaped differently.
    private static func readTokenizerType(data: Data) throws -> String {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TokenizerError.malformed("tokenizer.json: not a JSON object")
        }
        if let model = json["model"] as? [String: Any],
           let type = model["type"] as? String {
            return type
        }
        return ""
    }

    /// Finds a `*.model` file in `directory` (used by SentencePiece
    /// tokenizers). Returns nil if none is present.
    private static func findSentencePieceModel(in directory: URL) throws -> URL? {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(at: directory,
                                                       includingPropertiesForKeys: nil) else {
            return nil
        }
        return entries.first { $0.pathExtension == "model" }
    }

    /// Heuristic detector for DeepSeek-V4 tokenizers. Prefers an
    /// explicit `model_type == "deepseek_v4"` in tokenizer_config.json,
    /// falls back to checking the V4 added_tokens shape in
    /// tokenizer.json.
    private static func isDSV4(modelType: String?, tokenizerData: Data?) -> Bool {
        if let m = modelType?.lowercased(), m.contains("deepseek") {
            return true
        }
        guard let data = tokenizerData,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        // DSV4-specific added tokens, e.g. <｜begin▁of▁sentence｜>.
        if let added = json["added_tokens"] as? [[String: Any]] {
            for entry in added {
                if let s = entry["content"] as? String,
                   s.contains("begin▁of▁sentence") {
                    return true
                }
            }
        }
        return false
    }
}
