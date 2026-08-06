import Foundation
import Metal
import DS4Core

/// Fase M1 dello speculative decode con testa MTP (docs/SELF-SPECULATIVE.md,
/// sezione "Fase M"): apertura, classificazione e VALIDAZIONE del GGUF sidecar
/// Multi-Token Prediction — il componente 'mtp' del catalogo download
/// (DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf, ~4 GB).
///
/// Questa fase NON esegue nulla sul percorso di decode: produce l'inventario
/// dei tensori (nome, quant, forma) e il confronto con le dimensioni del
/// modello principale — il ground truth necessario per cablare il forward del
/// draft (Fase M2) sulla conversione REALE invece che su ipotesi. L'interfaccia
/// attesa è quella della testa MTP di DeepSeek:
///
///   h' = blocco([eh_proj(concat(enorm(emb(token)); hnorm(hidden)))])
///   logits = shared_head(shared_head_norm(h'))
///
/// dove `hidden` è l'hidden finale pre-head del modello principale alla
/// posizione corrente e il blocco è un layer transformer proprio del modulo
/// MTP (con la SUA attention e quindi un suo KV incrementale).
public struct MTPSidecar: DS4Logging {
    public static let logTag = "mtp"

    /// Ruoli d'interfaccia riconosciuti. Le conversioni llama.cpp chiamano i
    /// tensori blk.N.nextn.* (eh_proj, embed_tokens, enorm, hnorm,
    /// shared_head.*); altri converter usano nomi mtp.* o i nomi del modello
    /// principale (token_embd / output). Tutto il resto è il blocco
    /// transformer del modulo MTP.
    public struct Inventory {
        public var ehProj: GGUFModel.Tensor?
        public var embedTokens: GGUFModel.Tensor?
        public var enorm: GGUFModel.Tensor?
        public var hnorm: GGUFModel.Tensor?
        public var sharedHeadNorm: GGUFModel.Tensor?
        public var sharedHead: GGUFModel.Tensor?
        public var blockTensors: [GGUFModel.Tensor] = []
    }

    public let model: GGUFModel
    public let inventory: Inventory

    /// Cerca il sidecar accanto al modello principale (DS4_MTP_GGUF=1):
    /// preferisce il nome esatto del catalogo, poi qualunque *mtp*.gguf
    /// nella stessa cartella.
    public static func locate(near mainModelPath: String) -> String? {
        let dir = (mainModelPath as NSString).deletingLastPathComponent
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        let catalog = "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
        if names.contains(catalog) { return dir + "/" + catalog }
        let cands = names.filter {
            $0.lowercased().contains("mtp") && $0.lowercased().hasSuffix(".gguf")
        }.sorted()
        return cands.first.map { dir + "/" + $0 }
    }

    public init(path: String) throws {
        // metalMapping: false — Fase M1 legge solo i METADATI; i buffer GPU
        // arrivano con la Fase M2 (caricamento residente).
        model = try GGUFModel(path: path, metalMapping: false, prefetchCPU: false)
        var inv = Inventory()
        for t in model.tensors {
            let n = t.name.lowercased()
            if n.contains("eh_proj") { inv.ehProj = t }
            else if n.contains("embed_tokens") || n.contains("token_embd") { inv.embedTokens = t }
            else if n.contains("enorm") { inv.enorm = t }
            else if n.contains("hnorm") { inv.hnorm = t }
            else if n.contains("shared_head.norm") || n.contains("shared_head_norm") { inv.sharedHeadNorm = t }
            else if n.contains("shared_head") || n.contains("output.weight") || n.hasSuffix("lm_head.weight") { inv.sharedHead = t }
            else { inv.blockTensors.append(t) }
        }
        inventory = inv
    }

    /// Inventario completo + verifiche d'interfaccia contro le dimensioni del
    /// modello principale (`vocab`/`nEmbd` <= 0 = salta il confronto). Le
    /// verifiche sono informative, non bloccanti: la Fase M2 deciderà sulla
    /// base di QUESTO output.
    public func report(vocab: Int, nEmbd: Int) -> String {
        func shape(_ t: GGUFModel.Tensor) -> String { t.dims.map(String.init).joined(separator: "×") }
        func line(_ label: String, _ t: GGUFModel.Tensor?) -> String {
            guard let t else { return "    \(label): MANCANTE" }
            return "    \(label): \(t.name)  \(t.typeName)  [\(shape(t))]  (\(t.bytes >> 20) MB)"
        }
        var s = DS4Log.line(Self.logTag, "sidecar \((model.path as NSString).lastPathComponent) — "
            + "\(model.tensors.count) tensori, \(model.size >> 20) MB")
        s += "\n" + line("eh_proj      ", inventory.ehProj)
        s += "\n" + line("embed_tokens ", inventory.embedTokens)
        s += "\n" + line("enorm        ", inventory.enorm)
        s += "\n" + line("hnorm        ", inventory.hnorm)
        s += "\n" + line("head norm    ", inventory.sharedHeadNorm)
        s += "\n" + line("shared head  ", inventory.sharedHead)
        if !inventory.blockTensors.isEmpty {
            s += "\n    blocco transformer MTP (\(inventory.blockTensors.count) tensori):"
            for t in inventory.blockTensors {
                s += "\n      \(t.name)  \(t.typeName)  [\(shape(t))]"
            }
        }

        var checks: [String] = []
        if vocab > 0, nEmbd > 0 {
            if let e = inventory.embedTokens {
                let d = e.dims.map { Int($0) }
                if !d.contains(vocab) { checks.append("embed_tokens non contiene vocab=\(vocab): [\(shape(e))]") }
                if !d.contains(nEmbd) { checks.append("embed_tokens non contiene nEmbd=\(nEmbd): [\(shape(e))]") }
            }
            if let p = inventory.ehProj {
                let d = p.dims.map { Int($0) }
                if !(d.contains(2 * nEmbd) && d.contains(nEmbd)) {
                    checks.append("eh_proj attesa [2·nEmbd=\(2 * nEmbd) × nEmbd=\(nEmbd)], trovata [\(shape(p))]")
                }
            }
            if let h = inventory.sharedHead {
                let d = h.dims.map { Int($0) }
                if !d.contains(vocab) { checks.append("shared_head non contiene vocab=\(vocab): [\(shape(h))]") }
            }
        }
        let missing = [inventory.ehProj == nil ? "eh_proj" : nil,
                       inventory.embedTokens == nil ? "embed_tokens" : nil,
                       inventory.enorm == nil ? "enorm" : nil,
                       inventory.hnorm == nil ? "hnorm" : nil,
                       inventory.sharedHead == nil ? "shared_head" : nil].compactMap { $0 }
        if !missing.isEmpty { checks.append("ruoli d'interfaccia mancanti: " + missing.joined(separator: ", ")) }
        s += checks.isEmpty
            ? "\n    interfaccia: COMPLETA e coerente con vocab/nEmbd — pronta per il forward (Fase M2)"
            : "\n    interfaccia: DA VERIFICARE —\n      " + checks.joined(separator: "\n      ")
        return s
    }
}

// MARK: - DSpark support model

/// Metadata and tensor contract for DeepSeek V4 Flash DSpark support GGUFs.
///
/// DSpark is deliberately represented separately from the legacy one-stage
/// MTP sidecar above.  Both artifacts use the `mtp.N.*` tensor namespace, but
/// DSpark contains multiple DeepSpec stages, target-hidden projections, a
/// Markov block head and a confidence head.  Treating it as an `MTPSidecar`
/// would therefore accept a file that the runtime cannot execute correctly.
///
/// This type performs the same strict metadata/tensor-layout gate as upstream
/// ds4 before any weight is uploaded to Metal.  `DSparkStage0Runtime` below
/// consumes this contract and owns the first executable resident GPU state.
public struct DSparkSupportModel: DS4Logging {
    public static let logTag = "dspark"
    public static let maxStages = 8
    public static let maxTargetLayers = 8
    public static let maxBlockSize = 16

    public struct Metadata: Sendable, Equatable {
        public let architecture: String?
        public let name: String?
        public let blockSize: Int?
        public let markovRank: Int?
        public let noiseTokenID: Int?
        public let targetLayerIDs: [Int]?
        public let declaredStageCount: Int?
        public let discoveredStageCount: Int

        public var stageCount: Int {
            declaredStageCount ?? discoveredStageCount
        }
    }

    public struct Stage: Sendable {
        public let index: Int
        /// Keyed by the part after `mtp.<stage>.`.
        public let tensors: [String: GGUFModel.Tensor]

        public subscript(_ suffix: String) -> GGUFModel.Tensor? {
            tensors[suffix]
        }
    }

    public struct ValidationIssue: Sendable, Equatable, CustomStringConvertible {
        public enum Severity: String, Sendable {
            case error
            case warning
        }

        public let severity: Severity
        public let message: String
        public var description: String { "\(severity.rawValue): \(message)" }
    }

    public struct Validation: Sendable {
        public let issues: [ValidationIssue]
        public var errors: [ValidationIssue] { issues.filter { $0.severity == .error } }
        public var warnings: [ValidationIssue] { issues.filter { $0.severity == .warning } }
        public var isRunnable: Bool { errors.isEmpty }
    }

    public let model: GGUFModel
    public let metadata: Metadata
    public let stages: [Stage]
    public let validation: Validation
    /// True when the backing GGUF uses MAP_SHARED and can safely be exposed to
    /// Metal through `makeBuffer(bytesNoCopy:)`.
    public private(set) var metalMappingEnabled: Bool

    /// A layout-valid support model can be wired into the executable runtime.
    public var isRunnable: Bool { validation.isRunnable }

    public func isRunnable(withMainModelPath path: String) -> Bool {
        validation.isRunnable && checkpointCompatibilityIssue(mainModelPath: path) == nil
    }

    /// Locate the checkpoint-matched canonical support model next to the main
    /// GGUF. Never silently mix the 0730 and 0731 checkpoints: they have
    /// identical geometry, so a shape-only check cannot catch that mismatch.
    public static func locate(near mainModelPath: String) -> String? {
        let dir = (mainModelPath as NSString).deletingLastPathComponent
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        let mainIs0731 = canonicalCheckpointIs0731(mainModelPath)
        let canonical = mainIs0731
            ? "DeepSeek-V4-Flash-DSpark-support-0731.gguf"
            : "DeepSeek-V4-Flash-DSpark-support.gguf"
        if names.contains(canonical) { return dir + "/" + canonical }

        // Custom conversions are allowed, but only when their filename does
        // not positively identify the other checkpoint.
        let candidates = names.filter {
            let lower = $0.lowercased()
            guard lower.contains("dspark"), lower.hasSuffix(".gguf") else { return false }
            return lower.contains("0731") == mainIs0731
        }.sorted()
        return candidates.first.map { dir + "/" + $0 }
    }

    public init(path: String, targetDims: DSV4Dims, targetLayerCount: Int,
                metalMapping: Bool = false) throws {
        let model = try GGUFModel(path: path, metalMapping: metalMapping, prefetchCPU: false)
        self.init(model: model, targetDims: targetDims, targetLayerCount: targetLayerCount)
        self.metalMappingEnabled = metalMapping
    }

    public init(model: GGUFModel, targetDims: DSV4Dims, targetLayerCount: Int) {
        let discovered = Self.stageTensors(model)
        let discoveredCount = (discovered.keys.max().map { $0 + 1 }) ?? 0
        let declared = Self.firstU32(model, ["dspark.stage_count", "dspark.n_layers"])
        let metadata = Metadata(
            architecture: model.string("general.architecture"),
            name: model.string("general.name"),
            blockSize: Self.firstU32(model, [
                "deepseek4.dspark.block_size", "deepseek4.dspark_block_size",
                "dspark.block_size",
            ]),
            markovRank: Self.firstU32(model, [
                "deepseek4.dspark.markov_rank", "deepseek4.dspark_markov_rank",
                "dspark.markov_rank",
            ]),
            noiseTokenID: Self.firstU32(model, [
                "deepseek4.dspark.noise_token_id", "deepseek4.dspark_noise_token_id",
                "dspark.noise_token_id",
            ]),
            targetLayerIDs: Self.firstIntArray(model, [
                "deepseek4.dspark.target_layer_ids", "deepseek4.dspark_target_layer_ids",
                "dspark.target_layer_ids",
            ]),
            declaredStageCount: declared,
            discoveredStageCount: discoveredCount
        )
        let stages = (0..<max(metadata.stageCount, discoveredCount)).map {
            Stage(index: $0, tensors: discovered[$0] ?? [:])
        }

        self.model = model
        self.metadata = metadata
        self.stages = stages
        self.validation = Validation(issues: Self.validate(
            metadata: metadata,
            stages: stages,
            targetDims: targetDims,
            targetLayerCount: targetLayerCount
        ))
        self.metalMappingEnabled = false
    }

    /// Check the only compatibility signal currently published outside the
    /// tensor geometry: the 0730/0731 naming in canonical file
    /// names. Explicitly named custom files remain the caller's responsibility.
    public func checkpointCompatibilityIssue(mainModelPath: String) -> String? {
        let supportName = (model.path as NSString).lastPathComponent.lowercased()
        let mainName = (mainModelPath as NSString).lastPathComponent.lowercased()
        guard supportName.contains("dspark-support") else { return nil }
        let support0731 = supportName.contains("0731")
        let main0731 = mainName.contains("0731")
        guard support0731 != main0731 else { return nil }
        return "checkpoint non compatibile: modello principale "
            + (main0731 ? "0731" : "0730") + ", supporto DSpark "
            + (support0731 ? "0731" : "0730")
    }

    public func report(mainModelPath: String? = nil) -> String {
        let sizeGiB = Double(model.size) / Double(1 << 30)
        var lines = [DS4Log.line(Self.logTag,
            "supporto \((model.path as NSString).lastPathComponent) — "
                + "\(metadata.stageCount) stadi, \(model.tensors.count) tensori, "
                + String(format: "%.2f GiB", sizeGiB))]
        lines.append("    architettura=\(metadata.architecture ?? "MANCANTE") "
            + "block=\(metadata.blockSize.map(String.init) ?? "MANCANTE") "
            + "markov=\(metadata.markovRank.map(String.init) ?? "MANCANTE") "
            + "noise=\(metadata.noiseTokenID.map(String.init) ?? "MANCANTE")")
        let targetLayers = metadata.targetLayerIDs?.map(String.init).joined(separator: ",")
            ?? "MANCANTI"
        lines.append("    target layers=[\(targetLayers)]")
        if let mainModelPath, let issue = checkpointCompatibilityIssue(mainModelPath: mainModelPath) {
            lines.append("    ERRORE: \(issue)")
        }
        for issue in validation.issues {
            let prefix = issue.severity == .error ? "ERRORE" : "avviso"
            lines.append("    \(prefix): \(issue.message)")
        }
        if validation.isRunnable,
           mainModelPath.map({ checkpointCompatibilityIssue(mainModelPath: $0) == nil }) ?? true {
            lines.append("    contratto GGUF: COMPLETO — compatibile con il runtime DSpark")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: validation

    private enum LayoutKind {
        case f32, plain, dense, routed
    }

    private struct Rule {
        let suffix: String
        let kind: LayoutKind
        let dims: [UInt64]
    }

    private static func validate(metadata m: Metadata, stages: [Stage],
                                 targetDims d: DSV4Dims,
                                 targetLayerCount: Int) -> [ValidationIssue] {
        var issues: [ValidationIssue] = []
        func error(_ message: String) {
            issues.append(.init(severity: .error, message: message))
        }

        if m.architecture != "deepseek4-dspark" {
            error("general.architecture attesa deepseek4-dspark, trovata \(m.architecture ?? "MANCANTE")")
        }
        if m.blockSize == nil || m.blockSize == 0
            || (m.blockSize ?? 0) > maxBlockSize {
            error("dspark.block_size mancante o fuori 1...\(maxBlockSize)")
        }
        if m.markovRank == nil || m.markovRank == 0 {
            error("dspark.markov_rank mancante o zero")
        }
        if let noise = m.noiseTokenID {
            if noise < 0 || noise >= d.vocab {
                error("dspark.noise_token_id \(noise) fuori dal vocabolario 0..<\(d.vocab)")
            }
        } else {
            error("dspark.noise_token_id mancante")
        }

        if let targetLayers = m.targetLayerIDs, !targetLayers.isEmpty {
            if targetLayers.count > maxTargetLayers {
                error("troppi target layer: \(targetLayers.count), massimo \(maxTargetLayers)")
            }
            var previous: Int?
            for layer in targetLayers {
                if layer < 0 || layer >= targetLayerCount {
                    error("target layer \(layer) fuori da 0..<\(targetLayerCount)")
                }
                if let previous, layer <= previous {
                    error("i target layer non sono strettamente crescenti")
                    break
                }
                previous = layer
            }
        } else {
            error("dspark.target_layer_ids mancante o vuoto")
        }

        if m.stageCount <= 0 || m.stageCount > maxStages {
            error("numero stadi \(m.stageCount) fuori 1...\(maxStages)")
        }
        if let declared = m.declaredStageCount,
           declared != m.discoveredStageCount {
            error("dspark.stage_count=\(declared), ma i tensori descrivono \(m.discoveredStageCount) stadi")
        }
        return validateTensors(metadata: m, stages: stages, dims: d,
                               existing: issues)
    }

    private static func validateTensors(metadata m: Metadata, stages: [Stage],
                                        dims d: DSV4Dims,
                                        existing: [ValidationIssue]) -> [ValidationIssue] {
        var issues = existing
        func error(_ message: String) {
            issues.append(.init(severity: .error, message: message))
        }

        guard m.stageCount > 0, m.stageCount <= maxStages else { return issues }
        let expectedStages = m.stageCount
        for stageIndex in 0..<expectedStages {
            guard stageIndex < stages.count else {
                error("stadio mtp.\(stageIndex) mancante")
                continue
            }
            let stage = stages[stageIndex]
            for rule in blockRules(d) {
                validate(rule, in: stage, error: error)
            }
            if let gate = stage["ffn_gate_exps.weight"],
               let up = stage["ffn_up_exps.weight"], gate.type != up.type {
                error("mtp.\(stageIndex): gate/up routed usano quantizzazioni diverse")
            }
        }

        if let first = stages.first {
            let targetCount = m.targetLayerIDs?.count ?? 0
            validate(.init(suffix: "main_proj.weight", kind: .dense,
                           dims: [UInt64(targetCount * d.nEmbd), UInt64(d.nEmbd)]),
                     in: first, error: error)
            validate(.init(suffix: "main_norm.weight", kind: .f32,
                           dims: [UInt64(d.nEmbd)]), in: first, error: error)
        }
        if expectedStages - 1 < stages.count {
            let final = stages[expectedStages - 1]
            let rank = max(0, m.markovRank ?? 0)
            let hcDim = d.nHC * d.nEmbd
            for rule in [
                Rule(suffix: "norm.weight", kind: .f32, dims: [UInt64(d.nEmbd)]),
                Rule(suffix: "hc_head_base.weight", kind: .f32, dims: [UInt64(d.nHC)]),
                Rule(suffix: "hc_head_fn.weight", kind: .plain,
                     dims: [UInt64(hcDim), UInt64(d.nHC)]),
                Rule(suffix: "hc_head_scale.weight", kind: .f32, dims: [1]),
                Rule(suffix: "markov_head.markov_w1.weight", kind: .dense,
                     dims: [UInt64(rank), UInt64(d.vocab)]),
                Rule(suffix: "markov_head.markov_w2.weight", kind: .dense,
                     dims: [UInt64(rank), UInt64(d.vocab)]),
                Rule(suffix: "confidence_head.proj.weight", kind: .dense,
                     dims: [UInt64(d.nEmbd + rank), 1]),
            ] {
                validate(rule, in: final, error: error)
            }
        }
        return issues
    }

    private static func validate(_ rule: Rule, in stage: Stage,
                                 error: (String) -> Void) {
        guard let tensor = stage[rule.suffix] else {
            error("mtp.\(stage.index).\(rule.suffix) mancante")
            return
        }
        if tensor.dims != rule.dims {
            error("\(tensor.name): forma \(shape(tensor.dims)), attesa \(shape(rule.dims))")
        }
        guard typeMatches(tensor.type, rule.kind) else {
            error("\(tensor.name): tipo \(tensor.typeName) non supportato")
            return
        }
    }

    private static func blockRules(_ d: DSV4Dims) -> [Rule] {
        let hcDim = d.nHC * d.nEmbd
        let hcMix = 2 * d.nHC + d.nHC * d.nHC
        let qDim = d.nHead * d.headDim
        let attentionGroup = d.headDim * (d.nHead / d.nOutGroup)
        let outputLow = d.nOutGroup * d.nLoraO
        return [
            .init(suffix: "hc_attn_fn.weight", kind: .plain, dims: [UInt64(hcDim), UInt64(hcMix)]),
            .init(suffix: "hc_attn_scale.weight", kind: .f32, dims: [3]),
            .init(suffix: "hc_attn_base.weight", kind: .f32, dims: [UInt64(hcMix)]),
            .init(suffix: "attn_norm.weight", kind: .f32, dims: [UInt64(d.nEmbd)]),
            .init(suffix: "attn_q_a.weight", kind: .dense, dims: [UInt64(d.nEmbd), UInt64(d.qRank)]),
            .init(suffix: "attn_q_a_norm.weight", kind: .f32, dims: [UInt64(d.qRank)]),
            .init(suffix: "attn_q_b.weight", kind: .dense, dims: [UInt64(d.qRank), UInt64(qDim)]),
            .init(suffix: "attn_kv.weight", kind: .dense, dims: [UInt64(d.nEmbd), UInt64(d.headDim)]),
            .init(suffix: "attn_kv_a_norm.weight", kind: .f32, dims: [UInt64(d.headDim)]),
            .init(suffix: "attn_sinks.weight", kind: .f32, dims: [UInt64(d.nHead)]),
            .init(suffix: "attn_output_a.weight", kind: .dense,
                  dims: [UInt64(attentionGroup), UInt64(outputLow)]),
            .init(suffix: "attn_output_b.weight", kind: .dense,
                  dims: [UInt64(outputLow), UInt64(d.nEmbd)]),
            .init(suffix: "hc_ffn_fn.weight", kind: .plain, dims: [UInt64(hcDim), UInt64(hcMix)]),
            .init(suffix: "hc_ffn_scale.weight", kind: .f32, dims: [3]),
            .init(suffix: "hc_ffn_base.weight", kind: .f32, dims: [UInt64(hcMix)]),
            .init(suffix: "ffn_norm.weight", kind: .f32, dims: [UInt64(d.nEmbd)]),
            .init(suffix: "ffn_gate_inp.weight", kind: .dense,
                  dims: [UInt64(d.nEmbd), UInt64(d.nExperts)]),
            .init(suffix: "exp_probs_b.bias", kind: .f32, dims: [UInt64(d.nExperts)]),
            .init(suffix: "ffn_gate_exps.weight", kind: .routed,
                  dims: [UInt64(d.nEmbd), UInt64(d.expertFfn), UInt64(d.nExperts)]),
            .init(suffix: "ffn_up_exps.weight", kind: .routed,
                  dims: [UInt64(d.nEmbd), UInt64(d.expertFfn), UInt64(d.nExperts)]),
            .init(suffix: "ffn_down_exps.weight", kind: .routed,
                  dims: [UInt64(d.expertFfn), UInt64(d.nEmbd), UInt64(d.nExperts)]),
            .init(suffix: "ffn_gate_shexp.weight", kind: .dense,
                  dims: [UInt64(d.nEmbd), UInt64(d.sharedFfn)]),
            .init(suffix: "ffn_up_shexp.weight", kind: .dense,
                  dims: [UInt64(d.nEmbd), UInt64(d.sharedFfn)]),
            .init(suffix: "ffn_down_shexp.weight", kind: .dense,
                  dims: [UInt64(d.sharedFfn), UInt64(d.nEmbd)]),
        ]
    }

    private static func typeMatches(_ type: UInt32, _ kind: LayoutKind) -> Bool {
        switch kind {
        case .f32: return type == 0
        case .plain: return type == 0 || type == 1
        case .dense: return type == 0 || type == 1 || type == 8
        // MXFP4 (GGUF 39) is intentionally excluded until DS4Core can compute
        // its block byte size and the Swift Metal path can execute it.
        case .routed: return [8, 10, 12, 13, 14, 16].contains(type)
        }
    }

    private static func stageTensors(_ model: GGUFModel)
        -> [Int: [String: GGUFModel.Tensor]] {
        var stages: [Int: [String: GGUFModel.Tensor]] = [:]
        for tensor in model.tensors {
            guard let parsed = parseStageTensorName(tensor.name) else { continue }
            if stages[parsed.stage] == nil { stages[parsed.stage] = [:] }
            stages[parsed.stage]?[parsed.suffix] = tensor
        }
        return stages
    }

    private static func parseStageTensorName(_ name: String)
        -> (stage: Int, suffix: String)? {
        guard name.hasPrefix("mtp.") else { return nil }
        let rest = name.dropFirst(4)
        guard let dot = rest.firstIndex(of: "."),
              let stage = Int(rest[..<dot]), stage >= 0 else { return nil }
        let suffix = String(rest[rest.index(after: dot)...])
        return suffix.isEmpty ? nil : (stage, suffix)
    }

    private static func firstU32(_ model: GGUFModel, _ keys: [String]) -> Int? {
        keys.lazy.compactMap { model.u32($0).map(Int.init) }.first
    }

    private static func firstIntArray(_ model: GGUFModel, _ keys: [String]) -> [Int]? {
        for key in keys {
            if let values = model.intArray(key) {
                return values.compactMap { Int(exactly: $0) }
            }
        }
        return nil
    }

    private static func shape(_ dims: [UInt64]) -> String {
        "[" + dims.map(String.init).joined(separator: "×") + "]"
    }

    private static func canonicalCheckpointIs0731(_ path: String) -> Bool {
        (path as NSString).lastPathComponent.lowercased().contains("0731")
    }
}

// MARK: - DSpark mapped transformer weights

/// Optional output heads carried only by the last DSpark stage.
public struct DSparkFinalHeadWeights {
    public let norm: GPUTensor
    public let hcHeadBase: GPUTensor
    public let hcHeadFn: GPUTensor
    public let hcHeadScale: GPUTensor
    public let markovW1: GPUTensor
    public let markovW2: GPUTensor
    public let confidenceProjection: GPUTensor
}

/// Kernel-ready weight binding for one `mtp.N` transformer block.
///
/// Large projections and the full expert arrays remain zero-copy views of the
/// support GGUF. Small F32 norms/scales/biases are copied once into anonymous
/// Metal buffers, so they cannot repeatedly fault under memory pressure. The
/// retained owner guarantees that the mmap outlives every no-copy MTLBuffer.
public final class DSparkMappedStageWeights {
    public let index: Int
    public let block: LayerWeights
    public let tensorTypes: [String: UInt32]
    public let mappedBytes: Int
    public let residentBytes: Int
    public let finalHead: DSparkFinalHeadWeights?

    private let modelOwner: GGUFModel

    fileprivate init(index: Int, block: LayerWeights,
                     tensorTypes: [String: UInt32], mappedBytes: Int,
                     residentBytes: Int, finalHead: DSparkFinalHeadWeights?,
                     modelOwner: GGUFModel) {
        self.index = index
        self.block = block
        self.tensorTypes = tensorTypes
        self.mappedBytes = mappedBytes
        self.residentBytes = residentBytes
        self.finalHead = finalHead
        self.modelOwner = modelOwner
    }

    public func type(of suffix: String) -> UInt32? { tensorTypes[suffix] }
}

extension DSparkSupportModel {
    /// Bind one validated support transformer without duplicating its large
    /// weights. This does not execute the stage and does not alter target decode.
    public func loadMappedStageWeights(_ rt: MetalRuntime, stage stageIndex: Int) throws
        -> DSparkMappedStageWeights {
        guard validation.isRunnable else {
            throw MetalError.unsupported("contratto DSpark non valido")
        }
        guard metalMappingEnabled else {
            throw MetalError.unsupported(
                "il supporto DSpark deve essere aperto con metalMapping=true")
        }
        guard stages.indices.contains(stageIndex) else {
            throw MetalError.unsupported("stadio DSpark \(stageIndex) fuori range")
        }
        let prefix = "mtp.\(stageIndex)."
        let stage = stages[stageIndex]

        func descriptor(_ suffix: String) throws -> GGUFModel.Tensor {
            guard let tensor = stage[suffix] else {
                throw GGUFWeights.LoadError.missing(prefix + suffix)
            }
            guard tensor.bytes <= UInt64(Int.max) else {
                throw MetalError.unsupported("tensore DSpark troppo grande: \(tensor.name)")
            }
            return tensor
        }
        func mapped(_ suffix: String) throws -> GPUTensor {
            let tensor = try descriptor(suffix)
            return try GPUTensor.mappedNoCopy(
                rt,
                ptr: model.mapBase + Int(tensor.absOffset),
                byteLength: Int(tensor.bytes),
                elementCount: Int(tensor.bytes))
        }
        func resident(_ suffix: String) throws -> GPUTensor {
            let tensor = try descriptor(suffix)
            return try GPUTensor.raw(
                rt,
                ptr: model.mapBase + Int(tensor.absOffset),
                byteLength: Int(tensor.bytes),
                elementCount: Int(tensor.bytes))
        }

        let mappedSuffixes = [
            "hc_attn_fn.weight", "attn_q_a.weight", "attn_q_b.weight",
            "attn_kv.weight", "attn_output_a.weight", "attn_output_b.weight",
            "hc_ffn_fn.weight", "ffn_gate_inp.weight",
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
        ]
        let residentSuffixes = [
            "hc_attn_scale.weight", "hc_attn_base.weight", "attn_norm.weight",
            "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "attn_sinks.weight",
            "hc_ffn_scale.weight", "hc_ffn_base.weight", "ffn_norm.weight",
            "exp_probs_b.bias",
        ]

        var block = LayerWeights(
            hcAttnFn: try mapped("hc_attn_fn.weight"),
            attnScale: try resident("hc_attn_scale.weight"),
            attnBase: try resident("hc_attn_base.weight"),
            attnNorm: try resident("attn_norm.weight"),
            qA: try mapped("attn_q_a.weight"),
            qANorm: try resident("attn_q_a_norm.weight"),
            qB: try mapped("attn_q_b.weight"),
            kvW: try mapped("attn_kv.weight"),
            kvNorm: try resident("attn_kv_a_norm.weight"),
            attnSinks: try resident("attn_sinks.weight"),
            attnOutA: try mapped("attn_output_a.weight"),
            attnOut: try mapped("attn_output_b.weight"),
            hcFfnFn: try mapped("hc_ffn_fn.weight"),
            ffnScale: try resident("hc_ffn_scale.weight"),
            ffnBase: try resident("hc_ffn_base.weight"),
            ffnNorm: try resident("ffn_norm.weight"),
            sharedGate: try mapped("ffn_gate_shexp.weight"),
            sharedUp: try mapped("ffn_up_shexp.weight"),
            sharedDown: try mapped("ffn_down_shexp.weight"),
            routerW: try mapped("ffn_gate_inp.weight"),
            expGate: try mapped("ffn_gate_exps.weight"),
            expUp: try mapped("ffn_up_exps.weight"),
            expDown: try mapped("ffn_down_exps.weight"))
        block.expBias = try resident("exp_probs_b.bias")
        block.tid2eid = nil
        block.tid2eidRows = 0
        guard let gateQuant = MoEQuant.from(
                ggufType: try descriptor("ffn_gate_exps.weight").type),
              let upQuant = MoEQuant.from(
                ggufType: try descriptor("ffn_up_exps.weight").type),
              let downQuant = MoEQuant.from(
                ggufType: try descriptor("ffn_down_exps.weight").type) else {
            throw MetalError.unsupported(
                "mtp.\(stageIndex): quantizzazione esperti non eseguibile in Swift")
        }
        block.gateQuant = gateQuant
        block.upQuant = upQuant
        block.downQuant = downQuant
        var mappedBytes = try mappedSuffixes.reduce(0) {
            $0 + Int(try descriptor($1).bytes)
        }
        var residentBytes = try residentSuffixes.reduce(0) {
            $0 + Int(try descriptor($1).bytes)
        }
        var finalHead: DSparkFinalHeadWeights?
        if stageIndex == metadata.stageCount - 1 {
            let finalMapped = [
                "hc_head_fn.weight", "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight", "confidence_head.proj.weight",
            ]
            let finalResident = [
                "norm.weight", "hc_head_base.weight", "hc_head_scale.weight",
            ]
            finalHead = DSparkFinalHeadWeights(
                norm: try resident("norm.weight"),
                hcHeadBase: try resident("hc_head_base.weight"),
                hcHeadFn: try mapped("hc_head_fn.weight"),
                hcHeadScale: try resident("hc_head_scale.weight"),
                markovW1: try mapped("markov_head.markov_w1.weight"),
                markovW2: try mapped("markov_head.markov_w2.weight"),
                confidenceProjection: try mapped("confidence_head.proj.weight"))
            mappedBytes += try finalMapped.reduce(0) {
                $0 + Int(try descriptor($1).bytes)
            }
            residentBytes += try finalResident.reduce(0) {
                $0 + Int(try descriptor($1).bytes)
            }
        }

        return DSparkMappedStageWeights(
            index: stageIndex,
            block: block,
            tensorTypes: stage.tensors.mapValues(\.type),
            mappedBytes: mappedBytes,
            residentBytes: residentBytes,
            finalHead: finalHead,
            modelOwner: model)
    }
}

/// Logical window shared by the private raw-KV rings of all DSpark stages.
/// Absolute token positions determine physical rows (`position % capacity`),
/// so cropping after verification never copies KV data: it only moves the
/// logical frontier. This mirrors upstream's cache set/crop/append contract.
public struct DSparkKVWindow: Sendable, Equatable {
    public let capacity: Int
    public private(set) var tokenStart = 0
    public private(set) var length = 0

    public init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
    }

    public var physicalStart: Int { length == 0 ? 0 : tokenStart % capacity }
    public var tokenEnd: Int? {
        guard tokenStart <= Int.max - length else { return nil }
        return tokenStart + length
    }

    @discardableResult
    public mutating func set(tokenStart: Int, length: Int) -> Bool {
        guard tokenStart >= 0, length >= 0, length <= capacity,
              tokenStart <= Int.max - length else { return false }
        self.tokenStart = length == 0 ? 0 : tokenStart
        self.length = length
        return true
    }

    public mutating func reset() {
        tokenStart = 0
        length = 0
    }

    /// Keep only KV rows strictly before `prefixLength`. A prefix outside the
    /// current logical interval invalidates the window, matching the C runtime.
    @discardableResult
    public mutating func crop(toPrefixLength prefixLength: Int) -> Bool {
        guard prefixLength >= 0 else { return false }
        guard length > 0 else { return true }
        guard let end = tokenEnd else { return false }
        if prefixLength <= tokenStart || prefixLength > end {
            reset()
        } else {
            length = prefixLength - tokenStart
        }
        return true
    }

    public func ends(at position: Int) -> Bool {
        guard position >= 0 else { return false }
        if length == 0 { return true }
        return tokenEnd == position
    }

    /// Claim the row just written at `position`. When full, the ring drops the
    /// oldest row and advances its absolute start without moving bytes.
    @discardableResult
    public mutating func appendRow(at position: Int) -> Bool {
        guard length > 0, ends(at: position) else { return false }
        length += 1
        if length > capacity {
            let excess = length - capacity
            guard tokenStart <= Int.max - excess else { return false }
            tokenStart += excess
            length = capacity
        }
        return true
    }
}

/// Pure setup contract for one DSpark proposal block.  Keeping the arithmetic
/// outside the Metal executor makes the cache/ring boundary independently
/// testable (and prevents a malformed speculative block from reaching a GPU
/// precondition).
struct DSparkStageBlockPlan: Sendable, Equatable {
    let position: Int
    let draftTokens: [Int]
    let positionIDs: [Int]
    let supportLength: Int
    let attentionRawStart: Int
    let appendPosition: Int
    let visibleRows: Int

    var draftCount: Int { draftTokens.count }
    var rowCount: Int { draftCount + 1 }

    static func make(currentToken: Int, noiseToken: Int, position: Int,
                     blockSize: Int, vocabularySize: Int,
                     window: DSparkKVWindow) -> Self? {
        guard currentToken >= 0, currentToken < vocabularySize,
              noiseToken >= 0, noiseToken < vocabularySize,
              position >= 0, blockSize > 0,
              position <= Int.max - blockSize,
              window.ends(at: position),
              blockSize + 1 <= window.capacity - window.length else {
            return nil
        }
        let tokens = [currentToken]
            + Array(repeating: noiseToken, count: blockSize - 1)
        let positions = [position] + (0..<blockSize).map { position + $0 }
        let rawStart = window.length > 0 ? window.tokenStart : position
        let append = window.length > 0
            ? (window.tokenStart + window.length) % window.capacity
            : position % window.capacity
        return Self(position: position, draftTokens: tokens,
                    positionIDs: positions, supportLength: window.length,
                    attentionRawStart: rawStart,
                    appendPosition: append,
                    visibleRows: window.length + blockSize + 1)
    }
}

/// Scratch and graph encoder for the three DSpark transformer stages.
///
/// The support checkpoint keeps all 256 routed experts for every stage in one
/// mmap-backed tensor.  DSpark evaluates five draft rows together, therefore
/// the existing prefill MM kernels are the right execution shape: each expert
/// tile is read once for the whole proposal instead of once per draft token.
/// This object owns only a few MiB of reusable scratch; stage weights and raw
/// KV rings remain owned by `DSparkStage0Runtime`.
private final class DSparkTransformerExecutor {
    private static let argmaxPartials = 64

    let draftCount: Int
    let rowCount: Int
    let stageInputHC: GPUTensor
    let stageOutputHC: GPUTensor

    private let rt: MetalRuntime
    private let d: DSV4Dims
    private let noiseToken: Int
    private let stage: StreamingDecoder.PrefillStage
    private let embeddingScratch: GPUTensor
    private let targetLayerCount: Int
    private let cacheCapacity: Int
    private let seedCapacity: Int
    private let seedPacked: GPUTensor
    private let seedMain: GPUTensor
    private let seedMainHC: GPUTensor
    private let seedFlatHC: GPUTensor
    private let seedMix: GPUTensor
    private let seedSplit: GPUTensor
    private let seedEmbd: GPUTensor
    private let seedAttnNorm: GPUTensor
    private let seedKV: GPUTensor
    private let markovRank: Int
    private let finalFlat: GPUTensor
    private let finalPre: GPUTensor
    private let finalWeights: GPUTensor
    private let finalWeightTmp: GPUTensor
    private let finalEmbd: GPUTensor
    private let finalNorm: GPUTensor
    private let baseLogits: GPUTensor
    private let markovState: GPUTensor
    private let markovBias: GPUTensor
    private let correctedLogits: GPUTensor
    private let confidenceFeatures: GPUTensor
    private let confidenceLogit: GPUTensor
    private let argmaxPartialValues: GPUTensor
    private let argmaxPartialIndices: GPUTensor
    private let argmaxResult: GPUTensor
    private let rope = RopeParams(
        nCtxOrig: 0,
        freqBase: DSV4Shape.ropeFreqBase,
        freqScale: 1,
        extFactor: 0,
        attnFactor: 1,
        betaFast: DSV4Shape.ropeBetaFast,
        betaSlow: DSV4Shape.ropeBetaSlow)

    init(rt: MetalRuntime, dims d: DSV4Dims, blockSize: Int,
         noiseToken: Int, markovRank: Int, cacheCapacity: Int,
         targetLayerCount: Int,
         stages: [DSparkMappedStageWeights]) throws {
        guard d.nHC == 4, d.headDim == 512, d.k == 6,
              d.nExperts == 256, d.qDim == d.nHead * d.headDim,
              d.nEmbd % 256 == 0, d.expertFfn % 256 == 0,
              blockSize > 0, blockSize + 1 <= cacheCapacity,
              markovRank > 0, markovRank % 32 == 0,
              targetLayerCount > 0 else {
            throw MetalError.unsupported(
                "geometria DSpark non compatibile con i kernel Metal batch")
        }
        guard !stages.isEmpty else {
            throw MetalError.unsupported("supporto DSpark senza transformer")
        }
        for weights in stages {
            try Self.validateKernelContract(weights, dims: d)
        }
        guard let finalStage = stages.last,
              finalStage.finalHead != nil else {
            throw MetalError.unsupported("supporto DSpark senza teste finali")
        }
        try Self.validateFinalHeadContract(finalStage)

        self.rt = rt
        self.d = d
        self.noiseToken = noiseToken
        self.markovRank = markovRank
        self.targetLayerCount = targetLayerCount
        self.cacheCapacity = cacheCapacity
        self.draftCount = blockSize
        self.rowCount = blockSize + 1
        let hcDim = d.nHC * d.nEmbd
        self.stageInputHC = try .zeros(
            rt, floatCount: (blockSize + 1) * hcDim)
        self.stageOutputHC = try .zeros(
            rt, floatCount: blockSize * hcDim)
        self.embeddingScratch = try .zeros(rt, floatCount: d.nEmbd)
        self.stage = try StreamingDecoder.PrefillStage(
            rt, n: blockSize, d: d, mmPath: true,
            maxUnion: d.nExperts,
            flashBatch: (nq: blockSize + 1, maxKv: cacheCapacity))
        guard stage.flash != nil, stage.mm != nil else {
            throw MetalError.unsupported("scratch DSpark batch non disponibile")
        }

        // Cache seeding is a one-shot operation after prompt prefill. A bounded
        // 128-row tile avoids committing hundreds of MiB of temporary HC
        // matrices on 16-GB Macs while still amortizing each support weight
        // over many target rows.
        let seedCapacity = min(cacheCapacity, 128)
        self.seedCapacity = seedCapacity
        self.seedPacked = try .zeros(
            rt, floatCount: seedCapacity * targetLayerCount * d.nEmbd)
        self.seedMain = try .zeros(rt, floatCount: seedCapacity * d.nEmbd)
        self.seedMainHC = try .zeros(rt, floatCount: seedCapacity * hcDim)
        self.seedFlatHC = try .zeros(rt, floatCount: seedCapacity * hcDim)
        self.seedMix = try .zeros(rt, floatCount: seedCapacity * 24)
        self.seedSplit = try .zeros(rt, floatCount: seedCapacity * 24)
        self.seedEmbd = try .zeros(rt, floatCount: seedCapacity * d.nEmbd)
        self.seedAttnNorm = try .zeros(rt, floatCount: seedCapacity * d.nEmbd)
        self.seedKV = try .zeros(rt, floatCount: seedCapacity * d.headDim)

        self.finalFlat = try .zeros(rt, floatCount: blockSize * hcDim)
        self.finalPre = try .zeros(rt, floatCount: blockSize * d.nHC)
        self.finalWeights = try .zeros(rt, floatCount: blockSize * d.nHC)
        self.finalWeightTmp = try .zeros(rt, floatCount: blockSize * d.nHC)
        self.finalEmbd = try .zeros(rt, floatCount: blockSize * d.nEmbd)
        self.finalNorm = try .zeros(rt, floatCount: blockSize * d.nEmbd)
        self.baseLogits = try .zeros(rt, floatCount: blockSize * d.vocab)
        self.markovState = try .zeros(rt, floatCount: markovRank)
        self.markovBias = try .zeros(rt, floatCount: d.vocab)
        self.correctedLogits = try .zeros(rt, floatCount: d.vocab)
        self.confidenceFeatures = try .zeros(
            rt, floatCount: d.nEmbd + markovRank)
        self.confidenceLogit = try .zeros(rt, floatCount: 1)
        self.argmaxPartialValues = try .zeros(
            rt, floatCount: Self.argmaxPartials)
        self.argmaxPartialIndices = try .zerosBytes(
            rt, byteLength: Self.argmaxPartials * MemoryLayout<Int32>.stride)
        self.argmaxResult = try .zerosBytes(
            rt, byteLength: MemoryLayout<Int32>.stride)
    }

    private static func validateKernelContract(
        _ weights: DSparkMappedStageWeights, dims d: DSV4Dims
    ) throws {
        func require(_ suffix: String, _ allowed: Set<UInt32>) throws {
            guard let type = weights.type(of: suffix), allowed.contains(type) else {
                let actual = weights.type(of: suffix).map(String.init) ?? "mancante"
                throw MetalError.unsupported(
                    "mtp.\(weights.index).\(suffix): tipo GGUF \(actual) "
                        + "non eseguibile dal forward DSpark")
            }
        }
        for suffix in ["hc_attn_fn.weight", "hc_ffn_fn.weight"] {
            try require(suffix, [0, 1])
        }
        for suffix in ["attn_q_a.weight", "attn_q_b.weight",
                       "attn_kv.weight", "attn_output_b.weight",
                       "ffn_gate_inp.weight"] {
            try require(suffix, [0, 1, 8])
        }
        // The grouped projection has a strided Q8 MM kernel; the three shared
        // projections use the same Q8 batch path as target prefill.
        try require("attn_output_a.weight", [8])
        for suffix in ["ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                       "ffn_down_shexp.weight"] {
            try require(suffix, [8])
        }
        guard weights.block.gateQuant == .iq2_xxs,
              weights.block.upQuant == .iq2_xxs,
              weights.block.downQuant == .q2_K else {
            throw MetalError.unsupported(
                "mtp.\(weights.index): esperti attesi IQ2_XXS/IQ2_XXS/Q2_K")
        }
    }

    private static func validateFinalHeadContract(
        _ weights: DSparkMappedStageWeights
    ) throws {
        func require(_ suffix: String, _ allowed: Set<UInt32>) throws {
            guard let type = weights.type(of: suffix), allowed.contains(type) else {
                let actual = weights.type(of: suffix).map(String.init) ?? "mancante"
                throw MetalError.unsupported(
                    "mtp.\(weights.index).\(suffix): tipo GGUF \(actual) "
                        + "non eseguibile dalla testa DSpark")
            }
        }
        try require("hc_head_fn.weight", [0, 1])
        try require("markov_head.markov_w1.weight", [0, 1, 8])
        try require("markov_head.markov_w2.weight", [0, 1, 8])
        try require("confidence_head.proj.weight", [0, 1, 8])
    }

    private func dense(_ graph: GraphContext, weight: GPUTensor,
                       type: UInt32, act: GPUTensor, out: GPUTensor,
                       inDim: Int, outDim: Int, rows: Int) throws {
        switch type {
        case 1:
            try graph.encodeMMDenseF16(
                weight: weight, act: act, actBase: 0, out: out,
                inDim: inDim, outDim: outDim, nTok: rows)
        case 8:
            try graph.encodeMMDenseQ8(
                weight: weight, act: act, actBase: 0, out: out,
                inDim: inDim, outDim: outDim, nTok: rows)
        case 0:
            for row in 0..<rows {
                let input = act.subview(
                    byteOffset: row * inDim * 4,
                    byteLength: inDim * 4, count: inDim)
                let output = out.subview(
                    byteOffset: row * outDim * 4,
                    byteLength: outDim * 4, count: outDim)
                try graph.matmulF32(weight: weight, x: input, out: output,
                                    inDim: inDim, outDim: outDim)
            }
        default:
            throw MetalError.unsupported(
                "proiezione DSpark tipo GGUF \(type) non eseguibile")
        }
    }

    private func dense(_ graph: GraphContext,
                       weights: DSparkMappedStageWeights, suffix: String,
                       weight: GPUTensor, act: GPUTensor, out: GPUTensor,
                       inDim: Int, outDim: Int, rows: Int) throws {
        guard let type = weights.type(of: suffix) else {
            throw MetalError.unsupported("tipo DSpark mancante: \(suffix)")
        }
        try dense(graph, weight: weight, type: type, act: act, out: out,
                  inDim: inDim, outDim: outDim, rows: rows)
    }

    private func encodeHCReduce(
        _ graph: GraphContext, _ fb: StreamingDecoder.PrefillStage.FlashBatch,
        weights: DSparkMappedStageWeights,
        x: GPUTensor, mixer: GPUTensor, mixerSuffix: String,
        scale: GPUTensor, base: GPUTensor, norm: GPUTensor,
        split: GPUTensor, output: GPUTensor, rows: Int
    ) throws {
        let hcDim = d.nHC * d.nEmbd
        try graph.rmsNorm(x, weight: nil, out: fb.flatMat,
                          rows: rows, n: hcDim, eps: ModelDefaults.rmsEps)
        try dense(graph, weights: weights, suffix: mixerSuffix,
                  weight: mixer, act: fb.flatMat, out: fb.mixMat,
                  inDim: hcDim, outDim: 24, rows: rows)
        if d.fusedHC {
            try graph.hcSplitWeightedSumNorm4(
                mix: fb.mixMat, scale: scale, base: base, x: x,
                split: split, embd: fb.embdMat, normWeight: norm,
                normOut: output, nEmbd: d.nEmbd, nRows: rows,
                sinkhornIters: d.sinkhornIterations,
                eps: ModelDefaults.hcEps, normEps: ModelDefaults.rmsEps)
        } else {
            try graph.hcSplitSinkhorn(
                mix: fb.mixMat, scale: scale, base: base, out: split,
                nRows: rows, sinkhornIters: d.sinkhornIterations,
                eps: ModelDefaults.hcEps)
            try graph.hcWeightedSum(
                x: x, weights: split, out: fb.embdMat,
                nEmbd: d.nEmbd, nHC: d.nHC, nTokens: rows,
                weightsTokenStride: 24 * 4)
            try graph.rmsNorm(
                fb.embdMat, weight: norm, out: output,
                rows: rows, n: d.nEmbd, eps: ModelDefaults.rmsEps)
        }
    }

    /// Seed every stage's private raw ring from target features captured during
    /// prompt prefill. The source is slot-major and circular; the main
    /// projection expects token-major rows, so a bounded CPU pack linearizes
    /// one tile at a time before the batched Metal graph consumes it.
    @_optimize(none)
    func seedCaches(targetHistory: GPUTensor, capturedRange: Range<Int>,
                    mainProjection: GPUTensor, mainProjectionType: UInt32,
                    mainNorm: GPUTensor,
                    weights allWeights: [DSparkMappedStageWeights],
                    rawCaches: [GPUTensor]) throws {
        guard !capturedRange.isEmpty,
              capturedRange.count <= cacheCapacity,
              allWeights.count == rawCaches.count,
              allWeights.count > 0 else {
            throw MetalError.unsupported("storico DSpark non seminabile")
        }
        let source = (targetHistory.buffer.contents() + targetHistory.byteOffset)
            .bindMemory(to: Float.self,
                        capacity: targetLayerCount * cacheCapacity * d.nEmbd)
        let destination = (seedPacked.buffer.contents() + seedPacked.byteOffset)
            .bindMemory(to: Float.self,
                        capacity: seedCapacity * targetLayerCount * d.nEmbd)
        let inputWidth = targetLayerCount * d.nEmbd
        let hcDim = d.nHC * d.nEmbd

        var offset = 0
        while offset < capturedRange.count {
            let rows = min(seedCapacity, capturedRange.count - offset)
            let position = capturedRange.lowerBound + offset
            for row in 0..<rows {
                let physical = (position + row) % cacheCapacity
                for slot in 0..<targetLayerCount {
                    let src = source + (slot * cacheCapacity + physical) * d.nEmbd
                    let dst = destination + (row * targetLayerCount + slot) * d.nEmbd
                    memcpy(dst, src, d.nEmbd * MemoryLayout<Float>.stride)
                }
            }

            let packedView = seedPacked.subview(
                byteOffset: 0, byteLength: rows * inputWidth * 4,
                count: rows * inputWidth)
            let mainView = seedMain.subview(
                byteOffset: 0, byteLength: rows * d.nEmbd * 4,
                count: rows * d.nEmbd)
            let hcView = seedMainHC.subview(
                byteOffset: 0, byteLength: rows * hcDim * 4,
                count: rows * hcDim)
            let flatView = seedFlatHC.subview(
                byteOffset: 0, byteLength: rows * hcDim * 4,
                count: rows * hcDim)
            let mixView = seedMix.subview(
                byteOffset: 0, byteLength: rows * 24 * 4,
                count: rows * 24)
            let splitView = seedSplit.subview(
                byteOffset: 0, byteLength: rows * 24 * 4,
                count: rows * 24)
            let embdView = seedEmbd.subview(
                byteOffset: 0, byteLength: rows * d.nEmbd * 4,
                count: rows * d.nEmbd)
            let normView = seedAttnNorm.subview(
                byteOffset: 0, byteLength: rows * d.nEmbd * 4,
                count: rows * d.nEmbd)
            let kvView = seedKV.subview(
                byteOffset: 0, byteLength: rows * d.headDim * 4,
                count: rows * d.headDim)

            let graph = GraphContext(rt)
            try graph.begin()
            try dense(graph, weight: mainProjection,
                      type: mainProjectionType, act: packedView,
                      out: mainView, inDim: inputWidth,
                      outDim: d.nEmbd, rows: rows)
            try graph.rmsNorm(mainView, weight: mainNorm, out: mainView,
                              rows: rows, n: d.nEmbd,
                              eps: ModelDefaults.rmsEps)
            try graph.repeatHC(src: mainView, out: hcView,
                               nEmbd: d.nEmbd, nTokens: rows, nHC: d.nHC)

            for (stageIndex, mapped) in allWeights.enumerated() {
                let w = mapped.block
                try graph.rmsNorm(hcView, weight: nil, out: flatView,
                                  rows: rows, n: hcDim,
                                  eps: ModelDefaults.rmsEps)
                try dense(graph, weights: mapped,
                          suffix: "hc_attn_fn.weight", weight: w.hcAttnFn,
                          act: flatView, out: mixView,
                          inDim: hcDim, outDim: 24, rows: rows)
                if d.fusedHC {
                    try graph.hcSplitWeightedSumNorm4(
                        mix: mixView, scale: w.attnScale,
                        base: w.attnBase, x: hcView,
                        split: splitView, embd: embdView,
                        normWeight: w.attnNorm, normOut: normView,
                        nEmbd: d.nEmbd, nRows: rows,
                        sinkhornIters: d.sinkhornIterations,
                        eps: ModelDefaults.hcEps,
                        normEps: ModelDefaults.rmsEps)
                } else {
                    try graph.hcSplitSinkhorn(
                        mix: mixView, scale: w.attnScale,
                        base: w.attnBase, out: splitView,
                        nRows: rows,
                        sinkhornIters: d.sinkhornIterations,
                        eps: ModelDefaults.hcEps)
                    try graph.hcWeightedSum(
                        x: hcView, weights: splitView, out: embdView,
                        nEmbd: d.nEmbd, nHC: d.nHC, nTokens: rows,
                        weightsTokenStride: 24 * 4)
                    try graph.rmsNorm(
                        embdView, weight: w.attnNorm, out: normView,
                        rows: rows, n: d.nEmbd,
                        eps: ModelDefaults.rmsEps)
                }
                try dense(graph, weights: mapped,
                          suffix: "attn_kv.weight", weight: w.kvW,
                          act: normView, out: kvView,
                          inDim: d.nEmbd, outDim: d.headDim, rows: rows)
                try graph.rmsNorm(
                    kvView, weight: w.kvNorm, out: kvView,
                    rows: rows, n: d.headDim,
                    eps: ModelDefaults.rmsEps)
                try graph.ropeTail(
                    x: kvView, nTok: rows, nHead: 1,
                    headDim: d.headDim, nRot: d.nRot,
                    nCtxOrig: rope.nCtxOrig, freqBase: rope.freqBase,
                    freqScale: rope.freqScale, extFactor: rope.extFactor,
                    attnFactor: rope.attnFactor, betaFast: rope.betaFast,
                    betaSlow: rope.betaSlow,
                    // Upstream seeds the complete captured prompt block at
                    // one DSpark coordinate; chunking must not change it.
                    pos0: capturedRange.lowerBound, posStep: 0)
                try graph.kvFP8StoreBatch(
                    kv: kvView, rawCache: rawCaches[stageIndex],
                    headDim: d.headDim, nRot: d.nRot,
                    pos0: position, rawRows: cacheCapacity,
                    nTok: rows)
            }
            graph.commit()
            if let error = graph.lastError { throw error }
            offset += rows
        }
    }

    /// Append the just-committed target feature to every support-stage ring.
    /// This is the Swift equivalent of upstream's `dspark_ring_maintain`: it
    /// keeps accepted/replayed target tokens, never speculative draft rows.
    func appendCapturedTarget(mainHidden: GPUTensor, position: Int,
                              weights allWeights: [DSparkMappedStageWeights],
                              rawCaches: [GPUTensor]) throws {
        guard allWeights.count == rawCaches.count,
              !allWeights.isEmpty else {
            throw MetalError.unsupported("cache DSpark non mantenibile")
        }
        let kv = seedKV.subview(
            byteOffset: 0, byteLength: d.headDim * 4, count: d.headDim)
        let graph = GraphContext(rt)
        try graph.begin()
        for (stageIndex, mapped) in allWeights.enumerated() {
            try dense(graph, weights: mapped, suffix: "attn_kv.weight",
                      weight: mapped.block.kvW, act: mainHidden, out: kv,
                      inDim: d.nEmbd, outDim: d.headDim, rows: 1)
            try graph.rmsNorm(
                kv, weight: mapped.block.kvNorm, out: kv,
                rows: 1, n: d.headDim, eps: ModelDefaults.rmsEps)
            try graph.ropeTail(
                x: kv, nTok: 1, nHead: 1,
                headDim: d.headDim, nRot: d.nRot,
                nCtxOrig: rope.nCtxOrig, freqBase: rope.freqBase,
                freqScale: rope.freqScale, extFactor: rope.extFactor,
                attnFactor: rope.attnFactor, betaFast: rope.betaFast,
                betaSlow: rope.betaSlow, pos0: position, posStep: 0)
            try graph.kvFP8StoreBatch(
                kv: kv, rawCache: rawCaches[stageIndex],
                headDim: d.headDim, nRot: d.nRot,
                pos0: position, rawRows: cacheCapacity, nTok: 1)
        }
        graph.commit()
        if let error = graph.lastError { throw error }
    }

    @_optimize(none)
    func run(currentToken: Int, position: Int, vocabularySize: Int,
             embeddingTable: GPUTensor, mainHidden: GPUTensor,
             weights allWeights: [DSparkMappedStageWeights],
             rawCaches: [GPUTensor], window: DSparkKVWindow) throws {
        guard let plan = DSparkStageBlockPlan.make(
            currentToken: currentToken, noiseToken: noiseToken,
            position: position, blockSize: draftCount,
            vocabularySize: vocabularySize, window: window) else {
            throw MetalError.unsupported(
                "blocco DSpark non compatibile con la finestra KV corrente")
        }
        guard allWeights.count == rawCaches.count,
              allWeights.count > 0,
              let fb = stage.flash, let mm = stage.mm else {
            throw MetalError.unsupported("stadi/cache DSpark incompleti")
        }

        let hcDim = d.nHC * d.nEmbd
        let hcRowBytes = hcDim * MemoryLayout<Float>.stride
        let graph = GraphContext(rt)
        try graph.begin()

        // Row 0 is the projected target feature. Rows 1...B are the current
        // token followed by the learned noise token, all repeated over HC.
        let targetRow = stageInputHC.subview(
            byteOffset: 0, byteLength: hcRowBytes, count: hcDim)
        try graph.repeatHC(src: mainHidden, out: targetRow,
                           nEmbd: d.nEmbd, nTokens: 1, nHC: d.nHC)
        for (row, token) in plan.draftTokens.enumerated() {
            let draftRow = stageInputHC.subview(
                byteOffset: (row + 1) * hcRowBytes,
                byteLength: hcRowBytes, count: hcDim)
            try graph.embedTokenHC(
                table: embeddingTable, token: token,
                embd: embeddingScratch, hc: draftRow,
                nEmbd: d.nEmbd, nVocab: vocabularySize, nHC: d.nHC)
        }

        for (stageIndex, mapped) in allWeights.enumerated() {
            let w = mapped.block
            let rawCache = rawCaches[stageIndex]
            let rawRows = rawCache.count / d.headDim
            guard rawRows == window.capacity else {
                throw MetalError.unsupported(
                    "mtp.\(stageIndex): capacità KV incoerente")
            }

            // Pre-attention HC collapse for all [target + draft] rows.
            try encodeHCReduce(
                graph, fb, weights: mapped, x: stageInputHC,
                mixer: w.hcAttnFn, mixerSuffix: "hc_attn_fn.weight",
                scale: w.attnScale, base: w.attnBase, norm: w.attnNorm,
                split: fb.splitA, output: fb.curMat, rows: rowCount)

            let draftCur = fb.curMat.subview(
                byteOffset: d.nEmbd * 4,
                byteLength: draftCount * d.nEmbd * 4,
                count: draftCount * d.nEmbd)
            try dense(graph, weights: mapped, suffix: "attn_q_a.weight",
                      weight: w.qA, act: draftCur, out: fb.qrMat,
                      inDim: d.nEmbd, outDim: d.qRank, rows: draftCount)
            try graph.rmsNorm(
                fb.qrMat, weight: w.qANorm, out: fb.qrNormMat,
                rows: draftCount, n: d.qRank, eps: ModelDefaults.rmsEps)
            try dense(graph, weights: mapped, suffix: "attn_q_b.weight",
                      weight: w.qB, act: fb.qrNormMat, out: fb.qMat,
                      inDim: d.qRank, outDim: d.qDim, rows: draftCount)
            try graph.rmsNorm(
                fb.qMat, weight: nil, out: fb.qMat,
                rows: draftCount * d.nHead, n: d.headDim,
                eps: ModelDefaults.rmsEps)
            try graph.ropeTail(
                x: fb.qMat, nTok: draftCount, nHead: d.nHead,
                headDim: d.headDim, nRot: d.nRot,
                nCtxOrig: rope.nCtxOrig, freqBase: rope.freqBase,
                freqScale: rope.freqScale, extFactor: rope.extFactor,
                attnFactor: rope.attnFactor, betaFast: rope.betaFast,
                betaSlow: rope.betaSlow, pos0: position, posStep: 0)

            // KV includes the target row and every draft row. DSpark gives the
            // whole block the same RoPE coordinate and then attends it with an
            // all-zero (non-causal) mask.
            try dense(graph, weights: mapped, suffix: "attn_kv.weight",
                      weight: w.kvW, act: fb.curMat, out: fb.kvMat,
                      inDim: d.nEmbd, outDim: d.headDim, rows: rowCount)
            try graph.rmsNorm(
                fb.kvMat, weight: w.kvNorm, out: fb.kvMat,
                rows: rowCount, n: d.headDim, eps: ModelDefaults.rmsEps)
            try graph.ropeTail(
                x: fb.kvMat, nTok: rowCount, nHead: 1,
                headDim: d.headDim, nRot: d.nRot,
                nCtxOrig: rope.nCtxOrig, freqBase: rope.freqBase,
                freqScale: rope.freqScale, extFactor: rope.extFactor,
                attnFactor: rope.attnFactor, betaFast: rope.betaFast,
                betaSlow: rope.betaSlow, pos0: position, posStep: 0)
            try graph.kvFP8StoreBatch(
                kv: fb.kvMat, rawCache: rawCache,
                headDim: d.headDim, nRot: d.nRot,
                pos0: plan.appendPosition, rawRows: rawRows,
                nTok: rowCount)

            let maskPointer = (fb.mask.buffer.contents() + fb.mask.byteOffset)
                .bindMemory(to: UInt16.self,
                            capacity: draftCount * plan.visibleRows)
            GraphContext.fillDSparkNoncausalMask(
                maskPointer, queryRows: draftCount,
                keyRows: plan.visibleRows)
            try graph.flashAttnPrefill(
                q: fb.qMat, kvF32: rawCache, kvF16: fb.kvF16,
                mask: fb.mask, sinks: w.attnSinks, pad: fb.pad,
                blk: fb.blk, heads: fb.heads,
                nHead: d.nHead, nQ: draftCount,
                rawSpan: plan.visibleRows,
                rawStartRow: plan.attentionRawStart)
            try graph.ropeTail(
                x: fb.heads, nTok: draftCount, nHead: d.nHead,
                headDim: d.headDim, nRot: d.nRot,
                nCtxOrig: rope.nCtxOrig, freqBase: rope.freqBase,
                freqScale: rope.freqScale, extFactor: rope.extFactor,
                attnFactor: rope.attnFactor, betaFast: rope.betaFast,
                betaSlow: rope.betaSlow, pos0: position, posStep: 0,
                inverse: true)

            let groupRowBytes = (d.attnGroupDim / 32) * 34
            for group in 0..<d.nOutGroup {
                try graph.encodeMMDenseQ8Strided(
                    weight: w.attnOutA,
                    weightOffset: group * d.nLoraO * groupRowBytes,
                    act: fb.heads,
                    actBase: group * d.attnGroupDim * 4,
                    actRowStride: d.qDim * 4,
                    out: fb.lowMat,
                    outBase: group * d.nLoraO * 4,
                    outRowStrideElems: d.attnLowDim,
                    inDim: d.attnGroupDim, outDim: d.nLoraO,
                    nTok: draftCount)
            }
            try dense(graph, weights: mapped,
                      suffix: "attn_output_b.weight", weight: w.attnOut,
                      act: fb.lowMat, out: fb.blockOutMat,
                      inDim: d.attnLowDim, outDim: d.nEmbd,
                      rows: draftCount)

            let draftInput = stageInputHC.subview(
                byteOffset: hcRowBytes,
                byteLength: draftCount * hcRowBytes,
                count: draftCount * hcDim)
            let draftAttnSplit = fb.splitA.subview(
                byteOffset: 24 * 4,
                byteLength: draftCount * 24 * 4,
                count: draftCount * 24)
            try graph.hcExpand4(
                blockOut: fb.blockOutMat, residual: draftInput,
                post: draftAttnSplit, comb: draftAttnSplit,
                blockAdd: nil, out: fb.afterAttnMat,
                nEmbd: d.nEmbd, nTokens: draftCount,
                postByteOffset: 4 * 4, combByteOffset: 8 * 4,
                splitTokenStride: 24 * 4)

            try encodeHCReduce(
                graph, fb, weights: mapped, x: fb.afterAttnMat,
                mixer: w.hcFfnFn, mixerSuffix: "hc_ffn_fn.weight",
                scale: w.ffnScale, base: w.ffnBase, norm: w.ffnNorm,
                split: fb.splitF, output: fb.curMat2,
                rows: draftCount)
            try dense(graph, weights: mapped,
                      suffix: "ffn_gate_inp.weight", weight: w.routerW,
                      act: fb.curMat2, out: fb.logitsMat,
                      inDim: d.nEmbd, outDim: d.nExperts,
                      rows: draftCount)
            try graph.routerProbabilitiesBatch(
                logits: fb.logitsMat, probabilities: fb.probsMat,
                width: d.nExperts, rows: draftCount)
            try graph.routerFinalizeTop6Batch(
                probs: fb.probsMat, selected: stage.idsSlab,
                bias: w.expBias, hashTable: nil, hashRows: 0,
                tokens: nil, weights: stage.rwSlab,
                nExperts: d.nExperts, nTok: draftCount,
                probsRow: d.nExperts, selRow: d.k, weightsRow: d.k,
                expertWeightScale: d.expertWeightScale)

            try graph.encodeMoEMap0(
                ids: stage.idsSlab, htpe: mm.htpe, hids: mm.hids,
                nTok: draftCount, kPerTok: d.k,
                nExperts: d.nExperts)
            try graph.encodeMMIdPairSwiGLUIQ2(
                gate: w.expGate, up: w.expUp,
                act: fb.curMat2, actBase: 0,
                htpe: mm.htpe, hids: mm.hids,
                mid: mm.mid, weights: stage.rwSlab,
                nTok: draftCount, kPerTok: d.k,
                nExperts: d.nExperts, inDim: d.nEmbd,
                ffnDim: d.expertFfn, clamp: d.swigluClamp)
            try graph.encodeMMIdDownQ2K(
                down: w.expDown, mid: mm.mid,
                htpe: mm.htpe, hids: mm.hids, out: mm.down6,
                nTok: draftCount, kPerTok: d.k,
                nExperts: d.nExperts, ffnDim: d.expertFfn,
                outDim: d.nEmbd)

            try graph.encodeMMDenseQ8(
                weight: w.sharedGate, act: fb.curMat2, actBase: 0,
                out: mm.sGate, inDim: d.nEmbd,
                outDim: d.sharedFfn, nTok: draftCount)
            try graph.encodeMMDenseQ8(
                weight: w.sharedUp, act: fb.curMat2, actBase: 0,
                out: mm.sUp, inDim: d.nEmbd,
                outDim: d.sharedFfn, nTok: draftCount)
            try graph.moeSwiGLUWeight(
                gate: mm.sGate, up: mm.sUp, weights: mm.ones,
                mid: mm.sMid, width: d.sharedFfn,
                rows: draftCount, clampValue: d.swigluClamp)
            try graph.encodeMMDenseQ8(
                weight: w.sharedDown, act: mm.sMid, actBase: 0,
                out: mm.sOut, inDim: d.sharedFfn,
                outDim: d.nEmbd, nTok: draftCount)
            try graph.hcWeightedSum(
                x: mm.down6, weights: mm.ones, out: mm.routedMat,
                nEmbd: d.nEmbd, nHC: d.k, nTokens: draftCount,
                weightsTokenStride: 0)
            try graph.add(mm.sOut, mm.routedMat, out: mm.routedMat,
                          width: d.nEmbd, rows: draftCount)
            try graph.hcExpand4(
                blockOut: mm.routedMat, residual: fb.afterAttnMat,
                post: fb.splitF, comb: fb.splitF,
                blockAdd: nil, out: stageOutputHC,
                nEmbd: d.nEmbd, nTokens: draftCount,
                postByteOffset: 4 * 4, combByteOffset: 8 * 4,
                splitTokenStride: 24 * 4)

            if stageIndex + 1 < allWeights.count {
                try graph.blitCopies([(
                    src: stageOutputHC, srcOff: 0,
                    dst: stageInputHC, dstOff: hcRowBytes,
                    bytes: draftCount * hcRowBytes)])
            }
        }
        guard let finalStage = allWeights.last else {
            throw MetalError.unsupported("stadio finale DSpark non disponibile")
        }
        // Upstream fuses the final HC collapse into the stage chain. Keeping it
        // in this command buffer removes one full CPU/GPU boundary before the
        // confidence-zero admission check.
        try encodeFinalHidden(graph, finalStage: finalStage)
        graph.commit()
        if let error = graph.lastError { throw error }
    }

    private func encodeFinalHidden(_ graph: GraphContext,
                                   finalStage: DSparkMappedStageWeights) throws {
        guard let head = finalStage.finalHead,
              finalStage.type(of: "hc_head_fn.weight") != nil else {
            throw MetalError.unsupported("teste finali DSpark non disponibili")
        }
        let hcDim = d.nHC * d.nEmbd
        try graph.rmsNorm(stageOutputHC, weight: nil, out: finalFlat,
                          rows: draftCount, n: hcDim,
                          eps: ModelDefaults.rmsEps)
        try dense(graph, weights: finalStage, suffix: "hc_head_fn.weight",
                  weight: head.hcHeadFn, act: finalFlat, out: finalPre,
                  inDim: hcDim, outDim: d.nHC, rows: draftCount)

        let scale = (head.hcHeadScale.buffer.contents()
            + head.hcHeadScale.byteOffset)
            .bindMemory(to: Float.self, capacity: 1).pointee
        for row in 0..<draftCount {
            let offset = row * d.nHC * MemoryLayout<Float>.stride
            let pre = finalPre.subview(
                byteOffset: offset, byteLength: d.nHC * 4, count: d.nHC)
            let weights = finalWeights.subview(
                byteOffset: offset, byteLength: d.nHC * 4, count: d.nHC)
            let tmp = finalWeightTmp.subview(
                byteOffset: offset, byteLength: d.nHC * 4, count: d.nHC)
            try graph.outputHCWeights(
                pre: pre, scaleScalar: scale, base: head.hcHeadBase,
                weights: weights, tmp: tmp, nHC: d.nHC,
                eps: ModelDefaults.hcEps)
        }
        try graph.hcWeightedSum(
            x: stageOutputHC, weights: finalWeights, out: finalEmbd,
            nEmbd: d.nEmbd, nHC: d.nHC, nTokens: draftCount,
            weightsTokenStride: d.nHC * 4)
        try graph.rmsNorm(finalEmbd, weight: head.norm, out: finalNorm,
                          rows: draftCount, n: d.nEmbd,
                          eps: ModelDefaults.rmsEps)
    }

    /// Evaluate confidence row zero before reading the 530 MB target output
    /// head. Low-yield prompts therefore stop after the support transformer,
    /// matching upstream's confidence-first runtime path.
    private func evaluateConfidence(_ head: DSparkFinalHeadWeights,
                                    finalStage: DSparkMappedStageWeights,
                                    row: Int) throws -> Float {
        packConfidenceFeatures(hiddenRow: row)
        let graph = GraphContext(rt)
        try graph.begin()
        try dense(
            graph, weight: head.confidenceProjection,
            type: finalStage.type(of: "confidence_head.proj.weight")!,
            act: confidenceFeatures, out: confidenceLogit,
            inDim: d.nEmbd + markovRank, outDim: 1, rows: 1)
        graph.commit()
        if let error = graph.lastError { throw error }
        let logit = (confidenceLogit.buffer.contents()
            + confidenceLogit.byteOffset)
            .bindMemory(to: Float.self, capacity: 1).pointee
        return DSparkProposal.sigmoid(logit)
    }

    /// Decode a confidence-gated Markov chain exactly in draft order. The
    /// full-vocabulary argmax remains on Metal; CPU reads back only one
    /// confidence float and one token id per accepted candidate.
    @_optimize(none)
    func makeProposal(firstPreviousToken: Int,
                      finalStage: DSparkMappedStageWeights,
                      baseOutputHead: GPUTensor,
                      confidenceThreshold: Float) throws -> DSparkProposal {
        guard let head = finalStage.finalHead,
              firstPreviousToken >= 0, firstPreviousToken < d.vocab else {
            throw MetalError.unsupported("teste finali DSpark non disponibili")
        }

        let threshold = min(1, max(0, confidenceThreshold))
        var previous = firstPreviousToken
        var tokens: [Int] = []
        var confidences: [Float] = []
        var firstConfidence: Float?
        tokens.reserveCapacity(draftCount)
        confidences.reserveCapacity(draftCount)

        // Confidence depends on the current-token Markov row plus the final
        // hidden row, but not on target logits. Reject before the expensive
        // vocabulary head and Markov W2 whenever possible.
        if threshold > 0 {
            try loadDenseRow(
                head.markovW1,
                type: finalStage.type(of: "markov_head.markov_w1.weight")!,
                row: previous, width: markovRank, into: markovState)
            let confidence = try evaluateConfidence(
                head, finalStage: finalStage, row: 0)
            firstConfidence = confidence
            guard confidence >= threshold else {
                return DSparkProposal(
                    tokens: [], confidences: [], threshold: threshold,
                    firstConfidence: confidence)
            }
        }

        // The DeepSeek V4 shared output head is Q8_0 by model contract. It is
        // deliberately encoded only after confidence row zero passes.
        let logitsGraph = GraphContext(rt)
        try logitsGraph.begin()
        try logitsGraph.encodeMMDenseQ8(
            weight: baseOutputHead, act: finalNorm, actBase: 0,
            out: baseLogits, inDim: d.nEmbd, outDim: d.vocab,
            nTok: draftCount)
        logitsGraph.commit()
        if let error = logitsGraph.lastError { throw error }

        for row in 0..<draftCount {
            let reusedConfidence = row == 0 ? firstConfidence : nil
            if reusedConfidence == nil {
                try loadDenseRow(
                    head.markovW1,
                    type: finalStage.type(of: "markov_head.markov_w1.weight")!,
                    row: previous, width: markovRank, into: markovState)
                packConfidenceFeatures(hiddenRow: row)
            }

            let candidateGraph = GraphContext(rt)
            try candidateGraph.begin()
            if reusedConfidence == nil {
                try dense(
                    candidateGraph, weight: head.confidenceProjection,
                    type: finalStage.type(of: "confidence_head.proj.weight")!,
                    act: confidenceFeatures, out: confidenceLogit,
                    inDim: d.nEmbd + markovRank, outDim: 1, rows: 1)
            }
            try dense(
                candidateGraph, weight: head.markovW2,
                type: finalStage.type(of: "markov_head.markov_w2.weight")!,
                act: markovState, out: markovBias,
                inDim: markovRank, outDim: d.vocab, rows: 1)
            let logits = baseLogits.subview(
                byteOffset: row * d.vocab * 4,
                byteLength: d.vocab * 4, count: d.vocab)
            try candidateGraph.add(
                logits, markovBias, out: correctedLogits, width: d.vocab)
            try candidateGraph.dsparkArgmax(
                values: correctedLogits, count: d.vocab,
                partialValues: argmaxPartialValues,
                partialIndices: argmaxPartialIndices,
                result: argmaxResult,
                partialCount: Self.argmaxPartials)
            candidateGraph.commit()
            if let error = candidateGraph.lastError { throw error }

            let confidence: Float
            if let reusedConfidence {
                confidence = reusedConfidence
            } else {
                let logit = (confidenceLogit.buffer.contents()
                    + confidenceLogit.byteOffset)
                    .bindMemory(to: Float.self, capacity: 1).pointee
                confidence = DSparkProposal.sigmoid(logit)
                if firstConfidence == nil { firstConfidence = confidence }
            }
            guard confidence >= threshold else { break }
            let token = Int((argmaxResult.buffer.contents()
                + argmaxResult.byteOffset)
                .bindMemory(to: Int32.self, capacity: 1).pointee)
            guard token >= 0, token < d.vocab else {
                throw MetalError.unsupported("argmax DSpark fuori vocabolario")
            }
            tokens.append(token)
            confidences.append(confidence)
            previous = token
        }
        return DSparkProposal(tokens: tokens, confidences: confidences,
                              threshold: threshold,
                              firstConfidence: firstConfidence)
    }

    /// Load one logical row from F32/F16/Q8_0 without materializing the whole
    /// vocabulary-by-rank Markov table. Q8_0 stores 32 values per 34-byte block.
    private func loadDenseRow(_ weight: GPUTensor, type: UInt32,
                              row: Int, width: Int,
                              into output: GPUTensor) throws {
        let source = weight.buffer.contents() + weight.byteOffset
        let destination = (output.buffer.contents() + output.byteOffset)
            .bindMemory(to: Float.self, capacity: width)
        switch type {
        case 0:
            memcpy(destination, source + row * width * 4, width * 4)
        case 1:
            let bytes = source + row * width * 2
            for column in 0..<width {
                let bits = bytes.loadUnaligned(
                    fromByteOffset: column * 2, as: UInt16.self)
                destination[column] = Float(Float16(bitPattern: bits))
            }
        case 8:
            guard width % 32 == 0 else {
                throw MetalError.unsupported("riga Q8_0 DSpark non allineata")
            }
            let rowBytes = width / 32 * 34
            let bytes = source + row * rowBytes
            for block in 0..<(width / 32) {
                let base = block * 34
                let bits = bytes.loadUnaligned(
                    fromByteOffset: base, as: UInt16.self)
                let scale = Float(Float16(bitPattern: bits))
                for lane in 0..<32 {
                    let quantized = bytes.load(
                        fromByteOffset: base + 2 + lane, as: Int8.self)
                    destination[block * 32 + lane] = scale * Float(quantized)
                }
            }
        default:
            throw MetalError.unsupported(
                "riga DSpark tipo GGUF \(type) non eseguibile")
        }
    }

    private func packConfidenceFeatures(hiddenRow row: Int) {
        let destination = (confidenceFeatures.buffer.contents()
            + confidenceFeatures.byteOffset)
        let hidden = finalNorm.buffer.contents() + finalNorm.byteOffset
            + row * d.nEmbd * MemoryLayout<Float>.stride
        memcpy(destination, hidden, d.nEmbd * MemoryLayout<Float>.stride)
        memcpy(destination + d.nEmbd * MemoryLayout<Float>.stride,
               markovState.buffer.contents() + markovState.byteOffset,
               markovRank * MemoryLayout<Float>.stride)
    }
}

private extension GraphContext {
    /// Generic two-stage F32 argmax. These kernels are shared with the GLM head;
    /// their ABI is model-independent and returns the lowest id on exact ties.
    func dsparkArgmax(values: GPUTensor, count: Int,
                      partialValues: GPUTensor,
                      partialIndices: GPUTensor, result: GPUTensor,
                      partialCount: Int) throws {
        let chunk = (count + partialCount - 1) / partialCount
        let partialArgs = [UInt32(count), UInt32(chunk), 0, 0]
        let partial = try rt.pipeline("kernel_glm52_argmax_partial_f32")
        let e = encoder
        e.setComputePipelineState(partial)
        partialArgs.withUnsafeBytes {
            e.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        e.setBuffer(values.buffer, offset: values.byteOffset, index: 1)
        e.setBuffer(partialValues.buffer,
                    offset: partialValues.byteOffset, index: 2)
        e.setBuffer(partialIndices.buffer,
                    offset: partialIndices.byteOffset, index: 3)
        e.setThreadgroupMemoryLength(256 * 2 * 4, index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: partialCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let finalArgs = [UInt32(partialCount), 0, 0, 0]
        let final = try rt.pipeline("kernel_glm52_argmax_final_f32")
        e.setComputePipelineState(final)
        finalArgs.withUnsafeBytes {
            e.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        e.setBuffer(partialValues.buffer,
                    offset: partialValues.byteOffset, index: 1)
        e.setBuffer(partialIndices.buffer,
                    offset: partialIndices.byteOffset, index: 2)
        e.setBuffer(result.buffer, offset: result.byteOffset, index: 3)
        e.setThreadgroupMemoryLength(64 * 2 * 4, index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }
}

// MARK: - Executable target-hidden capture and stage 0

/// Confidence-gated draft sequence emitted by the DSpark support model.
public struct DSparkProposal: Sendable, Equatable {
    public let tokens: [Int]
    public let confidences: [Float]
    public let threshold: Float
    /// Confidence evaluated for row zero even when it is below `threshold`
    /// and therefore no token is proposed. The adaptive scheduler uses this
    /// signal to avoid repeatedly paying for predictably empty blocks.
    public let firstConfidence: Float?

    public init(tokens: [Int], confidences: [Float], threshold: Float,
                firstConfidence: Float? = nil) {
        precondition(tokens.count == confidences.count)
        self.tokens = tokens
        self.confidences = confidences
        self.threshold = threshold
        self.firstConfidence = firstConfidence ?? confidences.first
    }

    static func sigmoid(_ value: Float) -> Float {
        if value >= 0 { return 1 / (1 + exp(-value)) }
        let positive = exp(value)
        return positive / (1 + positive)
    }
}

/// Result of the target verifier for one greedy DSpark proposal.
///
/// The target logits already available before the verify window validate the
/// first draft.  Row `i` of the verifier then predicts draft `i + 1`.  Keeping
/// this indexing in one pure helper prevents the common off-by-one that accepts
/// a token against the logits produced *after* consuming that same token.
public enum DSparkGreedyVerifier {
    public static func acceptedPrefix(proposal: [Int], currentTarget: Int,
                                      verifiedNext: [Int]) -> Int {
        guard let first = proposal.first, first == currentTarget else { return 0 }
        var accepted = 1
        while accepted < proposal.count {
            let verifierIndex = accepted - 1
            guard verifierIndex < verifiedNext.count,
                  verifiedNext[verifierIndex] == proposal[accepted] else { break }
            accepted += 1
        }
        return accepted
    }
}

/// First executable part of the DSpark graph.
///
/// For each target layer published by the support GGUF it captures
/// `mean(HC[0..<nHC])`, exactly like upstream `metal_graph_dspark_capture_hc`.
/// Once all slots for a token are present it evaluates:
///
///     mainX = RMSNorm(main_proj(concat(targetHidden)), main_norm)
///
/// The two stage-0 weights are copied out of the GGUF mapping into anonymous
/// shared Metal buffers once, so subsequent tokens never fault them from the
/// support file. The draft transformer chain owns private KV rings and its
/// final HC, Markov and confidence heads emit a verified greedy proposal.
public final class DSparkStage0Runtime: DS4Logging {
    public static let logTag = "dspark"

    public let support: DSparkSupportModel
    public let targetLayerIDs: [Int]
    public let residentWeightBytes: Int
    public let mappedTransformerBytes: Int
    public let cacheCapacity: Int
    public let privateKVBytes: Int

    private let rt: MetalRuntime
    private let dims: DSV4Dims
    private let targetSlot: [Int: Int]
    private let targetHidden: GPUTensor
    private let targetHiddenBatch: GPUTensor
    private let meanHC: GPUTensor
    private let mainProj: GPUTensor
    private let mainProjType: UInt32
    private let mainNorm: GPUTensor
    private let transformerStages: [DSparkMappedStageWeights]
    private let projected: GPUTensor
    private let mainX: GPUTensor
    private let stageRawCaches: [GPUTensor]
    private let transformerExecutor: DSparkTransformerExecutor?
    public private(set) var cacheWindow: DSparkKVWindow

    private var pendingCaptures: [GraphContext] = []
    private var pendingBatchCaptures: [GraphContext] = []
    private var captureMask: UInt32 = 0
    private var batchCaptureMask: UInt32 = 0
    private var capturePosition: Int?
    private var batchCaptureStart: Int?
    private var batchCaptureCount = 0
    private var batchInputDrop = 0
    private var privateCacheNeedsSeed = false
    private var readyPosition: Int?
    public private(set) var capturedBatchRange: Range<Int>?

    public var isReady: Bool { readyPosition != nil }
    public var position: Int? { readyPosition }

    public init(rt: MetalRuntime, supportPath: String, mainModelPath: String,
                targetDims: DSV4Dims, targetLayerCount: Int,
                cacheCapacity requestedCacheCapacity: Int? = nil) throws {
        let support = try DSparkSupportModel(path: supportPath,
                                             targetDims: targetDims,
                                             targetLayerCount: targetLayerCount,
                                             metalMapping: true)
        guard support.isRunnable(withMainModelPath: mainModelPath) else {
            let errors = support.validation.errors.map(\.message)
            let mismatch = support.checkpointCompatibilityIssue(mainModelPath: mainModelPath)
            throw MetalError.unsupported(
                "supporto DSpark non eseguibile: "
                    + ([mismatch].compactMap { $0 } + errors).joined(separator: "; "))
        }
        guard let targetLayers = support.metadata.targetLayerIDs,
              !targetLayers.isEmpty,
              let stage0 = support.stages.first,
              let mainProjDescriptor = stage0["main_proj.weight"],
              let mainNormDescriptor = stage0["main_norm.weight"] else {
            throw MetalError.unsupported("supporto DSpark senza interfaccia stage 0")
        }
        let transformerStages = try support.stages.indices.map {
            try support.loadMappedStageWeights(rt, stage: $0)
        }

        func resident(_ tensor: GGUFModel.Tensor) throws -> GPUTensor {
            guard tensor.bytes <= UInt64(Int.max) else {
                throw MetalError.unsupported("tensore DSpark troppo grande: \(tensor.name)")
            }
            return try GPUTensor.raw(
                rt,
                ptr: support.model.mapBase + Int(tensor.absOffset),
                byteLength: Int(tensor.bytes),
                elementCount: Int(tensor.bytes))
        }

        self.rt = rt
        self.dims = targetDims
        self.support = support
        self.targetLayerIDs = targetLayers
        self.targetSlot = Dictionary(uniqueKeysWithValues:
            targetLayers.enumerated().map { ($0.element, $0.offset) })
        self.mainProj = try resident(mainProjDescriptor)
        self.mainProjType = mainProjDescriptor.type
        self.mainNorm = try resident(mainNormDescriptor)
        self.transformerStages = transformerStages
        self.mappedTransformerBytes = transformerStages.reduce(0) {
            $0 + $1.mappedBytes
        }
        self.residentWeightBytes = Int(mainProjDescriptor.bytes + mainNormDescriptor.bytes)
            + transformerStages.reduce(0) { $0 + $1.residentBytes }
        let minimumCache = max(1, (support.metadata.blockSize ?? 1) + 1)
        let cacheCapacity = max(minimumCache,
                                requestedCacheCapacity ?? targetDims.nSWA)
        self.cacheCapacity = cacheCapacity
        self.privateKVBytes = support.metadata.stageCount * cacheCapacity
            * targetDims.headDim * MemoryLayout<Float>.stride
        self.stageRawCaches = try (0..<support.metadata.stageCount).map { _ in
            try GPUTensor.lazyZeros(
                rt, floatCount: cacheCapacity * targetDims.headDim)
        }
        guard let blockSize = support.metadata.blockSize,
              let markovRank = support.metadata.markovRank,
              let noiseToken = support.metadata.noiseTokenID else {
            throw MetalError.unsupported("metadati DSpark incompleti")
        }
        // Tiny synthetic geometries are used by the metadata/capture tests and
        // intentionally cannot instantiate the Flash-only 512-wide kernels.
        // Production Flash geometry must pass the executable contract here.
        if targetDims.nHC == 4, targetDims.headDim == 512,
           targetDims.k == 6, targetDims.nExperts == 256 {
            self.transformerExecutor = try DSparkTransformerExecutor(
                rt: rt, dims: targetDims, blockSize: blockSize,
                noiseToken: noiseToken, markovRank: markovRank,
                cacheCapacity: cacheCapacity,
                targetLayerCount: targetLayers.count,
                stages: transformerStages)
        } else {
            self.transformerExecutor = nil
        }
        self.cacheWindow = DSparkKVWindow(capacity: cacheCapacity)
        self.targetHidden = try .zeros(rt,
            floatCount: targetLayers.count * targetDims.nEmbd)
        self.targetHiddenBatch = try .zeros(
            rt,
            floatCount: targetLayers.count * cacheCapacity * targetDims.nEmbd)
        self.meanHC = try .floats(rt,
            Array(repeating: 1 / Float(targetDims.nHC), count: targetDims.nHC))
        self.projected = try .zeros(rt, floatCount: targetDims.nEmbd)
        self.mainX = try .zeros(rt, floatCount: targetDims.nEmbd)

        if ProcessInfo.processInfo.environment["DS4_MLOCK"] == "1" {
            _ = mainProj.lockResident()
            _ = mainNorm.lockResident()
            _ = targetHidden.lockResident()
            _ = projected.lockResident()
            _ = mainX.lockResident()
        }
    }

    deinit {
        abortCapture()
        abortBatchCapture()
    }

    /// Start a capture transaction for one absolute target token position.
    public func beginCapture(position: Int) {
        abortCapture()
        capturePosition = position
        readyPosition = nil
        captureMask = 0
    }

    /// Encode one target-layer HC reduction behind the layer's command buffer.
    /// All contexts use the same in-order Metal queue, so the reduction observes
    /// the completed layer output without a CPU wait.
    public func capture(layer: Int, hiddenHC: GPUTensor) throws {
        guard let slot = targetSlot[layer], capturePosition != nil else { return }
        let rowBytes = dims.nEmbd * MemoryLayout<Float>.stride
        let destination = targetHidden.subview(
            byteOffset: slot * rowBytes,
            byteLength: rowBytes,
            count: dims.nEmbd)
        let graph = GraphContext(rt)
        try graph.begin()
        try graph.hcWeightedSum(x: hiddenHC, weights: meanHC, out: destination,
                                nEmbd: dims.nEmbd, nHC: dims.nHC, nTokens: 1)
        graph.commitAsync()
        pendingCaptures.append(graph)
        captureMask |= UInt32(1) << UInt32(slot)
    }

    /// Begin collection of all target-hidden rows needed to seed DSpark's
    /// private stage caches after prefill. If a chunk exceeds the ring, only
    /// its newest `cacheCapacity` rows are retained.
    public func beginBatchCapture(startPosition: Int, nTokens: Int) {
        abortBatchCapture()
        guard startPosition >= 0, nTokens > 0 else { return }
        let retained = min(nTokens, cacheCapacity)
        batchInputDrop = nTokens - retained
        batchCaptureStart = startPosition + batchInputDrop
        batchCaptureCount = retained
        batchCaptureMask = 0
        capturedBatchRange = nil
    }

    /// Capture a contiguous `[token][HC][embd]` slab for one configured target
    /// layer into that layer's slot-major circular history.
    public func captureBatch(layer: Int, hiddenHC: GPUTensor,
                             nTokens: Int) throws {
        guard let slot = targetSlot[layer],
              let start = batchCaptureStart,
              batchCaptureCount > 0,
              nTokens >= batchInputDrop + batchCaptureCount else { return }
        let hcRowBytes = dims.nHC * dims.nEmbd * MemoryLayout<Float>.stride
        let outputRowBytes = dims.nEmbd * MemoryLayout<Float>.stride
        let physicalStart = start % cacheCapacity
        let firstCount = min(batchCaptureCount, cacheCapacity - physicalStart)

        func encodeSegment(inputRow: Int, outputRow: Int, count: Int) throws {
            guard count > 0 else { return }
            let input = hiddenHC.subview(
                byteOffset: inputRow * hcRowBytes,
                byteLength: count * hcRowBytes,
                count: count * dims.nHC * dims.nEmbd)
            let slotBase = slot * cacheCapacity * outputRowBytes
            let destination = targetHiddenBatch.subview(
                byteOffset: slotBase + outputRow * outputRowBytes,
                byteLength: count * outputRowBytes,
                count: count * dims.nEmbd)
            let graph = GraphContext(rt)
            try graph.begin()
            try graph.hcWeightedSum(
                x: input, weights: meanHC, out: destination,
                nEmbd: dims.nEmbd, nHC: dims.nHC, nTokens: count,
                weightsTokenStride: 0)
            graph.commitAsync()
            pendingBatchCaptures.append(graph)
        }

        try encodeSegment(inputRow: batchInputDrop,
                          outputRow: physicalStart, count: firstCount)
        try encodeSegment(inputRow: batchInputDrop + firstCount,
                          outputRow: 0, count: batchCaptureCount - firstCount)
        batchCaptureMask |= UInt32(1) << UInt32(slot)
    }

    /// Variant used by the speculative verifier, whose HC rows are independent
    /// tensors rather than one prefill slab. Proposal windows are at most five
    /// rows, so one tiny reduction per row avoids a CPU HC readback/copy.
    public func captureBatchRows(layer: Int,
                                 hiddenHC rows: [GPUTensor]) throws {
        guard let slot = targetSlot[layer],
              let start = batchCaptureStart,
              batchCaptureCount > 0,
              rows.count >= batchInputDrop + batchCaptureCount else { return }
        let outputRowBytes = dims.nEmbd * MemoryLayout<Float>.stride
        for row in 0..<batchCaptureCount {
            let physical = (start + row) % cacheCapacity
            let destination = targetHiddenBatch.subview(
                byteOffset: (slot * cacheCapacity + physical) * outputRowBytes,
                byteLength: outputRowBytes, count: dims.nEmbd)
            let graph = GraphContext(rt)
            try graph.begin()
            try graph.hcWeightedSum(
                x: rows[batchInputDrop + row], weights: meanHC,
                out: destination, nEmbd: dims.nEmbd, nHC: dims.nHC,
                nTokens: 1, weightsTokenStride: 0)
            graph.commitAsync()
            pendingBatchCaptures.append(graph)
        }
        batchCaptureMask |= UInt32(1) << UInt32(slot)
    }

    @discardableResult
    public func finishBatchCapture() -> Bool {
        for graph in pendingBatchCaptures { graph.waitCompleted() }
        pendingBatchCaptures.removeAll(keepingCapacity: true)
        let count = targetLayerIDs.count
        let expectedMask = count >= 32 ? UInt32.max : (UInt32(1) << UInt32(count)) - 1
        guard batchCaptureMask == expectedMask,
              let start = batchCaptureStart,
              batchCaptureCount > 0 else {
            capturedBatchRange = nil
            privateCacheNeedsSeed = false
            return false
        }
        capturedBatchRange = start..<(start + batchCaptureCount)
        privateCacheNeedsSeed = true
        return true
    }

    /// Finish the transaction and run the resident stage-0 projection.
    /// Returns false for an incomplete capture without touching stale rows.
    @discardableResult
    public func finishCapture() throws -> Bool {
        for graph in pendingCaptures { graph.waitCompleted() }
        pendingCaptures.removeAll(keepingCapacity: true)

        let count = targetLayerIDs.count
        let expectedMask = count >= 32 ? UInt32.max : (UInt32(1) << UInt32(count)) - 1
        guard captureMask == expectedMask, let position = capturePosition else {
            readyPosition = nil
            return false
        }

        let inputWidth = count * dims.nEmbd
        let graph = GraphContext(rt)
        try graph.begin()
        switch mainProjType {
        case 0:
            try graph.matmulF32(weight: mainProj, x: targetHidden, out: projected,
                                inDim: inputWidth, outDim: dims.nEmbd)
        case 1:
            try graph.matmulF16(weight: mainProj, x: targetHidden, out: projected,
                                inDim: inputWidth, outDim: dims.nEmbd)
        case 8:
            try graph.matmulQ8_0(weight: mainProj, x: targetHidden, out: projected,
                                 inDim: inputWidth, outDim: dims.nEmbd)
        default:
            throw MetalError.unsupported(
                "main_proj DSpark tipo GGUF \(mainProjType) non eseguibile")
        }
        try graph.rmsNorm(projected, weight: mainNorm, out: mainX,
                          rows: 1, n: dims.nEmbd, eps: ModelDefaults.rmsEps)
        graph.commit()
        if let error = graph.lastError { throw error }
        readyPosition = position
        return true
    }

    /// Diagnostic CPU readback. The draft executor consumes the same tensor on
    /// GPU; normal generation never needs this allocation/copy.
    public func mainHidden() -> [Float]? {
        guard isReady else { return nil }
        return mainX.floatArray(dims.nEmbd)
    }

    /// Execute `[target + draft]` through every support transformer. This
    /// diagnostic entry point deliberately stops before proposal decoding, so
    /// it never changes target tokens or target KV.
    @discardableResult
    func runTransformerStages(currentToken: Int, position: Int,
                              embeddingTable: GPUTensor) throws -> Bool {
        guard readyPosition == position, let main = mainHiddenTensor else {
            return false
        }
        guard let transformerExecutor else {
            throw MetalError.unsupported(
                "forward transformer DSpark non disponibile per questa geometria")
        }
        if privateCacheNeedsSeed {
            guard let range = capturedBatchRange,
                  position < Int.max,
                  range.upperBound == position + 1 else {
                cacheWindow.reset()
                throw MetalError.unsupported(
                    "storico DSpark non allineato alla posizione \(position)")
            }
            do {
                // The current target feature is row 0 of the proposal block;
                // seed only rows strictly before it into the support history.
                let seedRange = range.lowerBound..<position
                if !seedRange.isEmpty {
                    try transformerExecutor.seedCaches(
                        targetHistory: targetHiddenBatch,
                        capturedRange: seedRange,
                        mainProjection: mainProj,
                        mainProjectionType: mainProjType,
                        mainNorm: mainNorm,
                        weights: transformerStages,
                        rawCaches: stageRawCaches)
                }
                guard cacheWindow.set(
                    tokenStart: seedRange.lowerBound,
                    length: seedRange.count) else {
                    throw MetalError.unsupported(
                        "finestra KV DSpark non installabile")
                }
                privateCacheNeedsSeed = false
            } catch {
                cacheWindow.reset()
                throw error
            }
        } else {
            _ = cacheWindow.crop(toPrefixLength: position)
        }
        try transformerExecutor.run(
            currentToken: currentToken, position: position,
            vocabularySize: dims.vocab, embeddingTable: embeddingTable,
            mainHidden: main, weights: transformerStages,
            rawCaches: stageRawCaches, window: cacheWindow)
        // Every stage wrote the real target row first at `position`; draft rows
        // remain outside the logical window until target verification.
        if cacheWindow.length == 0 {
            guard cacheWindow.set(tokenStart: position, length: 1) else {
                throw MetalError.unsupported("frontiera KV DSpark non avanzabile")
            }
        } else if !cacheWindow.appendRow(at: position) {
            throw MetalError.unsupported("frontiera KV DSpark non avanzabile")
        }
        return true
    }

    /// Maintain support KV after an ordinary target forward or verifier replay.
    /// It is a no-op before the initial prompt cache has been seeded.
    func maintainCapturedTarget(position: Int) throws {
        guard !privateCacheNeedsSeed,
              readyPosition == position,
              cacheWindow.length > 0,
              cacheWindow.ends(at: position),
              let transformerExecutor,
              let main = mainHiddenTensor else { return }
        try transformerExecutor.appendCapturedTarget(
            mainHidden: main, position: position,
            weights: transformerStages, rawCaches: stageRawCaches)
        guard cacheWindow.appendRow(at: position) else {
            throw MetalError.unsupported("manutenzione ring DSpark fallita")
        }
    }

    /// Execute the support transformer and its final HC/Markov/confidence
    /// heads. Target-model state is read-only until the caller explicitly
    /// sends the returned proposal to `dsparkVerifyAndCommit`.
    func makeProposal(currentToken: Int, position: Int,
                      embeddingTable: GPUTensor,
                      baseOutputHead: GPUTensor,
                      confidenceThreshold: Float) throws -> DSparkProposal? {
        guard try runTransformerStages(
            currentToken: currentToken, position: position,
            embeddingTable: embeddingTable) else { return nil }
        guard let transformerExecutor,
              let finalStage = transformerStages.last else {
            throw MetalError.unsupported("teste finali DSpark non disponibili")
        }
        return try transformerExecutor.makeProposal(
            firstPreviousToken: currentToken,
            finalStage: finalStage,
            baseOutputHead: baseOutputHead,
            confidenceThreshold: confidenceThreshold)
    }

    /// Internal GPU hand-off for the transformer-stage executor.
    var mainHiddenTensor: GPUTensor? { isReady ? mainX : nil }
    var targetHiddenHistoryTensor: GPUTensor? {
        capturedBatchRange == nil ? nil : targetHiddenBatch
    }

    /// Private raw-KV storage for the transformer executor. Never aliases the
    /// target decoder's cache, so rejected draft rows cannot corrupt it.
    func rawCache(forStage stage: Int) -> GPUTensor? {
        stageRawCaches.indices.contains(stage) ? stageRawCaches[stage] : nil
    }

    func weights(forStage stage: Int) -> DSparkMappedStageWeights? {
        transformerStages.indices.contains(stage) ? transformerStages[stage] : nil
    }

    public func resetPrivateCache() {
        cacheWindow.reset()
        privateCacheNeedsSeed = capturedBatchRange != nil
    }

    public func abortCapture() {
        for graph in pendingCaptures { graph.waitCompleted() }
        pendingCaptures.removeAll(keepingCapacity: true)
        captureMask = 0
        capturePosition = nil
        readyPosition = nil
    }

    public func abortBatchCapture() {
        for graph in pendingBatchCaptures { graph.waitCompleted() }
        pendingBatchCaptures.removeAll(keepingCapacity: true)
        batchCaptureMask = 0
        batchCaptureStart = nil
        batchCaptureCount = 0
        batchInputDrop = 0
        capturedBatchRange = nil
        privateCacheNeedsSeed = false
    }
}

extension StreamingDecoder {
    /// Attach the executable stage-0 graph. Calling this between forwards keeps
    /// the ordinary decoder completely unchanged when no support is configured.
    public func enableDSparkStage0(supportPath: String,
                                   mainModelPath: String) throws {
        drainFFN()
        dsparkStage0Runtime?.abortCapture()
        let configuredPrefill = max(64, ProcessInfo.processInfo.environment[
            "DS4_PREFILL_CHUNK"].flatMap(Int.init) ?? 512)
        let wanted = min(maxKeys, min(8192,
            ((d.nSWA + configuredPrefill + 255) / 256) * 256))
        let targetRawCapacity = rawCaches.first.map { $0.count / d.headDim } ?? 1
        dsparkStage0Runtime = try DSparkStage0Runtime(
            rt: rt,
            supportPath: supportPath,
            mainModelPath: mainModelPath,
            targetDims: d,
            targetLayerCount: nLayers,
            cacheCapacity: max(targetRawCapacity, wanted))
    }

    public func disableDSpark() {
        drainFFN()
        dsparkStage0Runtime?.abortCapture()
        dsparkStage0Runtime?.abortBatchCapture()
        dsparkStage0Runtime = nil
    }

    public var dsparkStage0IsReady: Bool {
        dsparkStage0Runtime?.isReady ?? false
    }

    public var dsparkStage0Position: Int? {
        dsparkStage0Runtime?.position
    }

    public func dsparkStage0Hidden() -> [Float]? {
        dsparkStage0Runtime?.mainHidden()
    }

    /// Diagnostic/next-step entry point for the complete support transformer
    /// chain. It seeds the private rings lazily from the last prompt batch and
    /// never mutates target-model KV or target logits.
    @discardableResult
    public func dsparkRunTransformerStages(currentToken: Int,
                                            position: Int) throws -> Bool {
        drainFFN()
        guard let dsparkStage0Runtime else { return false }
        return try dsparkStage0Runtime.runTransformerStages(
            currentToken: currentToken, position: position,
            embeddingTable: embedTable)
    }

    /// Build a complete DSpark draft without mutating the target-model decode
    /// frontier. A confidence below the configured threshold simply returns a
    /// shorter (possibly empty) proposal.
    public func dsparkPropose(currentToken: Int, position: Int,
                             confidenceThreshold: Float = 0.7) throws
        -> DSparkProposal? {
        drainFFN()
        guard let dsparkStage0Runtime else { return nil }
        return try dsparkStage0Runtime.makeProposal(
            currentToken: currentToken, position: position,
            embeddingTable: embedTable, baseOutputHead: out.head,
            confidenceThreshold: confidenceThreshold)
    }

    /// Optional failures disable only DSpark, never the target decoder.
    func disableDSparkAfterFailure(_ error: Error) {
        DSparkStage0Runtime.log("runtime disabilitato dopo errore: \(error)")
        dsparkStage0Runtime?.abortCapture()
        dsparkStage0Runtime = nil
    }
}
