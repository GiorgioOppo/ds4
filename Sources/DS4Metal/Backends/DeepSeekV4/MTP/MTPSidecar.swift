import Foundation
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
public struct MTPSidecar {
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
        var s = "DS4 mtp: sidecar \((model.path as NSString).lastPathComponent) — "
            + "\(model.tensors.count) tensori, \(model.size >> 20) MB"
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
