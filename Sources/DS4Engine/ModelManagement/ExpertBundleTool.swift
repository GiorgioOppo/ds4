import Foundation
import DS4Core
import DS4Metal

/// Generazione ESPLICITA dell'expert-bundle (bottone nei Settings della GUI):
/// verifica/crea il sidecar per un GGUF SENZA caricare il motore — apre il
/// modello (solo mmap + metadata), deriva la stessa geometria del factory del
/// decoder e chiama ExpertBundle.openOrBuild. Le posizioni sono le stesse del
/// load: lettura sibling → DS4_BUNDLE_DIR; build in DS4_BUNDLE_DIR quando
/// impostata (l'app la punta ad Application Support). Log su stderr come sempre.
public enum ExpertBundleTool {
    /// Ritorna un esito sintetico per la UI; i dettagli (percorsi, motivi di
    /// skip, progresso) finiscono nel Log motore con prefisso "DS4 expbundle:".
    public static func ensure(modelPath: String) -> String {
        guard !modelPath.isEmpty else { return "Nessun modello selezionato." }
        do {
            let model = try GGUFModel(path: modelPath, metalMapping: false, prefetchCPU: false)
            var dims = DSV4Shape.dims
            let mq = GGUFWeights.detectMoEQuant(model)
            dims.gateQuant = mq.gate; dims.upQuant = mq.up; dims.downQuant = mq.down
            let gateBytes = (dims.nEmbd / 256) * dims.gateQuant.blockBytes * dims.expertFfn
            let upBytes = (dims.nEmbd / 256) * dims.upQuant.blockBytes * dims.expertFfn
            let downBytes = (dims.expertFfn / 256) * dims.downQuant.blockBytes * dims.nEmbd
            let bundle = ExpertBundle.openOrBuild(model: model, layers: 0..<DSV4Shape.nLayer,
                                                  nExpert: dims.nExperts,
                                                  gateBytes: gateBytes, upBytes: upBytes,
                                                  downBytes: downBytes)
            if let bundle {
                return "Expert bundle pronto: \(bundle.path)"
            }
            return "Bundle NON creato: il Log motore riporta il motivo (spazio / permessi / modello)."
        } catch {
            return "Impossibile aprire il GGUF: \(error)"
        }
    }
}
