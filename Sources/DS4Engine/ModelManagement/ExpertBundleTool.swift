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
            let selection = try RuntimeBackendFactory.prepare(model: model)
            if selection.backend == .glm52 {
                // Sidecar GLM UNIFICATO: un file per layer sparse con i
                // tensori grossi riquantizzati Q4_K E i record esperti
                // contigui, accanto al GGUF (DS4_GLM_LAYERQ4_DIR per
                // spostarlo). Migra da solo i file legacy (bundle .experts
                // e sidecar v1) e resumabile: ripremere riprende dal primo
                // layer mancante.
                let map = try GLM52WeightMap(model: model)
                let reader = try GLM52PayloadReader(path: modelPath,
                                                    weightMap: map)
                let environment = ProcessInfo.processInfo.environment
                let directory = environment["DS4_GLM_LAYERQ4_DIR"]
                    ?? (modelPath + ".glm-layers-q4")
                let legacyBundles = environment["DS4_GLM_BUNDLE_DIR"]
                    ?? (modelPath + ".glm-experts")
                let summary = try GLM52LayerQuantSidecar.buildAvailable(
                    directory: directory, weightMap: map, reader: reader,
                    legacyBundleDirectory: legacyBundles) { layer, built in
                    DS4Log.info("expbundle", "GLM pack blk\(layer) "
                        + (built ? "creato" : "già valido"))
                }
                var outcome = "Sidecar GLM unificati in \(directory): "
                    + "\(summary.created) creati, "
                    + "\(summary.alreadyValid) già validi"
                if summary.remaining > 0 {
                    outcome += ", \(summary.remaining) mancanti (serviti dal "
                        + "GGUF)"
                }
                if let reason = summary.stoppedBecause {
                    outcome += " — interrotto: \(reason). Libera spazio e "
                        + "ripremi per continuare."
                }
                return outcome + "."
            }
            let geometry = DSV4RuntimeGeometry(configuration: try ModelConfig(model: model))
            var dims = geometry.dims
            let mq = GGUFWeights.detectMoEQuant(model)
            dims.gateQuant = mq.gate; dims.upQuant = mq.up; dims.downQuant = mq.down
            let gateBytes = (dims.nEmbd / 256) * dims.gateQuant.blockBytes * dims.expertFfn
            let upBytes = (dims.nEmbd / 256) * dims.upQuant.blockBytes * dims.expertFfn
            let downBytes = (dims.expertFfn / 256) * dims.downQuant.blockBytes * dims.nEmbd
            let bundle = ExpertBundle.openOrBuild(model: model, layers: 0..<geometry.nLayers,
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
