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
                // Sidecar GLM: un bundle contiguo per ogni layer sparse,
                // accanto al GGUF (o in DS4_GLM_BUNDLE_DIR). Resumabile: i
                // bundle già validi vengono saltati, quindi il bottone può
                // essere premuto di nuovo dopo un'interruzione.
                let map = try GLM52WeightMap(model: model)
                let reader = try GLM52PayloadReader(path: modelPath,
                                                    weightMap: map)
                let directory = ProcessInfo.processInfo
                    .environment["DS4_GLM_BUNDLE_DIR"]
                    ?? (modelPath + ".glm-experts")
                let shape = map.configuration.shape
                let sparse = Int(shape.nLeadingDense)..<Int(shape.inferenceLayerCount)
                var created = 0
                var skipped = 0
                for layer in sparse {
                    let built = try GLM52ExpertBundle.build(
                        directory: directory, layer: layer,
                        weightMap: map, reader: reader)
                    if built { created += 1 } else { skipped += 1 }
                    FileHandle.standardError.write(Data(
                        "DS4 expbundle: GLM blk\(layer) "
                        + "(\(created + skipped)/\(sparse.count))\n".utf8))
                }
                return "Bundle GLM pronti in \(directory): "
                    + "\(created) creati, \(skipped) già validi."
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
