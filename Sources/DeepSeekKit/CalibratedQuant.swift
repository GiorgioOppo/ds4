import Foundation

/// Metodi di quantizzazione disponibili per i pesi INT8/INT4.
/// `rtn` (round-to-nearest) è il baseline esistente; gli altri
/// metodi sono scaffold e richiedono statistiche di calibrazione.
///
/// Vedi `Sources/DeepSeekKit/Int8Quant.swift` per il path RTN
/// completo (`quantizeBF16ToInt8`) e
/// `Sources/DeepSeekKit/Kernels/int8_gemm.metal` per il GEMM
/// runtime. Questo modulo aggiunge l'interfaccia per AWQ/
/// SmoothQuant/GPTQ — l'algoritmo AWQ è implementato come "preview"
/// (vedi note in `awqQuantizeBF16ToInt8`), gli altri due sono
/// stub esplicitamente marcati `notImplemented`.
public enum QuantMethod: String, Sendable {
    /// Round-to-nearest, simmetrico, per-row × per-128 group. Baseline.
    case rtn
    /// Activation-aware Weight Quantization (Lin et al., 2023).
    /// Cerca uno scale per-canale che minimizza l'errore di
    /// quantizzazione *pesato* dalla magnitudine delle activations.
    /// Richiede `CalibrationStats` con per-channel absmax.
    case awq
    /// SmoothQuant (Xiao et al., 2022). Migra il "difficulty" da
    /// activations a weights tramite un smoothing factor per-canale.
    /// Stub.
    case smoothQuant
    /// GPTQ (Frantar et al., 2022). Quantizzazione layer-by-layer
    /// con aggiornamento OBS basato su Hessian approssimata.
    /// Stub.
    case gptq
}

/// Statistiche di calibrazione collezionate da un forward-pass su
/// un dataset di calibrazione. Per-channel absmax è il minimo
/// necessario per AWQ; medie e varianze servirebbero anche per
/// GPTQ e per smoothing più sofisticati.
///
/// La COLLEZIONE delle stats (hooking del forward pass) è esplicitamente
/// fuori scope di questo modulo — vedi `ActivationObserver` per il
/// pattern previsto (scaffold).
public struct CalibrationStats: Sendable {
    /// `[inDim]` — absmax per-canale dell'input al layer su tutto
    /// il calibration set.
    public let perChannelAbsMax: [Float]
    /// `[inDim]` opzionale — media per-canale (per smoothing più fini).
    public let perChannelMean: [Float]?
    /// Numero di token osservati durante la calibrazione (per
    /// diagnostica e per stimare la rumorosità delle stats).
    public let observedTokens: Int

    public init(perChannelAbsMax: [Float],
                perChannelMean: [Float]? = nil,
                observedTokens: Int = 0) {
        self.perChannelAbsMax = perChannelAbsMax
        self.perChannelMean = perChannelMean
        self.observedTokens = observedTokens
    }
}

/// Hook per la collezione di `CalibrationStats` durante un forward
/// pass. Scaffold — non ancora wirato in `Linear.swift` / `MLA.swift`
/// / etc.
///
/// Pattern d'uso previsto:
///
///     let observer = ActivationObserver()
///     // run calibration forward pass with observer.recordActivation
///     // hooked into each Linear.forward
///     let stats = observer.finalize(for: "layer.0.attn.wq")
///     let (w, s) = try quantizeBF16ToInt8Calibrated(
///         srcURL: ..., srcOffset: ..., outDim: ..., inDim: ...,
///         method: .awq, stats: stats)
public final class ActivationObserver: @unchecked Sendable {
    private let lock = NSLock()
    private var perLayerAbsMax: [String: [Float]] = [:]
    private var perLayerMean: [String: [Float]] = [:]
    private var perLayerCount: [String: Int] = [:]

    public init() {}

    /// Registra un'observation per `layerName`. `x` è `[N, inDim]`
    /// row-major. Update incrementale di absmax e somma per-canale.
    public func recordActivation(_ layerName: String,
                                  _ x: UnsafePointer<Float>,
                                  rows: Int,
                                  inDim: Int) {
        lock.lock(); defer { lock.unlock() }
        var absMax = perLayerAbsMax[layerName] ?? [Float](repeating: 0, count: inDim)
        var mean = perLayerMean[layerName] ?? [Float](repeating: 0, count: inDim)
        let prevCount = perLayerCount[layerName] ?? 0

        precondition(absMax.count == inDim,
                     "ActivationObserver: inDim mismatch for \(layerName)")

        for r in 0..<rows {
            for c in 0..<inDim {
                let v = x[r * inDim + c]
                let a = abs(v)
                if a > absMax[c] { absMax[c] = a }
                mean[c] += v
            }
        }
        perLayerAbsMax[layerName] = absMax
        perLayerMean[layerName] = mean
        perLayerCount[layerName] = prevCount + rows
    }

    /// Snapshot finale delle stats per `layerName`. Calcola la media
    /// dividendo la somma per il count. Restituisce `nil` se il
    /// layer non è mai stato osservato.
    public func finalize(for layerName: String) -> CalibrationStats? {
        lock.lock(); defer { lock.unlock() }
        guard let absMax = perLayerAbsMax[layerName],
              let sum = perLayerMean[layerName],
              let count = perLayerCount[layerName], count > 0 else {
            return nil
        }
        let mean = sum.map { $0 / Float(count) }
        return CalibrationStats(perChannelAbsMax: absMax,
                                perChannelMean: mean,
                                observedTokens: count)
    }
}

// MARK: - Calibrated quantization entry point

/// Errore restituito quando un metodo di calibrazione non è ancora
/// implementato.
public struct QuantNotImplemented: Error, CustomStringConvertible {
    public let method: QuantMethod
    public var description: String {
        "Quantization method `.\(method.rawValue)` is a scaffold and not yet implemented."
    }
}

/// Entry-point unificato per la quantizzazione INT8 dei pesi BF16,
/// con scelta del metodo.
///
/// - `.rtn`: delega a `quantizeBF16ToInt8` esistente. Le stats sono
///   ignorate.
/// - `.awq`: applica un per-channel smoothing pre-RTN basato sulla
///   formula AWQ `s = clip(act_amax^alpha · w_amax^(1-alpha), eps)`.
///   I pesi smussati vengono poi quantizzati con RTN. Vedi nota in
///   `awqQuantizeBF16ToInt8` sulla mancanza del runtime activation
///   inverse-scale (TODO follow-up: wirare l'`1/s` come pre-multiply
///   nel forward del layer).
/// - `.smoothQuant` / `.gptq`: throwa `QuantNotImplemented`.
public func quantizeBF16ToInt8Calibrated(
    srcURL: URL, srcOffset: Int,
    outDim: Int, inDim: Int,
    method: QuantMethod,
    stats: CalibrationStats?,
    awqAlpha: Float = 0.5
) throws -> (weight: Data, scale: Data) {
    switch method {
    case .rtn:
        return try quantizeBF16ToInt8(srcURL: srcURL,
                                       srcOffset: srcOffset,
                                       outDim: outDim,
                                       inDim: inDim)
    case .awq:
        guard let stats else {
            throw NSError(domain: "quantizeBF16ToInt8Calibrated",
                          code: 100,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "AWQ requires CalibrationStats"])
        }
        return try awqQuantizeBF16ToInt8(srcURL: srcURL,
                                          srcOffset: srcOffset,
                                          outDim: outDim, inDim: inDim,
                                          actAbsMax: stats.perChannelAbsMax,
                                          alpha: awqAlpha)
    case .smoothQuant, .gptq:
        throw QuantNotImplemented(method: method)
    }
}

// MARK: - AWQ implementation (preview)

/// Implementazione AWQ "preview": calcola i per-channel scales,
/// smooth-multiplica i pesi e poi delega a un quantizer RTN
/// in-memory (ricalcato dalla `quantizeRowFromFloat` di
/// `Int8Quant.swift`, qui replicato per non dover esporre il
/// privato).
///
/// ⚠️ LIMITAZIONE: AWQ per essere matematicamente corretto richiede
/// che al runtime le activations vengano divise per `s` per-channel
/// PRIMA di entrare nel layer (la moltiplicazione dei pesi per `s`
/// e la divisione delle activations per `s` si cancellano in
/// esatto). Questo modulo NON wira ancora quell'inverse-scale —
/// quindi il path produce un quant valido ma matematicamente
/// shiftato. Per il QAT scenario (re-train brevemente dopo la
/// quant) è comunque utile; per drop-in inference va aggiunto un
/// vettore di scales per-channel che `Linear.forward` legge e
/// applica via `Elementwise.scale` prima del GEMM.
///
/// Follow-up: aggiungere un campo `inverseChannelScale: Tensor?`
/// a `Linear` + un pre-mul nel forward quando presente.
func awqQuantizeBF16ToInt8(
    srcURL: URL, srcOffset: Int,
    outDim: Int, inDim: Int,
    actAbsMax: [Float],
    alpha: Float
) throws -> (weight: Data, scale: Data) {
    precondition(inDim % kInt8GroupK == 0,
                 "AWQ INT8 requires inDim % \(kInt8GroupK) == 0")
    precondition(actAbsMax.count == inDim,
                 "AWQ stats: expected actAbsMax.count == inDim, got \(actAbsMax.count) vs \(inDim)")

    let blocksIn = inDim / kInt8GroupK

    // 1) Carica i pesi BF16 e convertili a Float in-memory.
    guard let fh = FileHandle(forReadingAtPath: srcURL.path) else {
        throw NSError(domain: "awqQuantizeBF16ToInt8", code: 1)
    }
    defer { try? fh.close() }
    try fh.seek(toOffset: UInt64(srcOffset))
    let byteLen = outDim * inDim * 2
    guard let bf16Bytes = try fh.read(upToCount: byteLen),
          bf16Bytes.count == byteLen else {
        throw NSError(domain: "awqQuantizeBF16ToInt8", code: 2)
    }

    // 2) Calcola per-channel weight absmax (max over outDim per ogni inDim).
    var weightAbsMax = [Float](repeating: 0, count: inDim)
    bf16Bytes.withUnsafeBytes { raw in
        let bPtr = raw.bindMemory(to: UInt16.self).baseAddress!
        for r in 0..<outDim {
            for c in 0..<inDim {
                let v = abs(bf16ToFloat(bPtr[r * inDim + c]))
                if v > weightAbsMax[c] { weightAbsMax[c] = v }
            }
        }
    }

    // 3) Calcola lo scale AWQ per-channel:
    //    s[c] = max(actAmax[c]^alpha * weightAmax[c]^(1-alpha), eps)
    //
    // Lo scale è poi normalizzato dividendo per geom-mean per
    // mantenere la stessa scala media dei pesi (best practice
    // riportata nel paper AWQ, sezione 4.1).
    let eps: Float = 1e-4
    var channelScale = [Float](repeating: 1, count: inDim)
    var logSum: Float = 0
    for c in 0..<inDim {
        let aa = max(actAbsMax[c], eps)
        let ww = max(weightAbsMax[c], eps)
        let s = max(pow(aa, alpha) * pow(ww, 1 - alpha), eps)
        channelScale[c] = s
        logSum += log(s)
    }
    let geomMean = exp(logSum / Float(inDim))
    for c in 0..<inDim {
        channelScale[c] /= geomMean
    }

    // 4) Applica lo smoothing ai pesi e quantizza RTN per-row,
    //    per-blocco-128.
    var weight = Data(count: outDim * inDim)
    var scale = Data(count: outDim * blocksIn * 2)

    weight.withUnsafeMutableBytes { wRaw in
        scale.withUnsafeMutableBytes { sRaw in
            bf16Bytes.withUnsafeBytes { bRaw in
                let wPtr = wRaw.bindMemory(to: Int8.self).baseAddress!
                let sPtr = sRaw.bindMemory(to: UInt16.self).baseAddress!
                let bPtr = bRaw.bindMemory(to: UInt16.self).baseAddress!
                let csPtr = channelScale.withUnsafeBufferPointer { $0.baseAddress! }

                DispatchQueue.concurrentPerform(iterations: outDim) { r in
                    let rowIn = bPtr.advanced(by: r * inDim)
                    let outRow = wPtr.advanced(by: r * inDim)
                    let scaleRow = sPtr.advanced(by: r * blocksIn)

                    // Materializza la riga smussata: w'[c] = w[c] * s[c].
                    var rowF = [Float](repeating: 0, count: inDim)
                    rowF.withUnsafeMutableBufferPointer { buf in
                        for c in 0..<inDim {
                            buf[c] = bf16ToFloat(rowIn[c]) * csPtr[c]
                        }
                        // RTN per-blocco identico a `quantizeRowFromFloat`
                        // di Int8Quant.swift (replicato inline per
                        // non esporre il privato).
                        for sb in 0..<blocksIn {
                            let base = sb * kInt8GroupK
                            var maxAbs: Float = 0
                            for k in 0..<kInt8GroupK {
                                let v = abs(buf[base + k])
                                if v > maxAbs { maxAbs = v }
                            }
                            let s: Float; let invS: Float
                            if maxAbs == 0 { s = 0; invS = 0 }
                            else { s = maxAbs / 127.0; invS = 127.0 / maxAbs }
                            scaleRow[sb] = floatToF16Local(s)
                            for k in 0..<kInt8GroupK {
                                let q = buf[base + k] * invS
                                let rr = q.rounded(.toNearestOrEven)
                                let clamped = min(max(rr, -127), 127)
                                outRow[base + k] = Int8(clamped)
                            }
                        }
                    }
                }
            }
        }
    }

    return (weight, scale)
}
