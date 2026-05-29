import Foundation
import Accelerate

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
        // Drop the inverseChannelScale tuple element — the simple
        // public API stays a (weight, scale) tuple. Callers that
        // want the inverse scale call `awqQuantizeBF16ToInt8`
        // directly.
        let r = try awqQuantizeBF16ToInt8(srcURL: srcURL,
                                            srcOffset: srcOffset,
                                            outDim: outDim, inDim: inDim,
                                            actAbsMax: stats.perChannelAbsMax,
                                            alpha: awqAlpha)
        return (r.weight, r.scale)
    case .smoothQuant:
        guard let stats else {
            throw NSError(domain: "quantizeBF16ToInt8Calibrated", code: 101,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "SmoothQuant requires CalibrationStats"])
        }
        let r = try smoothQuantBF16ToInt8(srcURL: srcURL,
                                            srcOffset: srcOffset,
                                            outDim: outDim, inDim: inDim,
                                            actAbsMax: stats.perChannelAbsMax,
                                            alpha: 0.5)
        return (r.weight, r.scale)
    case .gptq:
        throw NSError(domain: "quantizeBF16ToInt8Calibrated", code: 102,
                      userInfo: [NSLocalizedDescriptionKey:
                                  "GPTQ requires the per-layer Hessian — call gptqQuantizeBF16ToInt8 directly"])
    }
}

/// Calibrated entry-point variant that accepts an explicit
/// `hessian: [Double]` for GPTQ. The other methods (`.rtn`, `.awq`,
/// `.smoothQuant`) ignore it and delegate to
/// `quantizeBF16ToInt8Calibrated`. Separate signature because GPTQ
/// needs the full `inDim × inDim` symmetric PD Hessian (collected
/// by `HessianObserver`), which is heavy enough not to want in the
/// common-path API.
public func quantizeBF16ToInt8Calibrated(
    srcURL: URL, srcOffset: Int,
    outDim: Int, inDim: Int,
    method: QuantMethod,
    stats: CalibrationStats?,
    hessian: [Double]?,
    awqAlpha: Float = 0.5,
    gptqPercentDamp: Float = 0.01,
    gptqActOrder: Bool = false
) throws -> (weight: Data, scale: Data) {
    if case .gptq = method {
        guard let hessian else {
            throw NSError(domain: "quantizeBF16ToInt8Calibrated", code: 103,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "GPTQ requires a Hessian"])
        }
        return try gptqQuantizeBF16ToInt8(
            srcURL: srcURL, srcOffset: srcOffset,
            outDim: outDim, inDim: inDim,
            hessian: hessian,
            percentDamp: gptqPercentDamp,
            actOrder: gptqActOrder)
    }
    return try quantizeBF16ToInt8Calibrated(
        srcURL: srcURL, srcOffset: srcOffset,
        outDim: outDim, inDim: inDim,
        method: method, stats: stats, awqAlpha: awqAlpha)
}

// MARK: - AWQ implementation (preview)

/// Implementazione AWQ "preview": calcola i per-channel scales,
/// smooth-multiplica i pesi e poi delega a un quantizer RTN
/// in-memory (ricalcato dalla `quantizeRowFromFloat` di
/// `Int8Quant.swift`, qui replicato per non dover esporre il
/// privato).
///
/// ⚠️ LIMITAZIONE (lato I/O, non runtime): AWQ per essere
/// matematicamente corretto richiede che al runtime le activations
/// vengano moltiplicate per `1/s` per-channel PRIMA del GEMM (la
/// moltiplicazione dei pesi per `s` e quella delle activations per
/// `1/s` si cancellano in esatto). Quel pre-mul ESISTE già:
/// `Linear` applica `inverseChannelScale` nel suo forward quando il
/// tensore è presente. Ciò che MANCA è la PERSISTENZA — il converter
/// non scrive ancora il vettore `1/s` come tensore sidecar nel
/// checkpoint e il loader non lo rilegge, quindi finché quel
/// round-trip I/O non è cablato i pesi sarebbero matematicamente
/// shiftati. Per questo il converter rifiuta `--quant-method
/// awq|smoothQuant` invece di emettere pesi silenziosamente errati.
/// AWQ output tuple. `inverseChannelScale` is `1 / s[c]` ready to
/// plug into `Linear.inverseChannelScale`: the runtime applies it
/// as `x' = x · inverseChannelScale` before the GEMM, recovering
/// the un-smoothed output exactly. Length is `inDim`.
public typealias CalibratedQuantResult = (
    weight: Data, scale: Data, inverseChannelScale: [Float])

public func awqQuantizeBF16ToInt8(
    srcURL: URL, srcOffset: Int,
    outDim: Int, inDim: Int,
    actAbsMax: [Float],
    alpha: Float
) throws -> CalibratedQuantResult {
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

    // Invert channelScale so the runtime side (`Linear.inverseChannelScale`)
    // can multiply directly. `s == 0` shouldn't happen after the eps
    // clamp above, but guard anyway.
    var inverseChannelScale = [Float](repeating: 1, count: inDim)
    for c in 0..<inDim {
        inverseChannelScale[c] = channelScale[c] == 0
            ? 1 : 1.0 / channelScale[c]
    }
    return (weight, scale, inverseChannelScale)
}

// MARK: - SmoothQuant implementation

/// SmoothQuant (Xiao et al., 2022). The mechanism is identical in
/// spirit to AWQ — migrate quantization difficulty from activations
/// to weights via a per-channel scaling factor — but the math
/// differs from the AWQ preview in two ways:
///
///   1. Formula. AWQ uses
///        s[c] = (act_amax[c]^α · w_amax[c]^(1-α))^(1/N)  (then
///                                                       geom-mean normalize)
///      SmoothQuant uses
///        s[c] = max(act_amax[c], eps)^α
///                / max(w_amax[c], eps)^(1-α)
///      then clamps s into a "safe" range so a single extreme
///      channel doesn't blow up the rest of the row's dynamic
///      range. Paper uses α = 0.5; harder layers might want
///      α ∈ {0.7, 0.8}.
///
///   2. Normalization. AWQ divides every scale by the layer's
///      geometric mean so the overall weight scale stays put;
///      SmoothQuant clips s into [1/clampRange, clampRange]
///      (default `clampRange = 5`) to keep individual scales
///      bounded.
///
/// Same caveat as AWQ — and it's a PERSISTENCE gap, not a runtime
/// one: the math is only exact when the runtime activations get
/// multiplied by `1/s` per channel before the GEMM. The engine
/// already does that (`Linear.inverseChannelScale`); what's missing
/// is the converter persisting `1/s` to the checkpoint, so the
/// converter refuses `--quant-method smoothQuant` for now.
public func smoothQuantBF16ToInt8(
    srcURL: URL, srcOffset: Int,
    outDim: Int, inDim: Int,
    actAbsMax: [Float],
    alpha: Float,
    clampRange: Float = 5.0
) throws -> CalibratedQuantResult {
    precondition(inDim % kInt8GroupK == 0,
                  "SmoothQuant INT8 requires inDim % \(kInt8GroupK) == 0")
    precondition(actAbsMax.count == inDim)
    let blocksIn = inDim / kInt8GroupK

    // 1) Load BF16 weights into memory.
    guard let fh = FileHandle(forReadingAtPath: srcURL.path) else {
        throw NSError(domain: "smoothQuantBF16ToInt8", code: 1)
    }
    defer { try? fh.close() }
    try fh.seek(toOffset: UInt64(srcOffset))
    let byteLen = outDim * inDim * 2
    guard let bf16Bytes = try fh.read(upToCount: byteLen),
          bf16Bytes.count == byteLen
    else {
        throw NSError(domain: "smoothQuantBF16ToInt8", code: 2)
    }

    // 2) Per-channel weight absmax.
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

    // 3) SmoothQuant scales: s[c] = act_amax[c]^α / w_amax[c]^(1-α).
    //    Clamp into [1/clampRange, clampRange] so a single extreme
    //    channel doesn't punish the rest of the row.
    let eps: Float = 1e-5
    var channelScale = [Float](repeating: 1, count: inDim)
    for c in 0..<inDim {
        let aa = pow(max(actAbsMax[c], eps), alpha)
        let ww = pow(max(weightAbsMax[c], eps), 1 - alpha)
        var s = aa / ww
        if s < 1.0 / clampRange { s = 1.0 / clampRange }
        if s > clampRange         { s = clampRange }
        channelScale[c] = s
    }

    // 4) Apply smoothing to weights + RTN per-row, per-128-group.
    //    The body is identical to the AWQ implementation — the
    //    only thing that changes between the two methods is how
    //    `channelScale` was computed.
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
                    var rowF = [Float](repeating: 0, count: inDim)
                    rowF.withUnsafeMutableBufferPointer { buf in
                        for c in 0..<inDim {
                            buf[c] = bf16ToFloat(rowIn[c]) * csPtr[c]
                        }
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
    var inverseChannelScale = [Float](repeating: 1, count: inDim)
    for c in 0..<inDim {
        inverseChannelScale[c] = channelScale[c] == 0
            ? 1 : 1.0 / channelScale[c]
    }
    return (weight, scale, inverseChannelScale)
}

// MARK: - GPTQ implementation

/// GPTQ (Frantar et al., 2022). Quantizes a `[outDim, inDim]` BF16
/// weight tensor into the same INT8 + F16 group-scale layout the
/// existing kernel uses, but instead of RTN it walks columns left
/// to right and propagates the quantization error backward into
/// the still-unquantized columns using the OBS update:
///
///     err[i]   = (w[i] - quantize(w[i])) / U[i, i]
///     w[j>i]  -= err[i] * U[i, j]
///
/// where `U` is the upper Cholesky factor of the (damped) inverse
/// Hessian `H^{-1}`. The Hessian is the gram matrix of input
/// activations averaged over the calibration set,
///   H = E[x x^T]
/// and is supplied by the caller (see `HessianObserver`).
///
/// Algorithm 1 of the paper, adapted to grouped per-row per-128-K
/// quantization:
///   - We process one row of `W` at a time, in parallel across
///     rows (`DispatchQueue.concurrentPerform`). The Hessian /
///     Cholesky machinery is row-agnostic, so the rows share the
///     same `U`.
///   - Within a row, we walk columns left-to-right. Each time we
///     cross a 128-column boundary, we re-compute the per-block
///     scale from the *current* row (which reflects all prior
///     error propagation).
///   - The error from quantizing column `i` propagates to every
///     column `j > i` in the same row via `U[i, j]`, including
///     across block boundaries — that's the whole point.
///
/// Cost (single-thread): ~4 × outDim × inDim^2 FLOPs per layer
/// (the inner OBS update is the bottleneck), plus one
/// `O(inDim^3)` Cholesky-of-inverse setup. For a Llama 7B Linear
/// (inDim = 4096, outDim = 4096) that's ~68 GFLOPs per row × 4k
/// rows / cores ≈ a couple minutes per layer on M-series. The
/// `feed_forward_length` ones (inDim ≈ 11k) are ~3 × heavier.
/// Plan an overnight run for a full 7B model.
///
/// `actOrder = true` enables the "act-order" variant from the
/// paper: columns are permuted by descending Hessian diagonal so
/// the highest-leverage columns get quantized first. Improves
/// quality at the cost of a permutation that the runtime kernel
/// would have to track — we leave the unpermuted output for now
/// (i.e. when `actOrder = true` we still permute internally but
/// un-permute before writing the output to match the existing
/// kernel layout). Set to `false` for the simpler path.
public func gptqQuantizeBF16ToInt8(
    srcURL: URL, srcOffset: Int,
    outDim: Int, inDim: Int,
    hessian: [Double],
    percentDamp: Float = 0.01,
    actOrder: Bool = false
) throws -> (weight: Data, scale: Data) {
    precondition(inDim % kInt8GroupK == 0,
                  "GPTQ INT8 requires inDim % \(kInt8GroupK) == 0")
    precondition(hessian.count == inDim * inDim,
                  "GPTQ expects Hessian sized \(inDim)×\(inDim)")
    // Per-group scales + act_order would need the kernel to track a
    // column permutation; the existing `gemm_int8_w8a16_*` kernels
    // don't. Block the combination until a permutation-aware kernel
    // lands (TODO §1 follow-up).
    precondition(!actOrder,
                  "GPTQ act_order is not yet supported alongside per-group "
                  + "scales — the runtime kernel doesn't track the column "
                  + "permutation. Set actOrder: false for now.")
    let blocksIn = inDim / kInt8GroupK

    // 1) Optional act-order permutation. Sort columns by
    //    descending Hessian diagonal magnitude (proxy for column
    //    importance: bigger diag → more leverage on output).
    //    `permutation[k]` is the original column index now sitting
    //    at position k in the permuted order.
    let n = inDim
    var permutation: [Int] = Array(0..<n)
    var inversePermutation: [Int] = Array(0..<n)
    if actOrder {
        permutation.sort {
            hessian[$0 * n + $0] > hessian[$1 * n + $1]
        }
        for (newPos, origPos) in permutation.enumerated() {
            inversePermutation[origPos] = newPos
        }
    }

    // 2) Build a permuted, damped Hessian H' = P H P^T + damp·I.
    //    Damping shifts the spectrum so the Cholesky is
    //    well-conditioned (Frantar uses 1% of mean(diag(H))).
    var H = [Double](repeating: 0, count: n * n)
    if actOrder {
        for i in 0..<n {
            let oi = permutation[i]
            for j in 0..<n {
                let oj = permutation[j]
                H[i * n + j] = hessian[oi * n + oj]
            }
        }
    } else {
        H = hessian
    }
    var diagMean: Double = 0
    for i in 0..<n { diagMean += H[i * n + i] }
    diagMean /= Double(n)
    let damp = Double(percentDamp) * diagMean
    for i in 0..<n { H[i * n + i] += damp }

    // 3) Cholesky factor of H^{-1}, upper. Three LAPACK steps:
    //      a. dpotrf(H, 'L')         → H = L L^T (lower in column-major)
    //      b. dpotri(H, 'L')         → H = H^{-1} (lower triangle filled)
    //      c. mirror lower → upper   → symmetric H^{-1}
    //      d. dpotrf(H, 'U')         → H = U^T U, U upper triangular
    //                                    in column-major == row-major
    //                                    upper. U^T U = H^{-1}.
    //    For symmetric input the column-major / row-major
    //    distinction collapses; the triangular outputs we use
    //    `U[i, j]` for i ≤ j, address as `U[i * n + j]` (row-major).
    var lapackN = __CLPK_integer(n)
    var lapackLDA = __CLPK_integer(n)
    var info: __CLPK_integer = 0
    var uplo: Int8 = 76 // 'L'
    H.withUnsafeMutableBufferPointer { ptr in
        dpotrf_(&uplo, &lapackN, ptr.baseAddress!, &lapackLDA, &info)
    }
    guard info == 0 else {
        throw NSError(domain: "gptqQuantizeBF16ToInt8", code: 10,
                       userInfo: [NSLocalizedDescriptionKey:
                                   "Cholesky factorization failed (info=\(info)); "
                                   + "increase --gptq-damp"])
    }
    H.withUnsafeMutableBufferPointer { ptr in
        dpotri_(&uplo, &lapackN, ptr.baseAddress!, &lapackLDA, &info)
    }
    guard info == 0 else {
        throw NSError(domain: "gptqQuantizeBF16ToInt8", code: 11)
    }
    // dpotri filled the column-major lower triangle (which is the
    // row-major upper triangle after the transposition flip). For
    // a symmetric matrix the data is the same either way; just
    // mirror to both halves.
    for i in 0..<n {
        for j in 0..<i {
            // column-major lower entry at (i, j), i > j, lives at
            // H[j * n + i] in column-major addressing. In our
            // storage that's the row-major upper triangle. Mirror
            // it to the row-major lower (and vice versa).
            let upper = H[j * n + i]
            H[i * n + j] = upper
        }
    }
    uplo = 85 // 'U'
    H.withUnsafeMutableBufferPointer { ptr in
        dpotrf_(&uplo, &lapackN, ptr.baseAddress!, &lapackLDA, &info)
    }
    guard info == 0 else {
        throw NSError(domain: "gptqQuantizeBF16ToInt8", code: 12)
    }
    // H now holds U in column-major upper. In row-major
    // addressing (storage[i * n + j]), `U[i, j]` for i ≤ j is the
    // value we want for the OBS update.

    // 4) Load BF16 weights. Apply column permutation if act-order
    //    is on so the row vectors line up with the permuted U.
    guard let fh = FileHandle(forReadingAtPath: srcURL.path) else {
        throw NSError(domain: "gptqQuantizeBF16ToInt8", code: 1)
    }
    defer { try? fh.close() }
    try fh.seek(toOffset: UInt64(srcOffset))
    let byteLen = outDim * n * 2
    guard let bf16Bytes = try fh.read(upToCount: byteLen),
          bf16Bytes.count == byteLen
    else {
        throw NSError(domain: "gptqQuantizeBF16ToInt8", code: 2)
    }

    // 5) Per-row GPTQ in parallel.
    var weight = Data(count: outDim * n)
    var scale = Data(count: outDim * blocksIn * 2)

    let permArr = permutation
    // `inversePermutation` would only matter if we kept the
    // quantized output in permuted order; today we un-permute via
    // `permArr` directly so the runtime kernel sees identity layout.
    _ = inversePermutation
    weight.withUnsafeMutableBytes { wRaw in
        scale.withUnsafeMutableBytes { sRaw in
            bf16Bytes.withUnsafeBytes { bRaw in
                H.withUnsafeBufferPointer { uPtr in
                    let wPtr = wRaw.bindMemory(to: Int8.self).baseAddress!
                    let sPtr = sRaw.bindMemory(to: UInt16.self).baseAddress!
                    let bPtr = bRaw.bindMemory(to: UInt16.self).baseAddress!
                    let UPtr = uPtr.baseAddress!

                    DispatchQueue.concurrentPerform(iterations: outDim) { r in
                        let rowIn = bPtr.advanced(by: r * n)
                        let outRow = wPtr.advanced(by: r * n)
                        let scaleRow = sPtr.advanced(by: r * blocksIn)

                        // Materialize row + apply column permutation.
                        var rowF = [Float](repeating: 0, count: n)
                        for c in 0..<n {
                            let src = permArr[c]
                            rowF[c] = bf16ToFloat(rowIn[src])
                        }

                        // Per-group buffers (in permuted order).
                        var qPermuted = [Int8](repeating: 0, count: n)
                        var sPermuted = [UInt16](repeating: 0, count: blocksIn)

                        for blockStart in stride(from: 0, to: n, by: kInt8GroupK) {
                            let blockEnd = min(blockStart + kInt8GroupK, n)
                            // Scale: absmax of the *current* row in
                            // this block (post-prior-error propagation).
                            var maxAbs: Float = 0
                            for k in blockStart..<blockEnd {
                                let v = abs(rowF[k])
                                if v > maxAbs { maxAbs = v }
                            }
                            let bScale: Float
                            let invS: Float
                            if maxAbs == 0 { bScale = 0; invS = 0 }
                            else { bScale = maxAbs / 127.0; invS = 127.0 / maxAbs }
                            sPermuted[blockStart / kInt8GroupK] = floatToF16Local(bScale)

                            for col in blockStart..<blockEnd {
                                let q = (rowF[col] * invS).rounded(.toNearestOrEven)
                                let clamped = Int8(min(max(q, -127), 127))
                                qPermuted[col] = clamped
                                let qFloat = Float(clamped) * bScale
                                // U[col, col] diagonal — column-major
                                // upper triangular, addressed row-major
                                // as U[col * n + col].
                                let d = Float(UPtr[col * n + col])
                                guard d != 0 else { continue }
                                let err = (rowF[col] - qFloat) / d
                                // Propagate to remaining columns in the
                                // row. `U[col, j]` for j > col lives at
                                // `UPtr[col * n + j]` in row-major
                                // address — but the column-major-upper
                                // storage of U writes the upper triangle
                                // at storage[j * n + col]. Mirror once
                                // (we already mirrored in step 3? No —
                                // step 3's mirror was on H^{-1}, before
                                // the final Cholesky). dpotrf with 'U'
                                // writes the col-major UPPER triangle,
                                // which is row-major LOWER → so U[col,
                                // j] for j > col, in our col-major
                                // upper storage, sits at storage[j * n
                                // + col].
                                for j in (col + 1)..<n {
                                    rowF[j] -= err * Float(UPtr[j * n + col])
                                }
                            }
                        }

                        // Un-permute back to the original column order
                        // so the kernel sees the same layout as the
                        // unpermuted RTN path. With actOrder=false
                        // (the only path we currently allow), permArr
                        // is the identity and this is a straight copy.
                        for c in 0..<n {
                            outRow[permArr[c]] = qPermuted[c]
                        }
                        for sb in 0..<blocksIn {
                            scaleRow[sb] = sPermuted[sb]
                        }
                    }
                }
            }
        }
    }
    return (weight, scale)
}

// MARK: - On-disk calibration format

/// Decodable mirror of the `stats.json` shape written by
/// `deepseek_calibrate`. Per-layer activation observations only;
/// Hessians live in a sibling `hessians/<name>.f64` directory.
public struct CalibrationStatsFile: Codable {
    public let model: String
    public let nLayers: Int
    public let hessianCollected: Bool
    public let layers: [LayerStats]

    public init(model: String, nLayers: Int,
                hessianCollected: Bool, layers: [LayerStats])
    {
        self.model = model
        self.nLayers = nLayers
        self.hessianCollected = hessianCollected
        self.layers = layers
    }

    public struct LayerStats: Codable {
        public let name: String
        public let inDim: Int
        public let observedTokens: Int
        public let perChannelAbsMax: [Float]
        public let perChannelMean: [Float]

        public init(name: String, inDim: Int, observedTokens: Int,
                    perChannelAbsMax: [Float],
                    perChannelMean: [Float])
        {
            self.name = name
            self.inDim = inDim
            self.observedTokens = observedTokens
            self.perChannelAbsMax = perChannelAbsMax
            self.perChannelMean = perChannelMean
        }
    }
}

/// In-memory handle on a `deepseek_calibrate` output directory.
/// Wraps the `stats.json` plus an optional path to the
/// `hessians/` subdirectory so the converter can look up
/// per-tensor calibration without re-parsing per call.
public struct CalibrationDir {
    public let statsByName: [String: CalibrationStats]
    public let inDimByName: [String: Int]
    public let hessianDirURL: URL?

    /// Read `stats.json` from `url` (or `<url>/stats.json` if a
    /// directory). If `stats.hessianCollected` is true, populate
    /// `hessianDirURL` with the sibling `hessians/` folder so
    /// `hessian(for:)` lookups can stream the binaries.
    public init(url: URL) throws {
        let fm = FileManager.default
        var statsURL = url
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: url.path, isDirectory: &isDir),
           isDir.boolValue
        {
            statsURL = url.appendingPathComponent("stats.json")
        }
        let data = try Data(contentsOf: statsURL)
        let parsed = try JSONDecoder().decode(
            CalibrationStatsFile.self, from: data)

        var byName: [String: CalibrationStats] = [:]
        var dimByName: [String: Int] = [:]
        for layer in parsed.layers {
            byName[layer.name] = CalibrationStats(
                perChannelAbsMax: layer.perChannelAbsMax,
                perChannelMean: layer.perChannelMean.isEmpty
                    ? nil : layer.perChannelMean,
                observedTokens: layer.observedTokens)
            dimByName[layer.name] = layer.inDim
        }
        self.statsByName = byName
        self.inDimByName = dimByName
        if parsed.hessianCollected {
            let dir = statsURL.deletingLastPathComponent()
                .appendingPathComponent("hessians")
            self.hessianDirURL = fm.fileExists(atPath: dir.path) ? dir : nil
        } else {
            self.hessianDirURL = nil
        }
    }

    /// Look up the Hessian for `layerName`. Returns nil when no
    /// hessian directory was paired with this calibration or when
    /// the per-layer file is missing. The file is `[inDim*inDim]`
    /// little-endian Doubles (no header) — we trust the caller to
    /// have the right `inDim` from the stats lookup.
    public func hessian(for layerName: String) -> [Double]? {
        guard let dir = hessianDirURL else { return nil }
        guard let inDim = inDimByName[layerName] else { return nil }
        let path = dir.appendingPathComponent("\(layerName).f64")
        guard let data = try? Data(contentsOf: path) else { return nil }
        let count = inDim * inDim
        let expectedBytes = count * MemoryLayout<Double>.size
        guard data.count == expectedBytes else { return nil }
        var out = [Double](repeating: 0, count: count)
        _ = out.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst)
        }
        return out
    }
}

// MARK: - Hessian collection

/// Activation observer specialized for GPTQ: instead of just
/// tracking per-channel statistics, accumulates the full
/// `inDim × inDim` Hessian H = E[x x^T] across the calibration
/// set. Use the BLAS rank-update kernel (`cblas_dgemm`) so each
/// batch contributes its outer-product in O(batch · inDim²) FLOPs
/// at peak BLAS throughput.
///
/// Memory: one `Double[inDim × inDim]` per layer. For a Llama 7B
/// `q/k/v/o` (inDim = 4096) that's 128 MB / layer; for `gate/up/
/// down` (inDim ≈ 11k) it's about 970 MB. Plan accordingly — the
/// observer keeps every layer's matrix in memory until you ask
/// for it back, so on a 32 GB Mac you'll want to finalize +
/// release after each layer's calibration block rather than
/// running all layers at once.
public final class HessianObserver: @unchecked Sendable {
    private let lock = NSLock()
    private var perLayerH: [String: [Double]] = [:]
    private var perLayerInDim: [String: Int] = [:]
    private var perLayerCount: [String: Int] = [:]

    public init() {}

    /// Accumulate one calibration batch's contribution. `x` is
    /// `[rows × inDim]` row-major F32 (the input to the layer's
    /// GEMM). Updates `H[layerName] += x^T x` via dgemm.
    public func recordBatch(_ layerName: String,
                              _ x: UnsafePointer<Float>,
                              rows: Int, inDim: Int) {
        lock.lock(); defer { lock.unlock() }
        precondition(rows > 0 && inDim > 0)
        if let existing = perLayerInDim[layerName] {
            precondition(existing == inDim,
                          "HessianObserver: inDim mismatch for \(layerName)")
        }
        var H = perLayerH[layerName]
            ?? [Double](repeating: 0, count: inDim * inDim)
        // Convert F32 batch to F64 for accumulation stability.
        var xD = [Double](repeating: 0, count: rows * inDim)
        for k in 0..<rows * inDim { xD[k] = Double(x[k]) }
        // H += X^T X. Row-major X[rows, inDim] reinterpreted as
        // col-major is X^T[inDim, rows]. dgemm('N', 'T', inDim,
        // inDim, rows, 1.0, X_data, inDim, X_data, inDim, 1.0, H,
        // inDim) computes (X^T) (X^T)^T = X^T X in the row-major
        // mental model.
        let m = __CLPK_integer(inDim)
        let k = __CLPK_integer(rows)
        let alpha: Double = 1.0
        let beta: Double = 1.0
        xD.withUnsafeBufferPointer { xBuf in
            H.withUnsafeMutableBufferPointer { hBuf in
                cblas_dgemm(
                    CblasRowMajor,
                    CblasTrans,        // op(A) = A^T
                    CblasNoTrans,      // op(B) = B
                    m, m, k,
                    alpha,
                    xBuf.baseAddress!, m,
                    xBuf.baseAddress!, m,
                    beta,
                    hBuf.baseAddress!, m)
            }
        }
        perLayerH[layerName] = H
        perLayerInDim[layerName] = inDim
        perLayerCount[layerName] = (perLayerCount[layerName] ?? 0) + rows
    }

    /// Snapshot the accumulated Hessian for `layerName`, normalized
    /// by the observed token count. Returns nil if no batches were
    /// recorded for that layer.
    public func finalize(for layerName: String) -> (hessian: [Double], inDim: Int, observedTokens: Int)? {
        lock.lock(); defer { lock.unlock() }
        guard let H = perLayerH[layerName],
              let inDim = perLayerInDim[layerName],
              let count = perLayerCount[layerName], count > 0
        else { return nil }
        let invN = 1.0 / Double(count)
        var Hn = H
        for k in 0..<Hn.count { Hn[k] *= invN }
        return (Hn, inDim, count)
    }

    /// Free the matrix for one layer once the caller is done with
    /// it — necessary on large models where holding every layer's
    /// Hessian simultaneously would blow the budget.
    public func releaseLayer(_ layerName: String) {
        lock.lock(); defer { lock.unlock() }
        perLayerH.removeValue(forKey: layerName)
        perLayerInDim.removeValue(forKey: layerName)
        perLayerCount.removeValue(forKey: layerName)
    }

    /// Names of layers observed so far.
    public func observedLayers() -> [String] {
        lock.lock(); defer { lock.unlock() }
        return Array(perLayerH.keys)
    }
}
