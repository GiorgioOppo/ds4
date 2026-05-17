import Foundation
import Metal

/// Wrapper Hashable per i function constants di una pipeline Metal.
/// `MTLFunctionConstantValues` è opaco (no API per ispezionarne il
/// contenuto), quindi non possiamo hashare quello: hashiamo invece
/// le tuple `(type, raw-bytes, index)` con cui lo costruiremmo. Il
/// metodo `mtl()` rigenera l'oggetto `MTLFunctionConstantValues` da
/// passare a `makeFunction(name:constantValues:)`.
public struct PipelineConstants: Hashable, Sendable {
    public struct Entry: Hashable, Sendable {
        public let typeRaw: UInt
        public let data: Data
        public let index: Int

        public init(typeRaw: UInt, data: Data, index: Int) {
            self.typeRaw = typeRaw
            self.data = data
            self.index = index
        }
    }

    public private(set) var entries: [Entry] = []

    public init() {}

    /// Costruttore ergonomico in stile builder:
    ///   `PipelineConstants { $0.setUInt32(64, at: 0) }`
    public init(_ builder: (inout PipelineConstants) -> Void) {
        self.init()
        builder(&self)
    }

    public mutating func setUInt32(_ value: UInt32, at index: Int) {
        var v = value
        let data = Data(bytes: &v, count: 4)
        entries.append(Entry(
            typeRaw: UInt(MTLDataType.uint.rawValue),
            data: data,
            index: index))
    }

    public mutating func setInt32(_ value: Int32, at index: Int) {
        var v = value
        let data = Data(bytes: &v, count: 4)
        entries.append(Entry(
            typeRaw: UInt(MTLDataType.int.rawValue),
            data: data,
            index: index))
    }

    public mutating func setFloat(_ value: Float, at index: Int) {
        var v = value
        let data = Data(bytes: &v, count: 4)
        entries.append(Entry(
            typeRaw: UInt(MTLDataType.float.rawValue),
            data: data,
            index: index))
    }

    public mutating func setBool(_ value: Bool, at index: Int) {
        var v: UInt8 = value ? 1 : 0
        let data = Data(bytes: &v, count: 1)
        entries.append(Entry(
            typeRaw: UInt(MTLDataType.bool.rawValue),
            data: data,
            index: index))
    }

    /// Rigenera un `MTLFunctionConstantValues` dagli entries.
    /// Da chiamare ogni volta che serve una nuova istanza per
    /// `makeFunction(name:constantValues:)` — non riusare.
    public func mtl() -> MTLFunctionConstantValues {
        let constants = MTLFunctionConstantValues()
        for entry in entries {
            guard let type = MTLDataType(rawValue: entry.typeRaw) else {
                // Skip silently: il tipo non è più valido in questa
                // versione di Metal. Non blocchiamo il build.
                continue
            }
            entry.data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                constants.setConstantValue(base,
                                            type: type,
                                            index: entry.index)
            }
        }
        return constants
    }
}

/// Chiave interna della cache di pipeline mantenuta da
/// `Device.shared`. Non esposta pubblicamente — i caller passano
/// `name` + `PipelineConstants?` e il `Device` costruisce la chiave.
struct PipelineCacheKey: Hashable, Sendable {
    let name: String
    let constants: PipelineConstants?
}

extension MTLComputePipelineState {
    /// Threadgroup tarato per questa pipeline, in funzione del
    /// grid totale che si vuole coprire.
    ///
    /// ⚠️ USARE SOLO con `dispatchThreads(_:threadsPerThreadgroup:)`.
    /// I siti che fanno `dispatchThreadgroups(...)` hanno un contratto
    /// con il kernel (SLM di dimensione fissa, simdgroup_matrix che
    /// vuole 32×1, riduzioni 256×1, ecc.) e DEVONO continuare a usare
    /// la TG hardcoded — vedi `Sources/DeepSeekKit/Layers/RMSNorm.swift`
    /// e `ActQuant.swift` come esempi.
    ///
    /// Garanzie:
    ///   - `tg.width % threadExecutionWidth == 0` quando
    ///     `grid.width >= threadExecutionWidth`;
    ///     altrimenti `tg.width == grid.width`.
    ///   - `tg.totalThreads <= maxTotalThreadsPerThreadgroup`.
    ///   - `tg.i <= grid.i` su ogni asse.
    ///
    /// Algoritmo (deterministico, no autotuning):
    ///   simd  = threadExecutionWidth   (32 su Apple Silicon)
    ///   maxT  = maxTotalThreadsPerThreadgroup
    ///   w     = largest simd-multiple ≤ min(grid.width, maxT)
    ///           (oppure grid.width se grid.width < simd)
    ///   h     = min(grid.height, maxT / w)
    ///   d     = min(grid.depth,  maxT / (w*h))
    ///   safety-net: se w*h*d > maxT, dimezza h, poi d, poi w.
    public func tunedThreadgroup(forGrid grid: MTLSize) -> MTLSize {
        let simd = max(1, threadExecutionWidth)
        let maxT = max(1, maxTotalThreadsPerThreadgroup)

        // Width: prefer simd-multiple, cap at grid.width and at maxT.
        var w: Int
        if grid.width <= 0 {
            w = 1
        } else if grid.width < simd {
            // Grid troppo piccolo per un'intera simdgroup —
            // accettiamo il "spreco" e usiamo grid.width così com'è.
            w = grid.width
        } else {
            let cap = min(grid.width, maxT)
            w = (cap / simd) * simd
            if w == 0 { w = simd }
        }

        // Height: occupa il budget residuo, clampa al grid.
        let remainingFromW = max(1, maxT / max(1, w))
        var h = max(1, min(max(1, grid.height), remainingFromW))

        // Depth: occupa il budget residuo dopo w*h, clampa al grid.
        let remainingFromWH = max(1, maxT / max(1, w * h))
        var d = max(1, min(max(1, grid.depth), remainingFromWH))

        // Safety net: in caso di edge case 3D molto piccoli o di
        // pipeline con maxT inferiore al solito, dimezza finché non
        // siamo entro il budget. h prima (meno costoso per la
        // località della cache), poi d, poi w come ultima risorsa.
        while w * h * d > maxT {
            if h > 1 { h /= 2; continue }
            if d > 1 { d /= 2; continue }
            if w > 1 { w /= 2 } else { break }
        }

        return MTLSize(width: w, height: h, depth: d)
    }
}
