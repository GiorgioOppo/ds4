import Foundation
import Metal

/// Parametri di problema passati al kernel — dimensioni positive
/// in `dims` (es. `[rows, cols]` per softmax, `[M, N, K]` per
/// GEMM) più un'mappa di extras per parametri ausiliari (es.
/// `eps` per RMSNorm, `flags` per kernel con varianti).
public struct KernelProblem: Sendable {
    public var dims: [Int]
    public var extras: [String: Int]

    public init(dims: [Int], extras: [String: Int] = [:]) {
        self.dims = dims
        self.extras = extras
    }
}

/// Classe astratta base per i kernel Metal. Incapsula il
/// pattern `pipeline cache → compute encoder → bind →
/// dispatchThreadgroups → endEncoding` che oggi è replicato
/// inline in `DeepSeekKit/Layers/Linear.swift:119`,
/// `RMSNorm.swift:39`, `Elementwise.swift:21`.
///
/// La subclass concreta decide:
/// - quale `MTLFunction` aprire (via `functionName`),
/// - come binding (assegnando buffer/bytes agli indici 0..N),
/// - come grid + threadgroup size in base al problema.
///
/// Il template method `dispatch(problem:buffers:offsets:in:)` è
/// `final` e garantisce un punto unico di orchestrazione —
/// possibile spot dove iniettare profiling/logging futuro.
///
/// Nota: NON è `Sendable` perché non serve esserlo (le
/// pipeline state vengono recuperate dal `Device.shared`
/// singleton al momento del dispatch). Se servirà passare un
/// kernel attraverso actor boundary, va aggiunto
/// `@unchecked Sendable` con invariante "no mutable state".
open class MetalKernel {
    public let functionName: String

    public init(functionName: String) {
        self.functionName = functionName
    }

    /// ASTRATTO — la subclass decide la threadgroup size.
    /// Esempi: `MTLSize(width: 256, height: 1, depth: 1)` per
    /// row-major softmax, `MTLSize(width: 16, height: 16, depth: 1)`
    /// per GEMM tiled, `MTLSize(width: 32, height: 1, depth: 1)`
    /// per simdgroup_matrix.
    open func threadgroupSize(for problem: KernelProblem) -> MTLSize {
        fatalError(
            "Subclass \(type(of: self)) must override threadgroupSize(for:)")
    }

    /// ASTRATTO — la subclass decide il grid (numero di
    /// threadgroup). Es. una threadgroup per riga in softmax.
    open func gridSize(for problem: KernelProblem) -> MTLSize {
        fatalError(
            "Subclass \(type(of: self)) must override gridSize(for:)")
    }

    /// ASTRATTO — la subclass associa buffer e bytes agli
    /// indici. Gli indici sono per-kernel (vedi `.metal`).
    open func bind(problem: KernelProblem,
                   buffers: [MTLBuffer?],
                   offsets: [Int],
                   to encoder: MTLComputeCommandEncoder) {
        fatalError(
            "Subclass \(type(of: self)) must override bind(problem:buffers:offsets:to:)")
    }

    /// Template method NON sovrascrivibile. Esegue:
    /// 1. lookup della pipeline state via `Device.shared`,
    /// 2. creazione del compute encoder,
    /// 3. binding (delegato alla subclass),
    /// 4. dispatch con grid/threadgroup (delegati alla subclass),
    /// 5. endEncoding.
    public final func dispatch(problem: KernelProblem,
                                buffers: [MTLBuffer?],
                                offsets: [Int],
                                in cmd: MTLCommandBuffer) {
        let pipeline = Device.shared.makePipeline(functionName)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            fatalError(
                "Could not create compute encoder for \(functionName)")
        }
        enc.setComputePipelineState(pipeline)
        bind(problem: problem,
             buffers: buffers,
             offsets: offsets,
             to: enc)
        enc.dispatchThreadgroups(gridSize(for: problem),
                                 threadsPerThreadgroup: threadgroupSize(for: problem))
        enc.endEncoding()
    }
}
