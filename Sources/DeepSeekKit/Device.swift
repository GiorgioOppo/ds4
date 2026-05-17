import Foundation
import Metal

public final class Device {
    public static let shared = Device()

    public let mtl: MTLDevice
    public let queue: MTLCommandQueue

    // Loaded on first use. The converter target only needs `mtl` to wrap
    // mmapped safetensors as MTLBuffers and never touches the kernel
    // library; eagerly loading the metallib here would make the converter
    // fail on builds that don't ship one (e.g. plain `swift build`, which
    // doesn't always compile the .metal resources into default.metallib).
    private lazy var _library: MTLLibrary = Self.loadLibrary(device: mtl)
    public var library: MTLLibrary { _library }

    // MARK: - Pipeline cache

    /// Cache lazy delle `MTLComputePipelineState` create durante la
    /// vita del processo. Ogni `makeComputePipelineState(function:)`
    /// costa 10-50 ms al primo invocazione (compilazione/specializzazione
    /// del kernel) — vedi `docs/PERFORMANCE.md` §2.4 / §5 e
    /// `TODO.md` §4. La cache evita di ripagare quel costo ad ogni
    /// `forward` o ad ogni istanza di layer che condivide la stessa
    /// pipeline.
    ///
    /// Universo limitato (~30 kernel × ≤4 combinazioni di constants):
    /// nessuna policy di eviction.
    private var pipelineCache: [PipelineCacheKey: MTLComputePipelineState] = [:]
    private let pipelineCacheLock = NSLock()

    private init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device. DeepSeekKit requires Apple Silicon.")
        }
        guard let q = dev.makeCommandQueue() else {
            fatalError("Could not create command queue.")
        }
        self.mtl = dev
        self.queue = q
    }

    private static func loadLibrary(device: MTLDevice) -> MTLLibrary {
        let bundle = Bundle.module
        do {
            return try device.makeDefaultLibrary(bundle: bundle)
        } catch {
            fatalError("Failed to load Metal library from bundle: \(error)")
        }
    }

    /// Restituisce la `MTLComputePipelineState` per il kernel `name`,
    /// senza function constants. Cached: chiamate successive con lo
    /// stesso `name` restituiscono lo stesso oggetto (identità `===`).
    public func makePipeline(_ name: String) -> MTLComputePipelineState {
        return makePipeline(name, constants: nil)
    }

    /// Variante con function constants. Lo stesso `name` con `constants`
    /// diversi (per bytes/types/index) restituisce pipeline diverse;
    /// con `constants` identici (anche fra `nil` e `nil`) restituisce
    /// la stessa pipeline cached.
    public func makePipeline(_ name: String,
                              constants: PipelineConstants?)
        -> MTLComputePipelineState
    {
        let key = PipelineCacheKey(name: name, constants: constants)

        // Fast path: lock minimo, una sola lookup.
        pipelineCacheLock.lock()
        if let cached = pipelineCache[key] {
            pipelineCacheLock.unlock()
            return cached
        }
        pipelineCacheLock.unlock()

        // Slow path: compilazione fuori dalla lock per non bloccare
        // altre lookup. C'è una piccola race teorica in cui due
        // thread compilano la stessa pipeline in parallelo —
        // accettabile (idempotente, una delle due perderà la corsa
        // al set finale e verrà scartata dal GC). Metal garantisce
        // che `MTLComputePipelineState` sia thread-safe per l'uso.
        let pipeline: MTLComputePipelineState
        do {
            let fn: MTLFunction?
            if let constants {
                fn = try library.makeFunction(name: name,
                                               constantValues: constants.mtl())
            } else {
                fn = library.makeFunction(name: name)
            }
            guard let fn else {
                fatalError("Metal function not found: \(name)")
            }
            pipeline = try mtl.makeComputePipelineState(function: fn)
        } catch {
            fatalError("Failed to create pipeline for \(name): \(error)")
        }

        // Inserisci sotto lock; se nel frattempo un altro thread ha
        // vinto la corsa, prendiamo il suo (così `===` resta stabile
        // per il caller).
        pipelineCacheLock.lock()
        let final = pipelineCache[key] ?? pipeline
        pipelineCache[key] = final
        pipelineCacheLock.unlock()
        return final
    }

    /// Diagnostica: numero di pipeline distinte attualmente in cache.
    /// Esposto principalmente per i test.
    public var pipelineCacheCount: Int {
        pipelineCacheLock.lock()
        defer { pipelineCacheLock.unlock() }
        return pipelineCache.count
    }
}
