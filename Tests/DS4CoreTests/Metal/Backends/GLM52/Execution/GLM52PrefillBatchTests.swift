import DS4Core
import XCTest
@testable import DS4Metal

/// Leva 1 del prefill GLM (GLM52PrefillBatch.swift): il percorso a GRUPPI
/// (due commit per gruppo) deve essere BIT-IDENTICO al per-token storico
/// (glm52ChainedDecodeLayer) — stessi kernel, stessi argomenti, stesso
/// ordine di dispatch. Stack sintetico nello spirito di
/// GLM52ResidentForwardTests: layer 0 full-indexer + dense, layer 6 full +
/// sparse (FFN routed differita come nella fase A del prefill), layer 7
/// IndexShare + sparse. indexerTopK=3 così i token oltre visible 3
/// esercitano anche il ramo top-k su device; gruppi da 4 su 6 token per
/// coprire il confine di gruppo (4+2).
final class GLM52PrefillBatchTests: XCTestCase {
    private static let tokens = 6
    private static let group = 4

    private static func geometry() -> GLM52DecodeGeometry {
        GLM52DecodeGeometry(
            layer: GLM52LayerGeometry(
                embeddingWidth: 256, headCount: 64, kvLoraRank: 512,
                ropeDimension: 64, valueDimension: 256,
                denseHiddenWidth: 128, expertHiddenWidth: 128,
                expertsUsed: 8),
            qLoraRank: 128, nopeDimension: 192,
            indexerHeadCount: 32, indexerHeadDimension: 128,
            indexerRotationDimension: 64, indexerTopK: 3)
    }

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private struct Generator {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func float(_ scale: Float) -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return (Float(state >> 33) / Float(1 << 30) - 1.0) * scale
        }
    }

    private static func floats(_ count: Int, seed: UInt64,
                               scale: Float) -> [Float] {
        var generator = Generator(seed: seed)
        return (0..<count).map { _ in generator.float(scale) }
    }

    private static func quantQ8(_ row: [Float]) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity((row.count / 32) * 34)
        var b = 0
        while b < row.count {
            let block = Array(row[b..<b + 32])
            let amax = block.map { abs($0) }.max() ?? 0
            let d = amax / 127.0
            withUnsafeBytes(of: Half.bits(d).littleEndian) {
                out.append(contentsOf: $0)
            }
            for x in block {
                out.append(UInt8(bitPattern: Int8(
                    clamping: d != 0 ? Int((x / d).rounded()) : 0)))
            }
            b += 32
        }
        return out
    }

    private static func attentionFixture(seed: UInt64)
        -> GLM52QuantizedDecodeAttention {
        let g = geometry()
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        return GLM52QuantizedDecodeAttention(
            attnNorm: floats(layer.embeddingWidth, seed: seed, scale: 0.5)
                .map { $0 + 1.5 },
            qA: quantQ8(floats(g.qLoraRank * layer.embeddingWidth,
                               seed: seed &+ 1, scale: 0.1)),
            qANorm: floats(g.qLoraRank, seed: seed &+ 2, scale: 0.5)
                .map { $0 + 1.5 },
            qB: quantQ8(floats(g.queryWidth * g.qLoraRank,
                               seed: seed &+ 3, scale: 0.1)),
            kvA: quantQ8(floats(layer.kvRawWidth * layer.embeddingWidth,
                                seed: seed &+ 4, scale: 0.1)),
            kvANorm: floats(layer.kvLoraRank, seed: seed &+ 5, scale: 0.5)
                .map { $0 + 1.5 },
            keyB: quantQ8(floats(
                layer.headCount * layer.kvLoraRank * g.nopeDimension,
                seed: seed &+ 6, scale: 0.05)),
            valueB: quantQ8(floats(headsWidth * layer.kvLoraRank,
                                   seed: seed &+ 7, scale: 0.05)),
            attnOutput: quantQ8(floats(layer.embeddingWidth * headsWidth,
                                       seed: seed &+ 8, scale: 0.05)))
    }

    private static func indexerFixture(seed: UInt64)
        -> GLM52QuantizedDecodeIndexer {
        let g = geometry()
        return GLM52QuantizedDecodeIndexer(
            key: quantQ8(floats(
                g.indexerHeadDimension * g.layer.embeddingWidth,
                seed: seed, scale: 0.1)),
            keyNorm: floats(g.indexerHeadDimension, seed: seed &+ 1,
                            scale: 0.4).map { $0 + 1.2 },
            keyNormBias: floats(g.indexerHeadDimension, seed: seed &+ 2,
                                scale: 0.2),
            queryB: quantQ8(floats(g.indexerQueryWidth * g.qLoraRank,
                                   seed: seed &+ 3, scale: 0.1)),
            proj: floats(g.indexerHeadCount * g.layer.embeddingWidth,
                         seed: seed &+ 4, scale: 0.3))
    }

    private static func sparseFFN(seed: UInt64) -> GLM52QuantizedLayerFFN {
        let layer = geometry().layer
        return .sparse(
            routerRows: floats(
                GLM52RouterReference.expertCount * layer.embeddingWidth,
                seed: seed, scale: 0.3),
            routerBias: floats(GLM52RouterReference.expertCount,
                               seed: seed &+ 1, scale: 0.1),
            sharedGate: quantQ8(floats(
                layer.expertHiddenWidth * layer.embeddingWidth,
                seed: seed &+ 2, scale: 0.1)),
            sharedUp: quantQ8(floats(
                layer.expertHiddenWidth * layer.embeddingWidth,
                seed: seed &+ 3, scale: 0.1)),
            sharedDown: quantQ8(floats(
                layer.embeddingWidth * layer.expertHiddenWidth,
                seed: seed &+ 4, scale: 0.1)),
            expertProvider: { _ in
                throw MetalError.unsupported(
                    "la fase A del prefill non deve toccare gli esperti")
            })
    }

    private static func denseFFN(seed: UInt64) -> GLM52QuantizedLayerFFN {
        let layer = geometry().layer
        return .dense(
            gate: quantQ8(floats(
                layer.denseHiddenWidth * layer.embeddingWidth,
                seed: seed, scale: 0.1)),
            up: quantQ8(floats(
                layer.denseHiddenWidth * layer.embeddingWidth,
                seed: seed &+ 1, scale: 0.1)),
            down: quantQ8(floats(
                layer.embeddingWidth * layer.denseHiddenWidth,
                seed: seed &+ 2, scale: 0.1)))
    }

    private static func bufferBytes(_ buffer: MTLBuffer) -> [UInt8] {
        let p = buffer.contents().bindMemory(to: UInt8.self,
                                             capacity: buffer.length)
        return Array(UnsafeBufferPointer(start: p, count: buffer.length))
    }

    func testGroupSweepMatchesPerTokenBitExact() throws {
        let runtime = try makeRuntime()
        let g = Self.geometry()
        let layer = g.layer
        let routerWasEnabled = GLM52GpuRouterDispatch.enabled
        GLM52GpuRouterDispatch.enabled = true
        defer { GLM52GpuRouterDispatch.enabled = routerWasEnabled }

        let attention = Self.attentionFixture(seed: 100)
        let indexer0 = Self.indexerFixture(seed: 200)
        let indexer6 = Self.indexerFixture(seed: 300)
        let layerSpecs: [(index: Int,
                          indexer: GLM52QuantizedDecodeIndexer?,
                          ffnSeed: UInt64, dense: Bool)] = [
            (0, indexer0, 400, true),
            (6, indexer6, 500, false),
            (7, nil, 600, false),
        ]
        let weights = try layerSpecs.map {
            try GLM52ResidentDecodeWeights(
                runtime: runtime, geometry: g, attention: attention,
                indexer: $0.indexer)
        }
        let ffns = try layerSpecs.map { spec in
            try GLM52ResidentFFN(
                runtime: runtime, geometry: g,
                ffnNorm: Self.floats(layer.embeddingWidth,
                                     seed: 700 &+ spec.ffnSeed, scale: 0.5)
                    .map { $0 + 1.5 },
                ffn: spec.dense ? Self.denseFFN(seed: spec.ffnSeed)
                                : Self.sparseFFN(seed: spec.ffnSeed))
        }
        func freshCaches() throws -> [GLM52ResidentDecodeCaches] {
            try layerSpecs.map {
                try GLM52ResidentDecodeCaches(
                    runtime: runtime, geometry: g, capacity: 8,
                    fullIndexer: $0.indexer != nil)
            }
        }
        let embedded = (0..<Self.tokens).map {
            Self.floats(layer.embeddingWidth, seed: 9_000 + UInt64($0),
                        scale: 0.5)
        }

        // ── Riferimento: per-token, fase A (routed FFN differita sui
        // layer sparse) — la semantica del percorso storico di sweep().
        let cachesR = try freshCaches()
        let scratch = try GLM52DecodeScratch(
            runtime: runtime, geometry: g, scoreCapacity: 8)
        var hiddensR = embedded
        var selectionsR: [[[UInt32]]] = []      // [layer][token]
        var routingsR: [[GLM52RouterOutput?]] = []
        var ffnInR: [[[Float]]] = []            // [layer][token] (sparse)
        var hiddenAfterR: [[[Float]]] = []      // [layer][token]
        var lastSelR: [[UInt32]?] = Array(repeating: nil,
                                          count: Self.tokens)
        for (l, spec) in layerSpecs.enumerated() {
            var sel: [[UInt32]] = []
            var rou: [GLM52RouterOutput?] = []
            var ffnIn: [[Float]] = []
            for t in 0..<Self.tokens {
                scratch.loadHidden(hiddensR[t])
                var noCarry: MTLCommandBuffer?
                let result = try runtime.glm52ChainedDecodeLayer(
                    weights: weights[l], ffn: ffns[l], caches: cachesR[l],
                    scratch: scratch,
                    reusedSelection: spec.indexer == nil ? lastSelR[t] : nil,
                    position: t, deferSparseFFN: !spec.dense,
                    carry: &noCarry)
                sel.append(result.selection)
                rou.append(result.routing)
                if spec.indexer != nil { lastSelR[t] = result.selection }
                hiddensR[t] = scratch.readHidden(
                    count: layer.embeddingWidth)
                if !spec.dense {
                    let p = scratch.ffnIn.contents().bindMemory(
                        to: Float.self, capacity: layer.embeddingWidth)
                    ffnIn.append(Array(UnsafeBufferPointer(
                        start: p, count: layer.embeddingWidth)))
                } else {
                    ffnIn.append([])
                }
            }
            selectionsR.append(sel)
            routingsR.append(rou)
            ffnInR.append(ffnIn)
            hiddenAfterR.append(hiddensR)
        }

        // ── Leva 1: stessi layer a gruppi di 4 (4+2) su cache fresche.
        let cachesB = try freshCaches()
        let pool = try GLM52PrefillScratchPool(
            runtime: runtime, geometry: g, scoreCapacity: 8,
            count: Self.group)
        var hiddensB = embedded
        var lastSelB: [[UInt32]?] = Array(repeating: nil,
                                          count: Self.tokens)
        for (l, spec) in layerSpecs.enumerated() {
            var g0 = 0
            while g0 < Self.tokens {
                let n = min(Self.group, Self.tokens - g0)
                let base = g0
                let outcome = try runtime.glm52PrefillGroupLayer(
                    weights: weights[l], ffn: ffns[l], caches: cachesB[l],
                    pool: pool, count: n,
                    hiddenAt: { hiddensB[base + $0] },
                    reusedSelectionAt: {
                        spec.indexer == nil ? lastSelB[base + $0] : nil
                    },
                    basePosition: g0)
                for i in 0..<n {
                    let t = g0 + i
                    XCTAssertEqual(outcome.selections[i],
                                   selectionsR[l][t],
                                   "selezione layer \(spec.index) token \(t)")
                    XCTAssertEqual(outcome.routings[i]?.selected,
                                   routingsR[l][t]?.selected,
                                   "esperti layer \(spec.index) token \(t)")
                    XCTAssertEqual(outcome.routings[i]?.weights,
                                   routingsR[l][t]?.weights,
                                   "pesi router layer \(spec.index) token \(t)")
                    if spec.indexer != nil {
                        lastSelB[t] = outcome.selections[i]
                    }
                    let got = pool.sets[i].readHidden(
                        count: layer.embeddingWidth)
                    XCTAssertEqual(got, hiddenAfterR[l][t],
                                   "hidden layer \(spec.index) token \(t)")
                    hiddensB[t] = got
                    if !spec.dense {
                        let p = pool.sets[i].ffnIn.contents().bindMemory(
                            to: Float.self, capacity: layer.embeddingWidth)
                        let ffnIn = Array(UnsafeBufferPointer(
                            start: p, count: layer.embeddingWidth))
                        XCTAssertEqual(ffnIn, ffnInR[l][t],
                                       "ffnIn layer \(spec.index) token \(t)")
                    }
                }
                g0 += n
            }
        }
        // Gli hidden finali e le cache devono coincidere bit-per-bit.
        for t in 0..<Self.tokens {
            XCTAssertEqual(hiddensB[t], hiddensR[t],
                           "hidden finale token \(t)")
        }
        for l in 0..<layerSpecs.count {
            XCTAssertEqual(Self.bufferBytes(cachesB[l].compact),
                           Self.bufferBytes(cachesR[l].compact),
                           "cache compact layer \(layerSpecs[l].index)")
            if let kB = cachesB[l].indexerKeys,
               let kR = cachesR[l].indexerKeys {
                XCTAssertEqual(Self.bufferBytes(kB), Self.bufferBytes(kR),
                               "cache indexer layer \(layerSpecs[l].index)")
            }
            XCTAssertEqual(cachesB[l].rows, cachesR[l].rows,
                           "righe cache layer \(layerSpecs[l].index)")
        }
    }
}
