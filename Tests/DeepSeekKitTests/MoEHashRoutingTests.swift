import XCTest
import Metal
@testable import DeepSeekKit

/// Verifies that Gate routes via the per-token-id `tid2eid` lookup table
/// when `nHashLayers > 0` and `layerId < nHashLayers`. The hash routing
/// branch (model.py:577-578) bypasses the score-based topk and emits the
/// expert ids stored in `tid2eid[input_id]`.
final class MoEHashRoutingTests: XCTestCase {

    func testHashRoutingPickFromLookupTable() throws {
        let nExperts = 4
        let topK = 2
        let dim = 8
        let vocab = 6

        var cfg = ModelConfig()
        cfg.nRoutedExperts = nExperts
        cfg.nActivatedExperts = topK
        cfg.dim = dim
        cfg.vocabSize = vocab
        cfg.nHashLayers = 1     // only layer 0 uses hash routing

        // Deterministic tid2eid table — token id `id` always routes to
        // experts (id % nExperts) and ((id+1) % nExperts).
        var tid2eid = [Int32](repeating: 0, count: vocab * topK)
        for id in 0..<vocab {
            tid2eid[id * topK + 0] = Int32(id % nExperts)
            tid2eid[id * topK + 1] = Int32((id + 1) % nExperts)
        }
        let tid2eidT = tid2eid.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [vocab, topK], dtype: .i32)
        }

        // Gate weight + bias don't matter (hash branch ignores them); we
        // pass anything well-formed.
        let dummyWeight = [Float](repeating: 0, count: nExperts * dim)
        let weightT = dummyWeight.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [nExperts, dim], dtype: .f32)
        }
        let weightLin = Linear(inFeatures: dim, outFeatures: nExperts,
                                weight: weightT, scale: nil)
        let gate = Gate(config: cfg, layerId: 0,
                        weight: weightLin, bias: nil, tid2eid: tid2eidT)
        XCTAssertTrue(gate.hashRouting)

        // Build input x and matching ids. `x` content is irrelevant for
        // hash routing (the gate ignores logits in this branch).
        let inputIds: [Int32] = [3, 1, 5]
        let N = inputIds.count
        let xData = [Float](repeating: 0.5, count: N * dim)
        let xT = xData.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, dim], dtype: .f32)
        }

        var cmd = Device.shared.queue.makeCommandBuffer()!
        let (_, indicesT) = gate(xT, inputIds: inputIds, in: &cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let idxPtr = indicesT.buffer.contents().bindMemory(to: Int32.self,
                                                            capacity: N * topK)
        for n in 0..<N {
            let id = Int(inputIds[n])
            XCTAssertEqual(idxPtr[n * topK + 0], Int32(id % nExperts),
                           "row \(n): first expert mismatch")
            XCTAssertEqual(idxPtr[n * topK + 1], Int32((id + 1) % nExperts),
                           "row \(n): second expert mismatch")
        }
    }
}
