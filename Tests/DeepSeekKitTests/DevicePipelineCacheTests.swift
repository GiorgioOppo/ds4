import XCTest
import Metal
@testable import DeepSeekKit

/// Test che la cache di pipeline in `Device.shared.makePipeline`
/// rispetti l'identita' per stessa chiave e produca pipeline
/// distinte per chiavi diverse. Tutti i test richiedono Metal e un
/// metallib caricato.
final class DevicePipelineCacheTests: XCTestCase {

    private func requireMetal() throws {
        try XCTSkipUnless(MTLCreateSystemDefaultDevice() != nil,
                          "Metal not available")
    }

    /// (a) Stesso `name`, nessun constants: stesso oggetto.
    func testCachingByNameOnly() throws {
        try requireMetal()
        let p1 = Device.shared.makePipeline("rms_norm_f32")
        let p2 = Device.shared.makePipeline("rms_norm_f32")
        XCTAssertTrue(p1 === p2,
                      "Stessa key (name only) deve restituire la stessa pipeline cached")
    }

    /// (b) Variante con constants vs senza constants: oggetti diversi.
    func testConstantsVsNoConstantsDiverge() throws {
        try requireMetal()
        // `moe_gate` ha function constants index 2 (score func) e 3
        // (route scale). Senza constants la pipeline si specializza
        // a valori default; con constants specifici si specializza
        // diversamente. Le due varianti DEVONO essere oggetti
        // distinti in cache (chiave PipelineCacheKey diversa).
        let consts = PipelineConstants { c in
            c.setUInt32(2, at: 2)   // score func id
            c.setFloat(1.0, at: 3)  // route scale
        }
        let pPlain = Device.shared.makePipeline("moe_gate", constants: nil)
        let pTuned = Device.shared.makePipeline("moe_gate", constants: consts)
        XCTAssertFalse(pPlain === pTuned,
                       "Varianti con/senza constants devono avere oggetti distinti")
    }

    /// (c) Stessi bytes → stesso oggetto; bytes diversi → diverso.
    func testConstantsHashByContent() throws {
        try requireMetal()
        let consts1 = PipelineConstants { c in
            c.setUInt32(2, at: 2)
            c.setFloat(1.0, at: 3)
        }
        let consts1bis = PipelineConstants { c in
            c.setUInt32(2, at: 2)
            c.setFloat(1.0, at: 3)
        }
        let consts2 = PipelineConstants { c in
            c.setUInt32(2, at: 2)
            c.setFloat(2.0, at: 3)   // route scale diverso
        }

        // Stessi bytes => stesso oggetto.
        let a = Device.shared.makePipeline("moe_gate", constants: consts1)
        let b = Device.shared.makePipeline("moe_gate", constants: consts1bis)
        XCTAssertTrue(a === b,
                      "Stessi function constants -> stessa pipeline cached")

        // Bytes diversi => oggetto diverso.
        let c = Device.shared.makePipeline("moe_gate", constants: consts2)
        XCTAssertFalse(a === c,
                       "Function constants diversi -> pipeline distinte")
    }

    /// (d) Concorrenza: 16 thread × 4 nomi ciascuno. Nessun crash,
    /// e dopo l'esecuzione la cache contiene esattamente le pipeline
    /// uniche viste.
    func testConcurrentCacheLookups() throws {
        try requireMetal()
        // Snapshot baseline: altri test in questa suite possono aver
        // popolato la cache. Calcoliamo il delta atteso.
        let baseline = Device.shared.pipelineCacheCount
        let names = ["rms_norm_f32", "silu_mul_f32", "axpy_f32", "scale_f32"]

        DispatchQueue.concurrentPerform(iterations: 16) { iter in
            for name in names {
                _ = Device.shared.makePipeline(name)
            }
        }

        // Dopo 16 × 4 lookup, la cache ha al massimo 4 nuove voci
        // (potrebbe averne 0 se i nomi erano gia' cached). In ogni
        // caso, il count attuale - baseline deve essere <= 4.
        let delta = Device.shared.pipelineCacheCount - baseline
        XCTAssertGreaterThanOrEqual(delta, 0)
        XCTAssertLessThanOrEqual(delta, names.count,
                                  "Race nella cache: il numero di pipeline uniche eccede i nomi distinti")
    }
}
