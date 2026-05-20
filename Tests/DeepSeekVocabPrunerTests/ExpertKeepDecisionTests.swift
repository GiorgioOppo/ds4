import XCTest
@testable import DeepSeekVocabPruner

/// Test della decision math dell'expert pruner: dato un grid di
/// counts per (layer, expert), verifica la selezione per coverage
/// threshold + il floor minimo.
final class ExpertKeepDecisionTests: XCTestCase {

    /// Caso base: 4 esperti, layer L=0 con counts [100, 50, 30, 20].
    /// Coverage 0.80 → kept {0, 1} (somma 150 / 200 = 75% non basta,
    /// includi anche {2} → 180/200 = 90% > 80%). Però il floor è
    /// `max(minKept=2, topK=2)` = 2, quindi non vincola.
    func testCoverageThresholdBasic() {
        let usage: [ExpertUsageRow] = [
            .init(layerId: 0, expertId: 0, count: 100),
            .init(layerId: 0, expertId: 1, count: 50),
            .init(layerId: 0, expertId: 2, count: 30),
            .init(layerId: 0, expertId: 3, count: 20),
        ]
        let dec = ExpertKeepDecision.build(
            usage: usage,
            nLayers: 1,
            nRoutedExperts: 4,
            nActivatedExperts: 2,
            coverage: 0.80,
            minKept: 2)
        XCTAssertEqual(dec.keepIds, [[0, 1, 2]])
        XCTAssertEqual(dec.droppedIds, [[3]])
        XCTAssertEqual(dec.actualCoveragePerLayer.count, 1)
        XCTAssertGreaterThanOrEqual(dec.actualCoveragePerLayer[0], 0.80)
    }

    /// Coverage 0.99: dovrebbe includere quasi tutti.
    func testCoverageVeryHighKeepsMost() {
        let usage: [ExpertUsageRow] = [
            .init(layerId: 0, expertId: 0, count: 100),
            .init(layerId: 0, expertId: 1, count: 50),
            .init(layerId: 0, expertId: 2, count: 30),
            .init(layerId: 0, expertId: 3, count: 1),
        ]
        let dec = ExpertKeepDecision.build(
            usage: usage,
            nLayers: 1,
            nRoutedExperts: 4,
            nActivatedExperts: 2,
            coverage: 0.99,
            minKept: 2)
        // 100 + 50 + 30 = 180 / 181 = 99.4% > 99% — può fermarsi
        // dopo 3 esperti, ma 180/181 ≥ 99% target=Int(0.99*181)+1=180.
        // Quindi {0, 1, 2}.
        XCTAssertEqual(Set(dec.keepIds[0]), Set([0, 1, 2]))
        XCTAssertEqual(Set(dec.droppedIds[0]), Set([3]))
    }

    /// Floor: se coverage permetterebbe meno esperti del floor, il
    /// floor vince.
    func testMinKeptFloorClampsUp() {
        // Layer molto skewed: expert 0 da solo copre 99% → coverage=0.95
        // potrebbe scegliere solo {0}, ma minKept=3 vincola a {0, 1, 2}.
        let usage: [ExpertUsageRow] = [
            .init(layerId: 0, expertId: 0, count: 990),
            .init(layerId: 0, expertId: 1, count: 5),
            .init(layerId: 0, expertId: 2, count: 3),
            .init(layerId: 0, expertId: 3, count: 1),
            .init(layerId: 0, expertId: 4, count: 1),
        ]
        let dec = ExpertKeepDecision.build(
            usage: usage,
            nLayers: 1,
            nRoutedExperts: 5,
            nActivatedExperts: 2,
            coverage: 0.95,
            minKept: 3)
        XCTAssertGreaterThanOrEqual(dec.keepIds[0].count, 3,
            "minKept floor non rispettato")
        // I 3 più frequenti sono [0, 1, 2].
        XCTAssertEqual(Set(dec.keepIds[0]).intersection([0, 1, 2]),
                        Set([0, 1, 2]))
    }

    /// Layer con zero token osservati: tieni tutto, niente da
    /// droppare (non c'è evidenza per giustificare un drop).
    func testZeroUsageLayerKeepsAll() {
        let usage: [ExpertUsageRow] = [
            .init(layerId: 0, expertId: 0, count: 0),
            .init(layerId: 0, expertId: 1, count: 0),
            .init(layerId: 0, expertId: 2, count: 0),
        ]
        let dec = ExpertKeepDecision.build(
            usage: usage,
            nLayers: 1,
            nRoutedExperts: 3,
            nActivatedExperts: 2,
            coverage: 0.99,
            minKept: 2)
        XCTAssertEqual(dec.keepIds, [[0, 1, 2]])
        XCTAssertEqual(dec.droppedIds, [[]])
        XCTAssertEqual(dec.actualCoveragePerLayer, [1.0])
    }

    /// Multi-layer: ogni layer ha la propria decisione indipendente.
    /// La struct esposta è uniforme `[[Int]]` con N layer.
    func testMultiLayerIndependentDecisions() {
        let usage: [ExpertUsageRow] = [
            // Layer 0: skewed verso expert 0.
            .init(layerId: 0, expertId: 0, count: 1000),
            .init(layerId: 0, expertId: 1, count: 1),
            .init(layerId: 0, expertId: 2, count: 1),
            // Layer 1: distribuito uniformemente.
            .init(layerId: 1, expertId: 0, count: 100),
            .init(layerId: 1, expertId: 1, count: 100),
            .init(layerId: 1, expertId: 2, count: 100),
        ]
        let dec = ExpertKeepDecision.build(
            usage: usage,
            nLayers: 2,
            nRoutedExperts: 3,
            nActivatedExperts: 2,
            coverage: 0.95,
            minKept: 2)
        // Layer 0: solo {0} basta per il 95%, ma minKept=2 → {0, 1}
        // (1 e 2 hanno lo stesso count, tie broken by ascending id).
        XCTAssertEqual(dec.keepIds[0].count, 2)
        XCTAssertTrue(dec.keepIds[0].contains(0))
        // Layer 1: 95% richiede 2/3 esperti (200/300 = 66% < 95% →
        // includi un terzo → 100% > 95%). Cosi tutti i 3 dovrebbero
        // sopravvivere.
        XCTAssertEqual(Set(dec.keepIds[1]), Set([0, 1, 2]))
    }

    /// Determinismo: stesso input produce stesso output.
    func testDeterministic() {
        let usage: [ExpertUsageRow] = [
            .init(layerId: 0, expertId: 0, count: 50),
            .init(layerId: 0, expertId: 1, count: 50),  // tie!
            .init(layerId: 0, expertId: 2, count: 30),
            .init(layerId: 0, expertId: 3, count: 20),
        ]
        let d1 = ExpertKeepDecision.build(
            usage: usage,
            nLayers: 1,
            nRoutedExperts: 4,
            nActivatedExperts: 2,
            coverage: 0.50,
            minKept: 2)
        let d2 = ExpertKeepDecision.build(
            usage: usage,
            nLayers: 1,
            nRoutedExperts: 4,
            nActivatedExperts: 2,
            coverage: 0.50,
            minKept: 2)
        XCTAssertEqual(d1.keepIds, d2.keepIds)
        XCTAssertEqual(d1.droppedIds, d2.droppedIds)
    }

    /// Convenience counts.
    func testConvenienceProperties() {
        let usage: [ExpertUsageRow] = [
            .init(layerId: 0, expertId: 0, count: 100),
            .init(layerId: 0, expertId: 1, count: 50),
            .init(layerId: 0, expertId: 2, count: 1),
            .init(layerId: 1, expertId: 0, count: 100),
            .init(layerId: 1, expertId: 1, count: 50),
            .init(layerId: 1, expertId: 2, count: 1),
        ]
        let dec = ExpertKeepDecision.build(
            usage: usage,
            nLayers: 2,
            nRoutedExperts: 3,
            nActivatedExperts: 2,
            coverage: 0.99,
            minKept: 2)
        XCTAssertEqual(dec.nLayers, 2)
        XCTAssertEqual(dec.totalKept + dec.totalDropped,
                        dec.nLayers * dec.nRoutedExperts)
    }
}
