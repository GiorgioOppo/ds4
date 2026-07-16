import XCTest
import DS4Core
@testable import DS4Metal

final class DSV4RuntimeGeometryTests: XCTestCase {
    func testFlashGeometryPreservesLegacyShape() {
        let geometry = DSV4RuntimeGeometry(shape: .flash)

        XCTAssertEqual(geometry.nLayers, DSV4Shape.nLayer)
        XCTAssertEqual(geometry.dims.nEmbd, DSV4Shape.nEmbd)
        XCTAssertEqual(geometry.dims.nHead, DSV4Shape.nHead)
        XCTAssertEqual(geometry.dims.qDim, DSV4Shape.qDim)
        XCTAssertEqual(geometry.dims.nOutGroup, DSV4Shape.nOutGroup)
        XCTAssertEqual(geometry.dims.nExperts, DSV4Shape.nExpert)
        XCTAssertEqual(geometry.dims.indexerTopK, 512)
        XCTAssertEqual(geometry.expertWeightScale, 1.5)
        XCTAssertEqual(geometry.nHashLayer, DSV4Shape.nHashLayer)
        XCTAssertEqual(geometry.sinkhornIterations, DSV4Shape.nHCSinkhornIter)
        XCTAssertEqual(geometry.dims.expertWeightScale, 1.5)
        XCTAssertEqual(geometry.dims.nHashLayers, 3)
        XCTAssertEqual(geometry.dims.sinkhornIterations, 20)

        for layer in 0..<geometry.nLayers {
            XCTAssertEqual(geometry.compressRatio(layer: layer),
                           DSV4Shape.compressRatio(layer: layer))
            XCTAssertEqual(geometry.ropeParams(layer: layer),
                           DSV4Shape.ropeParams(layer: layer))
        }
    }

    func testProGeometryDerivesAllChangedDimensions() {
        let geometry = DSV4RuntimeGeometry(shape: .pro)

        XCTAssertEqual(geometry.nLayers, 61)
        XCTAssertEqual(geometry.dims.nEmbd, 7168)
        XCTAssertEqual(geometry.dims.nHead, 128)
        XCTAssertEqual(geometry.dims.headDim, 512)
        XCTAssertEqual(geometry.dims.qRank, 1536)
        XCTAssertEqual(geometry.dims.qDim, 65_536)
        XCTAssertEqual(geometry.dims.nOutGroup, 16)
        XCTAssertEqual(geometry.dims.attnGroupDim, 4096)
        XCTAssertEqual(geometry.dims.nExperts, 384)
        XCTAssertEqual(geometry.dims.expertFfn, 3072)
        XCTAssertEqual(geometry.dims.indexerTopK, 1024)
        XCTAssertEqual(geometry.expertWeightScale, 2.5)
        XCTAssertEqual(geometry.dims.expertWeightScale, 2.5)

        XCTAssertEqual(geometry.compressRatio(layer: 0), 128)
        XCTAssertEqual(geometry.compressRatio(layer: 1), 128)
        XCTAssertEqual(geometry.compressRatio(layer: 2), 4)
        XCTAssertEqual(geometry.compressRatio(layer: 3), 128)
        XCTAssertEqual(geometry.compressRatio(layer: 60), 4)

        let rope = geometry.ropeParams(layer: 0)
        XCTAssertEqual(rope.nCtxOrig, 65_536)
        XCTAssertEqual(rope.freqBase, 160_000)
        XCTAssertEqual(rope.freqScale, 1.0 / 16.0)
        XCTAssertEqual(rope.extFactor, 1)
    }

    func testExistingDimsInitializerKeepsFlashCompatibleDefaults() {
        let dims = DSV4Dims(
            nEmbd: 16, nHC: 4, headDim: 8, nHead: 2,
            qRank: 8, qDim: 16, sharedFfn: 32,
            nExperts: 4, expertFfn: 32, k: 2, nRot: 4, vocab: 128
        )

        XCTAssertEqual(dims.expertWeightScale, 1.5)
        XCTAssertEqual(dims.nHashLayers, 3)
        XCTAssertEqual(dims.sinkhornIterations, 20)
    }

    func testDecoderGeometryContractAcceptsBothKnownProfiles() throws {
        for shape in [DeepSeekV4Shape.flash, .pro] {
            let geometry = DSV4RuntimeGeometry(shape: shape)
            XCTAssertNoThrow(
                try StreamingDecoder.validateRuntimeGeometry(
                    geometry, dims: geometry.dims, nLayers: geometry.nLayers),
                shape.name
            )
            XCTAssertTrue(
                StreamingDecoder.runtimeGeometryMismatches(
                    geometry, dims: geometry.dims, nLayers: geometry.nLayers
                ).isEmpty,
                shape.name
            )
        }
    }

    func testDecoderGeometryContractChecksEveryArchitectureField() throws {
        let geometry = DSV4RuntimeGeometry(shape: .pro)
        let mutations: [(field: String, mutate: (inout DSV4Dims) -> Void)] = [
            ("nEmbd", { $0.nEmbd += 1 }),
            ("nHC", { $0.nHC += 1 }),
            ("headDim", { $0.headDim += 1 }),
            ("nHead", { $0.nHead += 1 }),
            ("qRank", { $0.qRank += 1 }),
            ("qDim", { $0.qDim += 1 }),
            ("sharedFfn", { $0.sharedFfn += 1 }),
            ("nExperts", { $0.nExperts += 1 }),
            ("expertFfn", { $0.expertFfn += 1 }),
            ("k", { $0.k += 1 }),
            ("nRot", { $0.nRot += 1 }),
            ("vocab", { $0.vocab += 1 }),
            ("nOutGroup", { $0.nOutGroup += 1 }),
            ("nLoraO", { $0.nLoraO += 1 }),
            ("swigluClamp", { $0.swigluClamp += 1 }),
            ("nSWA", { $0.nSWA += 1 }),
            ("nIndexerHead", { $0.nIndexerHead += 1 }),
            ("nIndexerHeadDim", { $0.nIndexerHeadDim += 1 }),
            ("indexerTopK", { $0.indexerTopK += 1 }),
            ("expertWeightScale", { $0.expertWeightScale += 1 }),
            ("nHashLayers", { $0.nHashLayers += 1 }),
            ("sinkhornIterations", { $0.sinkhornIterations += 1 }),
        ]

        for mutation in mutations {
            var dims = geometry.dims
            mutation.mutate(&dims)
            XCTAssertEqual(
                StreamingDecoder.runtimeGeometryMismatches(
                    geometry, dims: dims, nLayers: geometry.nLayers),
                [mutation.field],
                mutation.field
            )
            XCTAssertThrowsError(
                try StreamingDecoder.validateRuntimeGeometry(
                    geometry, dims: dims, nLayers: geometry.nLayers)
            ) { error in
                XCTAssertTrue(String(describing: error).contains(mutation.field),
                              "\(mutation.field): \(error)")
            }
        }

        XCTAssertEqual(
            StreamingDecoder.runtimeGeometryMismatches(
                geometry, dims: geometry.dims, nLayers: geometry.nLayers - 1),
            ["nLayers"]
        )
        XCTAssertThrowsError(
            try StreamingDecoder.validateRuntimeGeometry(
                geometry, dims: geometry.dims, nLayers: geometry.nLayers - 1)
        ) { error in
            XCTAssertTrue(String(describing: error).contains("nLayers"))
        }
    }
}
