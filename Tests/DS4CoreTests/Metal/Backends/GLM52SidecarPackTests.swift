import XCTest
@testable import DS4Metal

/// Il PACK UNICO del sidecar GLM: append/scan di sezioni, riapertura dopo
/// una coda strappata, la vista finestrata del reader (una sezione deve
/// comportarsi esattamente come un file autonomo) e la migrazione dei nomi
/// storici a quelli standard DeepSeek (<gguf>.q4dense / <gguf>.expbundle).
/// Dati sintetici: qui si prova il contenitore, non il formato v2 che
/// trasporta (invariato).
final class GLM52SidecarPackTests: XCTestCase {
    private var directory: String = ""
    private let sourceFileSize: UInt64 = 123_456_789
    private let sourcePath = "/modelli/finto-modello.gguf"
    private var q4Name: String {
        GLM52SidecarPack.q4FileName(sourcePath: sourcePath)
    }
    private var expertsName: String {
        GLM52SidecarPack.expertsFileName(sourcePath: sourcePath)
    }

    override func setUpWithError() throws {
        directory = NSTemporaryDirectory()
            + "glm52-pack-tests-" + UUID().uuidString
        try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: directory)
    }

    private func writeSection(layer: Int, bytes: [UInt8],
                              fileName: String? = nil) throws -> String {
        let path = directory + "/blk\(layer)-\(fileName ?? "q4").payload"
        try Data(bytes).write(to: URL(fileURLWithPath: path))
        try GLM52SidecarPack.append(directory: directory,
                                    fileName: fileName ?? q4Name,
                                    layer: layer, contentsOf: path,
                                    sourceFileSize: sourceFileSize)
        return path
    }

    func testAppendScanAndWindowedRead() throws {
        let payloadA: [UInt8] = (0..<1000).map { UInt8($0 % 251) }
        let payloadB: [UInt8] = (0..<777).map { UInt8(($0 * 7) % 253) }
        _ = try writeSection(layer: 11, bytes: payloadA)
        _ = try writeSection(layer: 12, bytes: payloadB)

        let index = try XCTUnwrap(GLM52SidecarPack.scan(
            directory: directory, fileName: q4Name,
            sourceFileSize: sourceFileSize))
        XCTAssertEqual(index.sections.count, 2)
        XCTAssertTrue(index.path.hasSuffix(".q4dense"))
        let sectionA = try XCTUnwrap(index.sections[11])
        XCTAssertEqual(sectionA.length, UInt64(payloadA.count))

        // La vista finestrata riproduce la sezione come file autonomo:
        // stessi byte a offset relativi, e fuori-finestra rifiutato.
        let pack = try GLM52PayloadReader(path: index.path)
        let view = try pack.windowed(offset: sectionA.offset,
                                     length: sectionA.length)
        XCTAssertEqual(view.fileSize, UInt64(payloadA.count))
        let read = try view.bytes(of: GLM52WeightDescriptor(
            name: "test", type: GLM52TensorSchema.q8_0, dims: [],
            absOffset: 0, bytes: UInt64(payloadA.count)))
        XCTAssertEqual(read, payloadA)
        let tail = try view.bytes(of: GLM52WeightDescriptor(
            name: "tail", type: GLM52TensorSchema.q8_0, dims: [],
            absOffset: 990, bytes: 10))
        XCTAssertEqual(tail, Array(payloadA[990...]))
        XCTAssertThrowsError(try view.bytes(of: GLM52WeightDescriptor(
            name: "oltre", type: GLM52TensorSchema.q8_0, dims: [],
            absOffset: 0, bytes: UInt64(payloadA.count) + 1)))
    }

    func testIdentityMismatchIsRefusedLoudly() throws {
        _ = try writeSection(layer: 11, bytes: [1, 2, 3, 4])
        XCTAssertThrowsError(try GLM52SidecarPack.scan(
            directory: directory, fileName: q4Name,
            sourceFileSize: sourceFileSize + 1))
    }

    func testLegacyPackNameIsRenamedOnScan() throws {
        // Un pack costruito col nome storico deve essere ritrovato — e
        // rinominato al nome standard — alla prima scansione col legacy
        // name dichiarato. Le sezioni si conservano.
        let payload: [UInt8] = [4, 5, 6, 7, 8]
        _ = try writeSection(layer: 21, bytes: payload,
                             fileName: GLM52SidecarPack.legacyQ4FileName)
        let index = try XCTUnwrap(GLM52SidecarPack.scan(
            directory: directory, fileName: q4Name,
            legacyFileName: GLM52SidecarPack.legacyQ4FileName,
            sourceFileSize: sourceFileSize))
        XCTAssertTrue(index.path.hasSuffix(".q4dense"))
        XCTAssertEqual(index.sections[21]?.length, UInt64(payload.count))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: GLM52SidecarPack.path(
                directory: directory,
                fileName: GLM52SidecarPack.legacyQ4FileName)))
        XCTAssertTrue(FileManager.default.fileExists(atPath: index.path))
    }

    func testTornTailIsTruncatedOnNextAppend() throws {
        let payloadA: [UInt8] = Array(repeating: 7, count: 640)
        _ = try writeSection(layer: 11, bytes: payloadA)
        let packPath = GLM52SidecarPack.path(directory: directory,
                                             fileName: q4Name)
        let intact = try XCTUnwrap(GLM52SidecarPack.scan(
            directory: directory, fileName: q4Name,
            sourceFileSize: sourceFileSize))

        // Coda strappata: header di sezione valido ma contenuto troncato
        // (una build interrotta a metà scrittura).
        let handle = try XCTUnwrap(FileHandle(forWritingAtPath: packPath))
        try handle.seekToEnd()
        var torn = Data()
        withUnsafeBytes(of: UInt32(0x4345_5347).littleEndian) {
            torn.append(contentsOf: $0)
        }
        withUnsafeBytes(of: UInt32(12).littleEndian) {
            torn.append(contentsOf: $0)
        }
        withUnsafeBytes(of: UInt64(10_000).littleEndian) {
            torn.append(contentsOf: $0)
        }
        torn.append(Data(repeating: 9, count: 100))   // 100 << 10_000
        try handle.write(contentsOf: torn)
        try handle.close()

        // La scansione ignora la coda; l'append successivo la tronca e la
        // sezione nuova atterra pulita.
        let scanned = try XCTUnwrap(GLM52SidecarPack.scan(
            directory: directory, fileName: q4Name,
            sourceFileSize: sourceFileSize))
        XCTAssertEqual(scanned.sections.count, 1)
        XCTAssertEqual(scanned.validEnd, intact.validEnd)
        let payloadB: [UInt8] = Array(repeating: 3, count: 320)
        _ = try writeSection(layer: 12, bytes: payloadB)
        let final = try XCTUnwrap(GLM52SidecarPack.scan(
            directory: directory, fileName: q4Name,
            sourceFileSize: sourceFileSize))
        XCTAssertEqual(final.sections.count, 2)
        let sectionB = try XCTUnwrap(final.sections[12])
        let pack = try GLM52PayloadReader(path: packPath)
        let view = try pack.windowed(offset: sectionB.offset,
                                     length: sectionB.length)
        XCTAssertEqual(try view.bytes(of: GLM52WeightDescriptor(
            name: "b", type: GLM52TensorSchema.q8_0, dims: [],
            absOffset: 0, bytes: UInt64(payloadB.count))), payloadB)
    }

    func testTwoPackFilesCoexistIndependently() throws {
        // Sidecar Q4 (<gguf>.q4dense) e bundle esperti (<gguf>.expbundle):
        // stesso contenitore, file distinti nella stessa directory — le
        // sezioni non devono interferire.
        let q4Payload: [UInt8] = [10, 11, 12, 13]
        let expertsPayload: [UInt8] = [20, 21, 22]
        _ = try writeSection(layer: 11, bytes: q4Payload)
        _ = try writeSection(layer: 11, bytes: expertsPayload,
                             fileName: expertsName)
        let q4Index = try XCTUnwrap(GLM52SidecarPack.scan(
            directory: directory, fileName: q4Name,
            sourceFileSize: sourceFileSize))
        let expertsIndex = try XCTUnwrap(GLM52SidecarPack.scan(
            directory: directory, fileName: expertsName,
            sourceFileSize: sourceFileSize))
        XCTAssertEqual(q4Index.sections[11]?.length,
                       UInt64(q4Payload.count))
        XCTAssertEqual(expertsIndex.sections[11]?.length,
                       UInt64(expertsPayload.count))
        XCTAssertNotEqual(q4Index.path, expertsIndex.path)
        XCTAssertTrue(expertsIndex.path.hasSuffix(".expbundle"))
    }

    func testLastSectionWinsForUpgrades() throws {
        _ = try writeSection(layer: 11, bytes: [1, 1, 1, 1])
        _ = try writeSection(layer: 11, bytes: [2, 2, 2, 2, 2])
        let index = try XCTUnwrap(GLM52SidecarPack.scan(
            directory: directory, fileName: q4Name,
            sourceFileSize: sourceFileSize))
        let section = try XCTUnwrap(index.sections[11])
        XCTAssertEqual(section.length, 5)
        let pack = try GLM52PayloadReader(
            path: GLM52SidecarPack.path(directory: directory,
                                        fileName: q4Name))
        let view = try pack.windowed(offset: section.offset,
                                     length: section.length)
        XCTAssertEqual(try view.bytes(of: GLM52WeightDescriptor(
            name: "v2", type: GLM52TensorSchema.q8_0, dims: [],
            absOffset: 0, bytes: 5)), [2, 2, 2, 2, 2])
    }
}
