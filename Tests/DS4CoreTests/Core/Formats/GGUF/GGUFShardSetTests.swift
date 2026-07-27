import XCTest
@testable import DS4Core

/// Two synthetic layer-range shards (built with GGUFWriter) merged through
/// GGUFShardSet must behave like one model: tensor lookups resolve across shards
/// with the right bytes, and metadata is first-shard-wins.
final class GGUFShardSetTests: XCTestCase {

    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("shard-\(UUID().uuidString).gguf")
    }

    private func f32Data(_ xs: [Float]) -> Data {
        var d = Data()
        for x in xs { withUnsafeBytes(of: x.bitPattern.littleEndian) { d.append(contentsOf: $0) } }
        return d
    }

    /// shardA: global metadata + blk.0 ; shardB: blk.1 + output.
    private func writeShards() throws -> (a: URL, b: URL) {
        var a = try GGUFWriter(alignment: 32)
        a.put("general.architecture", .text("deepseek4"))
        a.put("deepseek4.block_count", .uint32(2))
        a.add(.init(name: "blk.0.attn.weight", dims: [4], type: 0, data: f32Data([1, 2, 3, 4])))
        let au = tempURL(); try a.write(to: au.path)

        var b = try GGUFWriter(alignment: 32)
        b.put("general.architecture", .text("deepseek4"))   // duplicated global (ok)
        b.add(.init(name: "blk.1.attn.weight", dims: [4], type: 0, data: f32Data([5, 6, 7, 8])))
        b.add(.init(name: "output.weight", dims: [2], type: 0, data: f32Data([9, 10])))
        let bu = tempURL(); try b.write(to: bu.path)
        return (au, bu)
    }

    func testMergeAcrossShards() throws {
        let (au, bu) = try writeShards()
        defer { try? FileManager.default.removeItem(at: au); try? FileManager.default.removeItem(at: bu) }

        let set = try GGUFShardSet(paths: [au.path, bu.path], metalMapping: false)
        XCTAssertEqual(set.shards.count, 2)
        XCTAssertEqual(set.n_tensors, 3)

        // Global metadata resolves (first shard that declares it).
        XCTAssertEqual(set.string("general.architecture"), "deepseek4")
        XCTAssertEqual(set.u32("deepseek4.block_count"), 2)

        // Tensors from both shards resolve, with correct bytes read from the
        // owning shard's mapping.
        let t0 = try XCTUnwrap(set.find("blk.0.attn.weight"))
        XCTAssertEqual(t0.shardIndex, 0)
        XCTAssertEqual(set.tensorData("blk.0.attn.weight"), f32Data([1, 2, 3, 4]))

        let t1 = try XCTUnwrap(set.find("blk.1.attn.weight"))
        XCTAssertEqual(t1.shardIndex, 1)
        XCTAssertEqual(set.tensorData("blk.1.attn.weight"), f32Data([5, 6, 7, 8]))

        let out = try XCTUnwrap(set.find("output.weight"))
        XCTAssertEqual(out.shardIndex, 1)
        XCTAssertEqual(set.tensorData("output.weight"), f32Data([9, 10]))

        XCTAssertNil(set.findTensor("missing"))
    }

    func testDuplicateTensorNameThrows() throws {
        var a = try GGUFWriter()
        a.add(.init(name: "dup.weight", dims: [4], type: 0, data: f32Data([1, 2, 3, 4])))
        let au = tempURL(); try a.write(to: au.path)
        var b = try GGUFWriter()
        b.add(.init(name: "dup.weight", dims: [4], type: 0, data: f32Data([5, 6, 7, 8])))
        let bu = tempURL(); try b.write(to: bu.path)
        defer { try? FileManager.default.removeItem(at: au); try? FileManager.default.removeItem(at: bu) }

        XCTAssertThrowsError(try GGUFShardSet(paths: [au.path, bu.path], metalMapping: false))
    }

    func testEmptyThrows() {
        XCTAssertThrowsError(try GGUFShardSet(paths: [], metalMapping: false))
    }

    /// A single shard behaves like the plain model for reads.
    func testSingleShardEquivalence() throws {
        let (au, bu) = try writeShards()
        defer { try? FileManager.default.removeItem(at: au); try? FileManager.default.removeItem(at: bu) }
        let set = try GGUFShardSet(paths: [au.path], metalMapping: false)
        XCTAssertEqual(set.n_tensors, 1)
        XCTAssertEqual(set.tensorData("blk.0.attn.weight"), f32Data([1, 2, 3, 4]))
        XCTAssertNil(set.findTensor("blk.1.attn.weight"))   // lives only in shard B
    }
}
