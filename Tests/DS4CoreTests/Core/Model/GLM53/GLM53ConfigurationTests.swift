import Foundation
import XCTest
import DS4Core

final class GLM53ConfigurationTests: XCTestCase {
    // Published glm5-next contract at upstream acf5c16; no tensor payloads.
    private func metadata() -> [(key: String, value: GGUFMetadataValue)] {
        let u: [(String, UInt32)] = [
            ("block_count",46),("trunk_block_count",45),("nextn_predict_layers",1),
            ("context_length",1048576),("embedding_length",4096),("vocab_size",154880),
            ("feed_forward_length",12288),("expert_feed_forward_length",2048),
            ("expert_count",288),("expert_used_count",8),("expert_shared_count",1),
            ("leading_dense_block_count",3),("attention.head_count",64),
            ("attention.key_length",256),("attention.value_length",256),
            ("attention.q_lora_rank",1536),("attention.kv_lora_rank",512),
            ("attention.rope_dimension_count",0),("attention.indexer.head_count",32),
            ("attention.indexer.key_length",128),("attention.indexer.top_k",2048),
            ("attention.indexer.pool_size",4),("linear_attention.head_count",64),
            ("linear_attention.head_dimension",128),("linear_attention.conv_kernel",4),
            ("hyper_connection.count",4),("hyper_connection.sinkhorn_iterations",20)
        ]
        let f: [(String, Float)] = [("expert_weights_scale",2.5),("swiglu_limit",10),
            ("attention.layer_norm_rms_epsilon",1e-5),("linear_attention.gate_lower_bound",-5),
            ("hyper_connection.epsilon",1e-6)]
        return [("general.architecture", .text("glm5-next"))]
            + u.map { ("glm5-next."+$0.0, .uint32($0.1)) }
            + f.map { ("glm5-next."+$0.0, .float32($0.1)) }
            + [("glm5-next.expert_weights_norm", .bool(true)),
               ("glm5-next.layer_types", .array(elementType: .uint32,
                    elements: (0..<46).map { .uint32($0 < 45 && $0 % 4 != 3 ? 0 : 1) }))]
    }
    private func load(_ metadata: [(key: String, value: GGUFMetadataValue)]) throws -> GLM53Configuration {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("glm53-config-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }
        try GGUFWriter(metadata: metadata).write(to: url.path)
        return try GLM53Configuration(model: GGUFModel(path: url.path, metalMapping: false, prefetchCPU: false))
    }
    func testPublishedGeometryAndAttentionSchedule() throws {
        let c = try load(metadata())
        XCTAssertEqual(c.layerCount,45)
        XCTAssertEqual(c.queryRank,1536)
        XCTAssertEqual(c.kvRank,512)
        XCTAssertEqual((0..<45).filter(GLM53Configuration.isKDALayer).count,34)
        XCTAssertFalse(GLM53Configuration.isKDALayer(45))
        XCTAssertFalse(GLM53Configuration.isKDALayer(-1))
    }
    func testMissingOrDifferentArchitectureAndRequiredFieldsReject() throws {
        for key in ["general.architecture","glm5-next.layer_types","glm5-next.attention.kv_lora_rank","glm5-next.hyper_connection.count"] {
            XCTAssertThrowsError(try load(metadata().filter { $0.key != key }), key)
        }
        for (key,value): (String,GGUFMetadataValue) in [
            ("general.architecture",.text("glm-dsa")),
            ("glm5-next.attention.kv_lora_rank",.uint32(256)),
            ("glm5-next.expert_weights_norm",.bool(false)),
            ("glm5-next.attention.layer_norm_rms_epsilon",.float32(1.09e-5)),
            ("glm5-next.layer_types",.array(elementType:.uint32,elements:Array(repeating:.uint32(0),count:46)))
        ] {
            XCTAssertThrowsError(try load(metadata().map { $0.key == key ? (key,value) : $0 }),key)
        }
    }
}
