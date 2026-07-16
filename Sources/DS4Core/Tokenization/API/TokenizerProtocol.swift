import Foundation

/// Architecture-neutral tokenizer surface needed by inference and benchmarks.
/// Role/reasoning/tool token ids intentionally do not belong to this protocol:
/// they are properties of a model's conversation backend.
public protocol TokenizerProtocol: AnyObject {
    var architecture: ModelArchitectureID { get }
    var nVocab: Int { get }

    func tokenize(_ text: String) -> [Int32]
    func tokenizeRenderedChat(_ text: String) -> [Int32]
    func tokenText(_ id: Int32) -> [UInt8]
}

