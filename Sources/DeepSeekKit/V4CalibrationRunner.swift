import Foundation

public final class V4CalibrationRunner {
    public let model: Transformer
    public let tokenizer: any Tokenizer
    public let activationObserver: ActivationObserver?
    public let hessianObserver: HessianObserver?

    public var maxTokensPerBatch: Int = 1024

    public init(model: Transformer,
                tokenizer: any Tokenizer,
                activation: ActivationObserver? = nil,
                hessian: HessianObserver? = nil)
    {
        self.model = model
        self.tokenizer = tokenizer
        self.activationObserver = activation
        self.hessianObserver = hessian
    }

    public func observe(_ text: String) {
        // Disabled for MLX migration
    }

    public func tagNames() -> [String] {
        return []
    }
}
