import Foundation

public struct GenerationOptions {
    public var maxNewTokens: Int = 128
    public var temperature: Float = 1.0      // V4 default per generation_config.json

    public init(maxNewTokens: Int = 128, temperature: Float = 1.0) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
    }
}

/// Generation loop. Mirrors `generate()` in
/// `Reference/inference/generate.py` lines 27–69.
///
/// Differences vs the reference:
///   - single batch only (the CLI is interactive)
///   - no MTP speculation (n_mtp_layers > 0 path is left unimplemented)
///   - greedy when temperature == 0; otherwise Gumbel-max sampling, same as
///     the reference's `sample()`
public final class Generator {
    public let model: Transformer
    public let tokenizer: Tokenizer

    public init(model: Transformer, tokenizer: Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    public func generate(prompt: String,
                         options: GenerationOptions = GenerationOptions(),
                         onToken: (String) -> Void) {
        // NOT IMPLEMENTED: model.forward is a stub. Once the chain
        // (act_quant → fp8/fp4 GEMM → MLA → HC → MoE → head) is wired up,
        // this becomes the standard prefill+decode loop:
        //
        //   ids = tokenizer.encode(prompt)
        //   logits = model.forward(ids, 0)
        //   for _ in 0..<options.maxNewTokens:
        //       next = sample(logits, options.temperature)
        //       onToken(tokenizer.decode([next]))
        //       if next == tokenizer.eosId: break
        //       logits = model.forward([next], cur_pos)
        fatalError("Generator.generate is a placeholder until Transformer.forward exists")
    }
}
