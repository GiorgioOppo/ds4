/// Single enablement switch for the future Kimi K3 engine.
///
/// Keep false until the virtual five-part GGUF reader, tokenizer, reference
/// layer and Metal decoder pass real-weight logits parity.
public enum KimiK3RuntimeGate {
    public static let enabled = false
}
