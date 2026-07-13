import Foundation
import Security

/// Keychain-backed storage for the user's Hugging Face token, edited from the
/// GUI (Settings → Hugging Face) and passed EXPLICITLY to
/// `ModelDownloader.download(token:)` — it never touches UserDefaults, which is
/// a plaintext plist on disk. `ModelDownloader.resolveToken` itself is left
/// unchanged (explicit > `HF_TOKEN` env > `~/.cache/huggingface/token`): the
/// keychain is app-private (a differently-signed CLI process would trigger a
/// keychain prompt), so the app reads it here and hands it over as the
/// "explicit" tier, which also keeps the demo/tests keychain-free.
public enum HFTokenStore {
    private static let service = "org.ds4.dwarfstar.hf-token"
    private static let account = "huggingface"

    /// The stored token, trimmed; nil when absent or empty.
    public static func load() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let s = String(data: data, encoding: .utf8) else { return nil }
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }

    /// Store (or replace) the token. Returns false on an empty token or a
    /// keychain error.
    @discardableResult
    public static func save(_ token: String) -> Bool {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8) else { return false }
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let update: [String: Any] = [kSecValueData as String: data]
        let status = SecItemUpdate(base as CFDictionary, update as CFDictionary)
        if status == errSecSuccess { return true }
        guard status == errSecItemNotFound else { return false }
        var add = base
        add[kSecValueData as String] = data
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }

    /// Delete the stored token. Returns true when it is gone (also when it was
    /// never there).
    @discardableResult
    public static func clear() -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    /// Redacted display form for the status line ("hf_ab…wxyz"): enough to
    /// recognize the token, never enough to copy it off the screen.
    public static func masked(_ token: String) -> String {
        guard token.count > 12 else { return "•••" }
        return "\(token.prefix(5))…\(token.suffix(4))"
    }

    /// Which token the next download would actually use, mirroring the resolve
    /// order (keychain is passed as the explicit tier, so it wins): a
    /// human-readable source label plus the redacted token, or nil when no
    /// token is configured anywhere.
    public static func activeSourceDescription() -> String? {
        if let t = load() { return "Keychain: \(masked(t))" }
        if let t = ProcessInfo.processInfo.environment["HF_TOKEN"],
           !t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "HF_TOKEN environment variable: \(masked(t.trimmingCharacters(in: .whitespacesAndNewlines)))"
        }
        let p = (NSHomeDirectory() as NSString).appendingPathComponent(".cache/huggingface/token")
        if let t = try? String(contentsOfFile: p, encoding: .utf8) {
            let trimmed = t.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return "~/.cache/huggingface/token: \(masked(trimmed))" }
        }
        return nil
    }
}
