import Foundation
import Security

/// Thin wrapper around Keychain Services. Used for the bits of
/// user data that can't sensibly live in `@AppStorage` because
/// the underlying `~/Library/Preferences/<bundle>.plist` is
/// plaintext readable by anything running as the user — API keys
/// being the obvious case.
///
/// Single shared service name (`com.deepseek.v4pro`) namespaces
/// every entry by an `account` string. Today's accounts:
///   - "openrouter.apiKey"
/// Future additions (Anthropic direct, OpenAI direct, etc.) drop
/// in here without schema changes.
///
/// Reading is sync — Keychain Services itself is sync — so the
/// callers should treat the lookup as a cheap memory read in the
/// happy path. macOS may prompt the user on first read after a
/// codesign change; subsequent reads are silent.
enum KeychainStore {
    private static let service = "com.deepseek.v4pro"

    /// Insert-or-update a string under `account`. The value is
    /// stored as UTF-8 bytes; reads on machines whose locale
    /// would re-interpret the bytes don't apply because we always
    /// round-trip through UTF-8.
    ///
    /// Throws `KeychainError.status` on any non-success OSStatus
    /// so the caller can surface a useful message in the UI
    /// instead of silently dropping the value.
    static func set(_ value: String, account: String) throws {
        let data = Data(value.utf8)
        let lookup: [String: Any] = [
            kSecClass as String:        kSecClassGenericPassword,
            kSecAttrService as String:  service,
            kSecAttrAccount as String:  account
        ]
        let attrs: [String: Any] = [
            kSecValueData as String: data
        ]
        let updateStatus = SecItemUpdate(
            lookup as CFDictionary, attrs as CFDictionary)
        switch updateStatus {
        case errSecSuccess:
            return
        case errSecItemNotFound:
            // No row yet — insert. Merge query + value attrs;
            // omit kSecReturnData (we don't care about the
            // inserted row).
            var insertItem = lookup
            insertItem[kSecValueData as String] = data
            let addStatus = SecItemAdd(insertItem as CFDictionary, nil)
            if addStatus != errSecSuccess {
                throw KeychainError.status(addStatus)
            }
        default:
            throw KeychainError.status(updateStatus)
        }
    }

    /// Fetch the string previously stored under `account`. Returns
    /// nil for both "no entry" and "couldn't decode as UTF-8" so
    /// callers can treat the path uniformly as a feature toggle
    /// ("is this thing configured?").
    static func get(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String:        kSecClassGenericPassword,
            kSecAttrService as String:  service,
            kSecAttrAccount as String:  account,
            kSecReturnData as String:   true,
            kSecMatchLimit as String:   kSecMatchLimitOne
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess,
              let data = item as? Data,
              let value = String(data: data, encoding: .utf8)
        else { return nil }
        return value
    }

    /// Drop the row for `account`. Idempotent — `errSecItemNotFound`
    /// is treated as success so "delete on logout" doesn't have
    /// to know whether the key was ever set.
    static func delete(account: String) throws {
        let query: [String: Any] = [
            kSecClass as String:        kSecClassGenericPassword,
            kSecAttrService as String:  service,
            kSecAttrAccount as String:  account
        ]
        let status = SecItemDelete(query as CFDictionary)
        if status != errSecSuccess && status != errSecItemNotFound {
            throw KeychainError.status(status)
        }
    }

    /// Cheaper than `get(...)` for the common "is this configured?"
    /// gate — uses `kSecReturnData: false` so the OS doesn't have
    /// to load + copy the secret into our address space.
    static func exists(account: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String:        kSecClassGenericPassword,
            kSecAttrService as String:  service,
            kSecAttrAccount as String:  account,
            kSecMatchLimit as String:   kSecMatchLimitOne
        ]
        return SecItemCopyMatching(query as CFDictionary, nil) == errSecSuccess
    }
}

enum KeychainError: LocalizedError {
    case status(OSStatus)

    var errorDescription: String? {
        switch self {
        case .status(let s):
            if let cf = SecCopyErrorMessageString(s, nil) as String? {
                return "Keychain: \(cf)"
            }
            return "Keychain error \(s)"
        }
    }
}

/// Named accounts used across the app — single source of truth so
/// typos in the account string don't silently shadow each other.
enum KeychainAccount {
    static let openRouterAPIKey = "openrouter.apiKey"
    /// Anthropic native API key, used by `AnthropicClient` for
    /// `api.anthropic.com/v1/messages` (TODO §10.4 / T4). Distinct
    /// from `openRouterAPIKey` so users can keep both backends
    /// configured side-by-side.
    static let anthropicAPIKey = "anthropic.apiKey"
    /// Optional bearer token required on requests to the local
    /// OpenAI-compatible server (TODO §10.1 / `LocalServer.swift`).
    /// Nil/empty = auth disabled (the default for localhost).
    static let serverBearerToken = "server.bearerToken"
}
