import SwiftUI

/// Settings → API Keys. Lets the user paste, save, test, and
/// delete API keys for the remote providers (OpenRouter,
/// Anthropic). Keys live in the Keychain under
/// `KeychainAccount.openRouterAPIKey` / `.anthropicAPIKey`; no plist
/// ever sees the value.
///
/// "Configured" state is reactive on view appearance; saves and
/// deletes immediately flip it. A "Test" button hits the provider's
/// validation endpoint so the user can verify a key before trying
/// it in the chat.
struct APIKeysSettingsTab: View {
    var body: some View {
        Form {
            ProviderKeySection(
                providerName: "OpenRouter",
                keychainAccount: KeychainAccount.openRouterAPIKey,
                placeholder: "API key (sk-or-…)",
                footerHint: "Get a key at https://openrouter.ai/keys. "
                          + "Stored in the macOS Keychain — never written to a plist.",
                successMessage: "Key accepted by OpenRouter",
                validator: { key in
                    try await OpenRouterClient().validateKey(key)
                })

            ProviderKeySection(
                providerName: "Anthropic",
                keychainAccount: KeychainAccount.anthropicAPIKey,
                placeholder: "API key (sk-ant-…)",
                footerHint: "Get a key at https://console.anthropic.com/settings/keys. "
                          + "Used for native `api.anthropic.com/v1/messages` calls so "
                          + "prompt caching (`cache_control`) is available — unreachable "
                          + "via OpenRouter.",
                successMessage: "Key accepted by Anthropic",
                validator: { key in
                    try await AnthropicClient().validateKey(key)
                })
        }
        .formStyle(.grouped)
        .padding(20)
    }
}

/// One provider's API-key pane. Generic over the validation closure
/// so the per-provider differences (label, hint, validator) stay in
/// one place.
private struct ProviderKeySection: View {
    let providerName: String
    let keychainAccount: String
    let placeholder: String
    let footerHint: String
    let successMessage: String
    let validator: @Sendable (String) async throws -> Void

    @State private var draftKey: String = ""
    @State private var hasKey: Bool = false
    @State private var testResult: TestResult?
    @State private var isTesting: Bool = false

    private enum TestResult {
        case success
        case failure(String)
    }

    var body: some View {
        Section {
            if hasKey {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.seal.fill")
                        .foregroundStyle(Color.green)
                    Text("API key configured — stored in macOS Keychain.")
                        .font(.callout)
                    Spacer()
                    Button("Delete", role: .destructive) {
                        deleteKey()
                    }
                    .controlSize(.small)
                }
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.circle")
                        .foregroundStyle(Color.orange)
                    Text("No API key configured.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            SecureField(placeholder, text: $draftKey)
                .textFieldStyle(.roundedBorder)
            HStack {
                Button("Save") { saveKey() }
                    .disabled(draftKey
                               .trimmingCharacters(in: .whitespaces)
                               .isEmpty)
                Button("Test") { testKey() }
                    .disabled(isTesting
                              || (draftKey.trimmingCharacters(in: .whitespaces).isEmpty
                                  && !hasKey))
                if isTesting {
                    ProgressView().controlSize(.small)
                }
                Spacer()
            }
            if let result = testResult {
                switch result {
                case .success:
                    Label(successMessage,
                           systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(Color.green)
                case .failure(let msg):
                    Label(msg, systemImage: "exclamationmark.octagon.fill")
                        .font(.caption)
                        .foregroundStyle(Color.red)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        } header: {
            Text(providerName)
        } footer: {
            Text(footerHint)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .onAppear { refreshHasKey() }
    }

    private func refreshHasKey() {
        hasKey = KeychainStore.exists(account: keychainAccount)
    }

    private func saveKey() {
        let trimmed = draftKey.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        do {
            try KeychainStore.set(trimmed, account: keychainAccount)
            draftKey = ""
            testResult = nil
            refreshHasKey()
        } catch {
            testResult = .failure(
                (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription)
        }
    }

    private func deleteKey() {
        do {
            try KeychainStore.delete(account: keychainAccount)
            testResult = nil
            refreshHasKey()
        } catch {
            testResult = .failure(
                (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription)
        }
    }

    private func testKey() {
        let key = draftKey.trimmingCharacters(in: .whitespaces).isEmpty
            ? (KeychainStore.get(account: keychainAccount) ?? "")
            : draftKey.trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty else { return }
        isTesting = true
        testResult = nil
        let v = validator
        Task {
            do {
                try await v(key)
                await MainActor.run {
                    self.testResult = .success
                    self.isTesting = false
                }
            } catch {
                let msg = (error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription
                await MainActor.run {
                    self.testResult = .failure(msg)
                    self.isTesting = false
                }
            }
        }
    }
}
