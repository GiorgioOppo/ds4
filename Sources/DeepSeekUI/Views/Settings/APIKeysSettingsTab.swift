import SwiftUI

/// Settings → API Keys. Lets the user paste, save, test, and
/// delete the OpenRouter API key without exposing the value in
/// any persistent plist (the key lives in Keychain under
/// `KeychainAccount.openRouterAPIKey`).
///
/// "Configured" state is reactive on view appearance; saves and
/// deletes immediately flip it. A "Test" button hits
/// `OpenRouter /auth/key` so the user can verify a key before
/// trying it in the chat.
struct APIKeysSettingsTab: View {
    @State private var draftKey: String = ""
    @State private var hasKey: Bool = false
    @State private var testResult: TestResult?
    @State private var isTesting: Bool = false

    private enum TestResult {
        case success
        case failure(String)
    }

    var body: some View {
        Form {
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
                SecureField("API key (sk-or-…)", text: $draftKey)
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Button("Save") { saveKey() }
                        .disabled(draftKey.trimmingCharacters(in: .whitespaces).isEmpty)
                        .keyboardShortcut(.return, modifiers: .command)
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
                        Label("Key accepted by OpenRouter",
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
                Text("OpenRouter")
            } footer: {
                Text("Get a key at https://openrouter.ai/keys. Stored in the macOS Keychain — never written to a plist.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding(20)
        .onAppear { refreshHasKey() }
    }

    private func refreshHasKey() {
        hasKey = KeychainStore.exists(account: KeychainAccount.openRouterAPIKey)
    }

    private func saveKey() {
        let trimmed = draftKey.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        do {
            try KeychainStore.set(trimmed, account: KeychainAccount.openRouterAPIKey)
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
            try KeychainStore.delete(account: KeychainAccount.openRouterAPIKey)
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
            ? (KeychainStore.get(account: KeychainAccount.openRouterAPIKey) ?? "")
            : draftKey.trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty else { return }
        isTesting = true
        testResult = nil
        Task {
            let client = OpenRouterClient()
            do {
                try await client.validateKey(key)
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
