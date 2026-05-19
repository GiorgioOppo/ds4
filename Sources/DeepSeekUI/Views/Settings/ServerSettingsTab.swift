import SwiftUI

/// Settings → Server. Toggles the local OpenAI-compatible HTTP
/// server on/off, exposes port + bind address, and lets the user
/// stash an optional bearer token in the Keychain. Status text
/// reflects the live `LocalServerController.isRunning` state and
/// surfaces the last bind error if the listener failed to come up.
///
/// The toggle is debounced through the controller's `start` / `stop`
/// actor calls — no spinner is shown because bind/teardown is
/// effectively instant on loopback. If a port collision happens
/// (`bindFailed`), the toggle bounces back off and the error label
/// explains why.
struct ServerSettingsTab: View {
    @ObservedObject var controller: LocalServerController

    @AppStorage(AppSettingsKey.serverEnabled) private var enabled: Bool = false
    @AppStorage(AppSettingsKey.serverPort) private var port: Int = 8080
    @AppStorage(AppSettingsKey.serverBindAddress)
        private var bindAddress: String = "127.0.0.1"

    @State private var draftToken: String = ""
    @State private var hasToken: Bool = false

    var body: some View {
        Form {
            Section {
                Toggle("Run local OpenAI-compatible server", isOn: $enabled)
                    .onChange(of: enabled) { _, newValue in
                        Task { await applyToggle(newValue) }
                    }

                HStack {
                    Image(systemName: controller.isRunning
                          ? "circle.fill" : "circle")
                        .foregroundStyle(controller.isRunning
                                          ? Color.green : Color.secondary)
                        .imageScale(.small)
                    Text(controller.isRunning
                         ? "Listening on http://\(bindAddress):\(port)"
                         : "Stopped")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }

                if let err = controller.lastError {
                    Label(err, systemImage: "exclamationmark.octagon.fill")
                        .font(.caption)
                        .foregroundStyle(Color.red)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } header: {
                Text("Status")
            } footer: {
                Text("Exposes `/v1/models` and `/v1/chat/completions` "
                     + "(streaming + buffered) backed by the currently "
                     + "loaded local model. Tools and JSON-schema "
                     + "constrained output are wired in follow-up commits.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section {
                HStack {
                    Text("Port")
                    Spacer()
                    TextField("Port", value: $port,
                               formatter: NumberFormatter())
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                        .textFieldStyle(.roundedBorder)
                        .disabled(controller.isRunning)
                }
                HStack {
                    Text("Bind address")
                    Spacer()
                    TextField("127.0.0.1", text: $bindAddress)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 160)
                        .textFieldStyle(.roundedBorder)
                        .disabled(controller.isRunning)
                }
            } header: {
                Text("Network")
            } footer: {
                Text("Stop the server before changing port or address. "
                     + "`127.0.0.1` is loopback-only and the safe default.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section {
                if hasToken {
                    HStack(spacing: 6) {
                        Image(systemName: "checkmark.seal.fill")
                            .foregroundStyle(Color.green)
                        Text("Bearer token configured — clients must send "
                             + "`Authorization: Bearer <token>`.")
                            .font(.callout)
                        Spacer()
                        Button("Delete", role: .destructive) { deleteToken() }
                            .controlSize(.small)
                    }
                } else {
                    HStack(spacing: 6) {
                        Image(systemName: "lock.open")
                            .foregroundStyle(Color.orange)
                        Text("No token — every localhost client can call the API.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                }
                SecureField("Bearer token (optional)", text: $draftToken)
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Button("Save token") { saveToken() }
                        .disabled(draftToken
                                   .trimmingCharacters(in: .whitespaces)
                                   .isEmpty)
                    Spacer()
                }
            } header: {
                Text("Authentication")
            } footer: {
                Text("Stored in macOS Keychain under "
                     + "`KeychainAccount.serverBearerToken`. Restart the "
                     + "server (toggle off then on) to pick up changes.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding(20)
        .onAppear { refreshTokenState() }
    }

    private func applyToggle(_ newValue: Bool) async {
        if newValue {
            await controller.start(
                port: UInt16(clamping: port),
                address: bindAddress)
            if !controller.isRunning {
                // Bind failed — revert the toggle so the persisted
                // preference matches the live state.
                enabled = false
            }
        } else {
            await controller.stop()
        }
    }

    private func refreshTokenState() {
        hasToken = KeychainStore.exists(
            account: KeychainAccount.serverBearerToken)
    }

    private func saveToken() {
        let trimmed = draftToken.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        try? KeychainStore.set(trimmed,
                                account: KeychainAccount.serverBearerToken)
        draftToken = ""
        refreshTokenState()
    }

    private func deleteToken() {
        try? KeychainStore.delete(
            account: KeychainAccount.serverBearerToken)
        refreshTokenState()
    }
}
