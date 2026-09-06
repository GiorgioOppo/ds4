import SwiftUI

/// Suggestions only prepare a draft, so a new conversation never sends a prompt
/// before the user has had a chance to personalize it.
struct ChatEmptyState: View {
    let agentName: String
    let selectPrompt: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Label("Pronto con \(agentName)", systemImage: "sparkles")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.tint)
                Text("Da dove iniziamo?")
                    .font(.largeTitle.weight(.semibold))
                Text("Scrivi un messaggio o scegli uno spunto da personalizzare.")
                    .foregroundStyle(.secondary)
            }
            VStack(spacing: 10) {
                suggestion("Spiega un concetto", detail: "Una spiegazione chiara, con esempi concreti.",
                           icon: "lightbulb", prompt: "Spiegami in modo semplice questo concetto, con un esempio concreto: ")
                suggestion("Rivedi del codice", detail: "Trova problemi e rendi il codice più leggibile.",
                           icon: "chevron.left.forwardslash.chevron.right", prompt: "Rivedi questo codice e suggerisci miglioramenti alla correttezza e alla leggibilità:\n\n")
                suggestion("Riassumi un documento", detail: "Allega un file e metti a fuoco i punti principali.",
                           icon: "doc.text", prompt: "Riassumi il documento allegato: evidenzia i punti principali e le azioni da intraprendere.")
            }
        }
        .frame(maxWidth: 520, alignment: .leading)
        .padding(.vertical, 32)
    }

    private func suggestion(_ title: String, detail: String, icon: String, prompt: String) -> some View {
        Button { selectPrompt(prompt) } label: {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundStyle(.tint)
                    .frame(width: 28)
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.callout.weight(.medium))
                    Text(detail).font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                Image(systemName: "arrow.up.left").foregroundStyle(.secondary)
                    .font(.caption)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.background, in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.primary.opacity(0.08)))
            .contentShape(RoundedRectangle(cornerRadius: 12))
        }
        .buttonStyle(.plain)
        .help("Inserisci questo spunto nel messaggio")
    }
}

struct ChatResponseSettings: View {
    @Bindable var store: ChatStore

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Impostazioni della risposta").font(.headline)
            if store.supportsReasoning {
                Toggle("Ragionamento", isOn: $store.think)
                    .toggleStyle(.switch)
                Text("Il modello dedica più tempo a ragionare prima di rispondere.")
                    .font(.caption).foregroundStyle(.secondary)
                Divider()
            }
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Creatività")
                    Spacer()
                    Text(store.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit().foregroundStyle(.secondary)
                }
                Slider(value: $store.temperature, in: 0...1.5, step: 0.05)
                    .accessibilityLabel("Temperatura di campionamento")
                HStack {
                    Text("Più precisa")
                    Spacer()
                    Text("Più creativa")
                }
                .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("Deterministica") { store.temperature = 0 }
                    Button("Precisa") { store.temperature = 0.3 }
                    Button("Predefinita") { store.temperature = 0.6 }
                }
                .controlSize(.small)
            }
            Divider()
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Penalità di ripetizione")
                    Spacer()
                    Text(store.repetitionPenalty, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit().foregroundStyle(.secondary)
                }
                Slider(value: $store.repetitionPenalty, in: 1.0...1.5, step: 0.05)
                    .accessibilityLabel("Penalità di ripetizione")
                Text("Aumenta questo valore se il modello ripete le stesse frasi.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Le modifiche si applicano al prossimo messaggio.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(20)
        .frame(width: 350)
    }
}
