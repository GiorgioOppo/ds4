import SwiftUI

/// Blocco grigio inserito fra il messaggio dell'utente e la
/// risposta dell'assistente: mostra esattamente cosa il modello
/// vede al turn corrente. Per i local chat = il prompt completo
/// decodificato dai token (system + tools + project + history +
/// turn). Per i remote chat = il body JSON inviato a OpenRouter
/// (messages array, tools, sampler, tool_choice).
///
/// Default ESPANSO e persistente come membro normale della
/// conversation — non auto-collassa quando la risposta arriva: il
/// trace è parte della chronology al pari del messaggio user e
/// della reply dell'assistente. Il pulsante di disclosure rimane
/// per nasconderlo manualmente se occupa troppo spazio, ma quella
/// scelta è dell'utente, non del flusso.
struct PrefillTraceDisclosure: View {
    let trace: String
    /// True mentre il trace si sta riempiendo (prefill in corso o
    /// dump remote in streaming). Solo cosmetico: cambia la label
    /// in "streaming…". Non più usato per decidere espanso/chiuso
    /// — quello è sempre espanso tranne se l'utente collassa.
    var isStreaming: Bool = false

    @State private var collapsed: Bool = false

    var body: some View {
        DisclosureGroup(isExpanded: Binding(
            get: { !collapsed },
            set: { collapsed = !$0 })
        ) {
            ScrollView(.vertical, showsIndicators: true) {
                Text(trace)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .frame(maxHeight: 320)
            .background(Color(NSColor.controlBackgroundColor),
                         in: RoundedRectangle(cornerRadius: 8))
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "eye.trianglebadge.exclamationmark")
                    .font(.caption2)
                Text(isStreaming
                      ? "Prompt al modello (streaming…)"
                      : "Prompt al modello · \(trace.count) char")
                    .font(.caption)
            }
            .foregroundStyle(.secondary)
        }
    }
}
