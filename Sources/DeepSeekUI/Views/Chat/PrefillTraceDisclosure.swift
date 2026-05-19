import SwiftUI

/// Blocco grigio collassabile inserito fra il messaggio dell'utente
/// e la risposta dell'assistente al primo turn (cold prefill) di una
/// conversazione. Mostra esattamente il testo del prompt che il
/// modello sta per vedere — system message, blocco tools, project
/// context, history e nuovo turn dell'utente — decodificato dal
/// tokenizer dai token che il transformer ingerisce.
///
/// Specchio strutturale di `ReasoningDisclosure`: stesso layout
/// (`DisclosureGroup` con scroll interno + monospaced + max height),
/// ma label diversa e tint più tenue per non rubare attenzione alla
/// risposta vera e propria. Stato di default:
/// - durante il prefill (`isStreaming == true`): espanso, così
///   l'utente vede il prompt scorrere mentre arriva;
/// - dopo il prefill: collassato, perché di solito non interessa più
///   — un tap riapre il blocco se serve ispezionare il prompt.
struct PrefillTraceDisclosure: View {
    let trace: String
    /// True mentre il prefill è in corso. Forza l'apertura del
    /// disclosure così il flusso del prompt è visibile senza un tap.
    var isStreaming: Bool = false

    @State private var manualExpanded: Bool = false

    var body: some View {
        DisclosureGroup(isExpanded: Binding(
            get: { isStreaming || manualExpanded },
            set: { manualExpanded = $0 })
        ) {
            ScrollView(.vertical, showsIndicators: true) {
                Text(trace)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .frame(maxHeight: 220)
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
