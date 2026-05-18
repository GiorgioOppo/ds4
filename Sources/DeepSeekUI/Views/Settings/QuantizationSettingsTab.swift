import SwiftUI

/// Settings tab "Quantization": riassume lo stato dei path
/// W8A16 / W8A8 / W4A16 / W2A16 e dei metodi di quantizzazione
/// (RTN active, AWQ scaffold, SmoothQuant/GPTQ stub).
///
/// I toggle qui sono preference SALVATE in UserDefaults; il loro
/// effetto runtime dipende dal wiring nel model loader e nel
/// converter (vedi commenti sotto). Sono esposti perchè facili da
/// passare dall'UI al loader quando il wiring arriverà.
struct QuantizationSettingsTab: View {

    @AppStorage(AppSettingsKey.useW8A8Activations)
    private var useW8A8Activations: Bool = false

    var body: some View {
        Form {
            // ---- Weight quantization status ----
            Section {
                row(method: "BF16 / FP8 / FP4",
                    state: .active,
                    note: "Default. Vedi Linear.swift e Quantization.swift.")
                row(method: "INT8 (W8A16)",
                    state: .active,
                    note: "Per-row, per-128, F16 group scales. " +
                          "Kernel int8_gemm.metal + simdgroup_matrix.")
                row(method: "INT4 (W4A16)",
                    state: .active,
                    note: "Packed nibbles, per-row per-128 F16 scales.")
                row(method: "INT2 (W2A16)",
                    state: .active,
                    note: "Packed 2-bit values. Sperimentale.")
                row(method: "INT8 W8A8 (activations int8)",
                    state: .optIn,
                    note: "Implementato lato kernel (act_quant_int8 + " +
                          "int8_gemm_w8a8). Toggle qui sotto è preference " +
                          "non ancora propagata al loader.")
            } header: {
                Text("Weight quantization formats")
            } footer: {
                Text("Lo status si riferisce ai kernel Metal. Il wiring " +
                     "del converter / loader può variare per format.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            // ---- W8A8 activation toggle ----
            Section {
                Toggle("Use W8A8 activations when loading INT8 models",
                       isOn: $useW8A8Activations)
                Text("Quando attivo (e il modello carica con pesi int8), " +
                     "`Linear` quantizza l'input a int8 prima del GEMM. " +
                     "Trade-off: ~2× throughput memory-bound, " +
                     "quantization-noise aggiuntivo. Default off.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Label {
                    Text("Status: preferenza salvata; il loader non la " +
                         "legge ancora. Wiring richiesto in " +
                         "Sources/DeepSeekKit/Model.swift al costruttore " +
                         "di Linear(...).")
                        .font(.caption)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                }
            } header: {
                Text("W8A8 path (opt-in)")
            }

            // ---- Calibrated quantization status ----
            Section {
                methodRow(name: "RTN (Round-to-Nearest)",
                          state: .active,
                          note: "Baseline. Implementato in Int8Quant / " +
                                "Int4Quant. Nessuna calibrazione richiesta.")
                methodRow(name: "AWQ (Activation-aware)",
                          state: .preview,
                          note: "Algoritmo in CalibratedQuant.swift. " +
                                "Manca runtime activation inverse-scale " +
                                "(richiede pre-mul nel Linear forward). " +
                                "ActivationObserver non ancora wirato.")
                methodRow(name: "SmoothQuant",
                          state: .scaffold,
                          note: "Stub: throw QuantNotImplemented.")
                methodRow(name: "GPTQ",
                          state: .scaffold,
                          note: "Stub: throw QuantNotImplemented (algoritmo " +
                                "OBS più complesso, follow-up).")
            } header: {
                Text("Calibrated quantization methods")
            } footer: {
                Text("La calibrazione gira a converter-time, non al " +
                     "runtime. Quando wirato, sarà esposto via flag " +
                     "`converter --quant-method awq|...`.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            // ---- Roadmap / pointers ----
            Section {
                LabeledContent("CalibratedQuant module") {
                    Text("Sources/DeepSeekKit/CalibratedQuant.swift")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("W8A8 kernels") {
                    Text("Sources/DeepSeekKit/Kernels/int8_gemm_w8a8.metal")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Roadmap") {
                    Text("TODO.md §0 Quantizzazione")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("References")
            }
        }
        .formStyle(.grouped)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // ---- Status rendering ----

    enum FeatureState {
        case active        // wirato end-to-end
        case optIn         // implementato ma toggle preference non propagato
        case preview       // algoritmo presente, integrazione incompleta
        case scaffold      // solo interfaccia / stub

        var label: String {
            switch self {
            case .active:   return "Active"
            case .optIn:    return "Opt-in"
            case .preview:  return "Preview"
            case .scaffold: return "Scaffold"
            }
        }
        var color: Color {
            switch self {
            case .active:   return .green
            case .optIn:    return .blue
            case .preview:  return .orange
            case .scaffold: return .gray
            }
        }
    }

    @ViewBuilder
    private func row(method: String, state: FeatureState, note: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(method).font(.callout)
                Spacer()
                stateBadge(state)
            }
            Text(note).font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private func methodRow(name: String, state: FeatureState, note: String) -> some View {
        row(method: name, state: state, note: note)
    }

    @ViewBuilder
    private func stateBadge(_ state: FeatureState) -> some View {
        Text(state.label)
            .font(.caption2.bold())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(state.color.opacity(0.15))
            .foregroundStyle(state.color)
            .cornerRadius(4)
    }
}
