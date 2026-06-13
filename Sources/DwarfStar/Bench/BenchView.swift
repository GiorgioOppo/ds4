import SwiftUI
import Charts

/// Native benchmark panel: run prefill + generation across context frontiers and
/// chart the throughput (no subprocess). The engine is either the local in-process
/// one or the already-connected distributed cluster (segmented control).
struct BenchView: View {
    @Bindable var controller: BenchController

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    Label(controller.mode == .local
                          ? "Benchmark nativo: misura prefill e generazione (token/s) del motore in-process a contesti crescenti. Carica una propria copia del modello (pesi mmap condivisi)."
                          : "Benchmark distribuito: misura prefill e generazione (token/s) sul cluster, riusando il coordinatore già connesso in Chat → Distribuito (nessuna seconda connessione).",
                          systemImage: "gauge.with.dots.needle.67percent")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Section("Motore") {
                    Picker("Motore", selection: $controller.mode) {
                        ForEach(BenchMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    if controller.mode == .distributed {
                        LabeledContent("Route", value: controller.distRoute)
                        if !controller.distConnected {
                            Label("Coordinatore non connesso: aprilo in Chat → Distribuito e premi «Connetti».",
                                  systemImage: "exclamationmark.triangle")
                                .font(.callout).foregroundStyle(.orange)
                        }
                    }
                }
                .disabled(controller.isRunning)
                Section("Modello (da Impostazioni)") {
                    LabeledContent("GGUF", value: (controller.modelPath as NSString).lastPathComponent)
                    LabeledContent("Contesto", value: "\(controller.contextSize) token")
                }
                Section("Frontiere di contesto") {
                    Stepper("Start: \(controller.ctxStart)", value: $controller.ctxStart, in: 64...200_000, step: 256)
                    Stepper("Max: \(controller.ctxMax)", value: $controller.ctxMax, in: 256...200_000, step: 256)
                    Stepper("Passo: \(controller.stepIncr)", value: $controller.stepIncr, in: 64...32_768, step: 256)
                    Stepper("Token generati per punto: \(controller.genTokens)",
                            value: $controller.genTokens, in: 1...512, step: 8)
                }
                .disabled(controller.isRunning)
                Section {
                    if controller.isRunning {
                        Button(role: .destructive) { controller.stop() } label: {
                            Label("Ferma", systemImage: "stop.fill")
                        }
                    } else {
                        Button { controller.run() } label: {
                            Label("Avvia benchmark", systemImage: "play.fill")
                        }
                        .disabled(controller.mode == .distributed && !controller.distConnected)
                    }
                }
            }
            .formStyle(.grouped)
            .frame(maxHeight: 320)

            Divider()

            if controller.rows.isEmpty {
                ContentUnavailableView("Nessun dato", systemImage: "chart.xyaxis.line",
                                       description: Text(controller.isRunning ? "Benchmark in corso…"
                                                                              : "Avvia un benchmark per vedere il throughput."))
                    .frame(maxHeight: .infinity)
            } else {
                throughputChart.padding()
            }

            if !controller.log.isEmpty {
                Divider()
                ScrollView {
                    Text(controller.log)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled).padding(8)
                }
                .frame(height: 110)
                .background(Color.black.opacity(0.05))
            }
        }
    }

    private var throughputChart: some View {
        Chart {
            ForEach(controller.rows) { row in
                LineMark(x: .value("Contesto", row.ctxTokens),
                         y: .value("t/s", row.prefillTps),
                         series: .value("Serie", "Prefill"))
                    .foregroundStyle(by: .value("Serie", "Prefill"))
                PointMark(x: .value("Contesto", row.ctxTokens),
                          y: .value("t/s", row.prefillTps))
                    .foregroundStyle(by: .value("Serie", "Prefill"))
            }
            ForEach(controller.rows) { row in
                LineMark(x: .value("Contesto", row.ctxTokens),
                         y: .value("t/s", row.genTps),
                         series: .value("Serie", "Generazione"))
                    .foregroundStyle(by: .value("Serie", "Generazione"))
                PointMark(x: .value("Contesto", row.ctxTokens),
                          y: .value("t/s", row.genTps))
                    .foregroundStyle(by: .value("Serie", "Generazione"))
            }
        }
        .chartXAxisLabel("Token di contesto")
        .chartYAxisLabel("Token/secondo")
        .chartLegend(position: .top)
    }
}
