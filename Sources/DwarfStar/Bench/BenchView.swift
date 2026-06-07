import SwiftUI
import Charts

/// Benchmark panel: configure and run ds4-bench, then chart prefill and
/// generation throughput across context frontiers.
struct BenchView: View {
    @Bindable var controller: BenchController

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("ds4-bench") {
                    TextField("Binario", text: $controller.binaryPath)
                    TextField("Modello GGUF", text: $controller.modelPath)
                    TextField("Prompt file", text: $controller.promptFile)
                }
                Section("Frontiere di contesto") {
                    Stepper("Start: \(controller.ctxStart)", value: $controller.ctxStart, in: 256...1_000_000, step: 1024)
                    Stepper("Max: \(controller.ctxMax)", value: $controller.ctxMax, in: 1024...1_000_000, step: 1024)
                    Stepper("Passo: \(controller.stepIncr)", value: $controller.stepIncr, in: 256...262_144, step: 1024)
                    Stepper("Token generati: \(controller.genTokens)", value: $controller.genTokens, in: 0...4096, step: 32)
                }
                Section {
                    if controller.isRunning {
                        Button(role: .destructive) { controller.stop() } label: {
                            Label("Ferma", systemImage: "stop.fill")
                        }
                    } else {
                        Button { controller.run() } label: {
                            Label("Avvia benchmark", systemImage: "gauge.with.dots.needle.67percent")
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .frame(maxHeight: 320)

            Divider()

            if controller.rows.isEmpty {
                ContentUnavailableView("Nessun dato", systemImage: "chart.xyaxis.line",
                                       description: Text(controller.isRunning ? "Benchmark in corso…" : "Avvia un benchmark per vedere il throughput."))
                    .frame(maxHeight: .infinity)
            } else {
                throughputChart
                    .padding()
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
