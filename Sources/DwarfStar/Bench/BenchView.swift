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
                          ? "Native benchmark: measures prefill and generation throughput (tokens/s) on the shared in-process engine loaded in Settings."
                          : "Distributed benchmark: measures prefill and generation throughput (tokens/s) on the cluster, reusing the coordinator already connected in Chat -> Distributed.",
                          systemImage: "gauge.with.dots.needle.67percent")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Section("Engine") {
                    Picker("Engine", selection: $controller.mode) {
                        ForEach(BenchMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    if controller.mode == .distributed {
                        LabeledContent("Route", value: controller.distRoute)
                        if !controller.distConnected {
                            Label("Coordinator not connected: open it in Chat -> Distributed and press Connect.",
                                  systemImage: "exclamationmark.triangle")
                                .font(.callout).foregroundStyle(.orange)
                        }
                    }
                }
                .disabled(controller.isRunning)
                Section("Model (from Settings)") {
                    LabeledContent("GGUF", value: (controller.modelPath as NSString).lastPathComponent)
                    LabeledContent("Context", value: "\(controller.contextSize) tokens")
                }
                Section("Context Frontiers") {
                    Stepper("Start: \(controller.ctxStart)", value: $controller.ctxStart, in: 64...200_000, step: 256)
                    Stepper("Max: \(controller.ctxMax)", value: $controller.ctxMax, in: 256...200_000, step: 256)
                    Stepper("Step: \(controller.stepIncr)", value: $controller.stepIncr, in: 64...32_768, step: 256)
                    Stepper("Generated tokens per point: \(controller.genTokens)",
                            value: $controller.genTokens, in: 1...512, step: 8)
                }
                .disabled(controller.isRunning)
                Section {
                    if controller.isRunning {
                        runningBadge
                        Button(role: .destructive) { controller.stop() } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                    } else {
                        Button { controller.run() } label: {
                            Label("Start Benchmark", systemImage: "play.fill")
                        }
                        .disabled(controller.mode == .distributed && !controller.distConnected)
                    }
                }
            }
            .formStyle(.grouped)
            .frame(maxHeight: 320)

            Divider()

            if controller.rows.isEmpty {
                ContentUnavailableView("No Data", systemImage: "chart.xyaxis.line",
                                       description: Text(controller.isRunning
                                                         ? "Benchmark running - \(controller.runningLabel ?? "")"
                                                         : "Start a benchmark to see throughput."))
                    .frame(maxHeight: .infinity)
            } else {
                VStack(spacing: 0) {
                    if controller.isRunning { runningBadge.padding(.horizontal).padding(.top, 8) }
                    throughputChart.padding()
                }
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

    /// Live "running on X" indicator: spinner + colored engine chip. Local is
    /// green/memorychip, distributed is blue/network so the two are unmistakable.
    private var runningBadge: some View {
        let distributed = controller.runningMode == .distributed
        return HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Label("Running: \(controller.runningLabel ?? "")",
                  systemImage: distributed ? "network" : "memorychip")
                .foregroundStyle(distributed ? Color.blue : Color.green)
                .font(.callout)
            Spacer(minLength: 0)
        }
    }

    private var throughputChart: some View {
        Chart {
            ForEach(controller.rows) { row in
                LineMark(x: .value("Context", row.ctxTokens),
                         y: .value("t/s", row.prefillTps),
                         series: .value("Series", "Prefill"))
                    .foregroundStyle(by: .value("Series", "Prefill"))
                PointMark(x: .value("Context", row.ctxTokens),
                          y: .value("t/s", row.prefillTps))
                    .foregroundStyle(by: .value("Series", "Prefill"))
            }
            ForEach(controller.rows) { row in
                LineMark(x: .value("Context", row.ctxTokens),
                         y: .value("t/s", row.genTps),
                         series: .value("Series", "Generation (p99)"))
                    .foregroundStyle(by: .value("Series", "Generation (p99)"))
                PointMark(x: .value("Context", row.ctxTokens),
                          y: .value("t/s", row.genTps))
                    .foregroundStyle(by: .value("Series", "Generation (p99)"))
            }
        }
        .chartXAxisLabel("Context Tokens")
        .chartYAxisLabel("Tokens/second")
        .chartLegend(position: .top)
    }
}
