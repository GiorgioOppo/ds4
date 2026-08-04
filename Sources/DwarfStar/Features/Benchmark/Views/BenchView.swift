import SwiftUI
import Charts
import DS4Engine

/// Native benchmark panel. Checkpoint fixtures are SHA-bound and never selected
/// from a filename alone, so a model revision cannot produce a false regression.
struct BenchView: View {
    @Bindable var controller: BenchController

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("Benchmark Type") {
                    Picker("Benchmark Type", selection: $controller.kind) {
                        ForEach(BenchKind.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                }
                .disabled(controller.isRunning)

                Section {
                    Label(benchmarkExplanation,
                          systemImage: benchmarkExplanationIcon)
                        .font(.callout).foregroundStyle(.secondary)
                }
                Section("Engine") {
                    Picker("Engine", selection: $controller.mode) {
                        ForEach(BenchMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    if controller.mode == .distributed {
                        LabeledContent("Route", value: controller.distRoute)
                        if controller.kind != .speed {
                            Label("Correctness and Checkpoint tests are available only on the Local engine. Select Local; no silent fallback will be used.",
                                  systemImage: "exclamationmark.triangle")
                                .font(.callout).foregroundStyle(.orange)
                        } else if !controller.distConnected {
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

                switch controller.kind {
                case .speed:
                    Section("Context Frontiers") {
                        Stepper("Start: \(controller.ctxStart)", value: $controller.ctxStart, in: 64...200_000, step: 256)
                        Stepper("Max: \(controller.ctxMax)", value: $controller.ctxMax, in: 256...200_000, step: 256)
                        Stepper("Step: \(controller.stepIncr)", value: $controller.stepIncr, in: 64...32_768, step: 256)
                        Stepper("Generated tokens per point: \(controller.genTokens)",
                                value: $controller.genTokens, in: 1...512, step: 8)
                    }
                    .disabled(controller.isRunning)
                case .accuracy:
                    Section("Evaluation Text") {
                        TextEditor(text: $controller.accuracyText)
                            .font(.body)
                            .frame(minHeight: 105)
                            .accessibilityLabel("Text used for next-token evaluation")
                        LabeledContent("Characters", value: "\(controller.accuracyText.count)")
                    }
                    .disabled(controller.isRunning)

                    Section("Correctness Parameters") {
                        Stepper("Minimum sampled context: \(controller.accuracyMinContextTokens)",
                                value: $controller.accuracyMinContextTokens,
                                in: 1...controller.accuracyContextLimit, step: 8)
                            .onChange(of: controller.accuracyMinContextTokens) { _, minimum in
                                controller.accuracyMaxContextTokens = max(
                                    minimum,
                                    min(controller.accuracyMaxContextTokens,
                                        controller.accuracyContextLimit)
                                )
                            }
                        Stepper("Maximum sampled context: \(controller.accuracyMaxContextTokens)",
                                value: $controller.accuracyMaxContextTokens,
                                in: 1...controller.accuracyContextLimit, step: 8)
                            .onChange(of: controller.accuracyMaxContextTokens) { _, maximum in
                                controller.accuracyMinContextTokens = min(
                                    max(1, maximum),
                                    controller.accuracyMinContextTokens
                                )
                            }
                        Stepper("Maximum evaluated tokens per piece: \(controller.accuracyMaxTokensPerPiece)",
                                value: $controller.accuracyMaxTokensPerPiece,
                                in: 1...8_192, step: 16)
                        Stepper("Random pieces: \(controller.accuracyPieceCount)",
                                value: $controller.accuracyPieceCount,
                                in: 1...256, step: 1)
                        LabeledContent("Random seed") {
                            HStack(spacing: 8) {
                                TextField("Seed", value: $controller.accuracySeed,
                                          format: .number.grouping(.never))
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 150)
                                    .multilineTextAlignment(.trailing)
                                Text("0x\(String(controller.accuracySeed, radix: 16, uppercase: true))")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                        }
                        Stepper("Tokens per chart block: \(controller.accuracyBucketSize)",
                                value: $controller.accuracyBucketSize, in: 1...256, step: 4)
                        Text("The same text, context range, piece count and seed reproduce the same sampled positions.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .disabled(controller.isRunning)
                case .official:
                    Section("Official checkpoint fixture") {
                        LabeledContent("Source", value: "antirez/ds4@b7e9f00")
                        LabeledContent("Cases", value: "3 short smoke vectors")
                        Label("The exact GGUF SHA-256 selects the fixture. A filename or an unknown model digest is rejected.",
                              systemImage: "lock.shield")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        Label("If the model has no verified download receipt or cached fingerprint, the first run hashes the complete GGUF once using uncached I/O.",
                              systemImage: "externaldrive")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section {
                    if controller.isRunning {
                        runningBadge
                        Button(role: .destructive) { controller.stop() } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                    } else {
                        Button { controller.run() } label: {
                            Label(startButtonTitle,
                                  systemImage: "play.fill")
                        }
                        .disabled(startDisabled)
                    }
                }
            }
            .formStyle(.grouped)
            .frame(maxHeight: formMaximumHeight)

            Divider()

            switch controller.kind {
            case .speed: speedResults
            case .accuracy: accuracyResults
            case .official: officialResults
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

    private var benchmarkExplanation: String {
        switch controller.kind {
        case .speed:
            return controller.mode == .local
                ? "Native benchmark: measures prefill and generation throughput (tokens/s) on the shared in-process engine loaded in Settings."
                : "Distributed benchmark: measures prefill and generation throughput (tokens/s) on the cluster, reusing the coordinator already connected in Chat -> Distributed."
        case .accuracy:
            return "Teacher-forced benchmark over reproducible random text pieces: each piece uses an independently sampled preceding context, then checks whether the source token appears among the model's first 1, 2 or 3 candidate tokens. These are token candidates, not MoE experts."
        case .official:
            return "Checkpoint-bound replay of official DeepSeek API smoke vectors. The model is identified by SHA-256, then generated token bytes are compared exactly with the fixture for that revision."
        }
    }

    private var benchmarkExplanationIcon: String {
        switch controller.kind {
        case .speed: return "gauge.with.dots.needle.67percent"
        case .accuracy: return "checkmark.circle"
        case .official: return "checkmark.shield"
        }
    }

    private var startButtonTitle: String {
        switch controller.kind {
        case .speed: return "Start Speed Benchmark"
        case .accuracy: return "Start Correctness Benchmark"
        case .official: return "Verify Official Checkpoint"
        }
    }

    private var formMaximumHeight: CGFloat {
        switch controller.kind {
        case .speed: return 320
        case .accuracy: return 650
        case .official: return 440
        }
    }

    private var startDisabled: Bool {
        switch controller.kind {
        case .speed:
            return controller.mode == .distributed && !controller.distConnected
        case .accuracy:
            return controller.accuracyUnavailableForSelectedMode || controller.accuracyTextIsEmpty
        case .official:
            return controller.accuracyUnavailableForSelectedMode
                || controller.modelPath.isEmpty
        }
    }

    @ViewBuilder
    private var speedResults: some View {
        if controller.rows.isEmpty {
            ContentUnavailableView("No Data", systemImage: "chart.xyaxis.line",
                                   description: Text(controller.isRunning
                                                     ? "Benchmark running - \(controller.runningLabel ?? "")"
                                                     : "Start a benchmark to see throughput."))
                .frame(maxHeight: .infinity)
        } else {
            VStack(spacing: 0) {
                if controller.isRunning { runningBadge.padding(.horizontal).padding(.top, 8) }
                HStack {
                    Spacer()
                    exportButtons(baseName: "dwarfstar-speed",
                                  title: "Prefill and generation throughput",
                                  subtitle: speedExportSubtitle,
                                  csv: speedCSV) { throughputChart }
                }
                .padding(.horizontal).padding(.top, 8)
                throughputChart.padding()
            }
        }
    }

    @ViewBuilder
    private var accuracyResults: some View {
        if controller.accuracyResult == nil && controller.accuracyLiveEvaluatedTokens == 0 {
            ContentUnavailableView("No Correctness Data", systemImage: "checkmark.circle",
                                   description: Text(controller.isRunning
                                                     ? "Sampling reproducible text pieces - \(controller.runningLabel ?? "")"
                                                     : "Paste a text and sample random pieces to measure Top-1, Top-2 and Top-3 candidate-token accuracy."))
                .frame(maxHeight: .infinity)
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if controller.isRunning {
                        runningBadge
                        liveAccuracyProgress
                    }
                    if let result = controller.accuracyResult {
                        accuracySummary(result)
                        samplingSummary(result)
                        if result.truncated {
                            Label("The source contained \(result.originalTokens) tokens. Sampling bounds/count were reduced, or one or more pieces were truncated, to fit the corpus and available context.",
                                  systemImage: "scissors")
                                .font(.callout)
                                .foregroundStyle(.orange)
                        }
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Cumulative candidate-token accuracy")
                                    .font(.headline)
                                Spacer()
                                exportButtons(baseName: "dwarfstar-correctness-cumulative",
                                              title: "Cumulative candidate-token accuracy",
                                              subtitle: accuracyExportSubtitle(result),
                                              csv: { bucketsCSV(result) }) {
                                    bucketTopKAccuracyChart(result)
                                }
                            }
                            Text("Each line shows whether the source token was found within the first 1, 2 or 3 candidates proposed by the model.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            bucketTopKAccuracyChart(result)
                                .frame(minHeight: 280)
                        }
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Accuracy by sampled piece")
                                    .font(.headline)
                                Spacer()
                                exportButtons(baseName: "dwarfstar-correctness-pieces",
                                              title: "Accuracy by sampled piece",
                                              subtitle: accuracyExportSubtitle(result),
                                              csv: { piecesCSV(result) }) {
                                    pieceTopKAccuracyChart(result)
                                }
                            }
                            Text("Each point is one independently sampled context and target segment.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            pieceTopKAccuracyChart(result)
                                .frame(minHeight: 280)
                        }
                    } else if controller.accuracyLiveEvaluatedTokens > 0 {
                        Text("Live cumulative candidate-token accuracy")
                            .font(.headline)
                        Text("The live graph is automatically decimated for large runs; counters and final metrics remain exact.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        liveCumulativeChart
                            .frame(minHeight: 280)
                    }
                }
                .padding()
            }
        }
    }

    @ViewBuilder
    private var officialResults: some View {
        if let result = controller.officialResult {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if controller.isRunning { runningBadge }
                    HStack(spacing: 12) {
                        metricCard(
                            title: "Checkpoint",
                            value: result.checkpoint,
                            systemImage: "shippingbox.fill",
                            tint: .blue)
                        metricCard(
                            title: "Exact token bytes",
                            value: "\(result.passedSteps)/\(result.totalSteps)",
                            systemImage: result.passed
                                ? "checkmark.seal.fill"
                                : "xmark.seal.fill",
                            tint: result.passed ? .green : .red)
                        metricCard(
                            title: "Duration",
                            value: result.duration.formatted(
                                .number.precision(.fractionLength(2))) + " s",
                            systemImage: "clock.fill",
                            tint: .secondary)
                    }
                    GroupBox("Verified model identity") {
                        VStack(alignment: .leading, spacing: 6) {
                            LabeledContent("SHA-256") {
                                Text(result.modelSHA256)
                                    .font(.caption.monospaced())
                                    .textSelection(.enabled)
                            }
                            LabeledContent("Evidence",
                                           value: result.digestSource.rawValue)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    GroupBox("Official smoke cases") {
                        VStack(alignment: .leading, spacing: 12) {
                            ForEach(result.cases) { fixture in
                                VStack(alignment: .leading, spacing: 5) {
                                    Label(
                                        "\(fixture.id) · \(fixture.passedSteps)/\(fixture.steps.count)",
                                        systemImage: fixture.passed
                                            ? "checkmark.circle.fill"
                                            : "xmark.circle.fill")
                                        .foregroundStyle(fixture.passed
                                                         ? Color.green
                                                         : Color.red)
                                    Text("Prompt: \(fixture.promptTokens) tokens")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    ForEach(fixture.steps.filter { !$0.matches }) { step in
                                        Text("Step \(step.index): expected \(step.expectedHex), got \(step.actualHex) (token \(step.actualTokenID))")
                                            .font(.caption.monospaced())
                                            .foregroundStyle(.red)
                                            .textSelection(.enabled)
                                    }
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding()
            }
        } else {
            ContentUnavailableView(
                "No Checkpoint Result",
                systemImage: "checkmark.shield",
                description: Text(controller.isRunning
                    ? "Identifying the GGUF and replaying official vectors - \(controller.runningLabel ?? "")"
                    : "Verify the loaded DeepSeek GGUF against the official fixture for its exact checkpoint."))
                .frame(maxHeight: .infinity)
        }
    }

    private var liveAccuracyProgress: some View {
        let evaluated = controller.accuracyLiveEvaluatedTokens
        let totalMaximum = max(
            1,
            controller.accuracyPieceCount * controller.accuracyMaxTokensPerPiece
        )
        let currentPiece = min(
            controller.accuracyPieceCount,
            max(1, controller.accuracyLivePieceIndex + 1)
        )
        let top1Correct = controller.accuracyLiveTop1CorrectTokens
        let top2Correct = controller.accuracyLiveTop2CorrectTokens
        let top3Correct = controller.accuracyLiveTop3CorrectTokens
        return VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: Double(evaluated),
                         total: Double(totalMaximum))
            Text("Piece \(currentPiece)/\(controller.accuracyPieceCount) · evaluated \(evaluated)/\(totalMaximum) source tokens (maximum)")
                .font(.callout)
                .foregroundStyle(.secondary)
            HStack(spacing: 18) {
                liveAccuracyMetric("Top-1", correct: top1Correct,
                                   evaluated: evaluated, color: .blue)
                liveAccuracyMetric("Top-2", correct: top2Correct,
                                   evaluated: evaluated, color: .orange)
                liveAccuracyMetric("Top-3", correct: top3Correct,
                                   evaluated: evaluated, color: .green)
            }
        }
    }

    private func liveAccuracyMetric(_ label: String, correct: Int,
                                    evaluated: Int, color: Color) -> some View {
        let accuracy = evaluated > 0 ? Double(correct) / Double(evaluated) : 0
        return HStack(spacing: 5) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text("\(label): \(correct)/\(evaluated) (\(accuracy.formatted(.percent.precision(.fractionLength(2)))))")
                .font(.callout.monospacedDigit())
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(label) candidate-token accuracy")
        .accessibilityValue("\(correct) of \(evaluated), \(accuracy.formatted(.percent))")
    }

    private func accuracySummary(_ result: InferenceService.AccuracyResult) -> some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                metricCard(title: "Top-1 token candidates",
                           value: topKMetricValue(correct: result.top1CorrectTokens,
                                                  evaluated: result.evaluatedTokens,
                                                  accuracy: result.top1Accuracy),
                           systemImage: "1.circle.fill", tint: .blue)
                metricCard(title: "Top-2 token candidates",
                           value: topKMetricValue(correct: result.top2CorrectTokens,
                                                  evaluated: result.evaluatedTokens,
                                                  accuracy: result.top2Accuracy),
                           systemImage: "2.circle.fill", tint: .orange)
                metricCard(title: "Top-3 token candidates",
                           value: topKMetricValue(correct: result.top3CorrectTokens,
                                                  evaluated: result.evaluatedTokens,
                                                  accuracy: result.top3Accuracy),
                           systemImage: "3.circle.fill", tint: .green)
            }
            HStack(spacing: 12) {
                metricCard(title: "Duration",
                           value: result.duration.formatted(.number.precision(.fractionLength(2))) + " s",
                           systemImage: "clock.fill", tint: .secondary)
                metricCard(title: "Evaluation speed",
                           value: result.evaluatedTps.formatted(.number.precision(.fractionLength(2))) + " t/s",
                           systemImage: "speedometer", tint: .purple)
            }
        }
    }

    private func samplingSummary(_ result: InferenceService.AccuracyResult) -> some View {
        GroupBox("Random sampling") {
            HStack(alignment: .top, spacing: 28) {
                samplingDetail("Pieces",
                               "\(result.pieces.count) / \(result.requestedPieceCount)")
                samplingDetail("Effective context",
                               "\(result.effectiveMinContextTokens)...\(result.effectiveMaxContextTokens) tokens")
                samplingDetail("Seed", seedLabel(result.seed))
                samplingDetail("Evaluated", "\(result.evaluatedTokens) tokens")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 4)
        }
    }

    private func samplingDetail(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.callout.monospacedDigit().weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func seedLabel(_ seed: UInt64) -> String {
        "\(seed) (0x\(String(seed, radix: 16, uppercase: true)))"
    }

    private func topKMetricValue(correct: Int, evaluated: Int, accuracy: Double) -> String {
        "\(correct)/\(evaluated) · \(accuracy.formatted(.percent.precision(.fractionLength(2))))"
    }

    private func metricCard(title: String, value: String,
                            systemImage: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: systemImage)
                .font(.caption)
                .foregroundStyle(tint)
            Text(value)
                .font(.title3.monospacedDigit().weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    /// A single cumulative chart makes the inclusive Top-K relationship easy to
    /// inspect: Top-2 cannot be below Top-1 and Top-3 cannot be below Top-2.
    private func bucketTopKAccuracyChart(_ result: InferenceService.AccuracyResult) -> some View {
        Chart {
            ForEach(result.buckets, id: \.index) { bucket in
                LineMark(x: .value("Block", bucket.index + 1),
                         y: .value("Accuracy (%)", bucket.cumulativeTop1Accuracy * 100),
                         series: .value("Candidate rank", "Top-1"))
                    .foregroundStyle(by: .value("Candidate rank", "Top-1"))
                    .lineStyle(StrokeStyle(lineWidth: 2.5))
                LineMark(x: .value("Block", bucket.index + 1),
                         y: .value("Accuracy (%)", bucket.cumulativeTop2Accuracy * 100),
                         series: .value("Candidate rank", "Top-2"))
                    .foregroundStyle(by: .value("Candidate rank", "Top-2"))
                    .lineStyle(StrokeStyle(lineWidth: 2.5, dash: [7, 3]))
                LineMark(x: .value("Block", bucket.index + 1),
                         y: .value("Accuracy (%)", bucket.cumulativeTop3Accuracy * 100),
                         series: .value("Candidate rank", "Top-3"))
                    .foregroundStyle(by: .value("Candidate rank", "Top-3"))
                    .lineStyle(StrokeStyle(lineWidth: 2.5, dash: [2, 3]))
            }
        }
        .chartYScale(domain: 0...100)
        .chartXAxisLabel("Block of \(result.bucketSize) evaluated tokens")
        .chartYAxisLabel("Cumulative candidate-token accuracy")
        .chartYAxis {
            AxisMarks(values: [0, 25, 50, 75, 100]) { value in
                AxisGridLine()
                AxisTick()
                AxisValueLabel {
                    if let percent = value.as(Int.self) { Text("\(percent)%") }
                }
            }
        }
        .chartForegroundStyleScale([
            "Top-1": Color.blue,
            "Top-2": Color.orange,
            "Top-3": Color.green
        ])
        .chartLegend(position: .top)
    }

    /// Per-piece lines expose variance between independently sampled regions.
    /// Point marks are always present so a one-piece run still produces a
    /// visible and useful chart.
    private func pieceTopKAccuracyChart(_ result: InferenceService.AccuracyResult) -> some View {
        Chart {
            ForEach(result.pieces, id: \.index) { piece in
                LineMark(x: .value("Piece", piece.index + 1),
                         y: .value("Accuracy (%)", piece.top1Accuracy * 100),
                         series: .value("Candidate rank", "Top-1"))
                    .foregroundStyle(by: .value("Candidate rank", "Top-1"))
                    .lineStyle(StrokeStyle(lineWidth: 2.5))
                PointMark(x: .value("Piece", piece.index + 1),
                          y: .value("Accuracy (%)", piece.top1Accuracy * 100))
                    .foregroundStyle(by: .value("Candidate rank", "Top-1"))
                    .symbolSize(42)

                LineMark(x: .value("Piece", piece.index + 1),
                         y: .value("Accuracy (%)", piece.top2Accuracy * 100),
                         series: .value("Candidate rank", "Top-2"))
                    .foregroundStyle(by: .value("Candidate rank", "Top-2"))
                    .lineStyle(StrokeStyle(lineWidth: 2.5, dash: [7, 3]))
                PointMark(x: .value("Piece", piece.index + 1),
                          y: .value("Accuracy (%)", piece.top2Accuracy * 100))
                    .foregroundStyle(by: .value("Candidate rank", "Top-2"))
                    .symbolSize(42)

                LineMark(x: .value("Piece", piece.index + 1),
                         y: .value("Accuracy (%)", piece.top3Accuracy * 100),
                         series: .value("Candidate rank", "Top-3"))
                    .foregroundStyle(by: .value("Candidate rank", "Top-3"))
                    .lineStyle(StrokeStyle(lineWidth: 2.5, dash: [2, 3]))
                PointMark(x: .value("Piece", piece.index + 1),
                          y: .value("Accuracy (%)", piece.top3Accuracy * 100))
                    .foregroundStyle(by: .value("Candidate rank", "Top-3"))
                    .symbolSize(42)
            }
        }
        .chartYScale(domain: 0...100)
        .chartXAxisLabel("Randomly sampled piece")
        .chartYAxisLabel("Per-piece candidate-token accuracy")
        .chartYAxis {
            AxisMarks(values: [0, 25, 50, 75, 100]) { value in
                AxisGridLine()
                AxisTick()
                AxisValueLabel {
                    if let percent = value.as(Int.self) { Text("\(percent)%") }
                }
            }
        }
        .chartForegroundStyleScale([
            "Top-1": Color.blue,
            "Top-2": Color.orange,
            "Top-3": Color.green
        ])
        .chartLegend(position: .top)
    }

    private var liveCumulativeChart: some View {
        Chart {
            ForEach(controller.accuracyObservations, id: \.index) { observation in
                LineMark(x: .value("Evaluated token", observation.index),
                         y: .value("Accuracy (%)", observation.cumulativeTop1Accuracy * 100),
                         series: .value("Candidate rank", "Top-1"))
                    .foregroundStyle(by: .value("Candidate rank", "Top-1"))
                    .lineStyle(StrokeStyle(lineWidth: 2.5))
                LineMark(x: .value("Evaluated token", observation.index),
                         y: .value("Accuracy (%)", observation.cumulativeTop2Accuracy * 100),
                         series: .value("Candidate rank", "Top-2"))
                    .foregroundStyle(by: .value("Candidate rank", "Top-2"))
                    .lineStyle(StrokeStyle(lineWidth: 2.5, dash: [7, 3]))
                LineMark(x: .value("Evaluated token", observation.index),
                         y: .value("Accuracy (%)", observation.cumulativeTop3Accuracy * 100),
                         series: .value("Candidate rank", "Top-3"))
                    .foregroundStyle(by: .value("Candidate rank", "Top-3"))
                    .lineStyle(StrokeStyle(lineWidth: 2.5, dash: [2, 3]))
            }
        }
        .chartYScale(domain: 0...100)
        .chartXAxisLabel("Evaluated token")
        .chartYAxisLabel("Cumulative candidate-token accuracy")
        .chartForegroundStyleScale([
            "Top-1": Color.blue,
            "Top-2": Color.orange,
            "Top-3": Color.green
        ])
        .chartLegend(position: .top)
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

    /// PNG + CSV export controls for one chart. Failures are appended to the
    /// benchmark log; a cancelled save panel is not an error. The chart closure
    /// re-renders the chart for export at a fixed size, independent of the
    /// on-screen layout.
    private func exportButtons<C: View>(baseName: String,
                                        title: String,
                                        subtitle: String,
                                        csv: @escaping () -> String,
                                        @ViewBuilder chart: @escaping () -> C) -> some View {
        HStack(spacing: 2) {
            Button {
                if let error = ChartExport.savePNG(title: title, subtitle: subtitle,
                                                   suggestedFileName: ChartExport.stampedName(baseName),
                                                   chart: chart()) {
                    controller.log += error
                }
            } label: {
                Label("PNG", systemImage: "photo")
            }
            .help("Export this chart as a PNG image")
            Button {
                if let error = ChartExport.saveCSV(csv(),
                                                   suggestedFileName: ChartExport.stampedName(baseName)) {
                    controller.log += error
                }
            } label: {
                Label("CSV", systemImage: "tablecells")
            }
            .help("Export this chart's data as CSV")
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
    }

    private var speedExportSubtitle: String {
        let model = (controller.modelPath as NSString).lastPathComponent
        return "\(model) · \(controller.mode.rawValue) engine" +
               " · \(controller.genTokens) generated tokens per point · generation is 1/p99 latency" +
               " · \(Date.now.formatted(date: .abbreviated, time: .shortened))"
    }

    private func accuracyExportSubtitle(_ result: InferenceService.AccuracyResult) -> String {
        let model = (controller.modelPath as NSString).lastPathComponent
        return "\(model) · seed \(result.seed) · \(result.pieces.count) pieces" +
               " · context \(result.effectiveMinContextTokens)...\(result.effectiveMaxContextTokens) tokens" +
               " · \(result.evaluatedTokens) evaluated tokens" +
               " · \(Date.now.formatted(date: .abbreviated, time: .shortened))"
    }

    private func speedCSV() -> String {
        var out = "context_tokens,prefill_tps,gen_tps_at_p99_latency,kvcache_bytes\n"
        for row in controller.rows {
            out += "\(row.ctxTokens),\(row.prefillTps),\(row.genTps),\(row.kvcacheBytes)\n"
        }
        return out
    }

    private func bucketsCSV(_ result: InferenceService.AccuracyResult) -> String {
        var out = "block,block_size,cumulative_top1_accuracy,cumulative_top2_accuracy,cumulative_top3_accuracy\n"
        for bucket in result.buckets {
            out += "\(bucket.index + 1),\(result.bucketSize)," +
                   "\(bucket.cumulativeTop1Accuracy),\(bucket.cumulativeTop2Accuracy)," +
                   "\(bucket.cumulativeTop3Accuracy)\n"
        }
        return out
    }

    private func piecesCSV(_ result: InferenceService.AccuracyResult) -> String {
        var out = "piece,source_start_token,target_start_token,context_tokens," +
                  "evaluated_tokens,top1_accuracy,top2_accuracy,top3_accuracy,truncated\n"
        for piece in result.pieces {
            out += "\(piece.index + 1),\(piece.sourceStartTokenIndex)," +
                   "\(piece.targetStartTokenIndex),\(piece.contextTokens)," +
                   "\(piece.evaluatedTokens),\(piece.top1Accuracy)," +
                   "\(piece.top2Accuracy),\(piece.top3Accuracy),\(piece.truncated)\n"
        }
        return out
    }

    // Colors shared between each series and its own axis, so the reader maps the
    // LEFT axis to prefill and the RIGHT axis to generation by color alone.
    private static let prefillColor = Color.blue
    private static let generationColor = Color.green

    /// Dual-scale throughput chart: prefill on the LEFT axis, generation
    /// (throughput equivalent to p99 latency) on
    /// the RIGHT axis. Prefill (tens–thousands t/s) and generation (~0.05–0.3 t/s)
    /// differ by orders of magnitude, so a single shared axis squashes generation
    /// flat. Swift Charts has no native second Y scale, so generation is plotted
    /// rescaled onto the prefill axis and the trailing axis relabels those
    /// positions back to real generation t/s.
    private var throughputChart: some View {
        let rows = controller.rows
        let maxPrefill = max(rows.map(\.prefillTps).max() ?? 1, 0.0001)
        let maxGen = max(rows.map(\.genTps).max() ?? 1, 0.0001)
        // Factor that maps a generation value onto the prefill axis (and its
        // inverse relabels the right axis). Same relative headroom on both axes.
        let genToPrefill = maxPrefill / maxGen
        let top = max(maxPrefill * 1.08, 0.0001)
        let prefillColor = Self.prefillColor
        let generationColor = Self.generationColor

        return Chart {
            ForEach(rows) { row in
                LineMark(x: .value("Context", row.ctxTokens),
                         y: .value("t/s", row.prefillTps),
                         series: .value("Series", "Prefill"))
                    .foregroundStyle(by: .value("Series", "Prefill"))
                PointMark(x: .value("Context", row.ctxTokens),
                          y: .value("t/s", row.prefillTps))
                    .foregroundStyle(by: .value("Series", "Prefill"))
            }
            ForEach(rows) { row in
                // Rescaled onto the prefill axis; the right axis shows real t/s.
                LineMark(x: .value("Context", row.ctxTokens),
                         y: .value("t/s", row.genTps * genToPrefill),
                         series: .value("Series", "Generation (1/p99 lat)"))
                    .foregroundStyle(by: .value("Series", "Generation (1/p99 lat)"))
                PointMark(x: .value("Context", row.ctxTokens),
                          y: .value("t/s", row.genTps * genToPrefill))
                    .foregroundStyle(by: .value("Series", "Generation (1/p99 lat)"))
            }
        }
        .chartForegroundStyleScale([
            "Prefill": prefillColor,
            "Generation (1/p99 lat)": generationColor,
        ])
        .chartXAxisLabel("Context Tokens")
        .chartYScale(domain: 0...top)
        .chartYAxisLabel("Prefill t/s (left) — Generation t/s (right)")
        .chartYAxis {
            // LEFT: prefill, real values.
            AxisMarks(position: .leading) { value in
                AxisGridLine().foregroundStyle(.quaternary)
                AxisTick()
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text(v, format: .number.precision(.fractionLength(2)))
                            .foregroundStyle(prefillColor)
                    }
                }
            }
            // RIGHT: same tick positions, relabeled to real generation t/s.
            AxisMarks(position: .trailing) { value in
                AxisTick()
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text(v / genToPrefill, format: .number.precision(.fractionLength(2)))
                            .foregroundStyle(generationColor)
                    }
                }
            }
        }
        .chartLegend(position: .top)
    }
}
