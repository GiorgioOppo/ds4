import SwiftUI
import DS4Engine

struct MiniToolBenchView: View {
    @Bindable var controller: MiniToolBenchController
    let openServer: () -> Void
    @State private var tab = Panel.preparation
    @State private var showAdvanced = false
    @State private var showSetup = false
    @State private var showManual = false

    private enum Panel: String, CaseIterable, Identifiable {
        case preparation = "Esegui test"
        case results = "Risultati"
        var id: String { rawValue }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack {
                Picker("Pannello", selection: $tab) {
                    ForEach(Panel.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 360)
                Spacer()
                if controller.isImporting {
                    ProgressView().controlSize(.small)
                    Text("Lettura dei risultati…").font(.caption).foregroundStyle(.secondary)
                }
            }.padding(.horizontal, 24).padding(.vertical, 12)
            switch tab {
            case .preparation: preparation
            case .results: results
            }
            if let status = controller.statusMessage {
                Divider()
                Label(status, systemImage: "checkmark.circle")
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 24).padding(.vertical, 10)
                    .accessibilityAddTraits(.updatesFrequently)
            }
        }
        .onChange(of: controller.reportRevision) { _, revision in
            if revision > 0 { tab = .results }
        }
        .onChange(of: controller.isManagedRunning) { _, running in
            if running { tab = .preparation }
        }
        .alert("Mini Tool Bench", isPresented: Binding(
            get: { controller.errorMessage != nil },
            set: { if !$0 { controller.errorMessage = nil } }
        )) {
            Button("OK") { controller.errorMessage = nil }
        } message: {
            Text(controller.errorMessage ?? "")
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: "terminal")
                .font(.system(size: 25, weight: .medium))
                .foregroundStyle(.tint)
                .frame(width: 44, height: 44)
                .background(.tint.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                Text("Mini Tool Bench").font(.title2.bold())
                Text("Terminal-Bench-Local · attività reali, verificate nei container")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
            Button { controller.chooseReportDirectory() } label: {
                Label("Importa risultati…", systemImage: "square.and.arrow.down")
            }
            .disabled(controller.isImporting || controller.isManagedRunning)
            .help("Importa la cartella con summary.json e i risultati dei task")
        }.padding(24)
    }

    private var preparation: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                managedRunPanel

                GroupBox("Configura la prova") {
                    VStack(alignment: .leading, spacing: 16) {
                        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 14) {
                            GridRow {
                                Text("Suite")
                                Picker("Suite", selection: $controller.configuration.suite) {
                                    Text("Core-19 · 19 task").tag(MiniToolBenchSuite.core19)
                                    Text("Legacy Mini-20 · storico").tag(MiniToolBenchSuite.legacyMini20)
                                }.labelsHidden().frame(maxWidth: 350, alignment: .leading)
                            }
                            GridRow {
                                Text("Estensione")
                                Picker("Estensione della prova", selection: $controller.configuration.tier) {
                                    Text("Full").tag(MiniToolBenchTier.full)
                                    Text("Smoke").tag(MiniToolBenchTier.smoke)
                                }.pickerStyle(.segmented).frame(maxWidth: 260)
                            }
                            GridRow {
                                Text("Tentativi per task")
                                Picker("Tentativi massimi per task", selection: $controller.configuration.attempts) {
                                    Text("1").tag(1)
                                    Text("Fino a 2").tag(2)
                                }.pickerStyle(.segmented).frame(maxWidth: 260)
                            }
                        }
                        Text(controller.configuration.tier == .full
                             ? "La suite completa può richiedere molte ore. Il secondo tentativo viene eseguito solo quando serve."
                             : "Smoke usa il sottoinsieme rapido della suite per verificare la configurazione prima di una prova completa.")
                            .font(.caption).foregroundStyle(.secondary)
                        if controller.configuration.suite == .legacyMini20 {
                            Label("La suite storica serve a riprodurre risultati precedenti; usa Core-19 per nuovi confronti.",
                                  systemImage: "clock.arrow.circlepath")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Divider()
                        labeledField("Piattaforma del modello", text: $controller.configuration.platform,
                            prompt: "es. apple-m4-max")
                        labeledField("Nome canonico del modello", text: $controller.configuration.modelName,
                            prompt: "es. DeepSeek-V4-Flash-Vision-Exp")
                        Text("Piattaforma e nome modello sono obbligatori e identificano il confronto nei risultati.")
                            .font(.caption).foregroundStyle(.secondary)
                        DisclosureGroup("Parametri avanzati del modello", isExpanded: $showAdvanced) {
                            VStack(alignment: .leading, spacing: 14) {
                                labeledField("Motore", text: $controller.configuration.engine, prompt: "DwarfStar")
                                labeledField("Backend", text: $controller.configuration.backend, prompt: "metal")
                                labeledField("Quantizzazione", text: $controller.configuration.quant, prompt: "facoltativa")
                                labeledField("Profilo di inferenza", text: $controller.configuration.inferenceProfile, prompt: "facoltativo")
                                labeledField("ID modello API", text: $controller.configuration.modelID, prompt: "rilevato dal runner")
                                labeledField("Limite di contesto", text: $controller.configuration.contextLength, prompt: "rilevato dal runner")
                            }.padding(.top, 10)
                        }
                    }.padding(8)
                }
                .disabled(controller.isManagedRunning)

                DisclosureGroup("Avanzate: comandi per un runner Linux esterno", isExpanded: $showManual) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Questa modalità è per un host Linux configurato manualmente: esegui i comandi in una shell Linux con Python 3.11+, Docker o Podman e uv oppure Harbor 0.20.0.")
                            .font(.callout).foregroundStyle(.secondary)
                        labeledField("Endpoint API del runner esterno", text: $controller.configuration.endpoint,
                            prompt: "http://indirizzo-del-mac:8080/v1")
                        Text("Usa l’indirizzo raggiungibile dal runner Linux. localhost indica il runner stesso. L’avvio con Podman configura automaticamente il proprio endpoint.")
                            .font(.caption).foregroundStyle(.secondary)
                        labeledField("Cartella sul runner Linux", text: $controller.configuration.runnerDirectory,
                            prompt: "~/terminal-bench-mini")
                        labeledField("Interprete Python sul runner", text: $controller.configuration.pythonExecutable,
                            prompt: "python3.12 oppure /percorso/python3")
                        DisclosureGroup("Preparazione del runner esterno", isExpanded: $showSetup) {
                            VStack(alignment: .leading, spacing: 10) {
                                commandText(MiniToolBenchCommandBuilder.setupCommand, maxHeight: 180)
                                Button { controller.copySetup() } label: {
                                    Label("Copia preparazione", systemImage: "doc.on.doc")
                                }
                                Text("Revisione: \(MiniToolBenchCommandBuilder.pinnedRevision)")
                                    .font(.caption2.monospaced()).foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }.padding(.top, 8)
                        }
                        Divider()
                        Text("doctor verifica endpoint e prerequisiti. Se i controlli passano, run avvia i task e i verificatori nei container. Per le prove lunghe usa una sessione tmux persistente.")
                            .font(.callout).foregroundStyle(.secondary)
                        if controller.validationErrors.isEmpty {
                            commandText(controller.command, maxHeight: 260)
                        } else {
                            VStack(alignment: .leading, spacing: 7) {
                                ForEach(controller.validationErrors, id: \.self) { error in
                                    Label(error, systemImage: "info.circle").font(.callout)
                                }
                            }
                            .foregroundStyle(.secondary)
                            .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                        }
                        HStack {
                            Button { controller.copyCommand() } label: {
                                Label("Copia doctor + run", systemImage: "doc.on.doc")
                            }
                            Button { controller.exportScript() } label: {
                                Label("Esporta script…", systemImage: "square.and.arrow.up")
                            }
                        }.disabled(!controller.canExportCommand)
                        Text("Al termine, copia dal runner la cartella dei risultati e importala qui: summary.json, results-*.json e transcript devono restare insieme.")
                            .font(.caption).foregroundStyle(.secondary)
                    }.padding(8)
                }
                .disabled(controller.isManagedRunning)
            }
            .frame(maxWidth: 940, alignment: .leading)
            .padding(.horizontal, 24).padding(.bottom, 24)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private var managedRunPanel: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                Label("Avvia il benchmark da DwarfStar", systemImage: "play.circle")
                    .font(.headline)
                Text("DwarfStar avvia il Server API sul modello già caricato, prepara l’ambiente Linux di Podman e lancia la suite. Al termine importa automaticamente i risultati dei verificatori.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Text("Il primo avvio può scaricare l’ambiente e le immagini dei task. Le prove complete possono richiedere molte ore; il Server API resta attivo al termine.")
                    .font(.caption).foregroundStyle(.secondary)
                if controller.modelIsReady {
                    Label(controller.servedModelName, systemImage: "checkmark.circle")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(2)
                } else {
                    Label("Carica un modello nelle impostazioni per avviare il test.", systemImage: "shippingbox")
                        .font(.callout).foregroundStyle(.secondary)
                }
                if !controller.isManagedRunning && !controller.managedValidationErrors.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(controller.managedValidationErrors, id: \.self) { message in
                            Label(message, systemImage: "info.circle").font(.caption)
                        }
                    }.foregroundStyle(.secondary)
                }
                HStack(spacing: 12) {
                    if controller.isManagedRunning || controller.isManagedStopping {
                        ProgressView().controlSize(.small)
                        Button(role: .destructive) { controller.stopManagedRun() } label: {
                            Label(controller.isManagedStopping ? "Interruzione…" : "Interrompi test",
                                  systemImage: "stop.fill")
                        }.disabled(controller.isManagedStopping)
                    } else {
                        Button { controller.startManagedRun() } label: {
                            Label("Avvia test con Podman", systemImage: "play.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!controller.canStartManagedRun)
                    }
                    Button(action: openServer) {
                        Label("Server API", systemImage: "server.rack")
                    }
                    Spacer()
                    Link("Istruzioni", destination: MiniToolBenchCommandBuilder.repositoryURL)
                }
                if controller.isManagedRunning || !controller.managedLog.isEmpty {
                    Text(controller.managedStatus)
                        .font(.callout.weight(.medium))
                        .accessibilityAddTraits(.updatesFrequently)
                }
                if let error = controller.managedError {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .font(.callout).foregroundStyle(.red)
                        .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                }
                if !controller.managedLog.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Attività del test").font(.caption.weight(.semibold))
                            Spacer()
                            Button { controller.copyManagedLog() } label: {
                                Label("Copia log", systemImage: "doc.on.doc")
                            }.font(.caption)
                        }
                        ScrollViewReader { proxy in
                            ScrollView([.horizontal, .vertical]) {
                                VStack(alignment: .leading, spacing: 0) {
                                    Text(controller.managedLog)
                                        .font(.system(.caption, design: .monospaced))
                                        .textSelection(.enabled)
                                        .fixedSize(horizontal: true, vertical: false)
                                    Color.clear.frame(height: 1).id("log-end")
                                }.padding(10)
                            }
                            .frame(height: 220)
                            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
                            .onChange(of: controller.managedLog) { _, _ in
                                proxy.scrollTo("log-end", anchor: .bottom)
                            }
                        }
                    }
                }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
        }
    }

    @ViewBuilder private var results: some View {
        if let report = controller.report {
            VStack(alignment: .leading, spacing: 16) {
                reportSummary(report)
                if !report.warnings.isEmpty {
                    DisclosureGroup("Note sul report (\(report.warnings.count))") {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(Array(report.warnings.enumerated()), id: \.offset) { _, warning in
                                Label(warning, systemImage: "info.circle").font(.caption)
                            }
                        }.padding(.top, 6)
                    }.foregroundStyle(.secondary)
                }
                HStack {
                    TextField("Cerca task", text: $controller.resultSearch)
                        .textFieldStyle(.roundedBorder).frame(maxWidth: 280)
                    Picker("Filtra esiti", selection: $controller.resultFilter) {
                        ForEach(MiniToolBenchResultFilter.allCases) { Text($0.rawValue).tag($0) }
                    }.frame(maxWidth: 210)
                    Spacer()
                    Text("\(controller.filteredTasks.count) di \(report.tasks.count) task")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Table(controller.filteredTasks) {
                    TableColumn("Task") { task in
                        Text(task.task).font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                    }.width(min: 210, ideal: 290)
                    TableColumn("Esito") { task in resultLabel(task) }
                        .width(min: 115, ideal: 140, max: 165)
                    TableColumn("Tentativi") { task in
                        Text(task.attempts.map { String($0.count) } ?? "—").monospacedDigit()
                    }.width(75)
                    TableColumn("Durata") { task in
                        Text(duration(task.durationMS)).monospacedDigit()
                    }.width(85)
                    TableColumn("Transcript") { task in
                        if report.transcriptURL(for: task) != nil {
                            Button { controller.openTranscript(for: task) } label: {
                                Label("Apri", systemImage: "doc.text")
                            }
                            .buttonStyle(.link)
                            .accessibilityLabel("Apri transcript di \(task.task)")
                        } else {
                            Text("Non presente").font(.caption).foregroundStyle(.secondary)
                        }
                    }.width(110)
                }
                .overlay {
                    if controller.filteredTasks.isEmpty {
                        ContentUnavailableView.search(text: controller.resultSearch)
                    }
                }
                .frame(minHeight: 240)
            }.padding(.horizontal, 24).padding(.bottom, 20)
        } else {
            ContentUnavailableView {
                Label("Importa una prova verificata", systemImage: "chart.bar.doc.horizontal")
            } description: {
                Text("Scegli la cartella dei risultati prodotta da Terminal-Bench-Local. Qui vedrai gli esiti reali dei verificatori, i tentativi e i transcript disponibili.")
            } actions: {
                Button("Importa cartella risultati…") { controller.chooseReportDirectory() }
                    .buttonStyle(.borderedProminent).disabled(controller.isImporting || controller.isManagedRunning)
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func reportSummary(_ report: MiniToolBenchReport) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .firstTextBaseline) {
                    Text("\(report.summary.passedTasks) / \(report.summary.totalTasks)")
                        .font(.system(size: 30, weight: .semibold, design: .rounded)).monospacedDigit()
                    Text("task superati").font(.headline).foregroundStyle(.secondary)
                    Spacer()
                    Text(report.summary.passRate, format: .percent.precision(.fractionLength(1)))
                        .font(.title2.weight(.semibold)).monospacedDigit()
                }
                Text("Esiti verificati sul totale esportato. Il report può contenere una prova parziale o tentativi di esecuzioni precedenti.")
                    .font(.caption).foregroundStyle(.secondary)
                Divider()
                Grid(alignment: .leading, horizontalSpacing: 28, verticalSpacing: 8) {
                    GridRow {
                        summaryField("Modello", value: report.summary.model.name ?? report.summary.model.id ?? "Non dichiarato")
                        summaryField("Piattaforma", value: report.summary.platform.name ?? report.summary.platform.id ?? "Non dichiarata")
                    }
                    GridRow {
                        summaryField("Suite", value: report.summary.suite?.name ?? report.summary.suite?.id ?? "Non dichiarata")
                        summaryField("Motore / backend", value: [report.summary.engine, report.summary.backend].compactMap { $0 }.joined(separator: " / "))
                    }
                    GridRow {
                        summaryField("Quantizzazione", value: report.summary.quant ?? "Non dichiarata")
                        summaryField("Durata totale", value: duration(report.summary.totalDurationMS))
                    }
                    if let budget = report.attemptBudget {
                        GridRow { summaryField("Budget tentativi dichiarato", value: String(budget)) }
                    }
                }
                HStack {
                    Text(report.directory.lastPathComponent).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    Spacer()
                    Button { controller.revealReport() } label: {
                        Label("Mostra nel Finder", systemImage: "folder")
                    }.font(.caption)
                }
            }.padding(8)
        }
    }

    private func summaryField(_ title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value.isEmpty ? "Non dichiarato" : value).font(.callout)
                .textSelection(.enabled).lineLimit(2)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private func resultLabel(_ task: MiniToolBenchTaskResult) -> some View {
        let title = task.passed ? "Superato" : (task.completed ? "Non superato" : "Incompleto")
        let icon = task.passed ? "checkmark.circle.fill" : (task.completed ? "xmark.circle" : "clock")
        return Label(title, systemImage: icon)
            .font(.callout).foregroundStyle(task.passed ? Color.green : Color.secondary)
    }

    private func labeledField(_ title: String, text: Binding<String>, prompt: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.callout.weight(.medium))
            TextField(title, text: text, prompt: Text(prompt))
                .labelsHidden().textFieldStyle(.roundedBorder)
        }
    }

    private func commandText(_ text: String, maxHeight: CGFloat) -> some View {
        ScrollView([.horizontal, .vertical]) {
            Text(text).font(.system(.caption, design: .monospaced))
                .textSelection(.enabled).fixedSize(horizontal: true, vertical: false)
                .frame(maxWidth: .infinity, alignment: .leading).padding(12)
        }
        .frame(maxHeight: maxHeight)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary))
        .accessibilityLabel("Comandi da eseguire sul runner Linux")
    }

    private func duration(_ milliseconds: Double?) -> String {
        guard let milliseconds, milliseconds.isFinite, milliseconds >= 0,
              milliseconds < Double(Int.max) else { return "—" }
        let seconds = Int(milliseconds / 1_000)
        if seconds >= 3_600 { return "\(seconds / 3_600) h \(seconds % 3_600 / 60) min" }
        if seconds >= 60 { return "\(seconds / 60) min \(seconds % 60) s" }
        return "\(seconds) s"
    }
}
