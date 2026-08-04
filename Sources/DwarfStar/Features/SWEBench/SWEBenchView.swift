import SwiftUI
import AppKit

struct SWEBenchView: View {
    @Bindable var controller: SWEBenchController
    @State private var confirmRunAll = false

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("SWE-bench Swift locale") {
                    Label("Download del task, checkout, applicazione della patch e report sono gestiti interamente in Swift. Non vengono usati Python, Docker, cloud, pip o pytest.",
                          systemImage: "swift")
                        .font(.callout).foregroundStyle(.orange)
                }
                Section("Esecuzione locale") {
                    Toggle("Solo verifica applicazione patch", isOn: $controller.patchOnlyMode)
                    if controller.patchOnlyMode {
                        Label("Clona il repository e applica la patch selezionata al commit di base. Non compila, non esegue test e non dichiara il task risolto.",
                              systemImage: "checkmark.seal")
                            .font(.caption).foregroundStyle(.secondary)
                        HStack {
                            Button(controller.goldCatalogLoading ? "Scaricamento…" : "Scarica Gold SWE-bench Lite") {
                                controller.downloadSWEGoldCatalog()
                            }.disabled(controller.goldCatalogLoading)
                            if controller.goldCatalogLoading { ProgressView().controlSize(.small) }
                            Text(controller.goldCatalogStatus).font(.caption).foregroundStyle(.secondary)
                        }
                        if !controller.goldCatalog.isEmpty {
                            TextField("Filtra Gold per repository o instance ID", text: $controller.goldCatalogFilter)
                            Picker("Gold task", selection: $controller.selectedGoldTaskID) {
                                Text("Seleziona…").tag("")
                                ForEach(controller.filteredGoldCatalog) { task in
                                    Text("\(task.id) · \(task.repo) \(task.version)").tag(task.id)
                                }
                            }.onChange(of: controller.selectedGoldTaskID) { _, id in
                                if !id.isEmpty { controller.selectGoldTask(id) }
                            }
                        }
                    }
                    HStack {
                        TextField("Task Swift-bench JSON", text: $controller.localTaskPath)
                        Button("Scegli…") { chooseLocalTask() }
                    }
                    if !controller.selectedTaskInstanceID.isEmpty {
                        LabeledContent("Istanza", value: controller.selectedTaskInstanceID)
                        LabeledContent("Patch Gold", value: controller.selectedTaskHasGoldPatch ? "Disponibile" : "Assente")
                    }
                    Picker("Soluzione", selection: $controller.predictionMode) {
                        ForEach(SWELocalPredictionMode.allCases) { Text($0.rawValue).tag($0) }
                    }.pickerStyle(.segmented)
                    if controller.predictionMode == .model {
                        Button {
                            controller.generateModelPrediction()
                        } label: {
                            Label(controller.isGeneratingPatch ? "Generazione in corso…" : "Esegui prompt e genera patch",
                                  systemImage: "brain.head.profile")
                        }
                        .disabled(controller.isGeneratingPatch || controller.localTaskPath.isEmpty)
                        HStack {
                            Button {
                                confirmRunAll = true
                            } label: {
                                Label("Avvia tutti i task", systemImage: "play.rectangle.on.rectangle")
                            }
                            .disabled(controller.goldCatalog.isEmpty || controller.isBatchRunning)
                            if controller.isBatchRunning {
                                Button("Stop batch", role: .destructive) { controller.stopModelBatch() }
                            }
                        }
                        if controller.batchTotal > 0 {
                            ProgressView(value: Double(controller.batchCompleted),
                                         total: Double(controller.batchTotal))
                            Text(controller.batchStatus).font(.caption).foregroundStyle(.secondary)
                        }
                        if controller.isGeneratingPatch { ProgressView() }
                        if !controller.modelGenerationStatus.isEmpty {
                            Text(controller.modelGenerationStatus).font(.caption).foregroundStyle(.secondary)
                        }
                        if !controller.predictionsPath.isEmpty {
                            LabeledContent("Model patch", value: controller.predictionsPath)
                        }
                        if !controller.lastModelTranscriptPath.isEmpty {
                            LabeledContent("Transcript JSON", value: controller.lastModelTranscriptPath)
                        }
                    } else if controller.predictionMode == .gold {
                        Label("La GUI converte automaticamente il campo `patch` del task in una prediction JSONL.",
                              systemImage: "wand.and.stars")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        HStack {
                            TextField("Prediction JSONL (una riga)", text: $controller.predictionsPath)
                            Button("Scegli…") { choosePredictions() }
                            Button("Valida") { controller.validatePredictions() }
                        }
                        if controller.predictionCount > 0 {
                            LabeledContent("Prediction valide", value: "\(controller.predictionCount)")
                            LabeledContent("Patch non vuote", value: "\(controller.nonEmptyPatchCount)")
                        }
                    }
                    LabeledContent("Workspace", value: controller.localWorkspace)
                    DisclosureGroup("Opzioni avanzate") {
                        HStack {
                            TextField("Eseguibile runner Swift", text: $controller.localRunnerPath)
                            Button("Scegli…") { chooseLocalRunner() }
                        }
                        TextField("Workspace repository", text: $controller.localWorkspace)
                        HStack {
                            TextField("Directory risultati", text: $controller.outputDirectory)
                            Button("Scegli…") { chooseOutputDirectory() }
                        }
                        TextField("Run ID", text: $controller.runID)
                        Stepper("Timeout: \(controller.timeoutSeconds) s",
                                value: $controller.timeoutSeconds, in: 60...14_400, step: 60)
                    }
                    if !controller.patchOnlyMode {
                        Toggle("Comprendo: i test eseguono codice non attendibile sul Mac",
                               isOn: $controller.allowUnsafeHostExecution)
                            .foregroundStyle(.red)
                    }
                }
                Section("Prompt e risposta del modello") {
                    if controller.currentModelPrompt.isEmpty {
                        Label("Esegui il prompt SWE-bench per visualizzare qui richiesta, thinking e risposta.",
                              systemImage: "text.bubble")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        modelTextDisclosure("Prompt inviato", text: controller.currentModelPrompt,
                                            icon: "text.quote")
                        if controller.isGeneratingPatch && controller.currentModelThinking.isEmpty {
                            HStack { ProgressView().controlSize(.small); Text("Attendo il thinking del modello…") }
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        if !controller.currentModelThinking.isEmpty {
                            modelTextDisclosure("Thinking", text: controller.currentModelThinking,
                                                icon: "brain")
                        }
                        if !controller.currentModelResponse.isEmpty {
                            modelTextDisclosure("Risposta del modello", text: controller.currentModelResponse,
                                                icon: "text.bubble")
                        }
                    }
                    if !controller.modelExchanges.isEmpty {
                        DisclosureGroup("Cronologia tentativi (\(controller.modelExchanges.count))") {
                            ForEach(controller.modelExchanges) { exchange in
                                DisclosureGroup("\(exchange.instanceID) · \(exchange.createdAt.formatted(date: .omitted, time: .standard))") {
                                    modelTextDisclosure("Prompt", text: exchange.prompt, icon: "text.quote")
                                    if !exchange.thinking.isEmpty {
                                        modelTextDisclosure("Thinking", text: exchange.thinking, icon: "brain")
                                    }
                                    modelTextDisclosure("Risposta", text: exchange.response, icon: "text.bubble")
                                    modelTextDisclosure("Patch estratta", text: exchange.extractedPatch,
                                                        icon: "doc.badge.gearshape")
                                }
                            }
                        }
                    }
                }
                Section {
                    if controller.isRunning {
                        HStack { ProgressView(); Text("Runner Swift locale in esecuzione…"); Spacer() }
                        Button("Stop", role: .destructive) { controller.stop() }
                    } else {
                        Button { controller.run() } label: {
                            Label("Avvia SWE-bench locale", systemImage: "play.fill")
                        }.disabled(controller.localTaskPath.isEmpty
                                   || (controller.predictionMode != .gold && controller.predictionsPath.isEmpty))
                    }
                    if let error = controller.lastError {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.orange).textSelection(.enabled)
                    }
                }
            }
            .formStyle(.grouped).frame(maxHeight: 560)

            Divider()
            if let report = controller.localReport {
                localReportView(report)
            } else {
                ContentUnavailableView("Nessun risultato", systemImage: "wrench.and.screwdriver",
                                       description: Text("Seleziona task e prediction, poi avvia il runner Swift locale."))
            }
            if !controller.log.isEmpty {
                Divider()
                ScrollView {
                    Text(controller.log).font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled).padding(8)
                }.frame(minHeight: 120)
            }
        }
        .alert("Avviare tutti i task SWE-bench Lite?", isPresented: $confirmRunAll) {
            Button("Annulla", role: .cancel) {}
            Button("Avvia tutti") { controller.generateAllModelPredictions() }
        } message: {
            Text("Il modello elaborerà in sequenza tutti i \(controller.goldCatalog.count) task scaricati. L'operazione può richiedere molte ore; ogni prompt, thinking e risposta verrà salvato.")
        }
    }

    private func localReportView(_ report: SWELocalGradeReport) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            if report.verificationOnly == true {
                Label(report.patchApplied ? "Gold patch applicabile" : "Gold patch non applicabile",
                      systemImage: report.patchApplied ? "checkmark.seal.fill" : "xmark.seal.fill")
                    .font(.headline).foregroundStyle(report.patchApplied ? .green : .red)
                Text("Verifica nativa completata. I test Python ufficiali non sono eseguiti; lo stato di risoluzione non è applicabile.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                Label(report.resolved ? "Task risolto" : "Task non risolto",
                      systemImage: report.resolved ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .font(.headline).foregroundStyle(report.resolved ? .green : .red)
            }
            LabeledContent("Istanza", value: report.instanceID)
            LabeledContent("Patch applicata", value: report.patchApplied ? "Sì" : "No")
            if report.verificationOnly != true {
                LabeledContent("FAIL_TO_PASS", value: "\(report.failToPassSuccess.count) riusciti · \(report.failToPassFailure.count) falliti")
                LabeledContent("PASS_TO_PASS", value: "\(report.passToPassSuccess.count) riusciti · \(report.passToPassFailure.count) falliti")
            }
            if report.timedOut { Label("Timeout", systemImage: "clock.badge.exclamationmark").foregroundStyle(.orange) }
        }.padding()
    }

    private func modelTextDisclosure(_ title: String, text: String, icon: String) -> some View {
        DisclosureGroup {
            ScrollView {
                Text(text)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(8)
            }
            .frame(minHeight: 80, maxHeight: 260)
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 6))
        } label: {
            Label(title, systemImage: icon)
        }
    }

    private func panel(file: Bool, action: (URL) -> Void) {
        let p = NSOpenPanel(); p.canChooseFiles = file; p.canChooseDirectories = !file
        p.canCreateDirectories = !file
        if file { p.allowedContentTypes = [.json, .plainText, .data] }
        if p.runModal() == .OK, let url = p.url { action(url) }
    }
    private func choosePredictions() { panel(file: true, action: controller.selectPredictions) }
    private func chooseLocalTask() { panel(file: true, action: controller.selectLocalTask) }
    private func chooseLocalRunner() { panel(file: true, action: controller.selectLocalRunner) }
    private func chooseOutputDirectory() { panel(file: false, action: controller.selectOutputDirectory) }
}
