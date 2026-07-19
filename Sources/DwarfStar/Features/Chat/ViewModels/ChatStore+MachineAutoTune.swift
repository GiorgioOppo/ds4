import Foundation
import DS4Engine
import DS4Core

private struct GUIAutoTuneTrial: Codable, Sendable {
    let pass: Int
    let parameter: String
    let fromValue: Int
    let toValue: Int
    let screenPassed: Bool
    let comparisonMode: String
    let balancedPrimaryRatio: Double?
    let balancedSecondaryRatio: Double?
    let transitionQualityExact: Bool
    let cumulativeQualityExact: Bool
    var selected: Bool
    let reasons: [String]
}

private struct GUIAutoTuneReport: Codable, Sendable {
    let schema: Int
    let createdAt: Date
    let updatedAt: Date
    let status: String
    let chip: String
    let ramGB: Int
    let modelPath: String
    let contextTokens: Int
    let initial: [String: Int]
    /// Best coordinate-search candidate, which may still be uncommitted.
    let bestCandidate: [String: Int]
    /// Engine actually live at the checkpoint; nil during teardown.
    let activeConfiguration: [String: Int]?
    /// Complete preference snapshot currently committed to UserDefaults.
    let persistedConfiguration: [String: Int]?
    let final: [String: Int]
    let fixedEnvironment: [String: String]
    let usageProfileHash: String?
    /// Immutable RAM policy derived from the already loaded root engine.
    let memoryEnvelope: MachineAutoTuneMemoryEnvelope?
    let pendingLabel: String?
    let pendingConfiguration: [String: Int]?
    let note: String?
    let validated: Bool
    let trials: [GUIAutoTuneTrial]
    /// Complete human-readable run log at this atomic checkpoint, including
    /// the separate cold-setup and steady-state swap windows.
    let log: String
}

private struct GUIAutoTuneCandidate {
    let configuration: MachineAutoTuneConfiguration
    let value: Int
    let evaluation: MachineAutoTuneEvaluationResult
    let trial: GUIAutoTuneTrial
}

/// Plain, copyable facts captured from the currently loaded decoder before the
/// first teardown. Keeping the service actor out of the long-running search
/// frame is important on memory-constrained Macs: an accidental strong
/// reference would retain the complete baseline engine while the first
/// candidate is being constructed.
private struct GUIAutoTuneRuntimeGeometry {
    let expertCacheBytesPerBaseSlot: UInt64
    let denseStagingBytesPerAheadSlot: UInt64
    let expertPreadSplitIsEffective: Bool
}

/// Result of the bounded barrier between cold setup and a policy measurement.
/// The snapshot is the only valid start of a steady-state swap window.
private struct GUIAutoTuneSwapSettlement {
    let snapshot: MemoryInfo.PressureSnapshot
    let elapsedSeconds: Double
    let lastRateMiBPerSecond: Double
}

/// Swap policy constants kept separate from the performance thresholds. The
/// 128 MiB promotion cap remains in `MachineAutoTunePolicy`; these values only
/// decide whether cold paging has stopped enough to start a fair measurement.
private enum GUIAutoTuneSwapSettlementPolicy {
    static let sampleIntervalNanoseconds: UInt64 = 1_000_000_000
    static let maximumIntervals = 10
    static let requiredQuietIntervals = 2
    static let maximumQuietRateMiBPerSecond = 16.0
    static let maximumSetupSwapoutMiB = 128.0
}

/// Mutable run-local state used only by the main-actor GUI orchestrator. The
/// on-disk journal is rewritten atomically before every model-sized load, so a
/// crash/OOM still leaves the last committed configuration, completed trials,
/// and the exact pending stage inspectable on the next launch.
private final class GUIAutoTuneJournal {
    let directory: URL
    let createdAt: Date
    let chip: String
    let ramGB: Int
    let modelPath: String
    let contextSize: Int
    let initial: MachineAutoTuneConfiguration
    var current: MachineAutoTuneConfiguration
    var active: MachineAutoTuneConfiguration?
    var persisted: MachineAutoTuneConfiguration?
    let fixedEnvironment: [String: String]
    let usageProfileHash: String?
    var memoryEnvelope: MachineAutoTuneMemoryEnvelope? = nil
    var trials: [GUIAutoTuneTrial] = []

    init(
        directory: URL,
        createdAt: Date,
        chip: String,
        ramGB: Int,
        modelPath: String,
        contextSize: Int,
        initial: MachineAutoTuneConfiguration,
        fixedEnvironment: [String: String],
        usageProfileHash: String?
    ) {
        self.directory = directory
        self.createdAt = createdAt
        self.chip = chip
        self.ramGB = ramGB
        self.modelPath = modelPath
        self.contextSize = contextSize
        self.initial = initial
        self.current = initial
        self.active = initial
        self.persisted = initial
        self.fixedEnvironment = fixedEnvironment
        self.usageProfileHash = usageProfileHash
    }
}

/// Conservative memory history for the in-process search. Measurements retain
/// the lowest observed free-RAM percentage per configuration; reloading a larger
/// cache is rejected before allocation when that history predicts less than the
/// safety floor.
private final class GUIAutoTuneMemoryGuard {
    var loadedConfiguration: MachineAutoTuneConfiguration?
    /// Exact legacy/mixed-cache byte budget added by one configured slot,
    /// queried from the currently loaded GGUF before the first teardown.
    var expertCacheBudgetBytesPerBaseSlot: UInt64 = 0
    /// Exact bytes of one DenseStreamer staging buffer for the loaded GGUF and
    /// fixed quantization/residency controls.
    var denseStagingBytesPerAheadSlot: UInt64 = 0
    var expertPreadSplitIsEffective = false
    /// Set exactly once from the live root before the first teardown. It must
    /// never follow a promoted candidate, otherwise a sequence of small
    /// regressions could ratchet the RAM floor downward.
    private(set) var envelope: MachineAutoTuneMemoryEnvelope?
    private(set) var hasStartedTeardown = false
    let baselineSignature: LoadedEngineSignature
    private(set) var observedFreePercent: [MachineAutoTuneConfiguration: Double] = [:]

    init(baselineSignature: LoadedEngineSignature) {
        self.baselineSignature = baselineSignature
        loadedConfiguration = baselineSignature.tuning
    }

    func installEnvelope(_ envelope: MachineAutoTuneMemoryEnvelope) {
        precondition(self.envelope == nil, "machine auto-tune memory envelope is immutable")
        self.envelope = envelope
    }

    var effectiveMinimumFreePercent: Double {
        envelope?.effectiveMinimumFreePercent ?? 10
    }

    var isConstrained: Bool { envelope?.isConstrained == true }

    func markTeardownStarted() { hasStartedTeardown = true }

    func record(_ freePercent: Double, for configuration: MachineAutoTuneConfiguration) {
        let bounded = min(100, max(0, freePercent))
        observedFreePercent[configuration] = min(observedFreePercent[configuration] ?? 100,
                                                 bounded)
    }

    func observedFree(for configuration: MachineAutoTuneConfiguration) -> Double? {
        observedFreePercent[configuration]
    }

    func signature(for configuration: MachineAutoTuneConfiguration) -> LoadedEngineSignature {
        LoadedEngineSignature(
            modelPath: baselineSignature.modelPath,
            contextSize: baselineSignature.contextSize,
            tuning: configuration,
            fixedEnvironment: baselineSignature.fixedEnvironment
        )
    }
}

/// One run-local measurement per exact configuration. Invalid observations are
/// retained as well, so revisiting a rejected point never reloads and reruns the
/// same model. Promoted configurations always point at their complete measured
/// observation; metrics from different samples are never merged.
private final class GUIAutoTuneBenchmarkLedger {
    private var observations: [MachineAutoTuneConfiguration: MachineAutoTuneObservation] = [:]
    private var failures: [MachineAutoTuneConfiguration: String] = [:]

    func observation(
        for configuration: MachineAutoTuneConfiguration
    ) -> MachineAutoTuneObservation? {
        observations[configuration]
    }

    func record(
        _ observation: MachineAutoTuneObservation,
        for configuration: MachineAutoTuneConfiguration
    ) {
        precondition(
            observations[configuration] == nil,
            "a machine auto-tune configuration must be benchmarked at most once"
        )
        observations[configuration] = observation
    }

    func failure(for configuration: MachineAutoTuneConfiguration) -> String? {
        failures[configuration]
    }

    func recordFailure(_ reason: String, for configuration: MachineAutoTuneConfiguration) {
        guard observations[configuration] == nil, failures[configuration] == nil else { return }
        failures[configuration] = reason
    }
}

private enum GUIAutoTuneError: LocalizedError {
    case baselineRejected([String])
    case missingQualityTrace
    case memoryCountersUnavailable
    case activeDownload
    case thermalPressure
    case expertCacheGeometryUnavailable
    case denseStagingGeometryUnavailable
    case insufficientMemoryHeadroom(String)
    case journalFailed(String)
    case reloadFailed(String)

    var errorDescription: String? {
        switch self {
        case .baselineRejected(let reasons):
            return "baseline non valida: " + reasons.joined(separator: "; ")
        case .missingQualityTrace:
            return "il benchmark non ha prodotto la trace di qualità"
        case .memoryCountersUnavailable:
            return "contatori RAM/swap non disponibili"
        case .activeDownload:
            return "download .part attivo nella cartella del modello"
        case .thermalPressure:
            return "pressione termica seria: attendere il raffreddamento del Mac"
        case .expertCacheGeometryUnavailable:
            return "geometria RAM dell'expert-cache non disponibile dal GGUF caricato"
        case .denseStagingGeometryUnavailable:
            return "geometria RAM del dense staging non disponibile dal motore caricato"
        case .insufficientMemoryHeadroom(let detail):
            return "preflight RAM respinto: \(detail)"
        case .journalFailed(let detail):
            return "journal auto-tune non scrivibile: \(detail)"
        case .reloadFailed(let detail):
            return "reload del modello fallito: \(detail)"
        }
    }

    var abortsWholeRun: Bool {
        switch self {
        case .memoryCountersUnavailable, .activeDownload, .thermalPressure,
             .expertCacheGeometryUnavailable, .denseStagingGeometryUnavailable,
             .journalFailed:
            return true
        case .baselineRejected, .missingQualityTrace, .insufficientMemoryHeadroom, .reloadFailed:
            return false
        }
    }
}

extension ChatStore {
    /// Starts the record-holder, in-process machine tuner.
    /// Candidate values live only in the process environment; persisted GUI
    /// settings are changed once, after the winning engine passes its final
    /// prompt-restoring RAM/swap probe.
    func runAutoTune(distributedRuntimeActive: Bool) {
        guard settings.mode == .local else {
            benchStatus = "L'auto-tune macchina richiede la modalità Locale."
            return
        }
        guard !distributedRuntimeActive else {
            benchStatus = "Arresta worker e coordinatore distribuiti prima dell'auto-tune: condividono GPU, RAM e knob Metal del processo."
            return
        }
        guard rawRingEnabled else {
            benchStatus = "Attiva RAW_RING prima dell'auto-tune: è un vincolo di memoria fisso e deve far parte della baseline."
            return
        }
        guard !profileRouteEnabled else {
            benchStatus = "Disattiva Profile route/attention prima dell'auto-tune: le sincronizzazioni diagnostiche falsano il throughput."
            return
        }
        guard service != nil else { benchStatus = "Carica prima il modello."; return }
        guard let loadedSignature = loadedEngineSignature else {
            benchStatus = "Impossibile ricostruire i knob del motore caricato: ricarica il modello prima dell'auto-tune."
            return
        }
        let requestedSignature = machineAutoTuneEngineSignature()
        guard loadedSignature == requestedSignature else {
            benchStatus = "Modello, contesto o setting di caricamento sono cambiati dopo il load. Ricarica il modello per rendere la baseline coerente, poi avvia l'auto-tune."
            return
        }
        guard phase == .ready else { benchStatus = "Attendi che il modello sia pronto."; return }
        guard glmService == nil else {
            benchStatus = "Auto-tune non necessario per GLM 5.2: la "
                + "residenza layer è già adattiva alla RAM; usa il "
                + "Benchmark per misurare, e gli stepper GLM per regolare."
            return
        }
        guard !isGenerating else { benchStatus = "Ferma la generazione prima dell'auto-tune."; return }
        guard !benchRunning else { return }

        let gate = EngineActivityGate.shared
        guard let lease = gate.acquire(.autoTune) else {
            let owner = gate.activeOwner?.displayName ?? "un'altra operazione"
            benchStatus = "Motore occupato da \(owner). Arrestalo prima dell'auto-tune."
            return
        }

        let initial = loadedSignature.tuning
        let path = modelPath
        let context = contextSize
        let agentID = selectedAgentId
        let modelName = (path as NSString).lastPathComponent
        let frozenUsage = InferenceService.usageDataSeeded(modelName: modelName,
                                                           agentId: agentID)
        let ramGB = max(1, Int(ProcessInfo.processInfo.physicalMemory >> 30))
        let manifest = MachineAutoTuneManifest.standard(ramGB: ramGB)
        let upperBoundCandidates = manifest.reduce(0) {
            $0 + max(0, $1.values.count - 1)
        }

        benchRunning = true
        benchSucceeded = nil
        benchResults = ""
        autoTuneReportURL = nil
        benchProgressDone = 0
        // One prompt-restoring root sample, at most one measurement per unique
        // candidate in two passes, then the final chat-engine installation.
        benchProgressTotal = 1 + upperBoundCandidates * 2 + 1
        benchStatus = "Auto-tune record-holder: preparazione…"

        benchTask?.cancel()
        benchTask = Task(priority: .userInitiated) { [weak self] in
            guard let self else {
                EngineActivityGate.shared.release(lease)
                return
            }
            defer {
                EngineActivityGate.shared.release(lease)
                self.benchTask = nil
            }
            await self.performMachineAutoTune(
                initial: initial,
                manifest: manifest,
                modelPath: path,
                contextSize: context,
                frozenUsage: frozenUsage,
                ramGB: ramGB,
                baselineSignature: loadedSignature
            )
        }
    }

    func cancelAutoTune() {
        guard benchRunning else { return }
        benchStatus = "Annullamento e ripristino della configurazione iniziale…"
        benchTask?.cancel()
    }

    private func performMachineAutoTune(
        initial: MachineAutoTuneConfiguration,
        manifest: [MachineAutoTuneParameter],
        modelPath: String,
        contextSize: Int,
        frozenUsage: Data?,
        ramGB: Int,
        baselineSignature: LoadedEngineSignature
    ) async {
        var policy = MachineAutoTunePolicy()
        let chip = Self.chipName()
        var current = initial
        var trials: [GUIAutoTuneTrial] = []
        let memoryGuard = GUIAutoTuneMemoryGuard(baselineSignature: baselineSignature)
        let benchmarkLedger = GUIAutoTuneBenchmarkLedger()
        let transactionStore = MachineAutoTuneTransactionStore()
        var activePersistenceTransactionID: UUID?
        let journal: GUIAutoTuneJournal

        do {
            journal = try makeMachineAutoTuneJournal(
                chip: chip,
                ramGB: ramGB,
                modelPath: modelPath,
                contextSize: contextSize,
                initial: initial,
                fixedEnvironment: baselineSignature.fixedEnvironment,
                usageProfileHash: frozenUsage.map(Self.machineAutoTuneDataHash)
            )
            autoTuneReportURL = try checkpointMachineAutoTune(
                journal,
                status: "PREPARING",
                pendingLabel: nil,
                pendingConfiguration: nil,
                note: "Journal creato prima del primo teardown.",
                validated: false
            )
        } catch {
            benchSucceeded = false
            benchStatus = "Auto-tune non avviato: impossibile creare il journal: \(error.localizedDescription)"
            autoTuneLog(benchStatus ?? "Auto-tune non avviato.")
            benchProgressTotal = max(benchProgressDone, 1)
            benchRunning = false
            return
        }

        autoTuneLog("Mac: \(chip), \(ramGB) GB RAM; contesto \(contextSize).")
        autoTuneLog("Qualità: token + top-3 + hash bit-esatto di tutti i logits; " +
                    "una misura per configurazione, confronto col record decode valido.")
        autoTuneLog("RAW_RING resta fisso a 1. I candidati non vengono persistiti.")
        if frozenUsage == nil {
            autoTuneLog("Nessun usage profile disponibile: allocation e look-ahead usage-driven saranno saltati.")
        } else {
            autoTuneLog("Usage profile congelato per tutte le misure.")
        }

        do {
            // A prior load/role change may still be warming the currently
            // visible service. Drain that owner before the first teardown so
            // the tuner never overlaps two model-sized engines.
            guard await waitForEngineSetup() else {
                throw GUIAutoTuneError.reloadFailed(
                    "warmup del ruolo attivo precedente fallito"
                )
            }
            try Task.checkCancellation()
            // Capture only scalar geometry in a short-lived helper. Do not bind
            // the model-sized service actor in this outer async scope: Swift is
            // otherwise allowed to retain it across all subsequent reloads.
            let runtimeGeometry = try await captureMachineAutoTuneRuntimeGeometry()
            let cacheBytesPerSlot = runtimeGeometry.expertCacheBytesPerBaseSlot
            guard cacheBytesPerSlot > 0 else {
                throw GUIAutoTuneError.expertCacheGeometryUnavailable
            }
            memoryGuard.expertCacheBudgetBytesPerBaseSlot = cacheBytesPerSlot
            let denseBytesPerAhead = runtimeGeometry.denseStagingBytesPerAheadSlot
            if baselineSignature.fixedEnvironment["DS4_DENSE_STREAM"] == "1",
               denseBytesPerAhead == 0 {
                throw GUIAutoTuneError.denseStagingGeometryUnavailable
            }
            memoryGuard.denseStagingBytesPerAheadSlot = denseBytesPerAhead
            memoryGuard.expertPreadSplitIsEffective = runtimeGeometry.expertPreadSplitIsEffective
            autoTuneLog(String(
                format: "Preflight dal GGUF: cache %.1f MiB/slot base · dense %.1f MiB/step ahead · pread split %@.",
                Double(cacheBytesPerSlot) / 1_048_576,
                Double(denseBytesPerAhead) / 1_048_576,
                memoryGuard.expertPreadSplitIsEffective ? "effettivo" : "bypassato"
            ))
            let startPressure = MemoryInfo.pressureSnapshot()
            guard let startFree = startPressure.freePercent,
                  startPressure.swapoutsBytes != nil else {
                let detail = GUIAutoTuneError.memoryCountersUnavailable.localizedDescription
                autoTuneReportURL = try? checkpointMachineAutoTune(
                    journal,
                    status: "PREFLIGHT_REJECTED",
                    pendingLabel: nil,
                    pendingConfiguration: nil,
                    note: detail,
                    validated: false
                )
                benchSucceeded = false
                benchStatus = "Auto-tune non avviato: \(detail). Il motore caricato resta attivo."
                benchProgressTotal = max(benchProgressDone, 1)
                benchRunning = false
                return
            }
            let envelope = MachineAutoTuneMemoryEnvelope(
                baselineFreePercent: startFree,
                physicalMemoryBytes: MemoryInfo.physicalBytes
            )
            guard envelope.isValid else {
                throw GUIAutoTuneError.memoryCountersUnavailable
            }
            memoryGuard.installEnvelope(envelope)
            memoryGuard.record(startFree, for: initial)
            journal.memoryEnvelope = envelope
            policy = envelope.policy(from: policy)

            guard envelope.canStart else {
                let freeGiB = Double(MemoryInfo.physicalBytes) * startFree / 100
                    / 1_073_741_824
                let detail = String(
                    format: "RAM libera col motore corrente %.1f%%/%.2f GiB; " +
                            "servono almeno %.0f MiB per un teardown sicuro",
                    startFree,
                    freeGiB,
                    Double(envelope.hardReserveBytes) / 1_048_576
                )
                autoTuneReportURL = try? checkpointMachineAutoTune(
                    journal,
                    status: "PREFLIGHT_REJECTED",
                    pendingLabel: nil,
                    pendingConfiguration: nil,
                    note: detail,
                    validated: false
                )
                benchSucceeded = false
                benchStatus = "Auto-tune non avviato: \(detail). Il motore caricato resta attivo."
                benchProgressTotal = max(benchProgressDone, 1)
                benchRunning = false
                return
            }
            if envelope.isConstrained {
                autoTuneLog(String(
                    format: "Modalità low-RAM vincolata: baseline %.1f%%, floor immutabile %.1f%% " +
                            "(riserva assoluta %.0f MiB). Saranno provati solo delta residenti ≤0.",
                    envelope.baselineFreePercent,
                    envelope.effectiveMinimumFreePercent,
                    Double(envelope.hardReserveBytes) / 1_048_576
                ))
            } else {
                autoTuneLog("Modalità RAM standard: floor 10%.")
            }
            // The loaded root is already warm and owns the frozen agent usage
            // profile. Measure it in place and restore its prompt state: there
            // is no reason to tear down and reload the same configuration just
            // to manufacture a second baseline sample.
            let root = try await measureLoadedMachineAutoTuneBaseline(
                configuration: initial,
                label: "baseline root",
                journal: journal,
                memoryGuard: memoryGuard
            )
            let rootQuality = root.quality
            let rootQualityReasons = MachineAutoTuneEvaluator.qualityValidationReasons(
                rootQuality,
                label: "immutable root quality"
            )
            guard rootQualityReasons.isEmpty else {
                throw GUIAutoTuneError.baselineRejected(rootQualityReasons)
            }
            let rootReasons = MachineAutoTuneEvaluator.observationValidationReasons(
                root,
                rootQuality: rootQuality,
                label: "baseline root",
                policy: policy
            )
            guard rootReasons.isEmpty,
                  MachineAutoTuneEvaluator.bestValidObservation(
                    in: [root], rootQuality: rootQuality, policy: policy
                  ) != nil else {
                throw GUIAutoTuneError.baselineRejected(rootReasons)
            }
            benchmarkLedger.record(root, for: initial)
            autoTuneLog(String(
                format: "Record iniziale fissato a %.3f t/s decode; la root non sarà rimisurata.",
                root.primaryTps
            ))

            for pass in 1...2 {
                try Task.checkCancellation()
                var passChanged = false
                autoTuneLog("── passata coordinata \(pass)/2 ──")

                for parameter in manifest {
                    try Task.checkCancellation()
                    if let reason = machineAutoTuneSkipReason(
                        parameter: parameter,
                        configuration: current,
                        frozenUsage: frozenUsage,
                        memoryGuard: memoryGuard
                    ) {
                        autoTuneLog("SKIP \(parameter.knob.rawValue): \(reason)")
                        continue
                    }
                    guard let fromValue = current.value(for: parameter.knob) else {
                        autoTuneLog("SKIP \(parameter.knob.rawValue): valore corrente assente.")
                        continue
                    }
                    if parameter.search == .walk && !parameter.values.contains(fromValue) {
                        autoTuneLog("SKIP \(parameter.knob.rawValue): valore corrente fuori manifest ordinato.")
                        continue
                    }

                    switch parameter.search {
                    case .sweep:
                        let candidates = machineAutoTuneSweepCandidates(
                            parameter: parameter,
                            fromValue: fromValue,
                            configuration: current,
                            memoryGuard: memoryGuard
                        )
                        var blockedGrowthAt: Int?
                        var qualified: [(candidate: GUIAutoTuneCandidate, trialIndex: Int)] = []
                        for value in candidates {
                            try Task.checkCancellation()
                            if let blockedGrowthAt,
                               parameter.memoryRisk,
                               value > fromValue,
                               value >= blockedGrowthAt {
                                autoTuneLog("SKIP \(parameter.knob.rawValue)=\(value): " +
                                            "un gradino cache più piccolo ha già fallito il preflight RAM.")
                                continue
                            }
                            let candidateConfig = current.setting(value, for: parameter.knob)
                            do {
                                let candidate = try await evaluateMachineAutoTuneCandidate(
                                    incumbent: current,
                                    candidate: candidateConfig,
                                    pass: pass,
                                    parameter: parameter.knob.rawValue,
                                    fromValue: fromValue,
                                    toValue: value,
                                    rootQuality: rootQuality,
                                    policy: policy,
                                    modelPath: modelPath,
                                    contextSize: contextSize,
                                    frozenUsage: frozenUsage,
                                    journal: journal,
                                    memoryGuard: memoryGuard,
                                    benchmarkLedger: benchmarkLedger
                                )
                                let index = trials.count
                                trials.append(candidate.trial)
                                journal.trials = trials
                                if candidate.evaluation.qualified {
                                    qualified.append((candidate, index))
                                }
                            } catch is CancellationError {
                                throw CancellationError()
                            } catch let fatal as GUIAutoTuneError where fatal.abortsWholeRun {
                                throw fatal
                            } catch {
                                if parameter.memoryRisk,
                                   value > fromValue,
                                   isMachineAutoTuneHeadroomError(error) {
                                    blockedGrowthAt = min(blockedGrowthAt ?? value, value)
                                }
                                autoTuneLog("REJECT \(parameter.knob.rawValue)=\(value): \(error.localizedDescription)")
                                trials.append(failedMachineAutoTuneTrial(
                                    pass: pass, parameter: parameter.knob.rawValue,
                                    fromValue: fromValue, toValue: value, error: error
                                ))
                                journal.trials = trials
                            }
                        }
                        if let winner = qualified.max(by: {
                            ($0.candidate.evaluation.balancedPrimaryRatio ?? 0)
                                < ($1.candidate.evaluation.balancedPrimaryRatio ?? 0)
                        }) {
                            trials[winner.trialIndex].selected = true
                            current = winner.candidate.configuration
                            passChanged = true
                            journal.current = current
                            journal.trials = trials
                            autoTuneLog("PROMOSSO \(parameter.knob.rawValue)=\(winner.candidate.value) " +
                                        ratioDescription(winner.candidate.evaluation))
                        }

                    case .walk:
                        var walking = current
                        var walk = MachineAutoTuneDirectionalWalk(
                            values: parameter.values,
                            current: fromValue
                        )
                        guard walk.isValid else {
                            autoTuneLog(
                                "SKIP \(parameter.knob.rawValue): scala direzionale non valida."
                            )
                            continue
                        }
                        while let neighbor = walk.nextCandidate() {
                            try Task.checkCancellation()
                            guard let walkValue = walking.value(for: parameter.knob) else { break }
                            let candidateConfig = walking.setting(
                                neighbor.value,
                                for: parameter.knob
                            )
                            if parameter.memoryRisk,
                               let envelope = memoryGuard.envelope {
                                let delta = estimatedMachineAutoTuneResidentDeltaBytes(
                                    from: memoryGuard.baselineSignature.tuning,
                                    to: candidateConfig,
                                    memoryGuard: memoryGuard
                                )
                                if !envelope.allowsResidentDeltaBytes(delta) {
                                    autoTuneLog(
                                        "SKIP \(parameter.knob.rawValue)=\(neighbor.value): " +
                                        "delta residente positivo in modalità low-RAM."
                                    )
                                    _ = walk.recordResult(qualified: false)
                                    logMachineAutoTuneWalkDecision(
                                        parameter: parameter.knob.rawValue,
                                        walk: walk
                                    )
                                    continue
                                }
                            }
                            do {
                                let candidate = try await evaluateMachineAutoTuneCandidate(
                                    incumbent: walking,
                                    candidate: candidateConfig,
                                    pass: pass,
                                    parameter: parameter.knob.rawValue,
                                    fromValue: walkValue,
                                    toValue: neighbor.value,
                                    rootQuality: rootQuality,
                                    policy: policy,
                                    modelPath: modelPath,
                                    contextSize: contextSize,
                                    frozenUsage: frozenUsage,
                                    journal: journal,
                                    memoryGuard: memoryGuard,
                                    benchmarkLedger: benchmarkLedger
                                )
                                let trialIndex = trials.count
                                trials.append(candidate.trial)
                                journal.trials = trials
                                let promoted = walk.recordResult(
                                    qualified: candidate.evaluation.qualified
                                )
                                if promoted {
                                    trials[trialIndex].selected = true
                                    walking = candidate.configuration
                                    passChanged = true
                                    journal.current = walking
                                    journal.trials = trials
                                    autoTuneLog(
                                        "PROMOSSO \(parameter.knob.rawValue)=\(candidate.value) " +
                                        ratioDescription(candidate.evaluation)
                                    )
                                } else {
                                    logMachineAutoTuneWalkDecision(
                                        parameter: parameter.knob.rawValue,
                                        walk: walk
                                    )
                                }
                            } catch is CancellationError {
                                throw CancellationError()
                            } catch let fatal as GUIAutoTuneError where fatal.abortsWholeRun {
                                throw fatal
                            } catch {
                                autoTuneLog(
                                    "REJECT \(parameter.knob.rawValue)=\(neighbor.value): " +
                                    error.localizedDescription
                                )
                                trials.append(failedMachineAutoTuneTrial(
                                    pass: pass, parameter: parameter.knob.rawValue,
                                    fromValue: walkValue, toValue: neighbor.value, error: error
                                ))
                                journal.trials = trials
                                _ = walk.recordResult(qualified: false)
                                logMachineAutoTuneWalkDecision(
                                    parameter: parameter.knob.rawValue,
                                    walk: walk
                                )
                            }
                        }
                        current = walking
                        journal.current = current
                    }
                }
                if !passChanged {
                    autoTuneLog("Nessuna promozione nella passata \(pass): massimo locale raggiunto.")
                    break
                }
            }

            guard let finalObservation = benchmarkLedger.observation(for: current) else {
                throw GUIAutoTuneError.baselineRejected([
                    "il finalista non ha una misura record in cache"
                ])
            }
            let finalReasons = MachineAutoTuneEvaluator.observationValidationReasons(
                finalObservation,
                rootQuality: rootQuality,
                label: "finalista record",
                policy: policy
            )
            guard finalReasons.isEmpty else {
                throw GUIAutoTuneError.baselineRejected(finalReasons)
            }
            autoTuneLog(String(
                format: "Finalista dal record già misurato: %.3f t/s decode; nessun benchmark duplicato.",
                finalObservation.primaryTps
            ))
            try Task.checkCancellation()
            journal.current = current
            journal.trials = trials
            let reportURL = try checkpointMachineAutoTune(
                journal,
                status: "VALIDATED",
                pendingLabel: nil,
                pendingConfiguration: nil,
                note: "Finalista record validato senza ripetere configurazioni già misurate.",
                validated: true
            )
            autoTuneReportURL = reportURL
            autoTuneLog("VALIDATO: \(machineAutoTuneSummary(current))")
            autoTuneLog("Report: \(reportURL.deletingLastPathComponent().path)")

            // Crash-safe two-phase adoption: persist nothing until the winner
            // has completed init and warmup. Startup recovery rolls any
            // interrupted prepared/installing/committing transaction back to
            // the complete initial snapshot.
            let transaction = try transactionStore.prepare(
                initial: initial,
                winner: current,
                modelPath: modelPath,
                contextTokens: contextSize,
                reportDirectory: reportURL.deletingLastPathComponent()
            )
            activePersistenceTransactionID = transaction.id
            _ = try checkpointMachineAutoTune(
                journal,
                status: "WINNER_INSTALL_PENDING",
                pendingLabel: "Installazione finalista validato",
                pendingConfiguration: current,
                note: "Preferenze ancora iniziali; commit solo dopo init e warmup.",
                validated: true
            )
            journal.active = nil
            benchStatus = "Auto-tune validato; installo e verifico il motore vincitore…"
            try await installChatEngineAfterAutoTune(
                configuration: current,
                modelPath: modelPath,
                contextSize: contextSize,
                memoryGuard: memoryGuard,
                allowCancellation: true
            )
            journal.active = current
            try Task.checkCancellation()
            try transactionStore.markInstalled(transactionID: transaction.id)
            try transactionStore.commit(transactionID: transaction.id) {
                self.persistMachineAutoTuneConfiguration(current)
            }
            activePersistenceTransactionID = nil
            journal.persisted = current
            _ = try? checkpointMachineAutoTune(
                journal,
                status: "COMPLETED",
                pendingLabel: nil,
                pendingConfiguration: nil,
                note: "Finalista persistito e motore chat ripristinato.",
                validated: true
            )
            benchSucceeded = true
            benchStatus = "Auto-tune completato e validato: \(machineAutoTuneSummary(current))"
        } catch is CancellationError {
            if !memoryGuard.hasStartedTeardown,
               service != nil,
               loadedEngineSignature == baselineSignature {
                // Stop arrived before the first teardown. The known-good root
                // is still live; if its probe ran, enginePrimed already forces
                // history re-prefill instead of an unnecessary model reload.
                applyMachineAutoTuneEnvironment(initial)
                journal.current = initial
                journal.active = initial
                phase = .ready
                benchSucceeded = false
                benchStatus = "Auto-tune annullato prima del teardown; motore root mantenuto."
                autoTuneReportURL = try? checkpointMachineAutoTune(
                    journal,
                    status: "CANCELLED_ROOT_RETAINED",
                    pendingLabel: nil,
                    pendingConfiguration: nil,
                    note: "Annullamento precedente al primo teardown; eventuale KV sintetica " +
                          "sarà ricostruita dalla cronologia al prossimo invio.",
                    validated: false
                )
                benchProgressTotal = max(benchProgressDone, 1)
                benchRunning = false
                return
            }
            autoTuneLog("ANNULLATO: ripristino configurazione e motore iniziali.")
            journal.current = current
            journal.trials = trials
            if let transactionID = activePersistenceTransactionID {
                do {
                    try transactionStore.rollback(
                        transactionID: transactionID,
                        reason: "Annullamento durante l'adozione del finalista"
                    ) {
                        self.persistMachineAutoTuneConfiguration(initial)
                    }
                    activePersistenceTransactionID = nil
                    journal.persisted = initial
                } catch {
                    // Reassert in-process state now; leave the durable record so
                    // launch recovery can retry the complete rollback.
                    persistMachineAutoTuneConfiguration(initial)
                    journal.persisted = initial
                    autoTuneLog("Rollback transazione incompleto; sarà ritentato all'avvio: \(error.localizedDescription)")
                }
            }
            do {
                autoTuneReportURL = try checkpointMachineAutoTune(
                    journal,
                    status: "CANCELLED_RESTORING",
                    pendingLabel: "Ripristino configurazione iniziale",
                    pendingConfiguration: initial,
                    note: "Annullamento richiesto; nessun candidato è stato persistito.",
                    validated: false
                )
            } catch {
                autoTuneLog("Impossibile aggiornare il journal di annullamento: \(error.localizedDescription)")
            }
            journal.active = nil
            applyMachineAutoTuneEnvironment(initial)
            do {
                try await installChatEngineAfterAutoTune(
                    configuration: initial,
                    modelPath: modelPath,
                    contextSize: contextSize,
                    memoryGuard: memoryGuard
                )
                journal.active = initial
                benchStatus = "Auto-tune annullato; configurazione iniziale ripristinata."
                _ = try? checkpointMachineAutoTune(
                    journal,
                    status: "CANCELLED_RESTORED",
                    pendingLabel: nil,
                    pendingConfiguration: nil,
                    note: "Configurazione e motore iniziali ripristinati.",
                    validated: false
                )
            } catch {
                phase = .failed(error.localizedDescription)
                benchStatus = "Annullato; ripristino motore fallito: \(error.localizedDescription)"
                _ = try? checkpointMachineAutoTune(
                    journal,
                    status: "CANCELLED_RESTORE_FAILED",
                    pendingLabel: nil,
                    pendingConfiguration: nil,
                    note: error.localizedDescription,
                    validated: false
                )
            }
            benchSucceeded = false
        } catch {
            if !memoryGuard.hasStartedTeardown,
               service != nil,
               loadedEngineSignature == baselineSignature {
                // A pre-teardown failure keeps the loaded root. If the prompt-
                // restoring probe ran, the next send rebuilds conversation KV.
                applyMachineAutoTuneEnvironment(initial)
                journal.current = initial
                journal.active = initial
                phase = .ready
                benchSucceeded = false
                benchStatus = "Auto-tune non avviato: \(error.localizedDescription). Motore root mantenuto."
                autoTuneReportURL = try? checkpointMachineAutoTune(
                    journal,
                    status: "FAILED_ROOT_RETAINED",
                    pendingLabel: nil,
                    pendingConfiguration: nil,
                    note: error.localizedDescription,
                    validated: false
                )
                benchProgressTotal = max(benchProgressDone, 1)
                benchRunning = false
                return
            }
            autoTuneLog("INTERROTTO: \(error.localizedDescription)")
            journal.current = current
            journal.trials = trials
            if let transactionID = activePersistenceTransactionID {
                do {
                    try transactionStore.rollback(
                        transactionID: transactionID,
                        reason: error.localizedDescription
                    ) {
                        self.persistMachineAutoTuneConfiguration(initial)
                    }
                    activePersistenceTransactionID = nil
                    journal.persisted = initial
                } catch let rollbackError {
                    persistMachineAutoTuneConfiguration(initial)
                    journal.persisted = initial
                    autoTuneLog("Rollback transazione incompleto; sarà ritentato all'avvio: \(rollbackError.localizedDescription)")
                }
            }
            applyMachineAutoTuneEnvironment(initial)
            do {
                let reportURL = try checkpointMachineAutoTune(
                    journal,
                    status: "FAILED_RESTORING",
                    pendingLabel: "Ripristino configurazione iniziale",
                    pendingConfiguration: initial,
                    note: error.localizedDescription,
                    validated: false
                )
                autoTuneReportURL = reportURL
            } catch {
                autoTuneLog("Impossibile scrivere il report parziale: \(error.localizedDescription)")
            }
            journal.active = nil
            do {
                try await installChatEngineAfterAutoTune(
                    configuration: initial,
                    modelPath: modelPath,
                    contextSize: contextSize,
                    memoryGuard: memoryGuard
                )
                journal.active = initial
                _ = try? checkpointMachineAutoTune(
                    journal,
                    status: "FAILED_RESTORED",
                    pendingLabel: nil,
                    pendingConfiguration: nil,
                    note: error.localizedDescription,
                    validated: false
                )
            } catch {
                phase = .failed(error.localizedDescription)
                autoTuneLog("Ripristino del motore fallito: \(error.localizedDescription)")
                _ = try? checkpointMachineAutoTune(
                    journal,
                    status: "FAILED_RESTORE_FAILED",
                    pendingLabel: nil,
                    pendingConfiguration: nil,
                    note: error.localizedDescription,
                    validated: false
                )
            }
            benchSucceeded = false
            benchStatus = "Auto-tune fallito: \(error.localizedDescription)"
        }

        benchProgressTotal = max(benchProgressDone, 1)
        benchRunning = false
    }

    private func evaluateMachineAutoTuneCandidate(
        incumbent: MachineAutoTuneConfiguration,
        candidate: MachineAutoTuneConfiguration,
        pass: Int,
        parameter: String,
        fromValue: Int,
        toValue: Int,
        rootQuality: MachineAutoTuneQualitySignature,
        policy: MachineAutoTunePolicy,
        modelPath: String,
        contextSize: Int,
        frozenUsage: Data?,
        journal: GUIAutoTuneJournal,
        memoryGuard: GUIAutoTuneMemoryGuard,
        benchmarkLedger: GUIAutoTuneBenchmarkLedger
    ) async throws -> GUIAutoTuneCandidate {
        guard let record = benchmarkLedger.observation(for: incumbent) else {
            throw GUIAutoTuneError.baselineRejected([
                "record incumbent assente per \(machineAutoTuneSummary(incumbent))"
            ])
        }
        if let cachedFailure = benchmarkLedger.failure(for: candidate) {
            throw GUIAutoTuneError.reloadFailed(
                "CACHE HIT: configurazione già respinta senza nuova misura (\(cachedFailure))"
            )
        }
        let measuredCandidate: MachineAutoTuneObservation
        if let cached = benchmarkLedger.observation(for: candidate) {
            measuredCandidate = cached
            autoTuneLog("CACHE HIT \(parameter)=\(toValue): configurazione già misurata, nessun reload.")
        } else {
            do {
                measuredCandidate = try await measureMachineAutoTune(
                    configuration: candidate,
                    label: "\(parameter) candidato=\(toValue)",
                    modelPath: modelPath,
                    contextSize: contextSize,
                    frozenUsage: frozenUsage,
                    journal: journal,
                    memoryGuard: memoryGuard
                )
            } catch {
                benchmarkLedger.recordFailure(error.localizedDescription, for: candidate)
                throw error
            }
            benchmarkLedger.record(measuredCandidate, for: candidate)
        }
        let result = MachineAutoTuneEvaluator.highWaterComparison(
            incumbent: record,
            candidate: measuredCandidate,
            rootQuality: rootQuality, policy: policy
        )
        autoTuneLog("\(result.qualified ? "RECORD" : "REJECT") \(parameter) " +
                    "\(fromValue)→\(toValue) \(ratioDescription(result))" +
                    (result.reasons.isEmpty ? "" : " — " + result.reasons.joined(separator: "; ")))
        return GUIAutoTuneCandidate(
            configuration: candidate,
            value: toValue,
            evaluation: result,
            trial: GUIAutoTuneTrial(
                pass: pass, parameter: parameter, fromValue: fromValue,
                toValue: toValue, screenPassed: result.qualified,
                comparisonMode: "cached-high-water",
                balancedPrimaryRatio: result.balancedPrimaryRatio,
                balancedSecondaryRatio: result.balancedSecondaryRatio,
                transitionQualityExact: result.transitionQualityExact,
                cumulativeQualityExact: result.cumulativeQualityExact,
                selected: false, reasons: result.reasons
            )
        )
    }

    /// Measures the already loaded, warmed root exactly once. The probe uses
    /// the same synthetic 96/32 workload as candidates and restores the active
    /// prompt. Synthetic KV cannot preserve an existing conversation, so every
    /// probe exit marks a non-empty chat for history re-prefill on its next send.
    private func measureLoadedMachineAutoTuneBaseline(
        configuration: MachineAutoTuneConfiguration,
        label: String,
        journal: GUIAutoTuneJournal,
        memoryGuard: GUIAutoTuneMemoryGuard
    ) async throws -> MachineAutoTuneObservation {
        try Task.checkCancellation()
        guard let svc = service,
              loadedEngineSignature == memoryGuard.baselineSignature else {
            throw GUIAutoTuneError.reloadFailed("motore root caricato non disponibile")
        }
        if Self.hasRecentPartialDownload(near: journal.modelPath) {
            throw GUIAutoTuneError.activeDownload
        }
        switch ProcessInfo.processInfo.thermalState {
        case .serious, .critical:
            throw GUIAutoTuneError.thermalPressure
        default:
            break
        }

        let before = MemoryInfo.pressureSnapshot()
        guard before.freePercent != nil, before.swapoutsBytes != nil else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        let settlement = try await waitForMachineAutoTuneSwapSettlement(
            snapshotBeforeLoad: before,
            configuration: configuration,
            label: label,
            memoryGuard: memoryGuard,
            allowCancellation: true
        )
        benchStatus = "Auto-tune: \(label) — unica misura greedy + qualità…"
        var probeStarted = false
        defer {
            if probeStarted {
                enginePrimed = messages.isEmpty
            }
        }
        probeStarted = true
        let point = try await svc.machineAutoTuneProbe(
            contextTokens: 96,
            genTokens: 32,
            captureQuality: true
        )
        let end = MemoryInfo.pressureSnapshot()
        guard let quality = point.qualitySignature,
              let freeBefore = before.freePercent,
              let freeAtStart = settlement.snapshot.freePercent,
              let freeAtEnd = end.freePercent,
              let swapBefore = before.swapoutsBytes,
              let swapAtStart = settlement.snapshot.swapoutsBytes,
              let swapAtEnd = end.swapoutsBytes else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        let windows = MachineAutoTuneSwapWindows(
            beforeLoadBytes: swapBefore,
            steadyStateStartBytes: swapAtStart,
            steadyStateEndBytes: swapAtEnd
        )
        guard let setupSwapMiB = windows.loadMiB,
              let steadySwapMiB = windows.policySwapoutMiB else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        let freePercent = min(freeBefore, min(freeAtStart, freeAtEnd))
        let steadySpeeds = point.genSpeeds.count > 12
            ? Array(point.genSpeeds.dropFirst(4))
            : point.genSpeeds
        guard let primary = MachineAutoTuneEvaluator.median(steadySpeeds),
              primary > 0 else {
            throw GUIAutoTuneError.reloadFailed("nessuna misura decode valida")
        }
        let half = steadySpeeds.count / 2
        let head = MachineAutoTuneEvaluator.median(Array(steadySpeeds.prefix(half))) ?? primary
        let tail = MachineAutoTuneEvaluator.median(Array(steadySpeeds.suffix(half))) ?? primary
        let stability = head > 0 ? tail / head : 0
        memoryGuard.record(freePercent, for: configuration)

        benchProgressDone += 1
        autoTuneLog(String(
            format: "%@: decode mediana %.3f t/s · prefill %.3f t/s · coda/testa %.1f%% · " +
                    "RAM libera %.1f%% · swap steady %.1f MiB · setup %.1f MiB " +
                    "(motore caldo, assestato %.1fs, %.1f MiB/s)",
            label, primary, point.prefillTps, stability * 100, freePercent,
            steadySwapMiB, setupSwapMiB, settlement.elapsedSeconds,
            settlement.lastRateMiBPerSecond
        ))
        autoTuneReportURL = try checkpointMachineAutoTune(
            journal,
            status: "MEASURED",
            pendingLabel: nil,
            pendingConfiguration: nil,
            note: "Root calda misurata una sola volta con probe che ripristina il prompt; " +
                  "una chat non vuota verrà riprefillata al prossimo invio.",
            validated: false
        )
        benchStatus = "Auto-tune \(benchProgressDone)/≤\(benchProgressTotal): \(label)"
        return MachineAutoTuneObservation(
            primaryTps: primary,
            secondaryTps: point.prefillTps,
            stability: stability,
            memoryFreePercent: freePercent,
            swapoutMiB: steadySwapMiB,
            quality: quality
        )
    }

    private func measureMachineAutoTune(
        configuration: MachineAutoTuneConfiguration,
        label: String,
        modelPath: String,
        contextSize: Int,
        frozenUsage: Data?,
        journal: GUIAutoTuneJournal,
        memoryGuard: GUIAutoTuneMemoryGuard
    ) async throws -> MachineAutoTuneObservation {
        try Task.checkCancellation()
        guard let memoryEnvelope = memoryGuard.envelope else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        let rootResidentDelta = estimatedMachineAutoTuneResidentDeltaBytes(
            from: journal.initial,
            to: configuration,
            memoryGuard: memoryGuard
        )
        guard memoryEnvelope.allowsResidentDeltaBytes(rootResidentDelta) else {
            throw GUIAutoTuneError.insufficientMemoryHeadroom(String(
                format: "modalità low-RAM: il candidato aggiunge %.1f MiB residenti " +
                        "rispetto alla baseline (vietato prima del teardown)",
                Double(rootResidentDelta) / 1_048_576
            ))
        }
        if Self.hasRecentPartialDownload(near: modelPath) {
            throw GUIAutoTuneError.activeDownload
        }
        switch ProcessInfo.processInfo.thermalState {
        case .serious, .critical:
            benchStatus = "Auto-tune in pausa: raffreddamento termico…"
            try await Task.sleep(nanoseconds: 10_000_000_000)
            switch ProcessInfo.processInfo.thermalState {
            case .serious, .critical: throw GUIAutoTuneError.thermalPressure
            default: break
            }
        default: break
        }

        do {
            autoTuneReportURL = try checkpointMachineAutoTune(
                journal,
                status: "LOAD_PENDING",
                pendingLabel: label,
                pendingConfiguration: configuration,
                note: "Checkpoint atomico scritto prima del teardown e del prossimo init.",
                validated: false
            )
        } catch {
            throw GUIAutoTuneError.journalFailed(error.localizedDescription)
        }

        let loadedConfiguration = memoryGuard.loadedConfiguration
        let loadedSnapshot = MemoryInfo.pressureSnapshot()
        if let loadedConfiguration,
           let loadedFree = loadedSnapshot.freePercent {
            memoryGuard.record(loadedFree, for: loadedConfiguration)
        }

        benchStatus = "Auto-tune: \(label) — teardown…"
        memoryGuard.markTeardownStarted()
        phase = .loading
        loadFraction = 0
        loadStage = label
        if let previousService = service {
            await previousService.quiesceForTeardown()
        }
        service = nil
        loadedEngineSignature = nil
        info = nil
        journal.active = nil
        memoryGuard.loadedConfiguration = nil
        try await Task.sleep(nanoseconds: 4_000_000_000)
        try Task.checkCancellation()
        applyMachineAutoTuneEnvironment(configuration)

        let memoryBefore = MemoryInfo.pressureSnapshot()
        guard memoryBefore.freePercent != nil, memoryBefore.swapoutsBytes != nil else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        try preflightMachineAutoTuneLoad(
            configuration: configuration,
            replacing: loadedConfiguration,
            snapshot: memoryBefore,
            memoryGuard: memoryGuard
        )
        benchStatus = "Auto-tune: \(label) — reload e warmup…"
        let slots = configuration.value(for: .expertCacheSlots) ?? expertCacheSlots
        let svc = try await makeMachineAutoTuneService(
            modelPath: modelPath,
            contextSize: contextSize,
            expertCacheSlots: slots,
            frozenUsage: frozenUsage
        )
        if Task.isCancelled {
            await svc.quiesceForTeardown()
            throw CancellationError()
        }
        do {
            try preflightMachineAutoTuneWarmup(
                configuration: configuration,
                snapshotBeforeInit: memoryBefore,
                memoryGuard: memoryGuard
            )
        } catch {
            // The candidate actor already owns model-sized Metal/VM buffers,
            // even though it has not yet been published in `service`.
            await svc.quiesceForTeardown()
            throw error
        }
        service = svc
        loadedEngineSignature = memoryGuard.signature(for: configuration)
        journal.active = configuration
        memoryGuard.loadedConfiguration = configuration
        guard await svc.warmup() else {
            await svc.quiesceForTeardown()
            throw GUIAutoTuneError.reloadFailed("warmup di prova fallito")
        }
        try Task.checkCancellation()
        _ = try await svc.benchmark(contextTokens: 16, genTokens: 4, greedy: true)

        let settlement = try await waitForMachineAutoTuneSwapSettlement(
            snapshotBeforeLoad: memoryBefore,
            configuration: configuration,
            label: label,
            memoryGuard: memoryGuard,
            allowCancellation: true
        )

        benchStatus = "Auto-tune: \(label) — misura greedy + qualità…"
        let point = try await svc.benchmark(
            contextTokens: 96,
            genTokens: 32,
            greedy: true,
            captureQuality: true
        )
        // End the policy window before quiesce/teardown: draining the engine is
        // lifecycle work and must not be charged to steady decode.
        let steadyEnd = MemoryInfo.pressureSnapshot()
        guard let quality = point.qualitySignature else {
            throw GUIAutoTuneError.missingQualityTrace
        }
        await svc.quiesceForTeardown()
        let memoryAfter = MemoryInfo.pressureSnapshot()
        guard let freeAfter = memoryAfter.freePercent,
              let freeAtStart = settlement.snapshot.freePercent,
              let freeAtEnd = steadyEnd.freePercent,
              let swapBefore = memoryBefore.swapoutsBytes,
              let swapAtStart = settlement.snapshot.swapoutsBytes,
              let swapAtEnd = steadyEnd.swapoutsBytes else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        let swapWindows = MachineAutoTuneSwapWindows(
            beforeLoadBytes: swapBefore,
            steadyStateStartBytes: swapAtStart,
            steadyStateEndBytes: swapAtEnd
        )
        guard let setupSwapMiB = swapWindows.loadMiB,
              let steadySwapMiB = swapWindows.policySwapoutMiB else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        let freePercent = min(freeAfter, min(freeAtStart, freeAtEnd))

        let steadySpeeds = point.genSpeeds.count > 12
            ? Array(point.genSpeeds.dropFirst(4))
            : point.genSpeeds
        guard let primary = MachineAutoTuneEvaluator.median(steadySpeeds),
              primary > 0 else {
            throw GUIAutoTuneError.reloadFailed("nessuna misura decode valida")
        }
        let half = steadySpeeds.count / 2
        let head = MachineAutoTuneEvaluator.median(Array(steadySpeeds.prefix(half))) ?? primary
        let tail = MachineAutoTuneEvaluator.median(Array(steadySpeeds.suffix(half))) ?? primary
        let stability = head > 0 ? tail / head : 0
        memoryGuard.record(freePercent, for: configuration)

        benchProgressDone += 1
        let metrics = String(
            format: "%@: decode mediana %.3f t/s · prefill %.3f t/s · coda/testa %.1f%% · " +
                    "RAM libera %.1f%% · swap steady %.1f MiB · setup %.1f MiB " +
                    "(assestato %.1fs, %.1f MiB/s)",
            label, primary, point.prefillTps, stability * 100, freePercent,
            steadySwapMiB, setupSwapMiB, settlement.elapsedSeconds,
            settlement.lastRateMiBPerSecond
        )
        autoTuneLog(metrics)
        do {
            autoTuneReportURL = try checkpointMachineAutoTune(
                journal,
                status: "MEASURED",
                pendingLabel: nil,
                pendingConfiguration: nil,
                note: "Misura completata; setup e swap steady salvati nel log.",
                validated: false
            )
        } catch {
            throw GUIAutoTuneError.journalFailed(error.localizedDescription)
        }
        benchStatus = "Auto-tune \(benchProgressDone)/≤\(benchProgressTotal): \(label)"
        return MachineAutoTuneObservation(
            primaryTps: primary,
            secondaryTps: point.prefillTps,
            stability: stability,
            memoryFreePercent: freePercent,
            // The record comparison consumes only the bounded steady-state window.
            // Cold init/warmup churn remains visible in the line above.
            swapoutMiB: steadySwapMiB,
            quality: quality
        )
    }

    private func makeMachineAutoTuneService(
        modelPath: String,
        contextSize: Int,
        expertCacheSlots: Int,
        frozenUsage: Data?
    ) async throws -> InferenceService {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let service = try InferenceService(
                        modelPath: modelPath,
                        contextSize: contextSize,
                        systemPrompt: nil,
                        expertCacheSlots: expertCacheSlots > 0 ? expertCacheSlots : nil,
                        frozenUsageSeed: frozenUsage
                    )
                    continuation.resume(returning: service)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Reads the few decoder facts needed by the RAM/search policy and returns
    /// without leaking a strong service reference into the caller's search
    /// frame. All model-sized ownership ends at this helper's return boundary.
    private func captureMachineAutoTuneRuntimeGeometry() async throws
        -> GUIAutoTuneRuntimeGeometry {
        guard let activeService = service else {
            throw GUIAutoTuneError.reloadFailed(
                "motore non disponibile dopo l'attesa del setup"
            )
        }
        let cacheBytes = await activeService.expertCacheBudgetBytesPerBaseSlot()
        let denseBytes = await activeService.denseStagingBytesPerAheadSlot()
        let preadEffective = await activeService.expertPreadSplitIsEffective()
        return GUIAutoTuneRuntimeGeometry(
            expertCacheBytesPerBaseSlot: cacheBytes,
            denseStagingBytesPerAheadSlot: denseBytes,
            expertPreadSplitIsEffective: preadEffective
        )
    }

    private func installChatEngineAfterAutoTune(
        configuration: MachineAutoTuneConfiguration,
        modelPath: String,
        contextSize: Int,
        memoryGuard: GUIAutoTuneMemoryGuard,
        allowCancellation: Bool = false
    ) async throws {
        // Cancellation requests stop the search, but restoration itself must
        // finish so the UI never remains `.ready` with a nil/stale service.
        applyMachineAutoTuneEnvironment(configuration)
        let loadedConfiguration = memoryGuard.loadedConfiguration
        if let loadedConfiguration,
           let freePercent = MemoryInfo.pressureSnapshot().freePercent {
            memoryGuard.record(freePercent, for: loadedConfiguration)
        }
        memoryGuard.markTeardownStarted()
        if let previousService = service {
            await previousService.quiesceForTeardown()
        }
        service = nil
        loadedEngineSignature = nil
        info = nil
        memoryGuard.loadedConfiguration = nil
        phase = .loading
        loadStage = "Ripristino motore chat"
        // This path also runs after Stop, from an already-cancelled search task.
        // Keep the Metal/VM release grace uncancelled just like the restoration
        // warmup, otherwise cancellation collapses the delay to zero and can
        // overlap two model-sized allocations on a tight-RAM Mac.
        if allowCancellation {
            try await Task.sleep(nanoseconds: 4_000_000_000)
        } else {
            await Task.detached(priority: .utility) {
                try? await Task.sleep(nanoseconds: 4_000_000_000)
            }.value
        }
        let memoryBefore = MemoryInfo.pressureSnapshot()
        try preflightMachineAutoTuneLoad(
            configuration: configuration,
            replacing: loadedConfiguration,
            snapshot: memoryBefore,
            memoryGuard: memoryGuard
        )
        let slots = configuration.value(for: .expertCacheSlots) ?? expertCacheSlots
        let svc: InferenceService
        do {
            svc = try await makeMachineAutoTuneService(
                modelPath: modelPath,
                contextSize: contextSize,
                expertCacheSlots: slots,
                frozenUsage: nil
            )
        } catch {
            throw GUIAutoTuneError.reloadFailed(error.localizedDescription)
        }
        if allowCancellation, Task.isCancelled {
            await svc.quiesceForTeardown()
            throw CancellationError()
        }
        do {
            try preflightMachineAutoTuneWarmup(
                configuration: configuration,
                snapshotBeforeInit: memoryBefore,
                memoryGuard: memoryGuard
            )
        } catch {
            await svc.quiesceForTeardown()
            throw error
        }
        await svc.setDiskKV(
            directory: diskKVEnabled ? Self.diskKVDirectory : nil,
            budgetTokens: diskKVBudgetKTok * 1000
        )
        memoryGuard.loadedConfiguration = configuration
        // Restoration may run from the already-cancelled search task. Start
        // warmup from a detached, explicitly uncancelled task so Stop cannot
        // publish a cold engine as ready.
        let warmupSucceeded = await Task.detached(priority: .userInitiated) {
            await svc.warmup()
        }.value
        guard warmupSucceeded else {
            await svc.quiesceForTeardown()
            throw GUIAutoTuneError.reloadFailed("warmup del motore chat fallito")
        }
        let loadedInfo = await svc.modelInfo()
        service = svc
        loadedEngineSignature = memoryGuard.signature(for: configuration)
        info = loadedInfo
        activate(activeSessionId)
        guard await waitForEngineSetup() else {
            await svc.quiesceForTeardown()
            service = nil
            loadedEngineSignature = nil
            info = nil
            memoryGuard.loadedConfiguration = nil
            throw GUIAutoTuneError.reloadFailed(
                "warmup del ruolo attivo fallito"
            )
        }
        do {
            // The benchmark service uses the frozen synthetic/root workload,
            // while the published chat engine additionally installs DiskKV and
            // warms the real agent profile. First let that cold work settle,
            // then gate the same 96/32 prompt-restoring inference workload.
            let settlement = try await waitForMachineAutoTuneSwapSettlement(
                snapshotBeforeLoad: memoryBefore,
                configuration: configuration,
                label: "motore chat pronto",
                memoryGuard: memoryGuard,
                allowCancellation: allowCancellation
            )
            if allowCancellation {
                _ = try await svc.machineAutoTuneProbe(
                    contextTokens: 96,
                    genTokens: 32
                )
            } else {
                // Root restoration can be entered by an already-cancelled
                // task; run the mandatory probe from a fresh task in that case.
                _ = try await Task.detached(priority: .userInitiated) {
                    try await svc.machineAutoTuneProbe(
                        contextTokens: 96,
                        genTokens: 32
                    )
                }.value
            }
            let probeEnd = MemoryInfo.pressureSnapshot()
            try validateMachineAutoTuneReadyMemory(
                configuration: configuration,
                snapshotBeforeInit: memoryBefore,
                steadyStateStart: settlement.snapshot,
                steadyStateEnd: probeEnd,
                settlement: settlement,
                memoryGuard: memoryGuard
            )
        } catch {
            await svc.quiesceForTeardown()
            service = nil
            loadedEngineSignature = nil
            info = nil
            memoryGuard.loadedConfiguration = nil
            throw error
        }
        if let freePercent = MemoryInfo.pressureSnapshot().freePercent {
            memoryGuard.record(freePercent, for: configuration)
        }
        phase = .ready
        benchProgressDone += 1
    }

    var machineAutoTuneConfiguration: MachineAutoTuneConfiguration {
        MachineAutoTuneConfiguration(settings: [
            .multiQuantCache: multiQuantCacheEnabled ? 1 : 0,
            .expertCacheSlots: expertCacheSlots,
            .expertCacheUniform: expertCacheUniform ? 1 : 0,
            .preadSplit: preadSplit,
            .denseAhead: denseAhead,
            .asyncFFN: asyncFFNEnabled ? 1 : 0,
            .expertLookahead: expertLookahead,
            .q8NSG: q8NSG,
            .moeNSG: moeNSG,
            .denseQ4NSG: denseQ4NSG,
        ])
    }

    /// Every DS4 environment value that remains fixed throughout the search.
    /// Unknown/new DS4 knobs are included automatically, while controls in the
    /// typed manifest are represented by `MachineAutoTuneConfiguration`.
    private var machineAutoTuneFixedEnvironment: [String: String] {
        let tuned: Set<String> = [
            "DS4_MULTI_QUANT_CACHE", "DS4_EXPERT_CACHE_SLOTS",
            "DS4_EXPERT_CACHE_UNIFORM", "DS4_PREAD_SPLIT",
            "DS4_DENSE_AHEAD", "DS4_ASYNC_FFN", "DS4_EXPERT_LOOKAHEAD",
            "DS4_Q8_NSG", "DS4_MOE_NSG", "DS4_DENSE_Q4_NSG",
        ]
        return ProcessInfo.processInfo.environment.reduce(into: [:]) { result, entry in
            guard entry.key.hasPrefix("DS4_"), !tuned.contains(entry.key) else { return }
            result[entry.key] = entry.value
        }
    }

    func machineAutoTuneEngineSignature() -> LoadedEngineSignature {
        let resolvedPath = URL(fileURLWithPath: modelPath)
            .standardizedFileURL.resolvingSymlinksInPath().path
        return LoadedEngineSignature(
            modelPath: resolvedPath,
            contextSize: contextSize,
            tuning: machineAutoTuneConfiguration,
            fixedEnvironment: machineAutoTuneFixedEnvironment
        )
    }

    private func applyMachineAutoTuneEnvironment(_ configuration: MachineAutoTuneConfiguration) {
        let envNames: [MachineAutoTuneKnob: String] = [
            .multiQuantCache: "DS4_MULTI_QUANT_CACHE",
            .expertCacheSlots: "DS4_EXPERT_CACHE_SLOTS",
            .expertCacheUniform: "DS4_EXPERT_CACHE_UNIFORM",
            .preadSplit: "DS4_PREAD_SPLIT",
            .denseAhead: "DS4_DENSE_AHEAD",
            .asyncFFN: "DS4_ASYNC_FFN",
            .expertLookahead: "DS4_EXPERT_LOOKAHEAD",
            .q8NSG: "DS4_Q8_NSG",
            .moeNSG: "DS4_MOE_NSG",
            .denseQ4NSG: "DS4_DENSE_Q4_NSG",
        ]
        for knob in MachineAutoTuneKnob.allCases {
            guard let value = configuration.value(for: knob),
                  let name = envNames[knob] else { continue }
            _ = setenv(name, String(value), 1)
        }
        // Fixed feasibility/quality constraints: this tuner never trades away
        // the constant-memory KV ring or changes active experts/quantization.
        _ = setenv("DS4_RAW_RING", "1", 1)
    }

    private func persistMachineAutoTuneConfiguration(
        _ configuration: MachineAutoTuneConfiguration
    ) {
        if let value = configuration.value(for: .multiQuantCache) {
            multiQuantCacheEnabled = value != 0
        }
        if let value = configuration.value(for: .expertCacheSlots) {
            expertCacheSlots = value
        }
        if let value = configuration.value(for: .expertCacheUniform) {
            expertCacheUniform = value != 0
        }
        if let value = configuration.value(for: .preadSplit) { preadSplit = value }
        if let value = configuration.value(for: .denseAhead) { denseAhead = value }
        if let value = configuration.value(for: .asyncFFN) { asyncFFNEnabled = value != 0 }
        if let value = configuration.value(for: .expertLookahead) { expertLookahead = value }
        if let value = configuration.value(for: .q8NSG) { q8NSG = value }
        if let value = configuration.value(for: .moeNSG) { moeNSG = value }
        if let value = configuration.value(for: .denseQ4NSG) { denseQ4NSG = value }
        rawRingEnabled = true
        applyMachineAutoTuneEnvironment(configuration)
    }

    private func machineAutoTuneSkipReason(
        parameter: MachineAutoTuneParameter,
        configuration: MachineAutoTuneConfiguration,
        frozenUsage: Data?,
        memoryGuard: GUIAutoTuneMemoryGuard
    ) -> String? {
        switch parameter.knob {
        case .multiQuantCache:
            return (configuration.value(for: .expertCacheSlots) ?? 0) > 0
                ? nil : "richiede una cache esperti attiva"
        case .expertCacheUniform:
            if (configuration.value(for: .expertCacheSlots) ?? 0) == 0 {
                return "richiede una cache esperti attiva"
            }
            return frozenUsage == nil ? "richiede un usage profile congelabile" : nil
        case .preadSplit:
            return memoryGuard.expertPreadSplitIsEffective
                ? nil
                : "il backend caricato non usa fill esperti GGUF F_NOCACHE; lo split non è effettivo"
        case .denseAhead:
            return denseStreamEnabled ? nil : "richiede dense streaming"
        case .expertLookahead:
            if (configuration.value(for: .expertCacheSlots) ?? 0) == 0 {
                return "richiede una cache esperti attiva"
            }
            if (configuration.value(for: .multiQuantCache) ?? 0) == 0 {
                return "richiede la mixed-quant cache"
            }
            return frozenUsage == nil ? "richiede un usage profile congelabile" : nil
        case .denseQ4NSG:
            return denseQ4Enabled ? nil : "richiede i kernel Q4 densi"
        case .expertCacheSlots, .asyncFFN, .q8NSG, .moeNSG:
            return nil
        }
    }

    /// Non-monotonic hardware grids remain full sweeps when memory is
    /// comfortable. Custom memory-risk sweeps are narrowed near RAM pressure;
    /// the standard expert-cache ladder uses `.walk` instead.
    private func machineAutoTuneSweepCandidates(
        parameter: MachineAutoTuneParameter,
        fromValue: Int,
        configuration: MachineAutoTuneConfiguration,
        memoryGuard: GUIAutoTuneMemoryGuard
    ) -> [Int] {
        let ordered = parameter.values
            .filter { $0 != fromValue }
            .sorted {
                let lhsDistance = abs($0 - fromValue)
                let rhsDistance = abs($1 - fromValue)
                return lhsDistance == rhsDistance ? $0 < $1 : lhsDistance < rhsDistance
            }
        if parameter.memoryRisk,
           memoryGuard.isConstrained,
           let rootValue = memoryGuard.baselineSignature.tuning.value(for: parameter.knob) {
            // Do not reduce a low-RAM sweep to one neighbour per pass: the
            // smaller cache points are precisely what may recover headroom.
            // Values up to the known-loadable root remain eligible even after
            // a reduction; anything above it is forbidden by the envelope.
            let bounded = ordered.filter { $0 <= rootValue }
            autoTuneLog(
                "LOW-RAM \(parameter.knob.rawValue): sweep entro baseline " +
                "\(rootValue) → \(bounded.map(String.init).joined(separator: ","))"
            )
            return bounded
        }
        guard parameter.memoryRisk,
              let observedFree = memoryGuard.observedFree(for: configuration),
              observedFree < 18 else {
            return ordered
        }

        let lower = parameter.values.filter { $0 < fromValue }.max()
        let upper = parameter.values.filter { $0 > fromValue }.min()
        let neighbours = Set([lower, upper].compactMap { $0 })
        let gradual = ordered.filter { neighbours.contains($0) }
        autoTuneLog(String(
            format: "HEADROOM %.1f%%: sweep %@ limitata ai gradini adiacenti %@; " +
                    "gli incrementi maggiori richiedono una promozione intermedia.",
            observedFree,
            parameter.knob.rawValue,
            gradual.map(String.init).joined(separator: ",")
        ))
        return gradual
    }

    private func isMachineAutoTuneHeadroomError(_ error: Error) -> Bool {
        guard let tuneError = error as? GUIAutoTuneError else { return false }
        if case .insufficientMemoryHeadroom = tuneError { return true }
        return false
    }

    /// Signed estimate of the resident-memory change relative to the engine
    /// whose post-load pressure was observed. The slot coefficient is the exact
    /// cache budget derived from the loaded GGUF, so Pro/Flash and mixed-quant
    /// layouts are handled without filename heuristics.
    private func estimatedMachineAutoTuneResidentDeltaBytes(
        from: MachineAutoTuneConfiguration?,
        to: MachineAutoTuneConfiguration,
        memoryGuard: GUIAutoTuneMemoryGuard
    ) -> Int64 {
        guard let from else { return 0 }
        let oldSlots = Int64(from.value(for: .expertCacheSlots) ?? 0)
        let newSlots = Int64(to.value(for: .expertCacheSlots) ?? 0)
        let cacheBytes = Int64(clamping: memoryGuard.expertCacheBudgetBytesPerBaseSlot)
        let slotDelta = (newSlots - oldSlots) * cacheBytes

        let oldAhead = Int64(from.value(for: .denseAhead) ?? 1)
        let newAhead = Int64(to.value(for: .denseAhead) ?? 1)
        let denseBytes = Int64(clamping: memoryGuard.denseStagingBytesPerAheadSlot)
        let denseDelta = (newAhead - oldAhead) * denseBytes
        return slotDelta + denseDelta
    }

    /// Fails before constructing another model-sized service. Unknown/growing
    /// candidates require the conservative 12%/1.5 GiB transient reserve.
    /// The exact root, and constrained candidates whose resident estimate does
    /// not exceed it, use the immutable root's 512 MiB known-loadable reserve;
    /// otherwise a successful teardown could make restoration impossible.
    private func preflightMachineAutoTuneLoad(
        configuration: MachineAutoTuneConfiguration,
        replacing reference: MachineAutoTuneConfiguration?,
        snapshot: MemoryInfo.PressureSnapshot,
        memoryGuard: GUIAutoTuneMemoryGuard
    ) throws {
        guard let freePercent = snapshot.freePercent,
              snapshot.swapoutsBytes != nil else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }

        let physicalBytes = Double(MemoryInfo.physicalBytes)
        let deltaBytes = estimatedMachineAutoTuneResidentDeltaBytes(
            from: reference,
            to: configuration,
            memoryGuard: memoryGuard
        )
        let freeBytes = physicalBytes * freePercent / 100
        let envelope = memoryGuard.envelope
        let rootDelta = estimatedMachineAutoTuneResidentDeltaBytes(
            from: memoryGuard.baselineSignature.tuning,
            to: configuration,
            memoryGuard: memoryGuard
        )
        let positiveRootDelta = Double(max(0, rootDelta))
        let usesKnownLoadableBudget = configuration == memoryGuard.baselineSignature.tuning ||
            (envelope?.isConstrained == true &&
             envelope?.allowsResidentDeltaBytes(rootDelta) == true)
        let transientReserve: Double
        let minimumPreInitPercent: Double
        if usesKnownLoadableBudget, let envelope {
            transientReserve = Double(envelope.hardReserveBytes)
            minimumPreInitPercent = envelope.hardReservePercent
        } else {
            transientReserve = max(1.5 * 1_073_741_824, physicalBytes * 0.12)
            minimumPreInitPercent = 12
        }
        let requiredBeforeInit = transientReserve + positiveRootDelta
        guard freePercent >= minimumPreInitPercent,
              freeBytes >= requiredBeforeInit else {
            throw GUIAutoTuneError.insufficientMemoryHeadroom(String(
                format: "prima dell'init liberi %.1f%%/%.2f GiB; servono almeno " +
                        "%.1f%%/%.2f GiB (%@, delta candidato %.2f GiB)",
                freePercent,
                freeBytes / 1_073_741_824,
                minimumPreInitPercent,
                requiredBeforeInit / 1_073_741_824,
                usesKnownLoadableBudget ? "budget root noto" : "reserve transiente",
                positiveRootDelta / 1_073_741_824
            ))
        }

        let predictedFreePercent: Double?
        if configuration == memoryGuard.baselineSignature.tuning,
           let envelope = memoryGuard.envelope {
            // The original root was already live when the envelope was
            // captured. Do not let a later noisy/lower VM sample make that
            // same configuration impossible to restore after teardown.
            predictedFreePercent = envelope.baselineFreePercent
        } else if let observed = memoryGuard.observedFree(for: configuration) {
            predictedFreePercent = observed
        } else if let reference,
                  let observed = memoryGuard.observedFree(for: reference) {
            predictedFreePercent = observed - Double(deltaBytes) / physicalBytes * 100
        } else {
            predictedFreePercent = nil
        }
        let floor = memoryGuard.effectiveMinimumFreePercent
        if let predictedFreePercent, predictedFreePercent < floor {
            throw GUIAutoTuneError.insufficientMemoryHeadroom(String(
                format: "RAM post-load prevista %.1f%% (<floor %.1f%%) per %@",
                predictedFreePercent,
                floor,
                machineAutoTuneSummary(configuration)
            ))
        }
    }

    /// `InferenceService.init` may allocate more than the slot/dense estimate.
    /// Recheck pressure before warmup touches the remaining lazy pages. In the
    /// constrained mode a known, non-growing configuration may incur cold
    /// setup swap here; it is allowed only because the later settlement and
    /// steady-state windows still fail closed. Standard/unknown/growing loads
    /// retain the original 128 MiB setup cap.
    private func preflightMachineAutoTuneWarmup(
        configuration: MachineAutoTuneConfiguration,
        snapshotBeforeInit: MemoryInfo.PressureSnapshot,
        memoryGuard: GUIAutoTuneMemoryGuard
    ) throws {
        let afterInit = MemoryInfo.pressureSnapshot()
        guard let freePercent = afterInit.freePercent,
              let swapBefore = snapshotBeforeInit.swapoutsBytes,
              let swapAfter = afterInit.swapoutsBytes else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        guard let newSwapBytes = MachineAutoTuneSwapWindows.delta(
            from: swapBefore,
            to: swapAfter
        ).bytes else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        let newSwapMiB = Double(newSwapBytes)
            / MachineAutoTuneSwapWindows.bytesPerMiB
        let floor = memoryGuard.effectiveMinimumFreePercent
        let residentDelta = estimatedMachineAutoTuneResidentDeltaBytes(
            from: memoryGuard.baselineSignature.tuning,
            to: configuration,
            memoryGuard: memoryGuard
        )
        let allowsColdSetupSwap = memoryGuard.isConstrained && residentDelta <= 0
        guard freePercent >= floor,
              allowsColdSetupSwap ||
                newSwapMiB <= GUIAutoTuneSwapSettlementPolicy.maximumSetupSwapoutMiB else {
            throw GUIAutoTuneError.insufficientMemoryHeadroom(String(
                format: "dopo init e prima del warmup: RAM libera %.1f%% " +
                        "(floor %.1f%%), swap setup %.1f MiB (%@)",
                freePercent,
                floor,
                newSwapMiB,
                machineAutoTuneSummary(configuration)
            ))
        }
    }

    /// Waits for cold paging to stop before opening a policy measurement. Swap
    /// counters are system-wide and cumulative, so the barrier is deliberately
    /// bounded and fail-closed: two consecutive one-second windows must remain
    /// at or below 16 MiB/s, the immutable RAM floor must hold on every sample,
    /// and a counter reset is never treated as zero.
    private func waitForMachineAutoTuneSwapSettlement(
        snapshotBeforeLoad: MemoryInfo.PressureSnapshot,
        configuration: MachineAutoTuneConfiguration,
        label: String,
        memoryGuard: GUIAutoTuneMemoryGuard,
        allowCancellation: Bool
    ) async throws -> GUIAutoTuneSwapSettlement {
        let floor = memoryGuard.effectiveMinimumFreePercent
        var previous = MemoryInfo.pressureSnapshot()
        guard let initialFree = previous.freePercent,
              let beforeLoadSwap = snapshotBeforeLoad.swapoutsBytes,
              let initialSwap = previous.swapoutsBytes else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        guard initialFree >= floor else {
            throw GUIAutoTuneError.insufficientMemoryHeadroom(String(
                format: "prima dell'assestamento: RAM libera %.1f%% " +
                        "(<floor %.1f%%) per %@",
                initialFree,
                floor,
                machineAutoTuneSummary(configuration)
            ))
        }
        guard MachineAutoTuneSwapWindows.delta(
            from: beforeLoadSwap,
            to: initialSwap
        ).bytes != nil else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }

        var previousUptime = ProcessInfo.processInfo.systemUptime
        let startedUptime = previousUptime
        var quietIntervals = 0
        var lastRate = Double.infinity

        for interval in 1...GUIAutoTuneSwapSettlementPolicy.maximumIntervals {
            benchStatus = "Auto-tune: \(label) — assestamento swap " +
                "\(interval)/\(GUIAutoTuneSwapSettlementPolicy.maximumIntervals)…"
            if allowCancellation {
                try await Task.sleep(
                    nanoseconds: GUIAutoTuneSwapSettlementPolicy.sampleIntervalNanoseconds
                )
                try Task.checkCancellation()
            } else {
                // Restoration can run inside an already-cancelled search task.
                // A detached sleep keeps the root recovery barrier effective.
                await Task.detached(priority: .utility) {
                    try? await Task.sleep(
                        nanoseconds: GUIAutoTuneSwapSettlementPolicy.sampleIntervalNanoseconds
                    )
                }.value
            }

            let current = MemoryInfo.pressureSnapshot()
            let now = ProcessInfo.processInfo.systemUptime
            guard let freePercent = current.freePercent,
                  let previousSwap = previous.swapoutsBytes,
                  let currentSwap = current.swapoutsBytes else {
                throw GUIAutoTuneError.memoryCountersUnavailable
            }
            guard freePercent >= floor else {
                throw GUIAutoTuneError.insufficientMemoryHeadroom(String(
                    format: "durante assestamento: RAM libera %.1f%% " +
                            "(<floor %.1f%%) per %@",
                    freePercent,
                    floor,
                    machineAutoTuneSummary(configuration)
                ))
            }
            guard let deltaBytes = MachineAutoTuneSwapWindows.delta(
                from: previousSwap,
                to: currentSwap
            ).bytes else {
                throw GUIAutoTuneError.memoryCountersUnavailable
            }
            let elapsed = max(0.001, now - previousUptime)
            lastRate = Double(deltaBytes) / MachineAutoTuneSwapWindows.bytesPerMiB
                / elapsed
            if lastRate <= GUIAutoTuneSwapSettlementPolicy.maximumQuietRateMiBPerSecond {
                quietIntervals += 1
            } else {
                quietIntervals = 0
            }

            if quietIntervals >= GUIAutoTuneSwapSettlementPolicy.requiredQuietIntervals {
                guard MachineAutoTuneSwapWindows(
                    beforeLoadBytes: beforeLoadSwap,
                    steadyStateStartBytes: currentSwap,
                    steadyStateEndBytes: currentSwap
                ).loadMiB != nil else {
                    throw GUIAutoTuneError.memoryCountersUnavailable
                }
                return GUIAutoTuneSwapSettlement(
                    snapshot: current,
                    elapsedSeconds: now - startedUptime,
                    lastRateMiBPerSecond: lastRate
                )
            }
            previous = current
            previousUptime = now
        }

        throw GUIAutoTuneError.insufficientMemoryHeadroom(String(
            format: "swapout system-wide non assestato entro %d s " +
                    "(ultima finestra %.1f MiB/s, limite %.1f MiB/s) per %@",
            GUIAutoTuneSwapSettlementPolicy.maximumIntervals,
            lastRate,
            GUIAutoTuneSwapSettlementPolicy.maximumQuietRateMiBPerSecond,
            machineAutoTuneSummary(configuration)
        ))
    }

    /// Final adoption barrier after DiskKV, the selected agent and its usage-
    /// driven pools are live. Setup swap remains diagnostic; only the bounded
    /// prompt-restoring probe is subject to the same 128 MiB cap used by
    /// candidate promotion. RAM is checked at both ends and after prompt-state
    /// restoration performed by the probe.
    private func validateMachineAutoTuneReadyMemory(
        configuration: MachineAutoTuneConfiguration,
        snapshotBeforeInit: MemoryInfo.PressureSnapshot,
        steadyStateStart: MemoryInfo.PressureSnapshot,
        steadyStateEnd: MemoryInfo.PressureSnapshot,
        settlement: GUIAutoTuneSwapSettlement,
        memoryGuard: GUIAutoTuneMemoryGuard
    ) throws {
        let ready = MemoryInfo.pressureSnapshot()
        guard let freeAtStart = steadyStateStart.freePercent,
              let freeAtEnd = steadyStateEnd.freePercent,
              let freeReady = ready.freePercent,
              let swapBefore = snapshotBeforeInit.swapoutsBytes,
              let swapAtStart = steadyStateStart.swapoutsBytes,
              let swapAtEnd = steadyStateEnd.swapoutsBytes else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        let windows = MachineAutoTuneSwapWindows(
            beforeLoadBytes: swapBefore,
            steadyStateStartBytes: swapAtStart,
            steadyStateEndBytes: swapAtEnd
        )
        guard let setupSwapMiB = windows.loadMiB,
              let steadySwapMiB = windows.policySwapoutMiB else {
            throw GUIAutoTuneError.memoryCountersUnavailable
        }
        let freePercent = min(freeReady, min(freeAtStart, freeAtEnd))
        let floor = memoryGuard.effectiveMinimumFreePercent
        guard freePercent >= floor,
              steadySwapMiB <= MachineAutoTunePolicy().maximumSwapoutMiB else {
            throw GUIAutoTuneError.insufficientMemoryHeadroom(String(
                format: "motore chat pronto: RAM libera %.1f%% (floor %.1f%%), " +
                        "swap probe steady %.1f MiB (limite %.0f MiB), " +
                        "setup %.1f MiB (%@)",
                freePercent,
                floor,
                steadySwapMiB,
                MachineAutoTunePolicy().maximumSwapoutMiB,
                setupSwapMiB,
                machineAutoTuneSummary(configuration)
            ))
        }
        autoTuneLog(String(
            format: "Motore chat verificato: RAM libera %.1f%% · swap probe steady %.1f MiB · " +
                    "setup %.1f MiB (assestato %.1fs, %.1f MiB/s).",
            freePercent,
            steadySwapMiB,
            setupSwapMiB,
            settlement.elapsedSeconds,
            settlement.lastRateMiBPerSecond
        ))
    }

    private func failedMachineAutoTuneTrial(
        pass: Int,
        parameter: String,
        fromValue: Int,
        toValue: Int,
        error: Error
    ) -> GUIAutoTuneTrial {
        GUIAutoTuneTrial(
            pass: pass, parameter: parameter, fromValue: fromValue,
            toValue: toValue, screenPassed: false, comparisonMode: "cached-high-water",
            balancedPrimaryRatio: nil, balancedSecondaryRatio: nil,
            transitionQualityExact: false, cumulativeQualityExact: false,
            selected: false, reasons: [error.localizedDescription]
        )
    }

    private func ratioDescription(_ result: MachineAutoTuneEvaluationResult) -> String {
        let primary = (result.balancedPrimaryRatio ?? 0) * 100 - 100
        let secondary = (result.balancedSecondaryRatio ?? 0) * 100 - 100
        return String(format: "record decode %+.2f%% · prefill %+.2f%% · exact=%@",
                      primary, secondary,
                      result.cumulativeQualityExact ? "sì" : "no")
    }

    private func logMachineAutoTuneWalkDecision(
        parameter: String,
        walk: MachineAutoTuneDirectionalWalk
    ) {
        if walk.isStopped {
            autoTuneLog(
                "STOP \(parameter): primo gradino non migliorativo; " +
                "nessun valore successivo nella stessa direzione sarà provato."
            )
        } else {
            autoTuneLog(
                "FALLBACK \(parameter): il primo gradino verso l'alto non ha vinto; " +
                "provo soltanto il vicino inferiore."
            )
        }
    }

    private func autoTuneLog(_ line: String) {
        benchResults += line + "\n"
        DS4Log.info("autotune", line)
    }

    private func machineAutoTuneSummary(_ configuration: MachineAutoTuneConfiguration) -> String {
        MachineAutoTuneKnob.allCases.compactMap { knob in
            configuration.value(for: knob).map { "\(knob.rawValue)=\($0)" }
        }.joined(separator: " · ")
    }

    private func machineAutoTuneDictionary(
        _ configuration: MachineAutoTuneConfiguration
    ) -> [String: Int] {
        Dictionary(uniqueKeysWithValues: MachineAutoTuneKnob.allCases.compactMap { knob in
            configuration.value(for: knob).map { (knob.rawValue, $0) }
        })
    }

    private func makeMachineAutoTuneJournal(
        chip: String,
        ramGB: Int,
        modelPath: String,
        contextSize: Int,
        initial: MachineAutoTuneConfiguration,
        fixedEnvironment: [String: String],
        usageProfileHash: String?
    ) throws -> GUIAutoTuneJournal {
        let createdAt = Date()
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let suffix = UUID().uuidString.prefix(8).lowercased()
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar/autotune", isDirectory: true)
            .appendingPathComponent("\(formatter.string(from: createdAt))-\(suffix)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return GUIAutoTuneJournal(
            directory: base,
            createdAt: createdAt,
            chip: chip,
            ramGB: ramGB,
            modelPath: modelPath,
            contextSize: contextSize,
            initial: initial,
            fixedEnvironment: fixedEnvironment,
            usageProfileHash: usageProfileHash
        )
    }

    /// Writes every artifact atomically into one stable run directory. The
    /// JSON journal is updated before each load; `results.json` mirrors it so
    /// external report consumers never need a separate partial-result parser.
    private func checkpointMachineAutoTune(
        _ journal: GUIAutoTuneJournal,
        status: String,
        pendingLabel: String?,
        pendingConfiguration: MachineAutoTuneConfiguration?,
        note: String?,
        validated: Bool
    ) throws -> URL {
        let updatedAt = Date()
        let report = GUIAutoTuneReport(
            schema: 7,
            createdAt: journal.createdAt,
            updatedAt: updatedAt,
            status: status,
            chip: journal.chip,
            ramGB: journal.ramGB,
            modelPath: journal.modelPath,
            contextTokens: journal.contextSize,
            initial: machineAutoTuneDictionary(journal.initial),
            bestCandidate: machineAutoTuneDictionary(journal.current),
            activeConfiguration: journal.active.map(machineAutoTuneDictionary),
            persistedConfiguration: journal.persisted.map(machineAutoTuneDictionary),
            final: machineAutoTuneDictionary(journal.current),
            fixedEnvironment: journal.fixedEnvironment,
            usageProfileHash: journal.usageProfileHash,
            memoryEnvelope: journal.memoryEnvelope,
            pendingLabel: pendingLabel,
            pendingConfiguration: pendingConfiguration.map(machineAutoTuneDictionary),
            note: note,
            validated: validated,
            trials: journal.trials,
            log: benchResults
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let encoded = try encoder.encode(report)
        try encoded.write(to: journal.directory.appendingPathComponent("journal.json"),
                          options: .atomic)
        try encoded.write(to: journal.directory.appendingPathComponent("results.json"),
                          options: .atomic)

        var markdown = "# DwarfStar machine auto-tune\n\n"
        markdown += "- Stato: **\(status)**\n"
        markdown += "- Validato: **\(validated ? "sì" : "no")**\n"
        markdown += "- Aggiornato: \(updatedAt.formatted(.iso8601))\n"
        markdown += "- Mac: \(journal.chip), \(journal.ramGB) GB\n"
        markdown += "- Modello: `\((journal.modelPath as NSString).lastPathComponent)`\n"
        markdown += "- Contesto: \(journal.contextSize) token\n"
        markdown += "- Qualità: greedy, full-logit Float32 bit-exact, root cumulativa\n"
        markdown += "- Usage profile hash: `\(journal.usageProfileHash ?? "nessuno")`\n"
        if let envelope = journal.memoryEnvelope {
            markdown += String(
                format: "- RAM root: %.1f%% · modalità: %@ · floor immutabile: %.1f%% · riserva root: %.0f MiB\n",
                envelope.baselineFreePercent,
                envelope.isConstrained ? "low-RAM" : "standard",
                envelope.effectiveMinimumFreePercent,
                Double(envelope.hardReserveBytes) / 1_048_576
            )
            markdown += String(
                format: "- Policy record-holder: decode strettamente > record, prefill ≥−8%%, " +
                        "stabilità ≥0.75, RAM ≥%.1f%%, swapout steady ≤128 MiB\n",
                envelope.effectiveMinimumFreePercent
            )
            markdown += "- Ricerca expert-cache slot: upward-first, un gradino per volta; " +
                        "dopo una promozione stop al primo peggioramento, " +
                        "fallback inferiore solo se il primo gradino superiore fallisce\n"
            markdown += String(
                format: "- Preflight OOM: root e candidati low-RAM non crescenti ≥512 MiB; " +
                        "altri candidati ≥12%%/1.5 GiB; RAM prevista/post-init/finale ≥%.1f%%\n",
                envelope.effectiveMinimumFreePercent
            )
        } else {
            markdown += "- Policy RAM: in attesa dello snapshot del root caricato\n"
        }
        if let pendingLabel, let pendingConfiguration {
            markdown += "- Caricamento pendente: **\(pendingLabel)** — "
            markdown += "\(machineAutoTuneSummary(pendingConfiguration))\n"
        }
        if let note { markdown += "- Nota: \(note)\n" }
        markdown += "\n"
        markdown += "## Configurazioni\n\n"
        markdown += "- Iniziale: \(machineAutoTuneSummary(journal.initial))\n"
        markdown += "- Miglior candidato: \(machineAutoTuneSummary(journal.current))\n"
        markdown += "- Motore attivo: \(journal.active.map(machineAutoTuneSummary) ?? "nessuno (teardown/load)")\n"
        markdown += "- Preferenze persistite: \(journal.persisted.map(machineAutoTuneSummary) ?? "sconosciute")\n\n"
        markdown += "## Ambiente DS4 fisso\n\n"
        for key in journal.fixedEnvironment.keys.sorted() {
            markdown += "- `\(key)=\(journal.fixedEnvironment[key] ?? "")`\n"
        }
        markdown += "\n"
        markdown += "## Prove\n\n"
        markdown += "| Pass | Parametro | Record | Candidato | Metodo | Decode | Prefill | Exact | Esito |\n"
        markdown += "|---:|---|---:|---:|---|---:|---:|:---:|---|\n"
        for trial in journal.trials {
            let decode = trial.balancedPrimaryRatio.map {
                String(format: "%+.2f%%", ($0 - 1) * 100)
            } ?? "—"
            let prefill = trial.balancedSecondaryRatio.map {
                String(format: "%+.2f%%", ($0 - 1) * 100)
            } ?? "—"
            let outcome = trial.selected ? "PROMOTED"
                : (trial.reasons.isEmpty ? "PASS" : trial.reasons.joined(separator: "; "))
            markdown += "| \(trial.pass) | \(trial.parameter) | \(trial.fromValue) | \(trial.toValue) | "
            markdown += "\(trial.comparisonMode) | \(decode) | \(prefill) | "
            markdown += "\(trial.cumulativeQualityExact ? "sì" : "no") | \(outcome.replacingOccurrences(of: "|", with: "/")) |\n"
        }
        markdown += "\n## Log misure\n\n```text\n"
        markdown += benchResults.replacingOccurrences(of: "```", with: "` ` `")
        if !benchResults.hasSuffix("\n") { markdown += "\n" }
        markdown += "```\n"
        let markdownURL = journal.directory.appendingPathComponent("report.md")
        try Data(markdown.utf8).write(to: markdownURL, options: .atomic)
        return markdownURL
    }

    private nonisolated static func hasRecentPartialDownload(near modelPath: String) -> Bool {
        let directory = URL(fileURLWithPath: modelPath).deletingLastPathComponent()
        let keys: Set<URLResourceKey> = [.contentModificationDateKey, .isRegularFileKey]
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]
        )) ?? []
        let cutoff = Date().addingTimeInterval(-600)
        return files.contains { url in
            guard url.pathExtension.lowercased() == "part",
                  let values = try? url.resourceValues(forKeys: keys),
                  values.isRegularFile == true,
                  let modified = values.contentModificationDate else { return false }
            return modified >= cutoff
        }
    }

    /// Stable FNV-1a digest for the frozen usage seed recorded in the report.
    /// It is an identity fingerprint (not a security primitive) and avoids
    /// copying/embedding the profile itself into every checkpoint.
    private nonisolated static func machineAutoTuneDataHash(_ data: Data) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in data {
            hash = (hash ^ UInt64(byte)) &* 0x1_0000_0000_01b3
        }
        return String(format: "fnv1a64:%016llx:%d", hash, data.count)
    }
}
