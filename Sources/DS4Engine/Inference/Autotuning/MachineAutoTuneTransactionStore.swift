import Foundation

/// Crash-safe persistence boundary for the GUI machine auto-tuner.
///
/// The search itself never writes preferences.  Once a winner has passed the
/// final quality/performance gate, the caller creates a transaction, installs
/// and warms that winner, marks it as installed, and only then commits the GUI
/// preferences.  An interrupted non-terminal transaction is rolled back to its
/// initial snapshot on the next app launch.
///
/// The active record intentionally lives outside an individual run report:
/// `Application Support/DwarfStar/autotune-transaction.json` is a stable path
/// that startup recovery can inspect before `ChatStore` reads `UserDefaults`.
public struct MachineAutoTuneTransactionStore {
    public static let schemaVersion = 1

    public enum Phase: String, Codable, Sendable {
        /// Winner validated, but its chat engine has not completed warmup yet.
        case prepared
        /// Winner engine completed init and warmup; preferences are untouched.
        case installed
        /// Preference commit started but its terminal record is not durable yet.
        case committing
        /// Winner preferences and commit marker were flushed successfully.
        case committed
        /// Initial preferences are being restored.
        case rollingBack
        /// Initial preferences and rollback marker were flushed successfully.
        case rolledBack

        var isTerminal: Bool {
            self == .committed || self == .rolledBack
        }
    }

    struct ConfigurationSnapshot: Codable, Equatable, Sendable {
        let values: [String: Int]

        init(_ configuration: MachineAutoTuneConfiguration) throws {
            var values: [String: Int] = [:]
            for knob in MachineAutoTuneKnob.allCases {
                guard let value = configuration.value(for: knob) else {
                    throw StoreError.incompleteConfiguration(knob.rawValue)
                }
                values[knob.rawValue] = value
            }
            self.values = values
        }

        func configuration() throws -> MachineAutoTuneConfiguration {
            var settings: [MachineAutoTuneKnob: Int] = [:]
            for knob in MachineAutoTuneKnob.allCases {
                guard let value = values[knob.rawValue] else {
                    throw StoreError.incompleteConfiguration(knob.rawValue)
                }
                settings[knob] = value
            }
            return MachineAutoTuneConfiguration(settings: settings)
        }
    }

    public struct Record: Codable, Equatable, Sendable {
        public let schema: Int
        public let id: UUID
        public let createdAt: Date
        public var updatedAt: Date
        public var revision: Int
        public var phase: Phase
        public let modelPath: String
        public let contextTokens: Int
        public let reportDirectory: String?
        let initial: ConfigurationSnapshot
        let winner: ConfigurationSnapshot
        public var note: String?
    }

    public enum RecoveryOutcome: Equatable, Sendable {
        case none
        case rolledBack(transactionID: UUID, interruptedPhase: Phase)
        case completedCommit(transactionID: UUID)
        case cleanedTerminal(transactionID: UUID, phase: Phase)
        case quarantined(path: String, reason: String)
        case failed(String)

        public var logMessage: String? {
            switch self {
            case .none:
                return nil
            case .rolledBack(let id, let phase):
                return "auto-tune transaction \(id.uuidString) interrotta in \(phase.rawValue); impostazioni iniziali ripristinate"
            case .completedCommit(let id):
                return "auto-tune transaction \(id.uuidString): commit terminale completato in recovery"
            case .cleanedTerminal(let id, let phase):
                return "auto-tune transaction \(id.uuidString) già \(phase.rawValue); record attivo ripulito"
            case .quarantined(let path, let reason):
                return "record auto-tune non leggibile spostato in \(path): \(reason)"
            case .failed(let reason):
                return "recovery transazione auto-tune fallita: \(reason)"
            }
        }
    }

    enum StoreError: LocalizedError {
        case incompleteConfiguration(String)
        case activeTransaction(UUID, Phase)
        case missingTransaction
        case transactionMismatch(expected: UUID, found: UUID)
        case invalidTransition(from: Phase, to: Phase)
        case unsupportedSchema(Int)

        var errorDescription: String? {
            switch self {
            case .incompleteConfiguration(let knob):
                return "snapshot auto-tune incompleto: manca \(knob)"
            case .activeTransaction(let id, let phase):
                return "esiste già la transazione auto-tune \(id.uuidString) (\(phase.rawValue))"
            case .missingTransaction:
                return "transazione auto-tune attiva non trovata"
            case .transactionMismatch(let expected, let found):
                return "ID transazione inatteso: atteso \(expected.uuidString), trovato \(found.uuidString)"
            case .invalidTransition(let from, let to):
                return "transizione auto-tune non valida: \(from.rawValue) → \(to.rawValue)"
            case .unsupportedSchema(let schema):
                return "schema transazione auto-tune non supportato: \(schema)"
            }
        }
    }

    private static let defaultsMarkerKey = "DS4MachineAutoTuneTransactionMarker"
    private static let committedMarkerPrefix = "committed:"
    private static let rolledBackMarkerPrefix = "rolled-back:"

    private let fileManager: FileManager
    private let defaults: UserDefaults
    let recordURL: URL

    public init(
        fileManager: FileManager = .default,
        defaults: UserDefaults = .standard,
        recordURL: URL? = nil
    ) {
        self.fileManager = fileManager
        self.defaults = defaults
        self.recordURL = recordURL ?? Self.defaultRecordURL(fileManager: fileManager)
    }

    /// Creates the durable `prepared` record before installing the winner.
    /// An unfinished prior transaction must be recovered first; silently
    /// replacing it would destroy the only rollback snapshot.
    @discardableResult
    public func prepare(
        initial: MachineAutoTuneConfiguration,
        winner: MachineAutoTuneConfiguration,
        modelPath: String,
        contextTokens: Int,
        reportDirectory: URL? = nil
    ) throws -> Record {
        if fileManager.fileExists(atPath: recordURL.path) {
            let existing = try readRecord()
            if existing.phase.isTerminal {
                try removeActiveRecord()
            } else {
                throw StoreError.activeTransaction(existing.id, existing.phase)
            }
        }

        let now = Date()
        let record = Record(
            schema: Self.schemaVersion,
            id: UUID(),
            createdAt: now,
            updatedAt: now,
            revision: 0,
            phase: .prepared,
            modelPath: modelPath,
            contextTokens: contextTokens,
            reportDirectory: reportDirectory?.path,
            initial: try ConfigurationSnapshot(initial),
            winner: try ConfigurationSnapshot(winner),
            note: "Finalista validato; installazione/warmup non ancora completati."
        )
        try writeRecord(record)
        return record
    }

    /// Records that init and warmup of the winner succeeded.  No preference is
    /// changed by this transition.
    @discardableResult
    public func markInstalled(transactionID: UUID) throws -> Record {
        try transition(
            transactionID: transactionID,
            expected: [.prepared],
            to: .installed,
            note: "Motore winner installato e warmup completato; commit impostazioni pendente."
        )
    }

    /// Commits the winner only after `markInstalled`.
    ///
    /// `adoptInProcess` lets `ChatStore` mirror the same values in its observable
    /// properties.  The helper first writes the complete winner snapshot to
    /// `UserDefaults`; the callback is therefore idempotent even if those
    /// properties persist again through `didSet`.
    @discardableResult
    public func commit(
        transactionID: UUID,
        adoptInProcess: () -> Void
    ) throws -> Record {
        var record = try transition(
            transactionID: transactionID,
            expected: [.installed],
            to: .committing,
            note: "Commit atomico logico dei knob in corso."
        )

        try apply(record.winner, marker: Self.committedMarker(for: record.id))
        adoptInProcess()
        // `synchronize()` is deprecated and its Bool is not a durability
        // contract (sandboxed/test suites may return false after valid writes).
        // The durable transaction record is the recovery authority instead.
        _ = defaults.synchronize()

        record = try transition(
            transactionID: transactionID,
            expected: [.committing],
            to: .committed,
            note: "Winner installato, warm e persistito."
        )
        return record
    }

    /// Reasserts the complete initial snapshot after an install/commit failure.
    /// The active record remains recoverable as `rollingBack` until the defaults
    /// flush and terminal `rolledBack` write both succeed.
    @discardableResult
    public func rollback(
        transactionID: UUID,
        reason: String,
        restoreInProcess: () -> Void
    ) throws -> Record {
        var record = try readRecord(transactionID: transactionID)
        guard !record.phase.isTerminal else {
            if record.phase == .rolledBack { return record }
            throw StoreError.invalidTransition(from: record.phase, to: .rollingBack)
        }

        if record.phase != .rollingBack {
            record = try transition(
                transactionID: transactionID,
                expected: [.prepared, .installed, .committing],
                to: .rollingBack,
                note: reason
            )
        }

        try apply(record.initial, marker: Self.rolledBackMarker(for: record.id))
        restoreInProcess()
        _ = defaults.synchronize()

        return try transition(
            transactionID: transactionID,
            expected: [.rollingBack],
            to: .rolledBack,
            note: "Configurazione iniziale ripristinata: \(reason)"
        )
    }

    /// Must run before `ChatStore` is constructed so its stored properties read
    /// a coherent preference snapshot.
    ///
    /// Non-terminal states are deliberately rolled back, including
    /// `committing`: a winner is durable only after the terminal record exists.
    /// A terminal commit with a missing defaults marker is completed from the
    /// record, covering a preferences-domain flush lost at process termination.
    func recoverInterruptedTransaction() throws -> RecoveryOutcome {
        guard fileManager.fileExists(atPath: recordURL.path) else { return .none }

        let record: Record
        do {
            record = try readRecord()
        } catch {
            let quarantined = try quarantineActiveRecord()
            return .quarantined(path: quarantined.path, reason: error.localizedDescription)
        }

        switch record.phase {
        case .prepared, .installed, .committing, .rollingBack:
            let interrupted = record.phase
            try apply(record.initial, marker: Self.rolledBackMarker(for: record.id))
            var terminal = record
            terminal.phase = .rolledBack
            terminal.revision += 1
            terminal.updatedAt = Date()
            terminal.note = "Recovery all'avvio da fase \(interrupted.rawValue): snapshot iniziale ripristinato."
            try writeRecord(terminal)
            try removeActiveRecord()
            return .rolledBack(transactionID: record.id, interruptedPhase: interrupted)

        case .committed:
            let marker = defaults.string(forKey: Self.defaultsMarkerKey)
            // Reassert the complete snapshot even when the marker exists: the
            // record, not asynchronous per-key defaults flushing, is atomic.
            try apply(record.winner, marker: Self.committedMarker(for: record.id))
            try removeActiveRecord()
            if marker != Self.committedMarker(for: record.id) {
                return .completedCommit(transactionID: record.id)
            }
            return .cleanedTerminal(transactionID: record.id, phase: record.phase)

        case .rolledBack:
            try apply(record.initial, marker: Self.rolledBackMarker(for: record.id))
            try removeActiveRecord()
            return .cleanedTerminal(transactionID: record.id, phase: record.phase)
        }
    }

    /// Launch-safe wrapper: recovery errors are reported without preventing the
    /// app from opening.  A malformed/unknown record is quarantined rather than
    /// guessed, because overwriting preferences without a trusted initial
    /// snapshot would be less safe than retaining their last durable values.
    public static func recoverAtLaunch(
        fileManager: FileManager = .default,
        defaults: UserDefaults = .standard
    ) -> RecoveryOutcome {
        do {
            return try MachineAutoTuneTransactionStore(
                fileManager: fileManager,
                defaults: defaults
            ).recoverInterruptedTransaction()
        } catch {
            return .failed(error.localizedDescription)
        }
    }

    private func transition(
        transactionID: UUID,
        expected: Set<Phase>,
        to phase: Phase,
        note: String
    ) throws -> Record {
        var record = try readRecord(transactionID: transactionID)
        guard expected.contains(record.phase) else {
            throw StoreError.invalidTransition(from: record.phase, to: phase)
        }
        record.phase = phase
        record.revision += 1
        record.updatedAt = Date()
        record.note = note
        try writeRecord(record)
        return record
    }

    private func readRecord(transactionID: UUID? = nil) throws -> Record {
        guard fileManager.fileExists(atPath: recordURL.path) else {
            throw StoreError.missingTransaction
        }
        let data = try Data(contentsOf: recordURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let record = try decoder.decode(Record.self, from: data)
        guard record.schema == Self.schemaVersion else {
            throw StoreError.unsupportedSchema(record.schema)
        }
        if let transactionID, transactionID != record.id {
            throw StoreError.transactionMismatch(expected: transactionID, found: record.id)
        }
        _ = try record.initial.configuration()
        _ = try record.winner.configuration()
        return record
    }

    private func writeRecord(_ record: Record) throws {
        guard record.schema == Self.schemaVersion else {
            throw StoreError.unsupportedSchema(record.schema)
        }
        try fileManager.createDirectory(
            at: recordURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(record).write(to: recordURL, options: .atomic)
    }

    private func removeActiveRecord() throws {
        guard fileManager.fileExists(atPath: recordURL.path) else { return }
        try fileManager.removeItem(at: recordURL)
    }

    private func quarantineActiveRecord() throws -> URL {
        let suffix = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let quarantineURL = recordURL.deletingPathExtension()
            .appendingPathExtension("corrupt-\(suffix).json")
        try fileManager.moveItem(at: recordURL, to: quarantineURL)
        return quarantineURL
    }

    private func apply(_ snapshot: ConfigurationSnapshot, marker: String) throws {
        let configuration = try snapshot.configuration()
        for knob in MachineAutoTuneKnob.allCases {
            guard let value = configuration.value(for: knob) else {
                throw StoreError.incompleteConfiguration(knob.rawValue)
            }
            switch knob {
            case .multiQuantCache:
                defaults.set(value != 0, forKey: "DS4MultiQuantCache")
            case .expertCacheSlots:
                defaults.set(value, forKey: "DS4ExpertCacheSlots")
            case .expertCacheUniform:
                defaults.set(value != 0, forKey: "DS4ExpertCacheUniform")
            case .preadSplit:
                defaults.set(value, forKey: "DS4PreadSplit")
            case .denseAhead:
                defaults.set(value, forKey: "DS4DenseAhead")
            case .asyncFFN:
                defaults.set(value != 0, forKey: "DS4AsyncFFN")
            case .expertLookahead:
                defaults.set(value, forKey: "DS4ExpertLookahead")
            case .q8NSG:
                defaults.set(value, forKey: "DS4Q8NSG")
            case .moeNSG:
                defaults.set(value, forKey: "DS4MoeNSG")
            case .denseQ4NSG:
                defaults.set(value, forKey: "DS4DenseQ4NSG")
            }
        }
        // RAW_RING is an invariant of this tuner, not an explored knob.
        defaults.set(true, forKey: "DS4RawRing")
        defaults.set(marker, forKey: Self.defaultsMarkerKey)
        _ = defaults.synchronize()
    }

    private static func committedMarker(for id: UUID) -> String {
        committedMarkerPrefix + id.uuidString.lowercased()
    }

    private static func rolledBackMarker(for id: UUID) -> String {
        rolledBackMarkerPrefix + id.uuidString.lowercased()
    }

    private static func defaultRecordURL(fileManager: FileManager) -> URL {
        let applicationSupport = fileManager.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Library/Application Support", isDirectory: true)
        return applicationSupport
            .appendingPathComponent("DwarfStar", isDirectory: true)
            .appendingPathComponent("autotune-transaction.json", isDirectory: false)
    }
}
