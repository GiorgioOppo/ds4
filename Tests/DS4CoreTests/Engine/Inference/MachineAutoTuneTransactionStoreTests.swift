import XCTest
@testable import DS4Engine

final class MachineAutoTuneTransactionStoreTests: XCTestCase {
    private var suiteName = ""
    private var defaults: UserDefaults!
    private var directory: URL!
    private var recordURL: URL!
    private var store: MachineAutoTuneTransactionStore!

    override func setUpWithError() throws {
        suiteName = "MachineAutoTuneTransactionStoreTests.\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(suiteName, isDirectory: true)
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        recordURL = directory.appendingPathComponent("transaction.json")
        store = MachineAutoTuneTransactionStore(
            defaults: defaults,
            recordURL: recordURL
        )
    }

    override func tearDownWithError() throws {
        defaults?.removePersistentDomain(forName: suiteName)
        if let directory {
            try? FileManager.default.removeItem(at: directory)
        }
        store = nil
        defaults = nil
    }

    func testRecoveryRollsEveryNonTerminalPhaseBackToInitial() throws {
        for phase in [
            MachineAutoTuneTransactionStore.Phase.prepared,
            .installed, .committing, .rollingBack,
        ] {
            try? FileManager.default.removeItem(at: recordURL)
            defaults.removePersistentDomain(forName: suiteName)
            let initial = configuration(seed: 0)
            let winner = configuration(seed: 1)
            var record = try store.prepare(
                initial: initial,
                winner: winner,
                modelPath: "/tmp/model.gguf",
                contextTokens: 4096
            )
            record.phase = phase
            record.revision += 1
            try write(record)
            applyToDefaults(winner)

            let outcome = try store.recoverInterruptedTransaction()
            guard case .rolledBack(_, let interrupted) = outcome else {
                return XCTFail("expected rollback recovery for \(phase), got \(outcome)")
            }
            XCTAssertEqual(interrupted, phase)
            assertDefaults(equal: initial)
            XCTAssertFalse(FileManager.default.fileExists(atPath: recordURL.path))
        }
    }

    func testCommittedRecordWithMissingMarkerCompletesWinner() throws {
        let initial = configuration(seed: 0)
        let winner = configuration(seed: 1)
        let record = try store.prepare(
            initial: initial, winner: winner,
            modelPath: "/tmp/model.gguf", contextTokens: 4096
        )
        _ = try store.markInstalled(transactionID: record.id)
        _ = try store.commit(transactionID: record.id) {}

        defaults.removeObject(forKey: "DS4MachineAutoTuneTransactionMarker")
        applyToDefaults(initial)
        let outcome = try store.recoverInterruptedTransaction()
        guard case .completedCommit(let recoveredID) = outcome else {
            return XCTFail("expected completed commit, got \(outcome)")
        }
        XCTAssertEqual(recoveredID, record.id)
        assertDefaults(equal: winner)
    }

    func testRolledBackRecordWithMissingMarkerReassertsInitial() throws {
        let initial = configuration(seed: 0)
        let winner = configuration(seed: 1)
        let record = try store.prepare(
            initial: initial, winner: winner,
            modelPath: "/tmp/model.gguf", contextTokens: 4096
        )
        _ = try store.rollback(transactionID: record.id, reason: "test") {}

        defaults.removeObject(forKey: "DS4MachineAutoTuneTransactionMarker")
        applyToDefaults(winner)
        _ = try store.recoverInterruptedTransaction()
        assertDefaults(equal: initial)
    }

    func testMalformedRecordIsQuarantinedWithoutChangingPreferences() throws {
        let initial = configuration(seed: 0)
        applyToDefaults(initial)
        try Data("{".utf8).write(to: recordURL, options: .atomic)

        let outcome = try store.recoverInterruptedTransaction()
        guard case .quarantined(let path, _) = outcome else {
            return XCTFail("expected quarantine, got \(outcome)")
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: recordURL.path))
        assertDefaults(equal: initial)
    }

    func testTransactionIDAndPhaseTransitionsAreRejected() throws {
        let record = try store.prepare(
            initial: configuration(seed: 0),
            winner: configuration(seed: 1),
            modelPath: "/tmp/model.gguf",
            contextTokens: 4096
        )
        XCTAssertThrowsError(try store.markInstalled(transactionID: UUID()))
        _ = try store.markInstalled(transactionID: record.id)
        XCTAssertThrowsError(try store.markInstalled(transactionID: record.id))
    }

    func testCommitPersistsCompleteWinnerSnapshotAndRawRingInvariant() throws {
        let initial = configuration(seed: 0)
        let winner = configuration(seed: 1)
        let record = try store.prepare(
            initial: initial, winner: winner,
            modelPath: "/tmp/model.gguf", contextTokens: 100_000
        )
        _ = try store.markInstalled(transactionID: record.id)
        let committed = try store.commit(transactionID: record.id) {}

        XCTAssertEqual(committed.phase, .committed)
        assertDefaults(equal: winner)
        XCTAssertTrue(defaults.bool(forKey: "DS4RawRing"))
        XCTAssertTrue((defaults.string(forKey: "DS4MachineAutoTuneTransactionMarker") ?? "")
            .hasPrefix("committed:"))
    }

    private func configuration(seed: Int) -> MachineAutoTuneConfiguration {
        MachineAutoTuneConfiguration(settings: [
            .multiQuantCache: seed,
            .expertCacheSlots: seed == 0 ? 16 : 22,
            .expertCacheUniform: seed,
            .preadSplit: seed == 0 ? 2 : 4,
            .denseAhead: seed == 0 ? 1 : 2,
            .asyncFFN: seed,
            .expertLookahead: seed == 0 ? 0 : 4,
            .q8NSG: seed == 0 ? 3 : 4,
            .moeNSG: seed == 0 ? 3 : 5,
            .denseQ4NSG: seed == 0 ? 3 : 6,
        ])
    }

    private func write(_ record: MachineAutoTuneTransactionStore.Record) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(record).write(to: recordURL, options: .atomic)
    }

    private func applyToDefaults(_ configuration: MachineAutoTuneConfiguration) {
        for knob in MachineAutoTuneKnob.allCases {
            let value = configuration.value(for: knob)!
            switch knob {
            case .multiQuantCache: defaults.set(value != 0, forKey: "DS4MultiQuantCache")
            case .expertCacheSlots: defaults.set(value, forKey: "DS4ExpertCacheSlots")
            case .expertCacheUniform: defaults.set(value != 0, forKey: "DS4ExpertCacheUniform")
            case .preadSplit: defaults.set(value, forKey: "DS4PreadSplit")
            case .denseAhead: defaults.set(value, forKey: "DS4DenseAhead")
            case .asyncFFN: defaults.set(value != 0, forKey: "DS4AsyncFFN")
            case .expertLookahead: defaults.set(value, forKey: "DS4ExpertLookahead")
            case .q8NSG: defaults.set(value, forKey: "DS4Q8NSG")
            case .moeNSG: defaults.set(value, forKey: "DS4MoeNSG")
            case .denseQ4NSG: defaults.set(value, forKey: "DS4DenseQ4NSG")
            }
        }
    }

    private func assertDefaults(
        equal configuration: MachineAutoTuneConfiguration,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for knob in MachineAutoTuneKnob.allCases {
            let expected = configuration.value(for: knob)!
            let actual: Int
            switch knob {
            case .multiQuantCache: actual = defaults.bool(forKey: "DS4MultiQuantCache") ? 1 : 0
            case .expertCacheSlots: actual = defaults.integer(forKey: "DS4ExpertCacheSlots")
            case .expertCacheUniform: actual = defaults.bool(forKey: "DS4ExpertCacheUniform") ? 1 : 0
            case .preadSplit: actual = defaults.integer(forKey: "DS4PreadSplit")
            case .denseAhead: actual = defaults.integer(forKey: "DS4DenseAhead")
            case .asyncFFN: actual = defaults.bool(forKey: "DS4AsyncFFN") ? 1 : 0
            case .expertLookahead: actual = defaults.integer(forKey: "DS4ExpertLookahead")
            case .q8NSG: actual = defaults.integer(forKey: "DS4Q8NSG")
            case .moeNSG: actual = defaults.integer(forKey: "DS4MoeNSG")
            case .denseQ4NSG: actual = defaults.integer(forKey: "DS4DenseQ4NSG")
            }
            XCTAssertEqual(actual, expected, "mismatch for \(knob)", file: file, line: line)
        }
    }
}
