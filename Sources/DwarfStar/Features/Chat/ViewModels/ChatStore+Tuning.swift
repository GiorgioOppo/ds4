import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

extension ChatStore {
    // MARK: - Tuning tab

    func refreshTuningInfo() {
        guard let service else { tuningInfo = nil; return }
        Task { tuningInfo = await service.tuningInfo() }
    }

    func saveExpertUsage() {
        guard let service else { return }
        Task { await service.saveExpertUsage(); refreshTuningInfo() }
    }

    func resetExpertUsage() {
        guard let service else { return }
        Task { await service.resetExpertUsage(); refreshTuningInfo() }
    }

}
