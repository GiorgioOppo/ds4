import Foundation
import DeepSeekKit

/// Lifecycle del KV cache persistente cross-restart. Ispirato a
/// antirez/ds4 che salva la KV cache a 4 trigger ben definiti:
///
///   - **Cold**: subito dopo il primo prefill di una conversation.
///     Cattura lo stato "pre-decode" — utile per resume dopo crash
///     dell'app durante la prima generazione, e per Step 5 (cold-save
///     alignment al chunk boundary 2048).
///
///   - **Continued**: ad ogni window boundary (window = 128 token in
///     V4) durante decode/prefill. Cattura snapshot periodici così
///     che un crash a metà generazione perda al più 128 token.
///
///   - **Evict**: prima di un model swap o cambio conversation —
///     serializza la KV corrente prima che il release la distrugga.
///     Permette di ripristinarla quando l'utente torna alla
///     conversation originale.
///
///   - **Shutdown**: al termine programma (app exit) — ultimo
///     opportunity per persistere lo stato vivente.
///
/// Trigger sono esposti come metodi sul `KVCacheLifecycle` che il
/// caller (InferenceService, ChatStore, AppDelegate) invoca al
/// momento opportuno. La classe NON osserva eventi autonomamente
/// per restare deterministic e testabile.
///
/// `@unchecked Sendable` perché lo stato mutabile (last save
/// timestamp/token count) è protetto da `NSLock` interno per
/// permettere chiamate da queue arbitrarie (es. il queue dedicato
/// di `InferenceService.q`).
public final class KVCacheLifecycle: @unchecked Sendable {

    /// Trigger di save. Documenta il "perché" del save corrente per
    /// logging e metric.
    public enum SaveTrigger: String, Sendable {
        case cold          // dopo primo prefill
        case continued     // window boundary durante decode
        case evict         // prima di swap conversation/model
        case shutdown      // app termination
    }

    /// Throttle: numero minimo di token aggiunti fra due save
    /// `continued` consecutivi. Sotto la soglia, skip silenziosamente
    /// (evita I/O storm durante decode rapido).
    public var continuedSaveTokenThreshold: Int = 128

    /// Throttle temporale: intervallo minimo fra due save `continued`.
    /// Combinato con `continuedSaveTokenThreshold` con OR — basta
    /// che UN solo throttle sia soddisfatto per saltare.
    public var continuedSaveMinInterval: TimeInterval = 5.0

    private let stateLock = NSLock()
    private var lastContinuedSaveTokenCount: Int = 0
    private var lastContinuedSaveAt: Date = Date.distantPast

    /// Hook invocato per persistere lo snapshot a disco. Il caller
    /// (InferenceService) inietta una closure che:
    ///   1. Chiama `model.snapshotKVCache()`
    ///   2. Salva via `snapshot.save(to: PersistencePaths.kvCacheURL(id:))`
    ///   3. Updates `KVCacheFile.writeManifest` se attivo
    /// Closure async per non bloccare il decode loop.
    public var save: (@Sendable (SaveTrigger) async throws -> Void)?

    public init() {}

    // MARK: - Triggers

    /// Da chiamare dopo il primo prefill di una conversation.
    /// Sempre forza il save (nessun throttle).
    public func triggerCold() async {
        await safeCall(.cold)
        stateLock.lock()
        lastContinuedSaveTokenCount = 0
        lastContinuedSaveAt = Date()
        stateLock.unlock()
    }

    /// Da chiamare durante decode quando il count cumulativo di
    /// token raggiunge un window boundary. Throttle applicato:
    /// non salva se sotto la soglia di token o tempo dal last save.
    public func triggerContinued(currentTokenCount: Int) async {
        stateLock.lock()
        let tokensDelta = currentTokenCount - lastContinuedSaveTokenCount
        let timeDelta = Date().timeIntervalSince(lastContinuedSaveAt)
        let shouldSave = tokensDelta >= continuedSaveTokenThreshold
                          || timeDelta >= continuedSaveMinInterval
        stateLock.unlock()
        if !shouldSave { return }
        await safeCall(.continued)
        stateLock.lock()
        lastContinuedSaveTokenCount = currentTokenCount
        lastContinuedSaveAt = Date()
        stateLock.unlock()
    }

    /// Da chiamare prima di un model swap o cambio conversation.
    /// Sempre forza il save.
    public func triggerEvict() async {
        await safeCall(.evict)
    }

    /// Da chiamare all'app termination (AppDelegate.applicationWillTerminate
    /// o equivalente SwiftUI scene phase change).
    public func triggerShutdown() async {
        await safeCall(.shutdown)
    }

    /// Resetta lo stato del throttle. Usa quando cambia
    /// conversation così il prossimo `continued` non viene
    /// "antichizzato" dallo stato della conversation precedente.
    public func reset() {
        stateLock.lock()
        lastContinuedSaveTokenCount = 0
        lastContinuedSaveAt = Date.distantPast
        stateLock.unlock()
    }

    // MARK: - Internal

    private func safeCall(_ trigger: SaveTrigger) async {
        guard let save = save else { return }
        do {
            try await save(trigger)
        } catch {
            let msg = "[kvlifecycle] save(\(trigger.rawValue)) failed: "
                + "\(error.localizedDescription)\n"
            FileHandle.standardError.write(Data(msg.utf8))
        }
    }
}
