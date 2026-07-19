import Foundation

/// Logger comune dei motori: UNA convenzione per le righe diagnostiche su
/// stderr (che la GUI raccoglie nel Log motore). Formato storico
/// "DS4 <tag>: <messaggio>" — già usato a mano ovunque; centralizzarlo rende
/// i prefissi uniformi e cercabili. Proprietà (identiche per ogni emissione):
/// stderr, una write POSIX singola senza buffering, nessun timestamp e nessun
/// livello — visibile in tempo reale e sicura da qualunque thread/actor.
/// Tag in uso: "engine", "gui", "server", "diskkv", "glm", "expbundle",
/// "lazy-idx", "dense-stream", "q4cache", "comp-q8", "multi-quant cache",
/// "mtp", "bench", "autotune", "mlock", "dist".
public enum DS4Log {
    /// Una write POSIX singola su stderr, senza buffering: visibile in
    /// tempo reale e sicura da qualunque thread.
    public static func info(_ tag: String, _ message: String) {
        raw(line(tag, message))
    }

    /// La riga nel formato canonico, senza emetterla: per i report costruiti
    /// come stringa (es. l'inventario MTP) la cui prima riga deve restare
    /// cercabile con lo stesso prefisso delle righe emesse direttamente.
    public static func line(_ tag: String, _ message: String) -> String {
        "DS4 \(tag): \(message)"
    }

    /// Testo già formato per esteso (report multi-riga dei profili, output del
    /// demo CLI): stessa destinazione e stesse proprietà, nessun prefisso —
    /// l'UNICO altro modo legittimo di scrivere diagnostica su stderr.
    public static func raw(_ text: String) {
        FileHandle.standardError.write(Data((text + "\n").utf8))
    }
}

/// Protocollo del log unificato: un sottosistema dichiara il proprio tag UNA
/// volta e ogni `log(...)` esce nel formato canonico "DS4 <tag>: <messaggio>"
/// con le proprietà di `DS4Log.info`. Sostituisce i wrapper locali che
/// replicavano il prefisso a mano (ExpertBundle.log, DenseStreamer.logQ4, …).
public protocol DS4Logging {
    /// Tag stabile e cercabile nel Log motore ("engine", "glm", "expbundle", …).
    static var logTag: String { get }
}

extension DS4Logging {
    /// Emissione dal tipo (factory, contesti statici).
    public static func log(_ message: String) { DS4Log.info(logTag, message) }
    /// Emissione dall'istanza — stesso tag del tipo.
    public func log(_ message: String) { DS4Log.info(Self.logTag, message) }
}
