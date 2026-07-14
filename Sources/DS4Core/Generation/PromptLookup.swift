import Foundation

/// Prompt-lookup decoding (draft n-gram — docs/SELF-SPECULATIVE.md § Fase N):
/// il draft speculativo non è un modello ma il TRANSCRIPT stesso. Si cerca
/// l'occorrenza più RECENTE del suffisso corrente (n-gramma più LUNGO prima,
/// poi via via più corto) nella storia prompt+generato, e si propongono come
/// candidati i token che la seguirono. Costo di modello: zero — dove il testo
/// non si ripete il lookup fallisce e il chiamante degrada al decode normale;
/// dove si ripete (codice, output di tool, citazioni del prompt) la verifica
/// batch full-config conferma i candidati con la stessa garanzia di parità
/// greedy del loop self-speculative.
///
/// Logica pura su [Int]: nessuna dipendenza dal motore, testabile ovunque.
public enum PromptLookup {
    /// Fino a `count` token candidati che seguono l'occorrenza più RECENTE del
    /// suffisso più LUNGO di `history` (lunghezze maxN→minN). Vuoto se nessun
    /// n-gramma del suffisso ricorre altrove nel transcript. Le occorrenze
    /// possono sovrapporsi al suffisso stesso (testo periodico: si copia il
    /// periodo). Complessità O((maxN−minN)·n·m): per transcript da chat/demo
    /// è trascurabile rispetto a un forward.
    public static func draft(history: [Int], maxN: Int = 4, minN: Int = 2,
                             count: Int) -> [Int] {
        let n = history.count
        guard count > 0, minN >= 1, maxN >= minN, n >= minN + 1 else { return [] }
        for m in stride(from: min(maxN, n - 1), through: minN, by: -1) {
            let sfx = history[(n - m)...]
            var i = n - m - 1
            while i >= 0 {
                if history[i..<(i + m)].elementsEqual(sfx) {
                    // i <= n-m-1 ⇒ start <= n-1: c'è SEMPRE almeno un token
                    // di continuazione da proporre.
                    let start = i + m
                    return Array(history[start..<min(start + count, n)])
                }
                i -= 1
            }
        }
        return []
    }
}
