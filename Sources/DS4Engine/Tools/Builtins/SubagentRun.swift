import Foundation
import DS4Core

extension ToolRegistry {
    /// Delegate a focused task to an isolated sub-agent. EXECUTED BY THE ENGINE
    /// (InferenceService.runSubAgent), which intercepts this call so the sub-agent
    /// runs in a separate context; this sentinel only applies if a non-engine path
    /// (HTTP server / distributed) emits the call, where sub-agents are unsupported.
    static let subagentRun = BuiltinTool(
        spec: ToolSpec(name: "subagent_run",
                       description: "Esegui un sub-agent ISOLATO su un TARGET (percorso file del progetto, oppure \"project\" per l'intero progetto) con una DOMANDA. Il sub-agent ha il contenuto già in contesto e restituisce SOLO la risposta. Con 'agent' (id da agents_list) il sub-agent assume quel RUOLO (system prompt + i suoi tool). In alternativa passa in 'tools' l'insieme MINIMO di tool. Precedenza: tools > tool del ruolo > sola lettura. Tool disponibili: project_list, project_read, project_search, project_edit, project_write, git.",
                       parametersJSON: #"{"type":"object","properties":{"target":{"type":"string","description":"percorso file relativo, oppure \"project\""},"question":{"type":"string","description":"compito o domanda per il sub-agent"},"agent":{"type":"string","description":"id di un agente (da agents_list): il sub-agent ne assume ruolo e tool. Opzionale."},"tools":{"type":"array","items":{"type":"string"},"description":"override opzionale: insieme MINIMO di tool concessi. Se assente usa i tool del ruolo 'agent', altrimenti sola lettura."}},"required":["target","question"]}"#),
        run: { _ in #"{"note":"subagent_run è gestito dall'engine (non disponibile in questo contesto)"}"# })
}
