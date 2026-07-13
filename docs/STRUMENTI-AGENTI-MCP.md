# Strumenti, agenti e MCP

Il sistema di tool permette al modello di richiedere operazioni strutturate
senza incorporare logica applicativa nel renderer o nel backend Metal. I tool
integrati e quelli MCP condividono il registro e il formato DSML esposto al
modello.

## Flusso di una chiamata

```text
ToolSpec -> ChatRenderer -> prompt DSML -> token generati
   -> ToolCallParser -> evento toolCall
   -> ChatStore+ToolLoop / DistributedController
   -> ToolRegistry -> esecuzione
   -> risultato strutturato -> nuovo turno tool_result -> inferenza
```

`DS4Core` conosce soltanto specifiche, rendering e parsing. `InferenceService`
emette la chiamata completa; `ChatStore+ToolLoop` e `DistributedController`
orchestrano i round. `DS4Engine` fornisce registro ed esecuzione. La GUI mostra
stato e risultati senza reinterpretare il protocollo DSML.

## Registro

`ToolRegistry` associa nome, descrizione, schema parametri ed handler. Ogni
agente usa una allow-list per filtrare le `ToolSpec` dichiarate nel prompt: un
tool escluso non viene presentato al modello nel turno corrente.

Questa allow-list è un filtro di esposizione, **non** un confine di sicurezza
né un controllo applicato automaticamente all'esecuzione. L'API
`ToolRegistry.executeAuto(_:)` risolve direttamente qualunque built-in
registrato o tool MCP con quel nome e non riceve l'insieme consentito. Un
chiamante che accetta `ToolCall` da una fonte non fidata e richiede enforcement
deve quindi verificare esplicitamente il nome contro la propria policy prima di
invocare `executeAuto`; registrazione e allow-list, da sole, non autorizzano la
chiamata.

Categorie integrate:

- aritmetica e orologio;
- file limitati al progetto;
- indice e modifica del progetto;
- Git e import GitHub;
- fetch e ricerca Web;
- elenco e avvio di sub-agent.

I tool sono raggruppati sotto `Sources/DS4Engine/Tools/Builtins`, uno per file o
responsabilità coesa.

## Sicurezza dei tool locali

I tool di progetto devono risolvere e normalizzare il percorso rispetto alla
root autorizzata. Symlink, `..` e percorsi assoluti non devono consentire fuga
dalla root. Le operazioni Git usano una whitelist; l'import GitHub non deve
ereditare credenziali arbitrarie.

Un nuovo tool che modifica file deve dichiarare chiaramente il proprio ambito
e restituire errori strutturati. Non usare output del modello come comando shell
generico.

## Agenti

Un profilo agente contiene:

- id e nome;
- system prompt;
- icona/metadati di presentazione;
- allow-list dei tool;
- profilo di uso degli esperti.

Cambiare agente modifica il contesto applicativo e il profilo di routing, non
crea un secondo decoder. I profili sono gestiti sotto `DS4Engine/Agents` e
presentati dalle feature Chat/Tuning.

## Sub-agent

I sub-agent ricevono domanda e contesto esplicitamente delegati. Eseguono una
conversazione isolata e restituiscono soltanto la risposta finale al chiamante.
La cache KV è content-keyed per riutilizzare prefissi compatibili senza
contaminare il contesto principale.

L'orchestratore deve limitare profondità, strumenti e quantità di contenuto
passato al sub-agent. Uno stop della chat principale deve propagare ai task
figli.

## MCP

`Sources/DS4Engine/Tools/MCP` implementa:

- configurazione persistibile;
- protocollo JSON-RPC;
- transport stdio;
- Streamable HTTP;
- discovery e chiamata dei tool;
- mapping dei nomi nel registro locale.

I tool remoti vengono esposti come `mcp_<server>_<tool>` per evitare collisioni.
Le configurazioni possono provenire da JSON compatibile con `mcpServers`.

## Transport stdio

Il manager avvia un processo figlio, invia messaggi JSON-RPC su stdin e legge
stdout. Log o testo non protocollare non devono essere interpretati come una
risposta valida. Ambiente e argomenti arrivano dalla configurazione esplicita.

## Streamable HTTP

Il transport invia JSON-RPC all'endpoint configurato, conserva eventuali
header e gestisce risposte/stream secondo il protocollo supportato. Token e
header sono dati sensibili e non devono comparire nei log diagnostici.

## Aggiungere un tool integrato

1. Creare l'implementazione nella categoria corretta.
2. Definire schema JSON stretto e descrizione breve.
3. Validare ogni parametro prima dell'effetto.
4. Registrare il tool nel punto centrale.
5. Decidere quali agenti possono usarlo.
6. Aggiungere test di successo, input errato e confine di sicurezza.
7. Aggiornare il README della categoria.

## Aggiungere funzionalità MCP

Separare configurazione, transport e mapping nel registro. Non introdurre
dipendenze MCP in `DS4Core` o `DS4Metal`. Coprire handshake, elenco tool,
chiamata, errore remoto, timeout e cancellazione.

## Documenti correlati

- [PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md)
- [GUI-SERVER-E-API.md](GUI-SERVER-E-API.md)
- [`templates/README.md`](../templates/README.md)
- [`Sources/DS4Engine/Tools/README.md`](../Sources/DS4Engine/Tools/README.md)
