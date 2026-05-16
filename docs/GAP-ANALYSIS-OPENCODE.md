# Gap Analysis: DeepSeek-V4-Pro-MacOS vs opencode

Confronto strutturato tra questo progetto e
[opencode](https://github.com/anomalyco/opencode), pensato per
individuare differenze, sovrapposizioni e possibili direzioni di
sviluppo.

> Sorgenti consultate: README e struttura del repo opencode
> (`packages/opencode/src/{agent,tool,session,plugin,ide,lsp,mcp,...}`)
> più il README e i sorgenti di questo progetto.

---

## TL;DR — sono progetti ortogonali

| | DeepSeek-V4-Pro-MacOS | opencode |
|---|---|---|
| **Scopo** | Eseguire DeepSeek-V4 in locale + chat generica | Coding agent open-source stile Claude Code / Aider |
| **Linguaggio / runtime** | Swift + Metal | TypeScript / Bun (monorepo) |
| **Target** | macOS Apple Silicon (native) | macOS, Windows, Linux (cross-platform) |
| **Interfaccia primaria** | App SwiftUI desktop + CLI di inferenza | TUI (terminal) + app desktop in beta |
| **Forte di** | Inferenza locale ottimizzata MoE/MLA, quantizzazione, streaming dei pesi | Tooling agentico (read/write/edit/bash/lsp), multi-provider, plugin |
| **Manca** | Strumenti per agire sul filesystem / shell / codice | Capacità di eseguire un modello in locale (delega ad altri runtime) |

In una frase: **noi sappiamo *eseguire* un LLM da 142 GB su un Mac da 16
GB; opencode sa *farlo lavorare sul codice***. I due insiemi di
competenze sono in larga parte disgiunti.

---

## 1. Confronto ad alto livello

### 1.1 Filosofia

| Asse | Questo progetto | opencode |
|---|---|---|
| **Distribuzione** | Build da sorgente (xcodegen + SPM) | `npm`, `brew`, `scoop`, `pacman`, `nix`, `mise`, script `curl \| sh`, app desktop scaricabile |
| **Provider** | Backend locale (DeepSeek-V4) + OpenRouter come unico ponte remoto | Provider-agnostic: Claude / OpenAI / Google / Azure / Cloudflare / Codex / DigitalOcean / GitHub Copilot / locale |
| **Apertura** | MIT, ma single-maintainer, tooling ad hoc | MIT, monorepo, plugin pubblici, SDK npm, estensioni IDE |
| **Stato** | "Experimental" | 161k+ stars su GitHub, ecosistema attivo |

### 1.2 Cosa fa **questo** progetto e opencode no

- **Inferenza locale di DeepSeek-V4** (1.6 T parametri / V4-Flash 284 B / 13 B attivati) in Swift + Metal nativo.
- **Quantizzazione su disco** (BF16 / INT8 / INT4 / INT2, FP8+FP4 nativi del checkpoint HuggingFace) tramite il binario `converter`.
- **Strategie di caricamento adattive**: `preload` / `mmap` / `streaming` (un solo shard layer-aligned in RAM alla volta, per girare V4-Flash su 16 GB).
- **Kernel Metal custom** (MLA, sparse attention, HyperConnections, MoE dispatch, sqrtsoftplus gating, YaRN RoPE, Compressor/Indexer, MTP).
- **KV-cache snapshot/restore** per delegation sub-agent senza pagare un re-prefill a freddo.
- **Encoding DSML** (token nativi DeepSeek `<｜begin▁of▁repo▁name｜>` ecc.) e parser di tool-call DSML.
- **Pre-tokenized projects**: collezioni di documenti pre-tokenizzati contro il tokenizer attivo, splittati nel primo turno con i delimitatori repo/file.
- **UI SwiftUI nativa** con feature model-aware: model-state banner, throughput bar, cumulative cost banner, brain-icon disclosure per `<think>` e `reasoning_content`, live delegation chain, resume banner per `pendingTurn`.

### 1.3 Cosa fa opencode e **questo** progetto no

Vedi §3 — è la parte sostanziosa.

---

## 2. Architettura a confronto

### 2.1 Macro-layer

| Livello | Questo progetto | opencode |
|---|---|---|
| **Runtime modello** | `DeepSeekKit` (Swift + Metal): tensor, kernels, sampler, KV cache, loader | Nessun runtime locale: si appoggia a provider via API |
| **Tool runtime** | `MCPClient` per server MCP stdio + parser DSML tool-call | `tool/` (~25 file): read, write, edit, shell, grep, glob, webfetch, websearch, plan, task, todo, apply_patch, repo_clone, repo_overview, lsp, … |
| **Session / orchestratore** | `InferenceService` + `ChatStore` + `Conversation` (JSON su disco) | `session/` con `session.sql.ts`, `compaction.ts`, `overflow.ts`, `revert.ts`, `retry.ts`, `processor.ts`, `summary.ts`, `todo.ts` |
| **Provider** | `OpenRouterAPI` (OpenAI-compatible) + `InferenceService.local*` | `provider/` configurabile + `plugin/{azure,cloudflare,codex,digitalocean,github-copilot}` |
| **UI** | SwiftUI desktop monolitico | Client/server decoupled: TUI, desktop, web tutti su API |
| **Integrazione IDE** | nessuna | VS Code SDK in `sdks/vscode`, estensione Zed in `extensions/zed` |
| **Distribuzione** | xcodegen → Xcode | turbo monorepo, package multipli (`opencode`, `tui`, `desktop`, `web`, `sdk`, `plugin`, `function`, `slack`, `enterprise`, `github`, `http-recorder`, …) |

### 2.2 Differenza di paradigma

- **Noi**: tutto è in-process nello stesso binario macOS. Il modello, la sessione, lo strumento MCP, la UI vivono nello stesso `NSApplication`.
- **opencode**: client/server. Il "core" (provider, tool runtime, session store) gira come server; TUI, desktop, web, IDE plugin sono client che parlano un'API. Conseguenza: posso lanciare l'agent su una macchina e controllarlo da un'altra, oppure incollare un permalink di sessione in chat e condividere la conversazione.

---

## 3. Cosa manca a noi rispetto a opencode

Raggruppato per area. Le voci più rilevanti sono in grassetto.

### 3.1 Strumenti agentici sul codice

Questo è il **gap più grande**. opencode è prima di tutto un coding
agent; noi siamo un chat client. Tool che opencode espone nativamente
e noi non abbiamo (a meno di delegarli a un MCP server esterno):

- **`read`** — lettura file con cat-n format.
- **`write`** — creazione file.
- **`edit`** — string-replace mirato (stile diff applicato).
- **`apply_patch`** — applicazione di patch unified diff.
- **`shell` / `bash`** — esecuzione comandi nella sandbox con working directory persistente.
- **`grep`** — ricerca pattern (probabilmente ripgrep-based).
- **`glob`** — match file con pattern.
- **`webfetch`** + **`websearch`** — fetch URL e ricerca web nativi.
- **`repo_clone`**, **`repo_overview`** — clone e analisi di repository.
- **`plan`**, **`task`**, **`task_status`**, **`todo`** — gestione strutturata del piano di lavoro.
- **`lsp`** — go-to-definition, hover, diagnostics via Language Server Protocol.

Per noi tutto questo dovrebbe arrivare da MCP server esterni
(filesystem, shell, ecc.), che è effettivamente quello che fa Claude
Desktop — ma significa che out-of-the-box il nostro client non *fa*
niente sul codice.

### 3.2 Modalità agente

- **Plan mode**: agent read-only che chiede conferma prima di
  comandi shell e nega edit per default. Noi non abbiamo distinzione
  tra "esegui" e "pianifica".
- **General subagent** invocabile con `@general` per query
  multi-step. Noi abbiamo `__delegate_to_agent` ma solo se l'utente
  ha definito a mano più di un agent.
- **Custom agent definitions** via file di configurazione versionabili
  in repo (`.opencode/` per-progetto). I nostri agent vivono in
  `AppSettings` (UserDefaults), non sono per-repo né versionabili in
  git.

### 3.3 Session management

opencode ha:

- **Persistence SQL** (`session.sql.ts`) — noi siamo su file JSON.
- **Compaction** automatica (`compaction.ts`) — quando il context si
  riempie, comprime gli scambi vecchi in un riassunto. Noi
  semplicemente tronchiamo (o lasciamo che il modello faccia tilt).
- **Overflow handling** (`overflow.ts`) — gestione strutturata del
  superamento del context window.
- **Revert / Retry** — undo di un turno, retry con seed diverso. Noi
  abbiamo solo `pendingTurn` resume su crash.
- **Summary** della sessione lato server.
- **Snapshot** di stato.
- **Sharing / permalinks** — pubblica una sessione, ottieni un URL.

### 3.4 Estendibilità

- **Plugin system** (`plugin/loader.ts`, `plugin/index.ts`) con hook
  documentati. Noi non abbiamo plugin esterni; tutto va aggiunto in
  Swift e ricompilato.
- **Provider plugins** già pronti per Azure, Cloudflare, Codex,
  DigitalOcean, GitHub Copilot. Per noi qualsiasi nuovo provider
  significa scrivere Swift in `InferenceService` o aggiungerlo a
  OpenRouter.
- **SDK npm** per scrivere automazioni esterne contro l'agent. Noi
  non abbiamo un'API pubblica.
- **Slash commands** custom definibili nel config.
- **Skills** (`tool/skill.ts`, `config/skills.ts`) — capability
  packaged riutilizzabili.

### 3.5 Integrazioni

- **VS Code extension** (`sdks/vscode/`).
- **Zed extension** (`extensions/zed/`).
- **GitHub Actions** (`github/` package, c'è anche una directory
  top-level dedicata).
- **Slack** (`packages/slack`).
- **Enterprise** features (`packages/enterprise`).
- **HTTP recorder** per registrare/rigiocare chiamate API.
- **Containers** package — esecuzione in sandbox containerizzata.

### 3.6 UX / TUI

- **TUI** primaria (sviluppata da utenti neovim), con focus su
  keybinding, tab-switch tra agent, motion-style nav. Per noi sarebbe
  l'equivalente di un client `deepseek-chat` headless che oggi non
  esiste — la `deepseek` CLI è single-turn.
- **Themes** e **keybindings** custom.
- **Permission system** (`config/permission.ts`) — granulare per
  tool. Noi abbiamo solo la tool-allowlist binaria per agent.

### 3.7 Tooling di sviluppo del progetto stesso

| | Nostro | opencode |
|---|---|---|
| Monorepo | no | sì (turbo + bun workspaces) |
| Lint/format CI | parziale | husky + lint-staged |
| Docs site | markdown statico nel repo | Astro/Starlight in `packages/web` |
| Storybook | no | sì (`packages/storybook`) |

---

## 4. Cosa abbiamo noi che opencode non ha

Per simmetria — sono cose che opencode non può fare *strutturalmente*
perché non ha un runtime di inferenza locale.

- **Eseguire DeepSeek-V4-Flash** (e in prospettiva V4-Pro) su Apple Silicon.
- **Quantizzare** un checkpoint HuggingFace verso INT8/INT4/INT2/BF16 con il `converter`.
- **Streaming dei pesi** per girare un modello da 142 GB su un Mac da 16 GB.
- **Caricamento `mmap`** che lascia al kernel macOS la gestione del page cache.
- **KV-cache snapshot/restore** per delegation sub-agent senza
  prefill cold.
- **Diagnostica low-level**: `--trace-norms`, `--list-tensors`,
  `--dump-tensor`, `--print-config`.
- **Encoding DSML** completo per DeepSeek (anche se restano i task
  tokens da emettere — vedi `TODO.md` §3).
- **Native macOS UX**: Keychain per le API key, NSDocument-style
  sidebar, system menu bar, Cmd+N / Cmd+,, focus ring SwiftUI,
  drag&drop file.
- **Inferenza offline** — un Mac senza rete fa generation. opencode
  senza rete non parla a nessun provider (a meno di puntare a Ollama
  o simili in localhost, ma non è il caso d'uso target).

---

## 5. Cosa hanno in comune

- **MCP server support** (stdio): entrambi parlano lo stesso
  protocollo, configurazione importabile da Claude Desktop.
- **Agent presets** con system prompt + tool allowlist + sampling
  override.
- **Sub-agent delegation** (per noi: `__delegate_to_agent`, locale; per
  opencode: `@general` e custom agents).
- **Tool-call loop** con OpenAI-compatible `tools` array (lato
  remote).
- **Streaming SSE** delle risposte.
- **Session persistence** (nostra: JSON file; loro: SQL).
- **Settings UI** per provider, agent, MCP, generation parameters.
- **Cumulative cost tracking** (noi solo su remote, loro su tutti i
  provider commerciali).

---

## 6. Possibili direzioni

Se l'obiettivo è ridurre il gap con opencode **senza snaturare** il
progetto (che ha una sua nicchia ben definita: inferenza locale
DeepSeek su Mac), le mosse che daranno più valore con meno sforzo:

### 6.1 Quick wins (settimane, non mesi)

1. **MCP server filesystem/shell built-in.** Compilare/distribuire i
   server MCP ufficiali (`@modelcontextprotocol/server-filesystem`,
   `…-shell`, `…-fetch`) come opzionali nell'app, in modo che
   out-of-the-box un agent possa leggere/scrivere/eseguire senza che
   l'utente debba configurarli a mano. È quello che differenzia un
   chat client da un coding agent senza riscrivere niente in Swift.

2. **Plan mode**. Aggiungere un flag a `AgentConfig`
   (`isReadOnly: Bool`) che (a) filtra i tool MCP che dichiarano
   side-effect, (b) inserisce una conferma utente per gli altri.
   Granularità minima: blocca `shell`/`bash` e tool `write_*`.

3. **Per-project agent config**. Leggere `.deepseek/agents.json` (o
   `.deepseek/config.json`) dal cwd del progetto attaccato, così gli
   agent diventano versionabili in git insieme al codice — è il
   pattern di `.opencode/` e di `CLAUDE.md`.

4. **Slash command palette** nel composer (`/help`, `/clear`,
   `/model`, `/agent`, `/reset-context`, custom). Bassa complessità,
   alta visibilità.

5. **Compaction automatica**. Quando il context > N token, chiedere
   al modello stesso un riassunto della prima metà e sostituirla. Già
   parzialmente abilitato da `pendingTurn` ma non automatico.

### 6.2 Medio termine

6. **Server mode**. Esporre `InferenceService` su `localhost:PORT`
   con un'API JSON (OpenAI-compatible è la scelta più ovvia). Questo
   sblocca:
   - **TUI** scritta in qualsiasi linguaggio (anche un wrapper su
     `opencode` come client puntato al nostro server).
   - Plugin VS Code / Zed che parlano al nostro runtime locale.
   - Test e automazioni esterne.

7. **LSP client minimale**. Anche solo go-to-definition e diagnostics
   sui linguaggi più comuni cambiano radicalmente l'utilità
   dell'agent su codice reale.

8. **Tool nativi in Swift** (non via MCP) per i casi più caldi:
   `read`, `edit` (string-replace), `glob`, `grep`. La latenza
   round-trip MCP per `read` di un file da 50 righe è inutilmente
   alta — un tool in-process costa zero.

9. **Snapshot / revert** del turno. Già metà del lavoro è fatto col
   KV snapshot; manca l'esposizione a livello UI ("rifai con seed
   diverso", "annulla l'ultimo turno").

### 6.3 Lungo termine / strutturale

10. **Plugin Swift**. Trasformare `MCPClient` in uno di N backend di
    tool; aggiungere un protocollo `ToolProvider` con caricamento da
    bundle `.bundle` o eseguibile esterno. Senza questo, ogni nuovo
    tool nativo è un commit nel core.

11. **Multi-provider non solo via OpenRouter**. Driver diretti per
    Anthropic, OpenAI, Google, Azure, ollama locale. OpenRouter resta
    valido come fallback ma toglie un livello di indirezione (e di
    margine sul costo).

12. **Sharing**. Esportare una conversazione come HTML o gist /
    permalink (anche solo locale a un server statico). Bassa
    priorità rispetto alle altre, ma è una delle "killer feature" di
    opencode/Claude.

13. **Sandboxed shell**. Se aggiungiamo un tool `shell` nativo, deve
    girare in una sandbox (sandbox-exec su macOS è disponibile).
    Saltare questo step è la differenza tra "tool utile" e "buco di
    sicurezza".

### 6.4 Quello che **non** ha senso copiare

- **TUI come interfaccia primaria**: il bacino macOS è SwiftUI-first;
  la TUI è una nicchia ortogonale e mantenerne due è doppio lavoro.
  Meglio una API server-mode e lasciare che chi vuole una TUI scriva
  un client.
- **Cross-platform** Windows/Linux: il vincolo Metal è inaggirabile
  per la parte locale, quindi un porting completo perderebbe il
  selling point principale del progetto (V4 su Mac).
- **Storybook / monorepo turbo**: oversize per la dimensione attuale
  del codebase Swift.
- **Enterprise / Slack** package: troppo specifici per la stage
  attuale.

---

## 7. Sintesi visiva

```
                        Strumenti agentici sul codice
                    (read/write/edit/shell/grep/lsp/...)
                                  ▲
                                  │
                                  │  opencode
                                  │
       ┌──────────────────────────┼───────────────────────────┐
       │                          │                           │
       │                          │                           │
       │       SOVRAPPOSIZIONE    │                           │
       │       (MCP, agent,       │                           │
       │        tool-call,        │                           │
       │        streaming)        │                           │
       │                          │                           │
       │                          │                           │
       └──────────────────────────┼───────────────────────────┘
                                  │
                                  │  Questo progetto
                                  │
                                  ▼
                    Inferenza locale DeepSeek-V4
                    (kernels Metal, quantizzazione,
                     streaming loader, KV snapshot)
```

Le due aree sono in larga parte disgiunte. La strategia che
massimizza ROI è:

- **tenere** il nostro asse (inferenza locale) come differenziatore;
- **prendere in prestito** da opencode il pattern client/server +
  tool nativi sul codice, così l'agent locale può finalmente *fare
  qualcosa* su un repo senza che l'utente configuri sei MCP server.

---

## 8. Riferimenti

- opencode repo: <https://github.com/anomalyco/opencode>
- opencode docs: <https://opencode.ai/docs>
- Nostro `README.md`, `docs/ARCHITECTURE.md`, `TODO.md`,
  `docs/ROADMAP.md`.
