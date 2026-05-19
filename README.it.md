# DeepSeek V4 su macOS

Port Swift + Metal del transformer Mixture-of-Experts [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
per Apple Silicon, più un client desktop SwiftUI nativo che fa anche
da chat surface generica per qualsiasi modello ospitato su OpenRouter.

L'app desktop supporta:

- **Inference locale** sui pesi V4-Flash (FP8 + FP4 nativi), con
  streaming del checkpoint da ~142 GB anche su Mac da 16 GB grazie a
  un rotating buffer per-layer.
- **Inference remota** tramite OpenRouter — qualsiasi modello
  OpenAI-compatible (Claude, GPT, DeepSeek-R1, Llama 3, ecc.) con una
  sola API key.
- **Dispatcher di chat template**: template nativo DSV4 + driver
  Jinja2 (subset) per modelli non-DeepSeek (Llama / Mistral / Qwen
  / qualsiasi `tokenizer_config.json` con un `chat_template`).
- **Tokenizer multi-formato**: byte-level BPE (V4), SentencePiece
  (Llama / Mistral), WordPiece (BERT-style). Routing in
  `TokenizerLoader.load`.
- **Lettore GGUF (MVP)** — parser header + dtype pass-through
  (`F32 / F16 / BF16 / I32 / I8`). I dtype GGUF quantizzati
  (`Q4_0 / Q4_K_M / …`) non sono ancora supportati. Dettagli in
  [`docs/GGUF.md`](docs/GGUF.md).
- **Sampler suite estesa**: top-K / top-P / min-p / tail-free /
  locally-typical / Mirostat v2 / repetition / frequency /
  presence penalty.
- **Tool nativi code-agent** (read / write / edit / glob / grep /
  shell / apply_patch / webfetch / websearch / repo_clone /
  repo_overview / plan / task / todo) utilizzabili da entrambi i
  backend senza round-trip MCP.
- **Server MCP (Model Context Protocol)** come fornitori di tool
  aggiuntivi, con la stessa config di Claude Desktop.
- **Preset agente**: system prompt + allowlist di tool + skill +
  sampling defaults + thinking mode fissati per chat, con modalità
  **Plan / Build** che filtrano i tool mutating / dangerous. Gli
  agenti possono delegare sotto-task ad altri agenti (nesting
  limitato, prevenzione cicli).
- **Sistema di permission**: ogni chiamata a tool dangerous /
  mutating / network passa per uno sheet di consenso con **Deny /
  Allow once / Always allow**, persistito tra i lanci.
- **Skill** e **slash command** picker (`/mode`, `/tools`,
  `/permissions`, `/skill <nome>`, …) nel composer.
- **Tema** e **keybinding** personalizzabili dalle Impostazioni.
- **Project**: codebase / collezioni di documenti pre-tokenizzati che
  puoi attaccare a una chat per portare il contesto già al primo turno.

> **Sperimentale.** V4-Pro (1.6T parametri, ~800 GB a FP4) non entra
> nella memoria unificata di nessun Mac; il target on-device realistico
> è **DeepSeek-V4-Flash** (284B / 13B attivati). L'inference remota
> tramite OpenRouter è la risposta quando vuoi Claude, GPT, o altri
> modelli non eseguibili in locale.

🇬🇧 [English version](README.md) · 🧠 [Il modello](docs/MODEL.it.md)
· 🎲 [Sampling](docs/SAMPLING.it.md) · 🔤 [Tokenizer](docs/TOKENIZERS.it.md)
· 📥 [Loading](docs/LOADING.it.md) · 🔁 [Converter](docs/CONVERTER.it.md)
· 🏗 [Architettura (dettagli)](docs/ARCHITECTURE.md)
· 🧪 [Test](docs/TESTING.md) · 🛠 [Sviluppo](docs/DEVELOPING.md)
· 🧰 [Tool nativi](docs/TOOLS.md) · 🚦 [Modalità agente](docs/AGENT-MODES.md)
· 📦 [Lettore GGUF](docs/GGUF.md)
· ✂️ [Vocab pruning (solo IT)](docs/VOCAB-PRUNING.md)
· 🔍 [Gap analysis vs opencode](docs/GAP-ANALYSIS-OPENCODE.md)
· 🔍 [Gap analysis vs llama.cpp](docs/GAP-ANALYSIS-LLAMACPP.md)

---

## Requisiti di sistema

| Cosa | Inference locale | Solo remoto (OpenRouter) |
|---|---|---|
| **CPU/GPU** | Apple Silicon (M1, M2, M3, M4…) | Qualsiasi Mac Apple Silicon |
| **macOS** | 14.0 Sonoma | 14.0 Sonoma |
| **RAM (unified)** | 16 GB (V4-Flash, streaming) — consigliati 64+ GB | 8 GB |
| **Disco** | 150 GB liberi per i pesi V4-Flash | Pochi MB per l'app |
| **Tool** | Swift 5.10 / Xcode 15+ | Swift 5.10 / Xcode 15+ |
| **Rete** | opzionale | obbligatoria |

Il loader sceglie automaticamente la strategia di inference locale in
base alla RAM disponibile:

| RAM disponibile | Strategia | Comportamento |
|---|---|---|
| ≥ 192 GB | `preload` | Tutto residente in RAM, velocità massima |
| 32–192 GB | `mmap` | Il sistema pagina su richiesta, veloce dopo warm-up |
| 16–32 GB | `streaming` | Un layer alla volta in RAM, primo token più lento |

I Mac Intel non sono supportati — le pipeline Metal richiedono
hardware `bfloat`-capable e `Sources/DeepSeekKit/Device.swift`
rifiuta di inizializzarsi.

---

## 1. Scaricare il progetto

```bash
git clone https://github.com/giorgiooppo/DeepSeek-V4-Pro-MacOS.git
cd DeepSeek-V4-Pro-MacOS
swift package resolve
```

Il repo non include pesi del modello, tokenizer o API key — sono
gitignored / salvati in Keychain.

## 2. Build

### Solo CLI

```bash
swift build -c release
```

Produce:

- `.build/release/deepseek` — CLI di inference locale
- `.build/release/converter` — transcoder pesi offline

La CLI parla solo a checkpoint locali — non c'è dispatch OpenRouter
nel `deepseek` stesso; il supporto remoto vive solo nella GUI.

### App GUI (Xcode)

```bash
brew install xcodegen        # una sola volta
./Tools/generate-xcodeproj.sh
open DeepSeekV4Pro.xcworkspace
```

Seleziona lo scheme **`DeepSeekApp`** (il workspace espone anche il
target SPM `DeepSeekUI` — scegli il target app, non quello) e premi
⌘R.

L'app si apre subito sulla chat surface. Nessun modello è richiesto
per partire — puoi sfogliare la history, modificare agenti / project
/ server MCP, e scrivere un draft prima ancora che un backend sia
caricato.

---

## 3. Usare un modello locale

### Scaricare i pesi

Il checkpoint on-device consigliato è **DeepSeek-V4-Flash** nel layout
nativo HuggingFace (FP8 per l'attention + FP4 per gli expert). Il
loader Swift legge quel layout direttamente — non serve nessuna
conversione.

```bash
pip install --upgrade huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
    --local-dir ~/Downloads/V4-Flash-HF
```

La cartella di destinazione deve contenere:

- `config.json`, `generation_config.json`
- `tokenizer.json`, `tokenizer_config.json`
- `model.safetensors.index.json`
- 46 shard `model-NNNNN-of-NNNNN.safetensors` (~142 GB totali)

Il binario `converter` serve **solo** se vuoi transcodificare il
checkpoint in BF16 / INT8 / INT4 / INT2 per varianti più compatte —
vedi [`docs/USAGE.md`](docs/USAGE.md).

### Caricare nella GUI

1. Apri il menu **Model** della toolbar della chat (icona cpu, il
   primo a sinistra).
2. **Choose model folder…** → seleziona `~/Downloads/V4-Flash-HF`.
3. Un banner sopra il composer mostra `Loading <nome>… <gb> GB across
   <n> shards · strategy: <preload|mmap|streaming>`. Aspetta che
   scompaia.
4. La label del model picker diventa il nome della cartella. Il Send
   è abilitato.
5. La cartella viene ricordata sotto **Recent** — al prossimo avvio
   l'app la ricarica automaticamente.

**Unload current model** dallo stesso menu rilascia la RAM senza
chiudere l'app. La history della chat è indipendente dal modello
caricato.

### Eseguire dalla CLI

```bash
./.build/release/deepseek ~/Downloads/V4-Flash-HF \
    "Qual è la capitale dell'Italia?" \
    --mode chat --max-tokens 50 --temperature 0.7
```

Vedi [§ Riferimento CLI](#riferimento-cli) sotto.

---

## 4. Usare un modello remoto tramite OpenRouter

OpenRouter è un gateway API OpenAI-compatible che instrada verso
~300 modelli di Anthropic, OpenAI, DeepSeek, Meta, Mistral e altri
con una singola key.

### Configurazione una tantum: aggiungere l'API key

1. Ottieni una key su <https://openrouter.ai/keys>.
2. Apri il tab **Settings → API Keys**.
3. Incolla la key nel SecureField OpenRouter. **Save**.
4. (Opzionale) Click su **Test** — chiama `/auth/key` e mostra un
   "Key accepted" verde se le credenziali funzionano.

La key è salvata nel **Keychain** di macOS (service
`com.deepseek.v4pro`, account `openrouter.apiKey`). Non viene mai
scritta in un plist o altrove leggibile in chiaro.

### Scegliere un modello

1. Menu **Model** della toolbar → **Add OpenRouter model…**.
2. Lo sheet carica il catalogo completo di OpenRouter (cachato per
   24 h sotto `Application Support/.../openrouter-catalog.json`).
   Cerca per provider / nome / slug; ogni riga mostra context length,
   pricing per-token, breve descrizione.
3. Click su una riga → `ModelState` valida la key + cambia
   l'endpoint della chat.
4. Il modello compare ora sotto **Recent** affianco alle eventuali
   cartelle locali, e la label della toolbar mostra lo slug (es.
   `claude-3.5-sonnet`).

### Invia e osserva i costi

- La risposta streamma via SSE come un'inference locale.
- DeepSeek-R1 e o-series mettono il reasoning in `reasoning_content`
  che viene renderizzato attraverso la stessa disclosure
  brain-icon dei blocchi `<think>` locali.
- La **ThroughputBar** sotto la bolla mostra `Turn cost: $0.0042`
  per il turno più recente.
- Un banner separato sopra il composer mostra `Chat total: $0.013`
  cumulativo per l'intera conversazione, persistito tra i riavvi.

### Cosa funziona su remoto

- Tool MCP (vedi § Agenti & tool): esposti automaticamente come
  array `tools` OpenAI; il loop di tool-call fa round-trip HTTP fino
  a 8 iterazioni.
- Preset agente (system prompt + sampling defaults + tool
  allowlist): applicati al body della richiesta.
- Reasoning mode (`high` / `max`): tradotto nell'hint
  `reasoning: {effort}` di OpenRouter. I provider che non lo
  supportano (la maggior parte dei non-R1/o-series) lo ignorano
  silenziosamente.
- Picker thinking-mode (sopra il composer): rispettato per turno.

### Cosa non funziona ancora su remoto

- **Delegation cross-agente** (`__delegate_to_agent`): non esposto
  sulle chat remote. Gli agenti locali gestiscono tutto — la variante
  remota richiederebbe un loop sub-agent remoto non ancora scritto.
- **Crash recovery**: i turni remoti non snapshottano un
  `pendingTurn`. Se l'app crasha mid-stream il turno si perde
  (rinvia il messaggio).
- **Prompt caching** (sconti Anthropic / OpenAI prompt-cache via
  OpenRouter): non ancora implementato.

---

## 5. L'app macOS

### Pickers della toolbar

Da sinistra a destra, quattro menu:

| Picker | Icona | Scopo |
|---|---|---|
| **Model** | cpu / disco / nuvola | Cambia backend. Cartella locale, modello OpenRouter remoto, browse, unload. |
| **Agent** | personalizzata per agente | Attacca un preset `AgentConfig` a questa chat. None / lista. |
| **Project** | folder | Attacca un project pre-tokenizzato per portare quel contesto al primo turno. None / lista. |
| **Convert** | bacchetta | Apre lo sheet di quantizzazione offline pesi. |

Ogni picker riflette la conversazione attiva; cambiare conversazione
nel sidebar aggiorna le label.

### Chat surface

Sopra il composer, dall'alto al basso:

1. **Banner stato modello**: nascosto quando ready; altrimenti
   "Loading …" / "No model" / "Could not load …" con retry /
   force-load.
2. **Banner costo cumulativo**: `Chat total: $X.XX` per chat remote
   che hanno fatturato qualcosa. Nascosto per chat locali.
3. **Catena delegation live**: card stack per ogni sub-agente in
   flight (ciascuno con la sua icona + task + reply streaming,
   indentati per profondità). Stack vuoto si collassa.
4. **Banner resume**: quando una generazione precedente è morta a
   metà, lo snapshot `pendingTurn` offre un resume one-click.
5. **Picker mode + thinking**: due segmented control — `Build /
   Plan` per la modalità agente (Plan filtra i tool mutating /
   dangerous), e `No think / High / Max` per il reasoning effort.
   Entrambi si bloccano sui valori dell'agente attached con un
   suggerimento 🔒 quando l'agente li fissa.
6. **Composer**: TextField (Cmd+↩ per inviare) + Send/Stop button.
   Digita `/` per aprire la palette degli slash command (mode,
   tools, permissions, skills, …). Send disabilitato quando non
   c'è modello caricato.

Durante lo streaming, la bolla assistant in corso mostra un caret
lampeggiante. La ThroughputBar sotto ticka `tok/min` e (per remoto)
`Turn cost`. Il reasoning content è ripiegato in una disclosure
brain-icon; tool call + output sono ripiegati in una disclosure
wrench-icon sotto la bolla.

### Sidebar

Lista conversazioni con timestamp. Click destro → Delete. Cmd+N crea
una nuova chat sotto il modello corrente. Un piccolo spinner segna la
chat che sta generando attivamente.

---

## 6. Agenti e tool

### Agenti (Settings → Agents)

Un **Agente** è un preset che raggruppa:

- `name`, `summary`, icona + tinta;
- system prompt iniettato prima di ogni turno della chat a cui è
  attaccato;
- sampling defaults (`temperature`, `topP`, `topK`, `repPenalty`,
  `maxTokens`, `defaultMode`) che sovrascrivono gli slider della
  tab Generation;
- allowlist di tool MCP (`nil` = tutti i tool, vuoto = nessuno, set
  esplicito = whitelist di nomi qualificati `<server>__<tool>`).

Attacca uno alla chat dal picker Agent in toolbar; stacca con
"None". Quando attaccato, il picker thinking si blocca sul
`defaultMode` dell'agente, e le sampling settings della chat
provengono dall'agente anche se tocchi gli slider globali.

Definisci gli agenti sotto **Settings → Agents**: master-detail con
uno sheet di edit per tutto quanto sopra più un segmented control
per la tool-policy.

### Delegation tra agenti

Quando sono registrati due o più agenti, quello attaccato alla chat
riceve un tool sintetico chiamato `__delegate_to_agent` il cui
schema include un elenco di ogni altro agente. Il modello può
chiamarlo con `{ agent_name, task }`; l'host esegue l'agente
nominato in isolamento attraverso uno snapshot/restore della KV
cache e ne restituisce la reply finale come output del tool. Limiti:

- **Cap di nesting**: fino a 3 livelli
  (`host → sub → sub-sub → sub-sub-sub`). Allo cap il schema non
  viene iniettato, quindi il modello non può nemmeno provare a
  delegare oltre.
- **Prevenzione cicli**: un agente già nella call stack attiva
  viene rifiutato con una stringa di errore strutturata. Il modello
  può auto-correggersi scegliendo un agente diverso.
- **Preservazione cache**: a ogni livello la KV cache viene
  snapshottata prima dell'esecuzione del sub-agente e ripristinata
  al ritorno, così l'host non paga un cold re-prefill.

La chat mostra la catena live (con il buffer streaming di ogni
livello) in una card pinned sopra il composer. La delegation è solo
per modelli locali per ora.

### Server MCP (Settings → MCP)

Configura server MCP stdio-based esattamente come Claude Desktop:
command + args + env. Importa da un config JSON di Claude Desktop
(`mcpServers: { … }`) con un click.

I server abilitati si spawnano all'avvio app attraverso il PATH
shell dell'utente (homebrew, fnm, pyenv, …) più un fallback esteso
così l'environment stripped di launchd non rompe la discovery dei
tool. Il footer di stato di ogni riga mostra lo stato della
connessione + l'elenco dei tool live. Il bottone Reconnect è
inline.

I tool sono esposti alla chat attiva tramite il backend su cui è
(blocchi DSML locali, o array `tools` OpenAI per remoto).

### Project (Settings → Projects) & Documenti (Settings → Documents)

Un **Documento** è un singolo file di testo ingestito una volta e
pre-tokenizzato contro il tokenizer del modello attivo. Un
**Project** è una collezione nominata di documenti.

Attacca un project a una chat dal picker Project in toolbar. Al
primo turno di quella chat i file del project vengono inseriti nel
prompt con i token nativi di delimitatore repo / file
(`<｜begin▁of▁repo▁name｜>` ecc.), così il modello li tratta come
contesto code-aware invece che come testo free-form. Solo per
modelli locali.

---

## 7. Tool nativi, Plan / Build, skill

Il target `DeepSeekTools` fornisce una toolbox code-agent che il
modello può invocare direttamente — niente round-trip MCP.
Riferimento completo: [`docs/TOOLS.md`](docs/TOOLS.md).

### Catalogo

| Categoria | Tool |
|---|---|
| **readOnly** | `read`, `glob`, `grep`, `repo_overview`, `lsp` (stub) |
| **planning** | `plan`, `task`, `todo` |
| **mutating** | `write`, `edit`, `apply_patch` |
| **dangerous** | `shell`, `repo_clone` |
| **network** | `webfetch`, `websearch` |

Il tool `lsp` è registrato come stub oggi; lo spawn di un vero
client `sourcekit-lsp` è in roadmap.

### Modalità agente Plan vs Build

Ogni agente opera in una di due modalità coarse (riferimento:
[`docs/AGENT-MODES.md`](docs/AGENT-MODES.md)):

- **Build** — ogni tool eleggibile. I tool mutating / dangerous /
  network passano attraverso la policy delle permission.
- **Plan** — i tool `.mutating` + `.dangerous` sono filtrati via
  dallo schema che il modello vede, quindi non può nemmeno
  proporli. I tool `.network` richiedono comunque consenso.

Cambiare modalità in tre posti:

1. Default per-agente in **Settings → Agents** → edit → segmented
   control "Agent mode" (persistito in `agents.json`).
2. Flip per-conversazione dal mode picker della toolbar.
3. Slash command inline `/mode plan` o `/mode build`.

### Flusso di permission

Il dispatch dei tool attraversa, in ordine: filtro mode →
default durabili `PermissionStore` (`alwaysAllow / alwaysDeny /
ask`) → cache di sessione → modal `PermissionPromptView` con tre
azioni: **Deny**, **Allow once**, **Always allow**. I grant
"always" si modificano da **Settings → Permissions**.

### Skill e slash command

Una **Skill** è un template di istruzioni che l'agente può
attivare. Gli id delle skill built-in sono UUID stabili; gli
agenti dichiarano quali consentire via `allowedSkillIDs`.
Attivane una inline con `/skill <nome>` o dalla palette degli
slash command.

Il testo prefissato `/` nel composer apre la palette dei slash
command. I built-in includono `/mode`, `/tools`, `/permissions`,
`/skill`, `/clear`, `/theme`. Slash command custom possono essere
aggiunti da **Settings → Slash Commands** (quando implementato —
vedi TODO).

---

## 8. Preferenze

Tutte le tab raggiungibili via `Cmd+,`. Le modifiche hanno effetto
sul prossimo Send (o, per `Model Config`, al prossimo caricamento
modello).

| Tab | Cosa controlla |
|---|---|
| **Generation** | Temperatura (slider 0.5–1.0, default 0.7), top-K (0 = disabilitato), top-P, max-tokens, thinking mode. Sovrascritto quando un agente è attaccato. |
| **Loading** | Override strategia loader, toggle force-load, percorso binario converter. |
| **Model Config** | Ogni campo di `ModelConfig`. Scrive in `~/Library/Application Support/<app>/config-overrides.json`. |
| **Agents** | CRUD per i preset agente — modalità Plan/Build, thinking mode, system prompt, allowlist skill, allowlist tool MCP, sampling defaults. |
| **Tools** | Inventario read-only di ogni tool nativo che l'agente può invocare + matrice di disponibilità Plan/Build. |
| **Permissions** | Default "ask / always allow / always deny" per-(tool, categoria) consultati prima di ogni dispatch. |
| **Skills** | Gestisce la libreria delle skill — built-in, custom, editor dell'allowlist usato dalla tab Agents. |
| **Theme** | Appearance (light / dark / system), tinting accent + bolle, import di tema custom. |
| **Keybindings** | Inventario read-only + reset ai default. UI di rebind inline in roadmap. |
| **Documents** | Importa singoli documenti (tokenizza contro il modello locale caricato). |
| **Projects** | Raggruppa documenti in project per injection one-shot di contesto. |
| **MCP** | Registra server MCP — vedi § Agenti e tool. |
| **API Keys** | API key OpenRouter (Keychain). Save / Test / Delete. |
| **Storage** | Posizione della history, dimensione su disco, Reveal / Clear all. |

---

## 9. Riferimento CLI

La CLI è solo locale (nessun dispatch OpenRouter).

```
deepseek <model-dir> "<prompt>" [opzioni]
```

Due argomenti posizionali; il secondo è opzionale solo in modalità
diagnostiche.

### Flag di generazione

| Flag | Tipo | Default | Cosa fa |
|---|---|---|---|
| `--mode` | `raw` \| `chat` | `chat` | `raw` antepone solo BOS; `chat` applica il template chat V4. |
| `--thinking` | `off` \| `high` \| `max` | `off` | Budget di ragionamento in chat. `off` appende `</think>`; `high` appende `<think>`; `max` aggiunge anche il blocco REASONING_EFFORT_MAX. |
| `--temperature` | float | `1.0` | Temperatura di sampling. **Imposta `0.7`** — vedi Valori consigliati. |
| `--max-tokens` | int | `32` | Numero massimo di token da generare. |

### Flag sampler (tutti default "disabilitato")

| Flag | Tipo | Default | Cosa fa |
|---|---|---|---|
| `--top-k` | int | `0` | Mantiene solo i top-K logit. |
| `--top-p` | float | `1.0` | Cutoff nucleus. |
| `--min-p` | float | `0.0` | Scarta token con `p < min_p × max_p`. |
| `--tfs` | float | `1.0` | Tail-free sampling, parametro `z`. |
| `--typical` | float | `1.0` | Locally-typical mass. |
| `--repetition-penalty` | float | `1.0` | Penalty stile HuggingFace su id in history. |
| `--frequency-penalty` | float | `0.0` | Stile OpenAI, scala con la count. |
| `--presence-penalty` | float | `0.0` | Stile OpenAI, binario sulla presenza. |
| `--mirostat` | float | off | Abilita Mirostat v2 con surprise target τ. |
| `--mirostat-eta` | float | `0.1` | Learning rate Mirostat. |

### Flag loader / memoria

| Flag | Tipo | Default | Cosa fa |
|---|---|---|---|
| `--load-strategy` | `auto` \| `preload` \| `mmap` \| `streaming` | `auto` | Forza una specifica strategia di caricamento pesi. |
| `--force-load` | flag | off | Bypassa i controlli di sicurezza RAM. |
| `--max-seq-len` | int | da `config.json` | Override delle righe KV-cache per-layer. Più basso = meno RAM, contesto più corto. |
| `--max-batch-size` | int | da `config.json` | Override della dimensione batch della KV cache. |

### Modalità diagnostiche

| Flag | Cosa fa |
|---|---|
| `--print-config` | Carica `config.json`, stampa il `ModelConfig` risolto ed esce. |
| `--trace-norms` | Stampa L2 norm + min/max/mean + contatori NaN/Inf del residual stream. |
| `--list-tensors [PREFIX]` | Elenca tutti i nomi di tensor nel checkpoint, eventualmente filtrati. |
| `--dump-tensor NAME[:row=R][:cols=A..B]` | Dequantizza una riga di slice e stampa i valori. |

### Valori consigliati

- **`--temperature 0.7`**. Sotto greedy argmax il routing MoE di
  V4-Flash cade in loop auto-rinforzanti (`好的好的好的…`). Valori
  intorno a 0.6–0.9 producono campioni più coerenti. La GUI vincola
  lo slider a `[0.5, 1.0]` per lo stesso motivo.
- **`--mode chat --thinking off`** per Q&A brevi; `--thinking high`
  per problemi in cui il modello deve "pensare ad alta voce".

### Esempio

```bash
./.build/release/deepseek ~/Downloads/V4-Flash-HF \
    "Spiega la fusione nucleare in due frasi." \
    --mode chat --thinking off \
    --temperature 0.7 --max-tokens 256
```

---

## 10. Risoluzione problemi

**Il primo token su modello locale impiega minuti.**
Atteso sotto `streaming` su un Mac da 16 GB — lo shard di ~3 GB di
ogni layer deve essere letto da disco prima dell'elaborazione. I
token successivi sono molto più rapidi (il rotating slot resta
caldo).

**La chat OpenRouter fallisce con "API key not configured".**
Settings → API Keys → incolla e Save. Il picker mostra anche un
warning inline quando la key non è configurata.

**OpenRouter restituisce 401 / 403.**
La key è invalida, scaduta o senza credito. Premi Test nel tab API
Keys; controlla limiti di uso sulla dashboard OpenRouter.

**Le tool call non si eseguono su remoto.**
Verifica che il footer di stato del server MCP mostri
`Connected · N tools` in Settings → MCP. Il path remoto usa lo
stesso pool del locale; se il server è offline il modello riceve un
errore strutturato.

**Il reasoning content non appare su remoto.**
Solo modelli come DeepSeek-R1 e o-series emettono
`reasoning_content`. Altri modelli includono il reasoning inline in
`content` (Claude extended-thinking) o non lo emettono affatto
(Llama). La disclosure brain-icon si renderizza solo quando il
campo è popolato.

**Errore di build: `precompiled file '…' was compiled with module
cache path '…'`.**
Cache Xcode stale, solitamente dopo aver spostato la cartella di
progetto. Pulisci:

```bash
rm -rf .build
swift package clean
swift build
```

Per Xcode svuota anche
`~/Library/Developer/Xcode/DerivedData/DeepSeekV4Pro-*`.

**Il modello locale continua a ripetere lo stesso token.**
Stai samplando con `--temperature 0`. V4-Flash richiede sampling
stocastico — passa `--temperature 0.7`. La GUI lo vincola.

**"No Metal device" / Mac Intel.**
Apple Silicon è richiesto per l'inference locale. Il remoto
(OpenRouter) funziona su qualunque Mac che soddisfi il minimo
macOS 14.

**Out of memory al caricamento locale.**
Prova `--load-strategy streaming` per forzare il rotating loader, o
`--max-seq-len 2048 --max-batch-size 1` per ridurre la KV cache.

**"Cannot find type 'MCPClientPool'" / simile dopo un git pull.**
Il progetto Xcode è generato da `project.yml`. Rigenera dopo aver
aggiunto nuovi file:

```bash
./Tools/generate-xcodeproj.sh
```

`swift build` non richiede rigenerazione — SPM raccoglie i nuovi
file automaticamente.

---

## 11. Licenza e crediti

Il codice Swift in questo repository è MIT (vedi [`LICENSE`](LICENSE)).
I pesi del modello DeepSeek e l'implementazione Python di
riferimento in `Reference/inference/` appartengono a DeepSeek —
vedi la loro [scheda Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
per i termini di licenza.

OpenRouter è un servizio di terze parti separato; il tuo uso è
governato dai loro termini.

Se vuoi capire come funziona il port sotto il cofano — mapping dei
kernel, amplificazione del residual stream, design dello streaming
pool, dispatch MoE — leggi [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Per il workflow contributore vedi [`docs/DEVELOPING.md`](docs/DEVELOPING.md),
per esempi e ricette pronte vedi [`docs/EXAMPLES.md`](docs/EXAMPLES.md).
