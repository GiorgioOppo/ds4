# Istruzioni passo-passo

Guida operativa in italiano dal Mac vuoto al primo token generato.
Tutto in ordine, niente di saltato.

Ci sono **tre percorsi** possibili. Scegli quello che fa per te:

- **Solo remoto (OpenRouter)** — la via più veloce. Niente download
  da 140 GB, niente Apple Silicon richiesta in modo stringente, paghi
  per token. Vai al [§ 6](#6-percorso-veloce-solo-openrouter).
- **Solo locale (V4-Flash)** — gira sul Mac, zero costi a regime,
  serve scaricare i pesi e avere abbastanza RAM. Vai al
  [§ 2](#2-prepara-i-pesi-del-modello-locale).
- **Entrambi** — local + remote convivono nello stesso client.
  L'utente sceglie da un picker nella toolbar al volo. Vai dal
  [§ 2](#2-prepara-i-pesi-del-modello-locale).

Se sei già dentro il progetto e cerchi solo il riferimento dei
flag, vai a [`USAGE.md`](USAGE.md). Questa guida è la versione
lunga, fatta per chi parte da zero.

---

## Prima di iniziare

Cosa serve avere a portata di mano:

- **macOS 14 (Sonoma) o superiore** con processore Apple Silicon
  (M1, M2, M3, M4 e relative varianti Pro/Max/Ultra). Su Intel Mac
  il path locale non gira (serve la GPU Apple + supporto BF16 di
  Metal 3+). Il path OpenRouter funziona comunque.
- **Xcode 15+** installato. Se non ce l'hai, scaricalo dall'App
  Store (gratis), poi nel Terminale:
  ```bash
  xcode-select --install
  ```
- **Spazio libero** (solo per il path locale): V4-Flash è ~140 GB
  in formato HuggingFace nativo. Pianifica almeno **150 GB liberi**,
  meglio su SSD interno o NVMe esterno veloce.
- **RAM** (solo per il path locale):
  - 16 GB: V4-Flash gira con strategia `streaming`, primo token
    lento (30 s – 3 min), poi tollerabile.
  - 64 GB: comodo con `mmap`.
  - 192 GB+: tutto residente (`preload`), velocità massima.
- **Connessione di rete** decente (per scaricare pesi o usare
  OpenRouter).

Strumenti aggiuntivi da installare con Homebrew (vai a
<https://brew.sh> se non ce l'hai):

```bash
brew install xcodegen           # serve per generare il progetto Xcode
brew install huggingface-cli    # serve solo per il download pesi locali
```

---

## 1. Clona e compila il progetto

Scegli una cartella di lavoro (l'esempio usa `~/Documents`):

```bash
cd ~/Documents
git clone https://github.com/giorgiooppo/DeepSeek-V4-Pro-MacOS.git
cd DeepSeek-V4-Pro-MacOS
```

Lo script che compila i kernel Metal deve essere eseguibile (una
sola volta):

```bash
chmod +x Plugins/MetalLibPlugin/build_metallib.sh
```

### 1a. Binari CLI

```bash
swift build -c release
```

La prima compilazione è lenta (~5–10 min) perché genera il
`default.metallib` con tutti gli shader Metal. Produce due binari:

- `.build/release/deepseek` — inference locale a riga di comando
- `.build/release/converter` — transcodifica offline dei pesi

### 1b. App GUI (Xcode)

```bash
./Tools/generate-xcodeproj.sh
open DeepSeekV4Pro.xcworkspace
```

Nella barra schema in alto a sinistra di Xcode seleziona
**DeepSeekApp** (NON "DeepSeekUI", che è il target SPM eseguibile).
Premi ⌘R per compilare e avviare.

L'app si apre direttamente sulla schermata della chat. Sopra il
composer vedi il banner `No model loaded` con l'invito a sceglierne
uno dalla toolbar — è normale al primo avvio.

---

## 2. Prepara i pesi del modello locale

> Salta questa sezione se vuoi usare solo OpenRouter — vai al
> [§ 6](#6-percorso-veloce-solo-openrouter).

### 2.1. Download da HuggingFace

Il checkpoint consigliato è **DeepSeek-V4-Flash** nel formato
nativo HuggingFace (FP8 + FP4). Il loader Swift legge questo
formato direttamente, senza conversioni.

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ~/Downloads/V4-Flash-HF
```

Il download è di circa 142 GB. Va lasciato lavorare; puoi
interromperlo e riprenderlo con lo stesso comando.

A fine download, nella cartella `~/Downloads/V4-Flash-HF/` devono
esserci:

- `config.json`, `generation_config.json`
- `tokenizer.json`, `tokenizer_config.json`
- `model.safetensors.index.json`
- 46 shard `model-NNNNN-of-NNNNN.safetensors`

### 2.2. (Opzionale) Conversione in variante compatta

Salta questa sezione se hai abbastanza disco — il path nativo HF è
più semplice e altrettanto veloce.

Se vuoi una versione INT4 (~160 GB) o INT8 (~290 GB) per risparmiare
spazio, il binario `converter` la produce:

```bash
.build/release/converter \
  --hf-ckpt-path ~/Downloads/V4-Flash-HF \
  --save-path   ~/Downloads/V4-Flash-int4 \
  --n-experts 256 \
  --target-dtype int4 \
  --shard-size-gb 5
```

A fine conversione devi copiare manualmente il `config.json` nella
cartella di output (il converter non lo fa):

```bash
cp ~/Downloads/V4-Flash-HF/config.json \
   ~/Downloads/V4-Flash-int4/config.json
```

Trade-off dei dtype: vedi
[`USAGE.md § 2.2`](USAGE.md#22-optional-convert-to-a-compact-variant).

---

## 3. Carica il modello nella GUI

1. Apri l'app (⌘R da Xcode, o doppio-click sull'.app prodotto in
   `~/Library/Developer/Xcode/DerivedData/.../Build/Products/Debug`).
2. Nella toolbar in alto, premi il menu **Model** (icona cpu, il
   primo a sinistra).
3. Scegli **Choose model folder…** → seleziona la cartella del
   checkpoint (`~/Downloads/V4-Flash-HF`).
4. Sopra il composer compare un banner:
   ```
   Loading V4-Flash-HF…
   142 GB across 46 shards · strategy: mmap
   ```
   La strategia (`preload` / `mmap` / `streaming`) è scelta
   automaticamente in base alla RAM disponibile.
5. Aspetta. Su un Mac da 16 GB il primo caricamento può richiedere
   diversi minuti perché il sistema deve mappare i pesi dal disco.
   Quando il banner sparisce, la label del picker mostra il nome
   della cartella e il pulsante Send si attiva.
6. La cartella viene ricordata. Al prossimo avvio l'app la
   ricarica automaticamente.

Vuoi liberare la RAM senza chiudere l'app? Stesso menu, **Unload
current model**. La history della chat resta — la chat smette solo
di rispondere finché non ricarichi un modello.

---

## 4. Invia il primo messaggio (locale)

1. Nel composer in basso scrivi una domanda, per esempio:
   ```
   Spiegami la fusione nucleare in tre frasi.
   ```
2. Premi **Send** (o `⌘↩`).
3. Sopra il composer compare un indicatore di prefill:
   ```
   Prefilling 256 tokens · 12.3s
   ```
   Il modello sta elaborando il prompt prima di campionare il primo
   token. Su Mac da 16 GB con strategia streaming questa fase è
   lenta (lecture del primo layer da SSD).
4. I token compaiono uno alla volta nella bolla dell'assistant,
   con un caret lampeggiante.
5. Sotto la bolla la **ThroughputBar** mostra:
   ```
   Prefill: 256 tok in 8.32s · 1850 tok/min
   Generation: 42 tok in 9.15s · 275 tok/min
   ```
6. Quando il modello finisce, il caret sparisce e compaiono i
   comandi (Copia, eventuali tool call, eventuale disclosure del
   reasoning).

### Cambiare il thinking mode

Sopra il composer c'è un picker `🧠 [No think | High | Max]`. Lo
trovi tra il banner cost e il composer.

- **No think**: il modello risponde direttamente.
- **High**: il modello produce un blocco `<think>...</think>` prima
  della risposta. Visibile come disclosure cliccabile (icona
  cervello).
- **Max**: come High ma aggiunge anche il blocco `REASONING_EFFORT_MAX`
  al system prompt.

Il valore è globale (cambia per il prossimo turno) e persistente.
Se un agente è attaccato alla chat, il picker si blocca sul
`defaultMode` dell'agente e mostra `🔒 set by <nome agente>`.

### Cambiare i parametri di sampling

Tutti i parametri (temperature, top-K, top-P, ripetition penalty,
max-tokens) stanno in **Settings (⌘,) → Generation**. Le modifiche
hanno effetto sul Send successivo.

> **Attenzione.** V4-Flash con `temperature = 0` cade in loop su un
> singolo token filler. La GUI vincola lo slider a `[0.5, 1.0]`
> proprio per questo. Il valore consigliato è **0.7**.

---

## 5. Conversazioni multiple e progetti

### Nuova chat

`⌘N` (o clic destro nella sidebar → Nuovo). Ogni chat ha la sua
storia, indipendente dalle altre.

### Cancellare una chat

Clic destro sulla riga della sidebar → Delete.

### Allegare un "project"

Se vuoi che il modello "ricordi" un set di file di codice o
documenti in modo permanente per quella chat:

1. **Settings → Documents** → importa i file singoli. L'app li
   tokenizza una volta sola contro il modello locale corrente.
2. **Settings → Projects** → crea un project, seleziona i documenti
   che ti interessano.
3. Nella toolbar della chat, picker **Project** → scegli il project.
4. Al primo Send di quella chat, i file del project vengono
   inseriti nel prompt con i token delimitatori nativi V4 (così il
   modello li tratta come codice indicizzabile, non come testo
   libero).

Solo per modelli locali — su OpenRouter i project non sono ancora
supportati.

---

## 6. Percorso veloce: solo OpenRouter

Se non vuoi scaricare 140 GB di pesi, puoi usare l'app come client
verso OpenRouter, che ospita Claude, GPT, DeepSeek-R1, Llama, ecc.

### 6.1. Crea una API key

1. Vai su <https://openrouter.ai>, registrati, ricarica credito.
2. Pagina **Keys** → crea una nuova key. Inizia con `sk-or-...`.
3. Copiala (sarà visibile solo una volta).

### 6.2. Salvala nell'app

1. App → **⌘,** (Settings) → tab **API Keys**.
2. Incolla la key nel campo OpenRouter (è un SecureField, non
   vedrai il contenuto).
3. **Save**.
4. (Consigliato) Click su **Test** — chiama `/auth/key` e mostra
   "Key accepted by OpenRouter" se va tutto bene.

La key viene salvata nel **Keychain di macOS** (servizio
`com.deepseek.v4pro`, account `openrouter.apiKey`). Non viene mai
scritta in un plist o in nessun altro file leggibile in chiaro.

### 6.3. Scegli un modello

1. Toolbar della chat → menu **Model** → **Add OpenRouter model…**.
2. Lo sheet che si apre mostra il catalogo completo di OpenRouter
   (~300 modelli). La prima volta lo scarica via rete; è cachato in
   locale per 24 ore.
3. Usa la barra di ricerca per filtrare: prova `anthropic`,
   `sonnet`, `deepseek`, `r1`.
4. Ogni riga mostra: nome, slug provider/modello, dimensione del
   contesto, prezzi per milione di token (`$X.XX/M in · $Y.YY/M
   out`) o `free` per i modelli gratuiti.
5. Click su una riga → l'app valida la API key, poi marca quel
   modello come attivo. Quasi instantaneo (niente pesi da caricare).

Il modello scelto appare ora sotto **Recent** nel menu Model e
verrà ricaricato automaticamente al prossimo avvio.

### 6.4. Invia il primo messaggio (remoto)

Identico al flusso locale: scrivi e premi Send. Differenze:

- La latenza del primo token è di 1–3 secondi (dipende dal
  provider upstream — OpenRouter aggiunge poco overhead).
- I modelli reasoning (DeepSeek-R1, o-series) emettono il
  ragionamento in `reasoning_content`, renderizzato nella stessa
  disclosure cervello dei `<think>` locali.
- Dopo la risposta, sotto la bolla compare `Turn cost: $0.0042`
  per quel turno. Sopra il composer un altro banner mostra il
  totale cumulativo della chat (`Chat total: $0.013`), persistente
  tra i riavvi.

---

## 7. Agenti e tool

### Definire un agente

1. **Settings → Agents** → premi **+** in basso a sinistra.
2. Compila:
   - **Name**: come si chiama (es. `Code reviewer`).
   - **Summary**: una riga che descrive a cosa serve (es. "Reviews
     Swift PRs"). Compare nei picker.
   - **System prompt**: testo libero che diventerà il primo system
     message di ogni chat a cui questo agente sarà attaccato.
   - **Default thinking mode**: chat / high / max.
   - **Sampling defaults**: temperature, top-K, top-P, max-tokens
     che sovrascrivono gli slider globali.
   - **Allowed MCP tools**: scegli quali tool MCP questo agente può
     usare (tutti / nessuno / whitelist esplicita).
   - Icon + tint per il riconoscimento visivo.

### Attaccare un agente alla chat

Toolbar → picker **Agent** → scegli quello che vuoi. Da quel
momento, ogni messaggio in quella chat passa attraverso le
impostazioni dell'agente.

Stacca con **None**.

### Server MCP (Model Context Protocol)

MCP è un protocollo standard per esporre tool al modello (browsing
filesystem, web search, database, ecc.). Stesso formato config di
Claude Desktop.

1. **Settings → MCP** → premi **+** in basso a sinistra.
2. Compila:
   - **Name**: identificatore arbitrario (es. `filesystem`).
   - **Command**: es. `npx`.
   - **Args** (uno per riga): es.
     ```
     @modelcontextprotocol/server-filesystem
     /path/to/files
     ```
   - **Env** (chiave=valore, uno per riga): variabili d'ambiente.
   - **Enabled**: deve essere ON per spawnare il server.
3. Salva. L'app fa lo spawn del processo, fa l'handshake JSON-RPC,
   chiede la lista dei tool.
4. Il footer della riga mostra lo stato: **Connected · N tools**
   con elenco dei tool disponibili.

Hai un `claude_desktop_config.json` esistente? Importalo con il
bottone **Import from Claude Desktop config…**.

I tool MCP vengono esposti automaticamente sia ai modelli locali
(tramite blocchi DSML) che a quelli OpenRouter (tramite array
`tools` OpenAI-style). Stessa configurazione vale per entrambi.

### Delegation tra agenti

Quando hai due o più agenti registrati, l'agente attaccato a una
chat riceve automaticamente un tool sintetico chiamato
`__delegate_to_agent`. Il modello può chiamarlo con
`{ agent_name: "...", task: "..." }` per affidare un sotto-task a
un altro agente.

Vedrai la chain di esecuzione in tempo reale in una card sopra il
composer: ogni livello mostra l'agente target, il task ricevuto, e
il buffer di risposta che si riempie.

Limiti: nesting massimo 3 livelli, prevenzione cicli automatica
(un agente già nella chain attiva non può essere richiamato).
Funziona solo sui modelli locali per ora.

---

## 8. Tool nativi, Plan / Build, slash command

Oltre ai server MCP, il target `DeepSeekTools` fornisce una
toolbox di tool nativi (`read`, `write`, `edit`, `glob`, `grep`,
`shell`, `apply_patch`, `webfetch`, `websearch`, `repo_clone`,
`repo_overview`, `plan`, `task`, `todo`) che il modello può
chiamare senza passare per MCP. Riferimento dettagliato:
[`TOOLS.md`](TOOLS.md).

### Modalità Plan / Build

Ogni agente ha una modalità di esercizio coarse:

- **Build** (default) — tutti i tool eleggibili. I tool
  `.mutating` / `.dangerous` / `.network` chiedono conferma la
  prima volta per sessione.
- **Plan** — i tool `.mutating` e `.dangerous` sono nascosti
  dallo schema che il modello vede; lui letteralmente non può
  proporli. I `.network` chiedono comunque consenso.

Tre modi per cambiare la modalità:

1. **Settings → Agents → edit → Agent mode** — default
   persistente sull'agente.
2. **Mode picker** sulla toolbar della chat — override
   per-conversazione, non tocca l'agente salvato.
3. **`/mode plan`** / **`/mode build`** — slash command inline.

### Permission

Quando il modello chiede di eseguire un tool dangerous /
mutating / network, parte un flusso a quattro step:

1. Filtro modalità: in Plan, mutating + dangerous sono già
   filtrati via dallo schema.
2. Default durabile in `PermissionStore`: se hai già messo
   "Sempre permetti" o "Sempre nega" per quel (tool, categoria),
   risposta immediata.
3. Cache di sessione: se hai già detto "permetti una volta" in
   questa sessione, il prossimo passa.
4. Sheet modale `PermissionPromptView`: tre bottoni — **Deny**,
   **Allow once**, **Always allow**. "Always allow" si memorizza
   in `PermissionStore` per le sessioni future.

Modifica i default in **Settings → Permissions**.

### Slash command

Digita `/` nel composer per aprire la palette degli slash
command. Built-in disponibili:

| Comando | Cosa fa |
|---|---|
| `/mode plan` · `/mode build` | Cambia la modalità della chat. |
| `/tools` | Apre la tab Tools. |
| `/permissions` | Apre la tab Permissions. |
| `/skill <nome>` | Attiva una skill (vedi sotto). |
| `/theme` | Apre la tab Theme. |
| `/clear` | Svuota la draft corrente (non cancella la chat). |
| `/help` | Lista i comandi disponibili. |

### Skill

Una **Skill** è un piccolo bundle di (system prompt aggiuntivo,
tool suggeriti, modalità default). Servono per chiedere all'agente
"ora comportati come un X". Gestione: **Settings → Skills**.

Gli agenti dichiarano quali skill consentire via il campo
"Allowed skills" nell'edit sheet. Lista vuota = nessuna
restrizione.

### Nuove tab Settings

| Tab | Cosa controlla |
|---|---|
| **Tools** | Inventario read-only dei tool nativi + matrice di disponibilità Plan/Build. |
| **Permissions** | Editor dei default ask / alwaysAllow / alwaysDeny. |
| **Skills** | CRUD della libreria delle skill. |
| **Theme** | Tema chiaro / scuro / system, accent, tinting bolle, import custom. |
| **Keybindings** | Lista read-only di azioni + scorciatoie. Reset ai default. |

---

## 9. Riepilogo dei menu della toolbar

Da sinistra a destra:

| Icona | Picker | Cosa fa |
|---|---|---|
| cpu / disco / nuvola | **Model** | Cambia backend: cartella locale, modello OpenRouter, Browse, Unload, Add OpenRouter… |
| variabile | **Agent** | Attacca / stacca un preset agente alla chat corrente. |
| folder | **Project** | Attacca / stacca un project pre-tokenizzato (solo locale). |
| bacchetta | **Convert** | Apre lo sheet di quantizzazione offline pesi. |

Sopra il composer trovi inoltre:

| Picker | Cosa fa |
|---|---|
| **Mode** (Build / Plan) | Override per-conversazione della modalità agente. Bloccato 🔒 se l'agente attached lo fissa. |
| **Thinking** (No think / High / Max) | Reasoning effort per il prossimo turno. Stesso lock sull'agente. |

---

## 10. Risoluzione problemi rapidi

| Sintomo | Causa | Soluzione |
|---|---|---|
| Build error `MTLLibraryErrorDomain code 6: no default library` | metallib non compilato | rilancia `swift build -c release` e verifica con `find .build -name '*.metallib'` |
| Build error sui kernel Metal | shell script senza permessi | `chmod +x Plugins/MetalLibPlugin/build_metallib.sh` |
| Primo token impiega minuti | strategia `streaming` su SSD freddo | normale al primo run; i successivi sono caldi |
| Il modello locale ripete sempre lo stesso token | sampling con temperature 0 | passa a `--temperature 0.7` da CLI, o la GUI lo vincola già |
| Chat OpenRouter: "API key not configured" | Keychain vuoto | Settings → API Keys → incolla e Save |
| OpenRouter 401 / 403 | key invalida / scaduta / credito finito | Settings → API Keys → Test, controlla dashboard OpenRouter |
| Tool MCP non si eseguono su remoto | server MCP offline | Settings → MCP → controlla che il footer dica "Connected" |
| Errore "Cannot find type 'MCPClientPool'" dopo `git pull` | xcodeproj fuori sync | rilancia `./Tools/generate-xcodeproj.sh` |
| Out of memory caricando locale | RAM insufficiente | usa `--load-strategy streaming` o riduci `--max-seq-len 2048` |

Quando qualcosa va male, sopra ogni cosa servono **le ultime ~20
righe di output prima del crash** e **il comando esatto che hai
eseguito**.

Lista più completa di errori e cause su
[`USAGE.md § 6 Troubleshooting`](USAGE.md#6-troubleshooting).

---

## 11. Dove continuare

- [`USAGE.md`](USAGE.md): riferimento conciso di tutti i flag CLI
  e dei tab Settings.
- [`EXAMPLES.md`](EXAMPLES.md): ricette pronte all'uso — codice di
  esempio per estendere l'app.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): come è organizzato il
  codice (engine + UI + backend).
- [`MODULES.md`](MODULES.md): mappa file-per-file del progetto.
- [`DEVELOPING.md`](DEVELOPING.md): convenzioni e workflow per
  contribuire.

Buon hacking.
