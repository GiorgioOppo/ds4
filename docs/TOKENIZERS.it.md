# Tokenizer e chat template

Tutto quello che l'engine fa tra una stringa scritta dall'utente e
l'array `[[Int]]` di token id che `Transformer.forward` consuma. Tre
formati di tokenizer, un dispatcher per i chat template, il template
nativo DSV4, e un subset Jinja2 fatto a mano per tutto il resto.

Documenti complementari:

- [`MODEL.md`](MODEL.md) — cosa l'engine fa con gli id.
- [`MODULES.md`](MODULES.md) — indice per-file di `Sources/`.
- [`USAGE.md`](USAGE.md) — walkthrough operativo (scarica + punta il
  loader alla directory giusta).
- [`GAP-ANALYSIS-LLAMACPP.md`](GAP-ANALYSIS-LLAMACPP.md) — come il
  dispatcher tokenizer + template si confronta con llama.cpp.

> 🇬🇧 La versione inglese è [`TOKENIZERS.md`](TOKENIZERS.md).

---

## 1. Il sottosistema

```
~/Downloads/Model/                       su disco
├── tokenizer.json                       (HF: BPE / WordPiece / Unigram)
├── tokenizer_config.json                (HF: chat_template, model_type)
├── tokenizer.model                      (SentencePiece protobuf — solo modelli SP)
└── …
        │
        ▼
TokenizerLoader.load(tokenizerDir:)
        │
        ├─ scegli il tokenizer
        │     · tokenizer.json model.type == "BPE"        → BPETokenizer
        │     · tokenizer.json model.type == "WordPiece"  → WordPieceTokenizer
        │     · tokenizer.json model.type == "Unigram"
        │       OPPURE *.model fratello                   → SentencePieceTokenizer
        │
        └─ scegli il chat template
              · firma vocab DSV4 (o model_type contiene "deepseek")
                                                          → DSV4Template (usa EncodingDSV4)
              · tokenizer_config.json.chat_template       → JinjaChatTemplate (subset Jinja2)
              · nessuno dei due                           → throw .unsupportedType
        ▼
LoadedTokenizer { tokenizer, chatTemplate, isDSV4 }
```

Tre pezzi indipendenti:

1. **Tokenizer**: testo → id `[Int]` e ritorno. Tre implementazioni,
   una per ogni formato che incontriamo.
2. **Chat template**: lista di messaggi → stringa di prompt. Due
   implementazioni: il path nativo DSV4 e un driver subset-Jinja2 per
   tutto il resto.
3. **`TokenizerLoader.load`**: il dispatcher che ispeziona una
   directory e istanzia la coppia giusta.

La chat surface vede solo `LoadedTokenizer`. La CLI usa sia il
tokenizer (per la raw mode) sia il template (per la chat mode).

---

## 2. Il protocollo `Tokenizer`

`Sources/DeepSeekKit/Tokenizer.swift:6`:

```swift
public protocol Tokenizer {
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
    var bosId: Int? { get }
    var eosId: Int? { get }
    var stopTokenIds: Set<Int> { get }
}
```

Cinque membri. I primi due sono ovvi. Gli ultimi tre guidano il
generation loop:

- `bosId` viene prepended al prompt in raw mode; i chat template lo
  embeddano esplicitamente quando vogliono.
- `eosId` è il token canonico "stop".
- `stopTokenIds` è il *set* di ogni id la cui sampling termina la
  generazione. Include sempre `eosId` più qualunque altro marker
  terminale che il vocab definisce (V4's `<|EOT|>`, Llama's
  `<|eot_id|>`).

Il decode loop controlla `stopTokenIds`, non solo `eosId`. Versioni
precedenti testavano solo contro `eosId` e le generazioni V4 chat
proseguivano oltre il marker end-of-turn dell'assistant, dove il
ranking residuale della LM head collassava in token filler in loop.
Quel bug è fixato trattando il check di stop come un set membership.

---

## 3. BPE (DeepSeek-V4 e amici)

`Sources/DeepSeekKit/BPETokenizer.swift:26`. BPE byte-level
compatibile con i file `tokenizer.json` HuggingFace prodotti dalla
libreria `tokenizers`. È il formato che DeepSeek-V4 spedisce e il
formato che il path di indexing Project / Document assume.

### Algoritmo

1. **Pre-tokenizzazione**: splitta l'input sul regex GPT-2 / cl100k
   (o qualunque pattern `model.pre_tokenizer.pattern.Regex` fornisca).
   Ogni match diventa una "parola" su cui BPE opera in modo
   indipendente.
2. **Encoding byte-to-unicode**: converte i byte UTF-8 della parola
   tramite la map byte-to-unicode GPT-2. Byte che sarebbero caratteri
   di controllo o whitespace vengono ruotati nel range stampabile
   Latin-1 così che i merge BPE lavorino su caratteri display-safe.
3. **BPE greedy lowest-rank**: partendo dai simboli per-carattere,
   guarda ogni coppia adiacente, scegli quella con il rank di merge
   più basso, fondila, ripeti finché non rimangono merge applicabili.
4. **Lookup vocab**: ogni simbolo finale ottiene `vocab[symbol] → Int`.

I token speciali (`added_tokens` in tokenizer.json) bypassano la
pipeline interamente — l'encoder splitta l'input su di essi prima e
emette l'id esatto direttamente.

### Decoding

Lookup `invVocab[id]` simbolo per simbolo, poi mappa ogni carattere
byte-encoded di nuovo tramite `unicodeToByte`, poi UTF-8 decode. I
token speciali decodano al loro contenuto testuale così come sono.

### Limitazioni vs lo spec HF completo

- Niente step di normalizzazione (il `tokenizer.json` di DeepSeek ha
  `normalizer: null`; se un modello richiede NFC/NFD/NMT prima di
  BPE, questa implementazione non la applica).
- Niente truncation/padding (gestiti dal caller).
- Il regex split usa `NSRegularExpression` con il pattern ByteLevel
  più comune; se un tokenizer specifica un
  `pretokenizer.pattern.Regex` diverso, usiamo quel pattern alla
  lettera — ma qualsiasi feature che `NSRegularExpression` non
  supporta (varianti lookbehind, Unicode property escapes oltre le
  basi) non parserà.

### Concorrenza

`BPETokenizer` è marcato `@unchecked Sendable`. Ogni stored property
è `let`, e `NSRegularExpression` è documentato thread-safe da Apple.
Il vocab-pruning tool chiama `encode(_:)` da più thread quando
`--concurrency` è > 1.

---

## 4. SentencePiece (Llama / Mistral / Qwen / Gemma)

`Sources/DeepSeekKit/SentencePieceTokenizer.swift:14`. Implementazione
minimale che legge il file protobuf binario `.model` ad-hoc — solo i
tag che servono vengono decodati, il resto è skippato, così spediamo
senza dipendenza da `swift-protobuf`.

### Algoritmo

Unigram language model con **decoding Viterbi** su un lattice:

1. **Normalizzazione whitespace**: ogni spazio diventa `▁` (U+2581,
   "lower one eighth block"). È la convenzione SentencePiece — i
   token con prefisso spazio preservano i confini di parola senza
   bisogno di un pre-tokenizer.
2. **Lattice**: a ogni posizione `i` nell'input (normalizzato),
   enumera ogni piece nel vocab che matcha il prefisso. Ogni piece è
   un edge con costo `-score(piece)` (score più basso = più
   probabile).
3. **Viterbi**: programmazione dinamica sugli edge per trovare il
   path a costo minimo da start a end.
4. **Byte fallback**: caratteri senza piece matching nel vocab cadono
   su token per-byte `<0xNN>` (`type == 6` nel proto). Ogni modello
   SentencePiece spedito con LLM HF include la tabella 256-byte BYTE
   di default.

### Decoding

Lookup `pieces[id].text`, byte-fallback per token `<0xNN>` (parsati
in un byte e ri-incollati), poi strip del `▁` leading e replace dei
`▁` rimanenti con spazi.

### Limitazioni

- SentencePiece in modalità BPE (Llama 1 vecchio) non è supportato —
  solo la variante Unigram. Lo schema proto è letto field-per-field;
  un wireType sconosciuto throwa.
- Niente subword regularization (il path nbest di SentencePiece è
  droppato per l'inference).

---

## 5. WordPiece (BERT-style, per modelli di embedding/classification)

`Sources/DeepSeekKit/WordPieceTokenizer.swift:9`. WordPiece stile BERT
che legge il campo `vocab` di un `tokenizer.json` HuggingFace il cui
`model.type == "WordPiece"`.

### Algoritmo

1. **Pre-tokenizzazione**: split su whitespace + punctuation. Ogni
   "parola" è encoded indipendentemente.
2. **Greedy longest-match**: per ogni parola, parti dalla testa e
   cerca il match più lungo nel vocab. I match successivi usano il
   prefisso di continuazione `##`.
3. **Gestione unknown**: parole che superano
   `maxInputCharsPerWord` (200) producono un singolo id `[UNK]`;
   parole senza match per il primo carattere producono anche `[UNK]`.

### Limitazioni

L'engine runtime non spedisce oggi un modello che usa WordPiece.
WordPiece esiste perché lo scope del lettore GGUF è stato esteso per
coprire i modelli di embedding BERT-class per use case di retrieval /
classification. Il tokenizer è implementato e testato, ma il *loop di
inference* per questi modelli non è ancora landato — vedi
`docs/GGUF.md` e la roadmap.

---

## 6. Il dispatcher: `TokenizerLoader`

`Sources/DeepSeekKit/Tokenizer.swift:57`. Un singolo entry point
statico: `TokenizerLoader.load(tokenizerDir: URL)` ritorna un
`LoadedTokenizer` con tokenizer + chat template + un flag che
distingue DSV4 da tutto il resto.

Logica di detection:

1. **Tokenizer**: leggi `model.type` di `tokenizer.json`.
   - `"BPE"` → `BPETokenizer`.
   - `"WordPiece"` → `WordPieceTokenizer`.
   - `"Unigram"` o mancante — cerca un file `*.model` fratello →
     `SentencePieceTokenizer`. Se nessuno (e neanche `tokenizer.json`),
     throw `missingFile`.
2. **Chat template**:
   - Se `tokenizer_config.json.model_type` contiene `"deepseek"`,
     OPPURE la lista `added_tokens` di `tokenizer.json` contiene
     `begin▁of▁sentence` → `DSV4Template()`.
   - Altrimenti, se `tokenizer_config.json.chat_template` è
     non-vuoto → `JinjaChatTemplate(src)`.
   - Altrimenti throw `unsupportedType`.

La detection DSV4 vince su un template Jinja generico anche quando
uno è presente: `EncodingDSV4` avvolge logica aggiuntiva oltre al
puro rendering dei messaggi (il prompt REASONING_EFFORT_MAX, il blocco
tools, il post-processing `<｜tool▁outputs｜>`) che il path Jinja-only
non riproduce. Modelli DeepSeek futuri non-V4 prendono ancora il loro
`chat_template`.

### Shim legacy

`TokenizerLoader.load(from url: URL)` prende un path direttamente a un
file `tokenizer.json` e ritorna solo il `Tokenizer` — niente chat
template. I caller più vecchi (e il converter) usano questo quando
non hanno bisogno del rendering.

---

## 7. Il protocollo `ChatTemplate`

`Sources/DeepSeekKit/Encoding/ChatTemplate.swift:11`:

```swift
public protocol ChatTemplate: Sendable {
    func render(messages: [Message], options: ChatTemplateOptions) throws -> String
}
```

Un metodo: renderizza una lista di `Message` nella stringa di prompt
che il modello si aspetta. Due implementazioni:

- **`DSV4Template`** — wrappa `EncodingDSV4.encodeMessages(...)`.
  Usato per ogni checkpoint DeepSeek-V4.
- **`JinjaChatTemplate`** — wrappa il driver subset-Jinja2. Usato
  quando il `tokenizer_config.json` del modello caricato porta un
  campo `chat_template` (Llama / Mistral / Qwen / ChatML / qualsiasi
  modello HF con un template).

### `ChatTemplateOptions`

```swift
public struct ChatTemplateOptions: Sendable {
    public var addGenerationPrompt: Bool   // append del role marker finale
    public var thinkingMode: ThinkingMode  // DSV4-specific
    public var toolSchemasJSON: String?    // DSV4-specific (formato DSML)
    public var tools: [JSONValue]?         // Jinja-specific (formato OpenAI)
    public var bosToken: String            // alcuni template Jinja ne hanno bisogno
    public var eosToken: String            // alcuni template Jinja ne hanno bisogno
}
```

Lo split tra `toolSchemasJSON` e `tools` è intenzionale: DSV4 si
aspetta JSON pre-serializzato che l'host ha costruito una volta e
incolla verbatim nella sezione `## Tools`, mentre i template Jinja si
aspettano la shape OpenAI (un array di oggetti con `name` /
`description` / `parameters`).

### `ChatTemplateError`

Tre case: `unsupportedFeature(msg)` (un costrutto Jinja che non
supportiamo), `templateRaise(msg)` (un template ha chiamato
`raise_exception(...)`), `parseFailure(msg)` (lexer / parser è
rotto). Tutti bubbleano al caller, che li traduce in un messaggio
d'errore user-facing.

---

## 8. Il template DSV4

`Sources/DeepSeekKit/Encoding/EncodingDSV4.swift` porta
`Reference/encoding/encoding_dsv4.py`. La shape ad alto livello:

```
<｜begin▁of▁sentence｜>{system_content}
<｜User｜>{user_content_1}<｜Assistant｜>[<think>{reasoning}</think>]{assistant_content_1}
[<｜DSML｜tool_calls>...</｜DSML｜tool_calls>]
<｜end▁of▁sentence｜>
[<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>{name}<｜tool▁sep｜>{body}<｜tool▁output▁end｜>...<｜tool▁outputs▁end｜>]
<｜User｜>{user_content_2}<｜Assistant｜>[<think>|</think>]
```

Il `[<think>|</think>]` finale è la parte che il modello riempie.
L'encoder emette uno dei due marker a seconda del thinking mode:

- `chat` (no-think): emetti `</think>` (dicendo al modello "no
  reasoning, ecco la risposta").
- `high` / `max`: emetti `<think>` (dicendo al modello "inizia il
  tuo reasoning"). I primi token generati dal modello vanno dentro
  il blocco; il path `parseCompletion` li estrae di ritorno in
  `Message.reasoningContent`.

### Token speciali

Il vocab DSV4 include ~25 marker usati dal template. Lista completa
(`Sources/DeepSeekKit/Encoding/EncodingDSV4.swift:9-44`):

```
<｜begin▁of▁sentence｜>     bosToken
<｜end▁of▁sentence｜>       eosToken
<｜User｜>                  userToken
<｜Assistant｜>             assistantToken
<think>, </think>           thinkOpen, thinkClose

# Delimitatori Project / repo (added_tokens reali 128815-820)
<｜begin▁of▁repo▁name｜>  / <｜end▁of▁repo▁name｜>
<｜begin▁of▁file▁name｜>  / <｜end▁of▁file▁name｜>
<｜begin▁of▁file｜>        / <｜end▁of▁file｜>

# Delimitatori tool-output (added_tokens reali 128810-814)
<｜tool▁outputs▁begin｜> / <｜tool▁outputs▁end｜>
<｜tool▁output▁begin｜>  / <｜tool▁output▁end｜>
<｜tool▁sep｜>
```

Tutti questi sono token reali nel vocab V4 — il BPE pre-splitta su di
essi così che ognuno emetta esattamente un id indipendentemente dai
byte circostanti.

### Thinking mode

Enum `ThinkingMode` (`Sources/DeepSeekKit/Encoding/Message.swift:46`):

| Mode | Marker finale emesso | Blocco system prepended |
|---|---|---|
| `.chat` | `</think>` | — |
| `.high` | `<think>` | — |
| `.max` | `<think>` | `REASONING_EFFORT_MAX` |

`REASONING_EFFORT_MAX` è un'istruzione multi-line hard-coded
prepended a (o merged in) il primo messaggio system — vedi
`EncodingDSV4.swift:48-54` per il testo esatto.

### Schemi tool — il blocco `TOOLS_TEMPLATE`

Quando `toolSchemasJSON` è non-vuoto, l'encoder prepend un blocco
`## Tools` al primo messaggio system (o crea un messaggio system se
non c'è). Il blocco istruisce il modello sulla syntax di invocazione
`<｜DSML｜tool_calls>` e elenca gli schemi che l'host ha reso
disponibili.

Gli schemi arrivano pre-serializzati (come una singola stringa JSON
che l'host ha già costruito, tipicamente da
`MCPClientPool.toolSchemasJSON(...)` più l'`availableSchemas(mode:)`
del registry tool nativi). Il template non prova a ri-renderizzarli —
incolla soltanto.

### Tool call (output modello → host)

L'encoder gestisce anche la direzione *uscente*. Dopo il contenuto
dell'assistant, appende `<｜DSML｜tool_calls>` contenente un block
`<｜DSML｜invoke>` per ogni `ToolCall`:

```
<｜DSML｜tool_calls>
<｜DSML｜invoke name="filesystem__read_file">
<｜DSML｜parameter name="path" string="true">/etc/hosts</｜DSML｜parameter>
<｜DSML｜parameter name="limit" string="false">50</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
```

`string="true"` per valori raw string; `string="false"` (con il valore
come JSON) per tutto il resto. XML-escaped dentro i body. Il pass di
parsing (`parseCompletion`) è l'inverso.

### Tool output (host → modello)

Quando un turn assistant ha `toolOutputs` popolati, l'encoder
appende un blocco in formato nativo dopo l'EOS di quel turn:

```
<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>filesystem__read_file<｜tool▁sep｜>...body...<｜tool▁output▁end｜><｜tool▁outputs▁end｜>
```

Un `<｜tool▁output▁begin｜>…<｜tool▁output▁end｜>` per output,
prefissato col nome del tool corrispondente quando noto. Usato come
splice point nel tool-call loop: dopo che l'host esegue una call,
scrive il risultato nel prompt via questo blocco, poi rilancia
`generateForConversation`.

### Parsing dei completion del modello

`EncodingDSV4.parseCompletion(_:mode:)` è l'inverso:

1. Strippa un `<｜end▁of▁sentence｜>` finale se presente.
2. Estrai un blocco opzionale `<think>…</think>` in
   `Message.reasoningContent`.
3. Estrai un blocco opzionale `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>`,
   parsa ogni `<｜DSML｜invoke>`, ricostruisci il JSON
   `ToolCall.args` dai figli `<｜DSML｜parameter>`.
4. Ritorna il rimanente (trimmato di whitespace) come
   `Message.content`.

La chat surface chiama `parseCompletion` solo a end-of-turn — durante
lo streaming mostra il testo raw così come arriva. La UI di streaming
sa come renderizzare un blocco `<think>` in corso (la disclosure con
icona cervello) riconoscendo il marker senza aspettare il close tag.

### Cosa non è portato

Il reference Python ha più superficie; abbiamo deliberatamente
deferred alcuni pezzi meno usati:

- `response_format_template` (il blocco constrained-output JSON
  schema) — il caller può prepend manualmente.
- `latest_reminder` — un "system reminder" once-per-turn che il
  Python appende. Valore negligibile per il chat flow realistico.
- Token task (`<task>...</task>`) — di nicchia, non nella
  distribuzione di training di V4-Flash per quanto ne sappiamo.

---

## 9. Il subset Jinja2

`Sources/DeepSeekKit/Encoding/JinjaTemplate.swift` è un'implementazione
fatta a mano del subset di Jinja2 che i chat template HuggingFace
usano davvero. ~900 LOC. È intenzionalmente stretto:

| Feature | Supportato |
|---|---|
| Interpolazione `{{ var }}` | ✅ |
| Accesso chained `.field` / `[index]` | ✅ |
| `{% if ... %}` / `{% elif %}` / `{% else %}` / `{% endif %}` | ✅ |
| `{% for x in xs %}` / `{% endfor %}` | ✅ |
| Filtri (`trim`, `length`, `lower`, `upper`, `default`, `tojson`, `replace`, `string`) | ✅ |
| `raise_exception("...")` | ✅ (→ `ChatTemplateError.templateRaise`) |
| Operatori di comparison + booleani (`==`, `!=`, `in`, `and`, `or`, `not`) | ✅ |
| Aritmetica | ❌ |
| `{% set %}` | ❌ |
| `{% macro %}` / `{% include %}` / template inheritance | ❌ |
| `loop.first` / `loop.last` / `loop.index` | ✅ |

Nessuna delle feature omesse appare nei chat template che abbiamo
incontrato. Se una compare, il messaggio d'errore
(`unsupportedFeature`) punta direttamente a cosa aggiungere.

### JinjaChatTemplate

`Sources/DeepSeekKit/Encoding/JinjaChatTemplate.swift` è l'adapter
che mappa i tipi `[Message]` / `[ToolCall]` dell'host nello scope
Jinja che i template HuggingFace si aspettano:

```
messages: lista di dict con chiavi
    role, content, tool_calls?, reasoning_content?
add_generation_prompt: bool
bos_token, eos_token: str
tools: list (stile OpenAI)
```

Ogni `ToolCall` è renderizzato come `{ type: "function", function: {
name, arguments } }` (shape OpenAI). Il template Jinja può poi
camminare `message.tool_calls` come fa il template ufficiale di
Llama 3.

### Quando i template falliscono

Il case d'errore `parseFailure` porta la diagnostica del lexer /
parser (numero di riga + token offending). La GUI lo espone come
"could not load chat template" con il messaggio interno — utile
quando una release HF spedisce un template con un costrutto che non
abbiamo ancora aggiunto.

---

## 10. Tokenizzazione Project / document

Per "attacca un progetto a una chat" l'engine pre-tokenizza una
directory una volta e splicea il risultato nel primo turn user. Il
path è local-model only (le chat remote OpenRouter non hanno accesso
ai token nativi).

`Sources/DeepSeekUI/Utility/ProjectIndexer.swift` percorre i file del
progetto; per ogni file costruisce:

```
<｜begin▁of▁repo▁name｜>{repo-name}<｜end▁of▁repo▁name｜>
<｜begin▁of▁file▁name｜>{path}<｜end▁of▁file▁name｜>
<｜begin▁of▁file｜>
{contents}
<｜end▁of▁file｜>
```

Questi sono token reali nel vocab V4; il BPE pre-splitta su di essi
così che il modello veda un boundary strutturato pulito. L'array di
id pre-tokenizzato è salvato sotto
`Application Support/.../projects/<id>/` keyed da
`ModelFingerprint.of(modelDirPath:)` — è richiesto re-importare
quando il modello caricato cambia (tokenizer diverso = id stale).

`InferenceService.tokenizeFirstTurnWithProject(...)` è l'entry point
che la chat surface chiama quando invia il primo messaggio di una
chat con progetto attaccato.

---

## 11. Vocab pruning (path IT-only)

Un secondo tool, `Sources/DeepSeekVocabPruner/`, percorre un corpus di
testo italiano contro il tokenizer BPE e trova il subset di id vocab
che copre ≥ N% dei token osservati. L'intento: prunare la coda
inutilizzata per ridurre la matrice di embedding.

È una feature sperimentale, solo per la lingua IT, e non ancora
collegata al runtime. Docs complete in
[`VOCAB-PRUNING.md`](VOCAB-PRUNING.md).

---

## 12. Cross-walk con il reference Python

| Python | Linee | Swift |
|---|---|---|
| `encode_messages` | 506–575 | `EncodingDSV4.encodeMessages` (`Sources/DeepSeekKit/Encoding/EncodingDSV4.swift:93`) |
| `parse_message_from_completion_text` | 687–744 | `EncodingDSV4.parseCompletion` (`:223`) |
| `tool_calls_template` / `tool_call_template` | 52–58 | `EncodingDSV4.encodeToolCalls` (`:185`) |
| `encode_arguments_to_dsml` | 139–180 | `EncodingDSV4.encodeArguments` (`:198`) |
| `REASONING_EFFORT_MAX` | 64–67 | `EncodingDSV4.reasoningEffortMax` (`:48`) |
| `TOOLS_TEMPLATE` | 70–95 | `EncodingDSV4.toolsBlock(toolSchemasJSON:)` (`:58`) |
| `response_format_template` | 49–51 | non portato — prepend nel caller |
| Costanti dei token speciali | 17–35 | top di `EncodingDSV4.swift` |
| `to_json` / `tools_from_openai_format` / etc. | 101–137 | non portati — il caller passa JSON pre-serializzato |

I test reference (`Reference/encoding/tests/test_input_*.json` più
`test_output_*.txt`) sono golden test per l'encoder. Lo Swift
`Tests/DeepSeekKitTests/EncodingDSV4Tests.swift` consuma il subset
che il port supporta; il resto è in roadmap.

---

## 13. Source map

| Topic | File |
|---|---|
| Protocollo + dispatcher | `Sources/DeepSeekKit/Tokenizer.swift` |
| BPE byte-level | `Sources/DeepSeekKit/BPETokenizer.swift` |
| SentencePiece (Unigram + byte-fallback) | `Sources/DeepSeekKit/SentencePieceTokenizer.swift` |
| WordPiece (BERT-style) | `Sources/DeepSeekKit/WordPieceTokenizer.swift` |
| `Message` / `Role` / `ToolCall` / `ThinkingMode` | `Sources/DeepSeekKit/Encoding/Message.swift` |
| Protocollo `ChatTemplate` + options + errors | `Sources/DeepSeekKit/Encoding/ChatTemplate.swift` |
| Template DSV4 (wrapper) | `Sources/DeepSeekKit/Encoding/DSV4Template.swift` |
| Encoder + parser DSV4 | `Sources/DeepSeekKit/Encoding/EncodingDSV4.swift` |
| Interprete subset Jinja2 | `Sources/DeepSeekKit/Encoding/JinjaTemplate.swift` |
| Adapter Jinja chat template | `Sources/DeepSeekKit/Encoding/JinjaChatTemplate.swift` |
| Tokenizzazione Project / Document | `Sources/DeepSeekUI/Utility/ProjectIndexer.swift` |
| Test | `Tests/DeepSeekKitTests/{BPETokenizerTests,EncodingDSV4Tests,JinjaTemplateTests}.swift` |
| Riferimento Python | `Reference/encoding/encoding_dsv4.py` |

---

## 14. Limitazioni e lavoro deferred

Tracciato in `TODO.md` (sezioni "Encoding" e "Multi-format / GGUF /
chat-template dispatcher") e [`ROADMAP.md`](ROADMAP.md). A colpo
d'occhio:

- **Loop di inference WordPiece**: il tokenizer è implementato; il
  forward pass del modello embedding che consumerebbe quegli id no.
- **`{% set %}` / `{% macro %}` / inheritance Jinja**: deferred —
  nessuno dei template in the wild che abbiamo incontrato ne ha
  bisogno. Il parser logga `unsupportedFeature` così aggiungere
  supporto è una patch pulita.
- **Constrained decoding (JSON schema)**: concern separato che pende
  più dal sampler che dal tokenizer; tracciato in `TODO.md §10.3`.
- **SentencePiece BPE-mode** (Llama 1 vecchio): non supportato. Solo
  Unigram + byte-fallback.
- **DSV4 `response_format_template` / `latest_reminder` / task token**:
  non portati; il caller può prepend la stringa letterale se serve.
