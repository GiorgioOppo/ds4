# Chat Template (Tool Calling)

[`chat_template.jinja`](chat_template.jinja) è il `chat_template` Jinja di
DeepSeek-V4 **con supporto al tool calling**, conservato in questo repository
come snapshot di riferimento leggibile. È stato confrontato con un
`tokenizer.chat_template` proveniente dal GGUF usato durante lo sviluppo e poi
riformattato/commentato. Il repository non registra un hash del modello e una
fixture che renderebbero l'identità byte per byte una garanzia universale per
ogni GGUF DeepSeek-V4, quindi va verificato rispetto al modello esatto in
distribuzione.

A runtime il GGUF selezionato fornisce il vocabolario e i metadati dei token
speciali, mentre il
[`ChatRenderer`](../Sources/DS4Core/Conversation/Backends/DeepSeekV4/DSML/ChatRenderer.swift)
compilato costruisce il testo della conversazione. DwarfStar può mostrare il
template Jinja incorporato nel GGUF a scopo diagnostico, ma non lo interpreta
durante la generazione. Analogamente questo file non viene caricato a runtime
e non sovrascrive il modello; i test di regressione del renderer, in
particolare per le dichiarazioni di tool complete e compatte, sono la
superficie di validazione eseguibile.

## Scopo

- **Riferimento per la revisione:** usalo per ispezionare il formato
  implementato dal renderer Swift e per progettare fixture di parità
  specifiche per modello.
- **Runtime esterni:** usalo con stack che consumano i chat template del
  tokenizer, come llama.cpp, vLLM o `transformers`, solo dopo averlo
  confrontato con i metadati del modello di destinazione; può anche essere
  re-incorporato intenzionalmente in un GGUF.
- **Supporto alle regressioni:** quando si modifica il renderer Swift, questo
  template è l'artefatto di confronto leggibile per spaziature, token
  speciali, schemi dei tool, invocazioni di tool e tag di thinking. Aggiungi o
  aggiorna fixture byte per byte invece di trattare questa copia come prova di
  per sé.

## Riepilogo del formato

- **Le dichiarazioni dei tool** vengono emesse dentro un blocco system con
  intestazione `## Tools`, seguita dagli schemi JSON delle funzioni
  (`tool['function'] | tojson`, chiavi ordinate).
- **Le chiamate ai tool** usano DSML simil-XML sul token `｜DSML｜`:

  ```xml
  <｜DSML｜tool_calls>
  <｜DSML｜invoke name="get_weather">
  <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
  <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
  </｜DSML｜invoke>
  </｜DSML｜tool_calls>
  ```

  Le stringhe vengono emesse con `string="true"` e valori di testo grezzo. Gli
  altri tipi vengono emessi con `string="false"` e valori JSON.
- **I risultati dei tool** vengono iniettati in un turno utente come
  `<｜User｜><tool_result>...</tool_result>`. Risultati di tool consecutivi non
  ripetono il prefisso `<｜User｜>`.
- **I turni dell'assistente** iniziano con `<｜Assistant｜>`, poi `</think>`
  oppure `<think>...</think>` quando `thinking` è abilitato ed è presente
  `reasoning_content`, seguiti dal contenuto dell'assistente, da eventuali
  chiamate ai tool e da `<｜end▁of▁sentence｜>`.
- **Non c'è newline** tra `BOS` e il contenuto system.

## Esempio con `transformers`

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("...")
with open("templates/chat_template.jinja") as f:
    tok.chat_template = f.read()

messages = [
    {"role": "user", "content": "What time is it?"},
]

tools = [{
    "type": "function",
    "function": {
        "name": "now",
        "description": "Current date/time (ISO-8601).",
        "parameters": {"type": "object", "properties": {}},
    },
}]

prompt = tok.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    thinking=False,
    tokenize=False,
)
print(prompt)
```

## Riferimenti

- Schema/specifica: paper DeepSeek-V4, Tabella 4 per lo schema delle chiamate
  ai tool e Tabella 5 per i token speciali.
- Implementazione Swift equivalente:
  [`ChatRenderer`](../Sources/DS4Core/Conversation/Backends/DeepSeekV4/DSML/ChatRenderer.swift).
- Dettagli sul motore:
  [`docs/ARCHITETTURA-MOTORE.md`](../docs/ARCHITETTURA-MOTORE.md), sezione 14.
- Ciclo di vita e ownership dei tool:
  [`docs/STRUMENTI-AGENTI-MCP.md`](../docs/STRUMENTI-AGENTI-MCP.md).
