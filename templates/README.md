# Chat template (tool calling)

[`chat_template.jinja`](chat_template.jinja) è il `chat_template` Jinja di
DeepSeek-V4 **con il supporto ai tool**, fedele al formato su cui il modello è
addestrato (verificato contro il `tokenizer.chat_template` del GGUF), riformattato
e commentato. Produce output **byte-identico** all'originale.

## A cosa serve

- **Riferimento/spec**: è il formato che il renderer Swift del progetto
  (`Sources/DS4Core/Inference/ChatTools.swift` → `ChatRenderer`) rispecchia 1:1.
- **Altri runtime**: usalo con stack che consumano il `chat_template`
  (llama.cpp, vLLM, `transformers`) o per **ri-incorporarlo** in un GGUF.

## Formato (riassunto)

- **Dichiarazione tool** in un blocco system `## Tools …` + schemi delle funzioni
  in JSON (`tool['function'] | tojson`, chiavi ordinate).
- **Chiamata** (XML sul token `｜DSML｜`):
  ```
  <｜DSML｜tool_calls>
  <｜DSML｜invoke name="get_weather">
  <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
  <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
  </｜DSML｜invoke>
  </｜DSML｜tool_calls>
  ```
  Stringhe → `string="true"` valore grezzo; altri tipi → `string="false"` valore JSON.
- **Risultato tool** dentro un turno utente: `<｜User｜><tool_result>…</tool_result>`
  (risultati consecutivi non ripetono `<｜User｜>`).
- Ogni turno assistant apre `<｜Assistant｜>` poi `</think>` (o `<think>…</think>`
  se `thinking` è attivo e `reasoning_content` è presente — *interleaved thinking*),
  contenuto, eventuali tool-call, e chiude con `<｜end▁of▁sentence｜>`.
- Nessuna newline tra `BOS` e system.

## Uso con `transformers`

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("…")
with open("templates/chat_template.jinja") as f:
    tok.chat_template = f.read()

messages = [
    {"role": "user", "content": "Che ore sono?"},
]
tools = [{"type": "function", "function": {
    "name": "now", "description": "Current date/time (ISO-8601).",
    "parameters": {"type": "object", "properties": {}}}}]

prompt = tok.apply_chat_template(
    messages, tools=tools, add_generation_prompt=True, thinking=False, tokenize=False)
print(prompt)
```

## Riferimenti

- Schema/spec: paper DeepSeek-V4, Table 4 (tool-call schema), Table 5 (token speciali).
- Implementazione Swift equivalente: `ChatRenderer` in
  [`Sources/DS4Core/Inference/ChatTools.swift`](../Sources/DS4Core/Inference/ChatTools.swift).
- Dettagli motore: [`docs/ARCHITETTURA-MOTORE.md`](../docs/ARCHITETTURA-MOTORE.md) §14.
