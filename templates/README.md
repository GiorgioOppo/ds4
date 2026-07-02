# Chat Template (Tool Calling)

[`chat_template.jinja`](chat_template.jinja) is the DeepSeek-V4 Jinja
`chat_template` **with tool-calling support**. It follows the format the model was
trained on, was checked against the `tokenizer.chat_template` stored in the GGUF,
and has only been reformatted/commented for readability. The rendered output is
byte-identical to the original template.

## Purpose

- **Reference/specification:** this is the format mirrored 1:1 by the Swift
  renderer in
  [`ChatRenderer`](../Sources/DS4Core/Inference/ChatTools.swift).
- **External runtimes:** use it with stacks that consume tokenizer chat templates,
  such as llama.cpp, vLLM, or `transformers`, or to re-embed the template into a
  GGUF.
- **Regression aid:** when changing the Swift renderer, this template is the
  canonical source for spacing, special tokens, tool schemas, tool invocations,
  and thinking tags.

## Format Summary

- **Tool declarations** are emitted inside a system block headed by `## Tools`,
  followed by JSON function schemas (`tool['function'] | tojson`, sorted keys).
- **Tool calls** use XML-like DSML on the `｜DSML｜` token:

  ```xml
  <｜DSML｜tool_calls>
  <｜DSML｜invoke name="get_weather">
  <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
  <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
  </｜DSML｜invoke>
  </｜DSML｜tool_calls>
  ```

  Strings are emitted with `string="true"` and raw text values. Other types are
  emitted with `string="false"` and JSON values.
- **Tool results** are injected into a user turn as
  `<｜User｜><tool_result>...</tool_result>`. Consecutive tool results do not repeat
  the `<｜User｜>` prefix.
- **Assistant turns** start with `<｜Assistant｜>`, then either `</think>` or
  `<think>...</think>` when `thinking` is enabled and `reasoning_content` is
  present, followed by assistant content, optional tool calls, and
  `<｜end▁of▁sentence｜>`.
- There is **no newline** between `BOS` and the system content.

## Example With `transformers`

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

## References

- Schema/specification: DeepSeek-V4 paper, Table 4 for the tool-call schema and
  Table 5 for special tokens.
- Equivalent Swift implementation:
  [`ChatRenderer`](../Sources/DS4Core/Inference/ChatTools.swift).
- Engine details:
  [`docs/ARCHITETTURA-MOTORE.md`](../docs/ARCHITETTURA-MOTORE.md), section 14.
