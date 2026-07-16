# Chat Template (Tool Calling)

[`chat_template.jinja`](chat_template.jinja) is the DeepSeek-V4 Jinja
`chat_template` **with tool-calling support** preserved in this repository as a
human-readable reference snapshot. It was compared with a
`tokenizer.chat_template` from the GGUF used during development and then
reformatted/commented. The repository does not record a model hash and fixture
that would make byte identity a universal guarantee across every DeepSeek-V4
GGUF, so verify it against the exact model being deployed.

At runtime the selected GGUF supplies vocabulary and special-token metadata,
while the compiled
[`ChatRenderer`](../Sources/DS4Core/Conversation/Backends/DeepSeekV4/DSML/ChatRenderer.swift)
constructs conversation text. DwarfStar can display the GGUF's embedded Jinja
template for diagnostics, but does not interpret it during generation. This
file is likewise not loaded at runtime and does not override the model; renderer
regression tests, especially for full and compact tool declarations, are the
executable validation surface.

## Purpose

- **Review reference:** use it to inspect the format implemented by the Swift
  renderer and to design model-specific parity fixtures.
- **External runtimes:** use it with stacks that consume tokenizer chat templates,
  such as llama.cpp, vLLM, or `transformers`, only after comparing it with the
  target model's metadata; it can also be re-embedded intentionally into a GGUF.
- **Regression aid:** when changing the Swift renderer, this template is the
  readable comparison artifact for spacing, special tokens, tool schemas, tool
  invocations, and thinking tags. Add or update byte-for-byte fixtures rather
  than treating this copy as proof by itself.

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
  [`ChatRenderer`](../Sources/DS4Core/Conversation/Backends/DeepSeekV4/DSML/ChatRenderer.swift).
- Engine details:
  [`docs/ARCHITETTURA-MOTORE.md`](../docs/ARCHITETTURA-MOTORE.md), section 14.
- Tool lifecycle and ownership:
  [`docs/STRUMENTI-AGENTI-MCP.md`](../docs/STRUMENTI-AGENTI-MCP.md).
