# DS4Core/Inference

Pezzi puri dell'inferenza (niente GPU).

- **`Tokenizer.swift`** — BPE con i token di controllo DeepSeek-V4 (BOS/EOS, `<｜User｜>`, `<think>`, `｜DSML｜`…); `tokenizeRenderedChat`, `tokenText`.
- **`ChatTools.swift`** — tipi `ToolSpec`/`ToolCall`/`ChatTurn`, rendering del prompt chat + dei tool nel formato DSML, e `ToolCallParser` (estrae le tool-call dal testo generato).
- **`Sampler.swift`** — campionamento (temperature, top-k/p, min-p, penalità di ripetizione).
- **`ModelShape.swift`** — descrizione della shape del modello.
