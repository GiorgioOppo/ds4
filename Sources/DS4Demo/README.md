# DS4Demo

Eseguibile CLI di demo/diagnostica che pilota `DS4Core` + `DS4Metal` direttamente (senza la GUI).

- **`main.swift`** — bring-up del runtime Metal + self-test GPU; con un argomento GGUF, streamma N token tramite `StreamingDecoder`.

Uso:
```sh
swift run DS4Demo                  # self-test Metal
swift run DS4Demo <model.gguf> 4   # streamma 4 token
```
