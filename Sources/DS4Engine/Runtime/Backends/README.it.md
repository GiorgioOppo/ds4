# Runtime/Backends

Ogni cartella registra capability e policy di selezione di una famiglia. I
decoder e i tensori restano nei rispettivi backend di DS4Metal; qui si mantiene
soltanto il bridge usato da DS4Engine.

- [`DeepSeekV4/`](DeepSeekV4/README.md): backend operativo.
- [`GLM52/`](GLM52/README.md): definizione/capability del modello con runtime
  ancora deliberatamente vuoto.
- [`Qwen/`](Qwen/README.md): famiglia riconosciuta ma non disponibile.

Un backend nuovo deve aggiungere insieme: rilevamento, configurazione, tokenizer
e formato chat, decoder, diagnostica, test numerici e capability UI. Riconoscere
il nome di un'architettura non equivale a dichiararla eseguibile.
