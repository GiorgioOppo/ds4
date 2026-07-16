# Runtime/Backends

Ogni cartella registra capability e policy di selezione di una famiglia. I
decoder e i tensori restano nei rispettivi backend di DS4Metal; qui si mantiene
soltanto il bridge usato da DS4Engine.

Un backend nuovo deve aggiungere insieme: rilevamento, configurazione, tokenizer
e formato chat, decoder, diagnostica, test numerici e capability UI. Riconoscere
il nome di un'architettura non equivale a dichiararla eseguibile.
