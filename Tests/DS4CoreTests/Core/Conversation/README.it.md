# Test di conversazione

[`Backends/DeepSeekV4/DSML/`](Backends/DeepSeekV4/DSML/README.md) valida il
rendering della chat, il markup DSML/tool, il parsing delle chiamate ai tool,
la gestione degli input malformati e il comportamento dei ruoli di
conversazione.

[`Backends/GLM52/`](Backends/GLM52/README.md) verifica ruoli GLM, reasoning,
tool XML tipizzati, parsing incrementale e contenimento dei marker.

Aggiungi casi di regressione per i token di confine e per output del modello
non validi. Le asserzioni dovrebbero coprire la struttura interpretata, non
solo la stringa finale visualizzata.
