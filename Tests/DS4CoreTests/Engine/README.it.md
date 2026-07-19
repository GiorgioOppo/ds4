[English](README.md) | **Italiano**

# Test dell'Engine

Test per i servizi di orchestrazione e integrazione in `DS4Engine`: profili
degli agenti, aggregazione dei benchmark, diagnostica, protocollo distribuito,
gestione dei modelli, persistenza, progetti e tool.

Preferisci transport iniettati, directory temporanee e fixture locali. Gli
unit test non devono richiedere credenziali, accesso a internet, i progetti
dell'utente o un modello di produzione caricato.

I test [`Benchmark`](Benchmark/README.it.md) esercitano l'aggregazione
deterministica delle osservazioni di accuratezza next-token. Intenzionalmente
non caricano un GGUF né eseguono Metal: la valutazione end-to-end del modello
appartiene a un'esecuzione di benchmark riproducibile, non alla suite di unit
test.
