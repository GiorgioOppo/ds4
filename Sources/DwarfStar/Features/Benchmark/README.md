# DwarfStar/Features/Benchmark

Benchmark nativo del motore condiviso, disponibile sia in modalità locale sia
attraverso il coordinator distribuito.

## Componenti

- [`Controllers/BenchController.swift`](Controllers/BenchController.swift)
  costruisce i punti di misura a contesti crescenti, coordina prefill e
  generazione e impedisce l'esecuzione locale mentre la chat usa lo stesso
  motore.
- [`Views/BenchView.swift`](Views/BenchView.swift) presenta selezione del motore,
  limiti del contesto, stato di avanzamento, report e grafico Swift Charts.

Il benchmark locale riusa l'unico `InferenceService` già caricato: non crea una
seconda copia del modello. Quello distribuito riusa la connessione attiva del
coordinator. Entrambi modificano lo stato KV e non devono sovrapporsi a una
generazione chat.

## Interpretazione

- Il prefill layer-major tende ad ammortizzare i costi fissi all'aumentare dei
  token, ma chunk, union, cache e pressione memoria possono cambiare il trend.
- Il decode è una sequenza token-per-token; sul profilo Flash in streaming può
  essere dominato dal gather SSD, mentre a contesti lunghi cresce il peso
  dell'attenzione e del KV.
- Il primo punto e i primi token possono includere warm-up, wiring e cache
  fredde. Confrontare sempre la stessa metrica di regime.
- I risultati locali e distribuiti non sono confrontabili se cambiano GGUF,
  contesto, activation bits, route, cache o knob del motore.

I vecchi numeri puntuali sono stati rimossi da questo README perché non avevano
data, build, hash del GGUF e linea completa dei knob. Le misure storiche per cui
è disponibile un contesto esplicito restano in
[`docs/VALUTAZIONE-DEMO-PERF.md`](../../../../docs/VALUTAZIONE-DEMO-PERF.md).

## Provenienza obbligatoria per nuove misure

Quando si aggiorna la documentazione con risultati reali, registrare insieme:

1. data, commit/build e modalità Local o Distributed;
2. modello del Mac, RAM, versione macOS e pressione memoria/swap;
3. nome, dimensione e SHA-256 del GGUF;
4. contesto, prompt o corpus, token generati e warm-up;
5. linea completa `DS4 engine:` e impostazioni non comprese in quella linea;
6. stato caldo/freddo di `.usage.json`, `.q4dense`, `.expbundle` e disk KV;
7. per la distribuzione, peer, topologia, activation bits e caratteristiche
   della rete;
8. media/percentile usato, numero di ripetizioni e variabilità osservata.

Conservare il report grezzo o un riferimento stabile ad esso. Una singola cifra
senza questi dati va descritta come osservazione non riproducibile, non come
default o regressione.

## Regole di modifica

Il controller possiede esecuzione e risultati; la view soltanto controlli e
rendering. Mantenere l'esclusione reciproca con Chat, non introdurre un secondo
engine e aggiornare i test quando cambia la definizione della metrica.

Vedere anche il [README dei controller](Controllers/README.md), il
[README delle view](Views/README.md) e la
[guida di testing](../../../../docs/TESTING-E-VALIDAZIONE.md).
