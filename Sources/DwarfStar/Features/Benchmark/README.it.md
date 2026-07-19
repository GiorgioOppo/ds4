[English](README.md) | **Italiano**

# DwarfStar/Features/Benchmark

Benchmark nativi del motore condiviso. Il pannello offre due misure distinte:

- **Speed** misura throughput di prefill e generazione a contesti crescenti ed
  è disponibile sia in locale sia attraverso il coordinator distribuito;
- **Correctness** misura quante volte il token successivo di un testo fornito
  dall'utente compare fra i primi uno, due o tre candidati del modello. Usa
  teacher forcing ed è al momento disponibile soltanto sul motore locale.

## Componenti

- [`Controllers/BenchController.swift`](Controllers/BenchController.swift)
  costruisce i punti delle due misure, coordina esecuzione e cancellazione e
  impedisce l'esecuzione locale mentre la chat usa lo stesso motore.
- [`Views/BenchView.swift`](Views/BenchView.swift) presenta selezione del motore,
  corpus e limiti, stato di avanzamento, KPI e grafici Swift Charts.

Il benchmark locale riusa l'unico `InferenceService` già caricato: non crea una
seconda copia del modello. Quello distribuito riusa la connessione attiva del
coordinator. Entrambi modificano lo stato KV e non devono sovrapporsi a una
generazione chat.

## Correctness: significato della metrica

Il testo viene tokenizzato dal tokenizer del GGUF. Il motore seleziona più
segmenti riproducibili tramite seed: il primo target di ciascun segmento è
distinto, anche se contesto e target dei segmenti possono sovrapporsi. Per ogni
segmento estrae uniformemente la lunghezza del contesto fra i limiti scelti e
valuta fino al massimo configurato di token. Preferisce segmenti completi e usa
la coda corta del corpus solo quando il numero richiesto lo rende necessario.

Il contesto non viene valutato; per ogni token successivo il motore riceve
sempre il token reale precedente e confronta il token reale seguente con i tre
candidati dal logit più alto. Questo teacher forcing evita che un singolo errore
faccia divergere il resto del test. I candidati sono token del vocabolario, non
i tre esperti MoE selezionati internamente dal router.

La top-1 conta i casi in cui il primo candidato è esatto, la top-2 quelli in cui
il token atteso compare nei primi due e la top-3 quelli in cui compare nei primi
tre. Per costruzione vale sempre `top-1 <= top-2 <= top-3`.
Non misura correttezza fattuale, capacità di ragionamento o qualità generale di
una risposta. I confronti tra modelli o configurazioni sono significativi solo
usando lo stesso testo, tokenizer, seed, numero di segmenti, intervallo di
contesto e limite per segmento. La UI mostra:

- corretti/totale e accuratezza top-1, top-2 e top-3, durata e token valutati al
  secondo;
- un grafico cumulativo dell'aggregazione progressiva dei token valutati;
- un grafico comparativo top-1, top-2 e top-3 per ciascun segmento;
- l'aggregato globale, calcolato sui conteggi complessivi e quindi ponderato per
  il numero di token realmente valutato in ogni segmento.

Per mantenere reattiva l'interfaccia anche con milioni di token richiesti, il
controller conserva al massimo circa 4.096 punti per il grafico live e aumenta
automaticamente la dimensione minima dei blocchi del grafico finale. I contatori
Top-1/2/3 e il risultato finale continuano a includere ogni token valutato: viene
ridotta solo la densità visiva dei punti.

Per riprodurre una prova vanno mantenuti uguali corpus, seed, numero di
segmenti, intervallo del contesto e token massimi per segmento. Segmenti più
corti non devono avere lo stesso peso di quelli completi nella metrica globale.

La modalità distribuita viene rifiutata esplicitamente per Correctness: il
protocollo corrente restituisce soltanto i logits dell'ultimo token di un chunk
e non esiste un fallback silenzioso al motore locale.

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
[`docs/VALUTAZIONE-DEMO-PERF.md`](../../../../docs/VALUTAZIONE-DEMO-PERF.it.md).

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
engine e aggiornare i test quando cambia la definizione delle metriche. Gli
aggiornamenti prodotti dal lavoro detached devono attraversare stream Sendable:
mai mutare direttamente stato osservabile MainActor dal motore.

Vedere anche il [README dei controller](Controllers/README.it.md), il
[README delle view](Views/README.it.md) e la
[guida di testing](../../../../docs/TESTING-E-VALIDAZIONE.it.md).
