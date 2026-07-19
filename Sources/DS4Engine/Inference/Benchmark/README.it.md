[English](README.md) | **Italiano**

# Inference/Benchmark

Misura il backend già caricato senza dipendere dalla GUI.

## Componente

`InferenceService+Benchmark.swift` espone due misure distinte:

- `benchmark` usa un prompt sintetico e misura throughput di prefill e decode;
- `accuracyBenchmark` tokenizza un corpus di testo semplice e misura la
  **top-1/top-2/top-3 next-token accuracy** con teacher forcing.

Il primo registra sia throughput medio sia velocità per-token; `warmup`
inizializza kernel e cache degli esperti una sola volta. Il secondo restituisce
osservazioni progressive e bucket pronti per i grafici, con accuratezza locale
e cumulativa top-1, top-2 e top-3 espresse come frazioni `0...1`.

## Correttezza next-token

Il corpus deve contenere almeno due token. Il benchmark costruisce più segmenti
del corpus usando un seed riproducibile. Ogni segmento ha un punto di partenza
distinto per il primo target; i relativi intervalli possono comunque
sovrapporsi. La lunghezza del contesto viene estratta uniformemente fra minimo
e massimo effettivi, senza oltrepassare né il target scelto né la capacità KV.
Il planner preferisce punti che permettono di valutare tutti i token richiesti
per segmento e usa la coda corta del corpus soltanto quando servono altri punti.
La selezione usa un Fisher-Yates parziale e sparso: tempo e memoria dipendono
dal numero di segmenti effettivamente richiesti, non dal numero di possibili
partenze nel corpus. A parità di seed, aumentare il numero di segmenti conserva
il prefisso già selezionato e le relative lunghezze di contesto.

Il contesto del segmento non viene valutato; per ogni posizione successiva il
modello estrae i tre token coi logits maggiori e verifica se il token vero è il
primo candidato, compare nei primi due o compare nei primi tre. Questi insiemi
sono annidati, quindi per costruzione `top-1 <= top-2 <= top-3`. Il benchmark
reinserisce poi sempre il token vero: un errore non sposta né contamina le
osservazioni seguenti. Il passaggio BOS → primo token non viene conteggiato.

I tre elementi sono **candidati token del vocabolario**, non esperti MoE. La
selezione degli esperti avviene internamente ai layer e non viene misurata da
questo benchmark.

`StreamingDecoder.prefillTopK` riusa il prefill layer-major: gli hidden finali
del chunk esistono già, l'output head viene applicato soltanto alle posizioni da
valutare e i tre massimi vengono letti dal buffer Metal condiviso. Non vengono
conservati `token × vocab` logits e non si degrada al costoso `forward`
token-per-token.

Contesto minimo/massimo, token massimi per segmento e numero di segmenti vengono
limitati al corpus e alla finestra KV; ogni segmento valuta sempre almeno un
token. Un numero di segmenti nullo o negativo viene normalizzato a uno e il piano
segnala il clamp tramite `truncated`. Il piano espone valori effettivi e
troncamento, così il chiamante non deve dedurli dal risultato. Le metriche
globali sommano corretti e valutati di tutti
i segmenti prima di dividere: sono quindi ponderate per i token realmente
valutati, non una media semplice delle percentuali dei segmenti. Il risultato
mantiene anche i valori per segmento, usati dal grafico comparativo.

L'overload storico `accuracyBenchmark` con un singolo `contextTokens` e un
singolo `maxTokens` resta disponibile e costruisce un piano a un solo segmento
con contesto fisso. Le percentuali mostrate dalla UI sono ciascuna
`topKAccuracy × 100`: il motore conserva sempre la rappresentazione `0...1`.

## Flusso e dipendenze

L'estensione opera sull'actor di [`Service`](../Service/README.it.md) e usa sampler
e decoder di `DS4Core`/`DS4Metal`. Entrambe le misure alterano la KV e la marcano
sporca. Il benchmark di correttezza non modifica `committedIds`, system prompt o
stato logico della conversazione: il turno reale successivo ricostruisce la KV.
Inoltre salva e ripristina l'usage-imatrix, così il corpus di prova non cambia il
profilo esperti dell'agente. Cancellazione ed errori seguono lo stesso cleanup.

## Estensione

Una nuova metrica deve separare chiaramente tempo di preparazione, prefill,
sampling CPU e decode GPU. Non cambiare parametri di qualità dell'utente durante
un benchmark e mantenere cancellabili i loop lunghi. Le accuracy top-k misurano
accordo esatto con quel corpus, non correttezza semantica: per confronti fra
build, GGUF o knob usare sempre lo stesso testo, prefisso e intervallo valutato.
