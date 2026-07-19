# Testing e validazione

I test verificano livelli diversi: logica pura, formati, protocollo, wrapper
Metal, composizione del grafo e servizi applicativi. Una build riuscita non
sostituisce i test numerici; uno skip GPU non equivale a un test superato.

## Struttura

```text
Tests/DS4CoreTests/
  Core/      tokenizer, conversazione, formati, sampling, modello, storage
  Metal/     runtime, kernel, grafo, decode e pesi
  Engine/    protocollo, persistenza, progetti, download e tool
```

Il nome storico del target è `DS4CoreTests`, ma il target dipende da Core,
Metal ed Engine. Le sottocartelle riflettono il dominio verificato.

## Comandi principali

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build --disable-sandbox

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift test --disable-sandbox

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --product DS4Demo --disable-sandbox

xcodegen generate

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build -project DwarfStar.xcodeproj -scheme DwarfStar \
  -destination 'platform=macOS' -derivedDataPath /tmp/DwarfStarDerivedData \
  CODE_SIGNING_ALLOWED=NO
```

Usare la build Release per le prestazioni; la Debug serve a correttezza e
diagnostica.

## Test puri

I test Core e buona parte di Engine non richiedono GPU. Devono coprire:

- input valido e malformato;
- limiti di dimensione e overflow;
- round-trip di serializzazione;
- compatibilità del formato persistente;
- errori espliciti, non crash o fallback silenziosi;
- determinismo quando seed e input sono fissati.

Per i protocolli aggiungere sempre test di payload troncato, conteggio
impossibile e campo fuori range.

## Test Metal

Un test kernel tipico prepara un input piccolo, calcola un riferimento CPU,
esegue il wrapper e confronta output e tolleranza. Coprire almeno:

- dimensione normale;
- coda non multipla del threadgroup quando supportata;
- offset o view se il wrapper li accetta;
- ogni quantizzazione dichiarata;
- valori estremi rappresentativi.

Le tolleranze devono derivare dal tipo e dall'ordine di accumulo, non essere
allargate finché il test passa.

## Skip GPU

I test Metal possono essere ignorati quando:

- non esiste un `MTLDevice` disponibile nell'ambiente;
- la sandbox impedisce accesso al compilatore/runtime necessario;
- un fixture esterno o un vecchio percorso kernel non è presente.

Ogni skip deve includere un motivo leggibile. In CI o su un Mac di validazione
si deve distinguere fra suite realmente eseguita e suite soltanto scoperta.

## Parità di grafo e decode

I test di grafo confrontano stadi completi, non solo primitive. Per modifiche a
route, attention, compressore o MoE verificare:

1. wrapper singolo;
2. composizione del grafo;
3. layer completo;
4. forward di più token quando interviene stato ricorrente;
5. testo/argmax su un fixture reale, se disponibile.

Il prefill e il decode devono convergere allo stesso stato per una sequenza
equivalente, salvo percorsi esplicitamente approssimati.

## Benchmark di velocità

Un benchmark utile mantiene costanti modello, prompt, contesto, sampling,
usage profile e stato delle cache. Registrare separatamente:

- tempo di caricamento;
- tok/s di prefill;
- tok/s di decode dopo warm-up;
- byte letti e banda effettiva;
- pressione memoria e swap;
- hit-rate della cache esperti.

Cambiare un solo knob per prova. `DS4_PROFILE_ROUTE` altera il timing e non va
usato per il throughput finale.

### Gate A/B per ottimizzazioni Metal

Per un nuovo percorso selezionabile tramite `DS4_*`, usare il runner di processo
prima di promuoverlo nei default:

```sh
scripts/metal_ab.sh model.gguf prompt.txt DS4_NUOVO_KNOB 0 1 8
```

Il runner usa due processi separati con prompt identico, greedy decode e
persistenza usage disabilitata. Confronta gli id generati e un numero limitato
di vettori logits completi: i vettori vengono trattenuti copy-on-write durante
la misura, scritti soltanto dopo i timer e analizzati via `mmap`. Il report
distingue `PASS_EXACT`, `PASS_NUMERIC` e `FAIL` e include tok/s di prefill,
profilo decode e regime. Impostare `DS4_AB_ATOL=0 DS4_AB_RTOL=0` per percorsi
che promettono parità bit-per-bit; tolleranze maggiori devono essere motivate
per fusioni con diverso ordine di riduzione o percorsi dichiaratamente lossy.

Page cache, temperatura e pressione di memoria possono favorire uno dei due
processi: ripetere con `DS4_AB_ORDER=candidate-first` e non promuovere variazioni
sotto il rumore della macchina. `python3 scripts/metal_ab_compare.py --self-test`
valida l'analizzatore senza richiedere Metal o un GGUF.

Per cercare una configurazione composta su più knob usare
`scripts/metal_autotune.py`: esegue sweep monodimensionali/coordinate ascent,
conferma i candidati in ordine ABBA, confronta anche ogni finalista con la root
iniziale e respinge run contaminati da pressione RAM o swap. Il formato trace è
validato fail-closed e top-3, hash e conteggio dei valori finiti vengono
ricalcolati dai `.f32`, non accettati sulla fiducia dal JSON. Procedura e
comandi completi sono in [AUTOTUNING-METAL.md](AUTOTUNING-METAL.md).

## Benchmark di correttezza next-token

Il benchmark di correttezza misura la **top-1/top-2/top-3 accuracy in teacher
forcing** su un testo fisso. Per ogni posizione il modello riceve il prefisso
reale, ordina i token per logit e verifica se il token successivo del corpus è
il primo candidato, compare nei primi due o compare nei primi tre; nel passo
seguente viene comunque inserito il token reale. Temperatura e sampling della
generazione non partecipano alla metrica. Il seed del benchmark sceglie soltanto
i segmenti del corpus e le loro lunghezze di contesto: a parità di input rende
il piano riproducibile, ma non altera logits o graduatoria dei candidati. I tre
candidati sono token del vocabolario, non esperti MoE.

La prova multi-segmento richiede numero di segmenti, contesto minimo/massimo,
token massimi per segmento e seed. Ogni primo target deve essere distinto; i
segmenti possono però sovrapporsi. Il contesto viene estratto uniformemente
entro l'intervallo effettivo e non può superare l'indice del target: perciò il
massimo del singolo segmento è `min(massimoEffettivo, targetStart)`. Il punto di
inizio del contesto deve restare non negativo. Il planner preferisce target che
permettono il numero massimo richiesto di valutazioni e ricorre alla coda corta
solo se sono necessari altri segmenti; ogni segmento valuta comunque fra uno e
il limite richiesto di token.

Se il testo produce `N` token e si usa il prefisso minimo di un token, le
predizioni verificabili sono `N - 1`: il primo token costituisce il contesto
iniziale e non è un target. Con un prefisso non valutato di `C` token, i target
disponibili diventano `N - C`, prima degli eventuali limiti di contesto o di
durata scelti dalla UI. Questi confini devono restare coperti da test espliciti,
perché valutare il prefisso o saltare l'ultimo target introduce un errore
off-by-one nella percentuale. Questa è anche la semantica dell'overload storico
a segmento singolo, che usa un contesto fisso.

Registrare almeno:

- testo/corpus o hash stabile del contenuto e lingua;
- GGUF, tokenizer, build e configurazione completa del motore;
- numero di token tokenizzati e numero di predizioni realmente valutate;
- seed, segmenti richiesti/effettivi, intervallo di contesto e massimo per
  segmento;
- token corretti e accuracy complessiva per top-1, top-2 e top-3;
- punto di partenza, contesto e token valutati per ciascun segmento;
- eventuale troncamento imposto dal contesto o dal limite scelto.

Le tre metriche devono essere annidate in ogni osservazione, bucket e risultato:
`top-1 <= top-2 <= top-3`. Le curve cumulative mostrano le percentuali
dall'inizio della prova; le serie locali misurano soltanto il bucket corrente.
L'ultimo bucket può contenere meno elementi e deve usare il proprio conteggio
reale come denominatore per tutti e tre i ranghi. Anche una top-3 accuracy bassa
non equivale automaticamente a una risposta semanticamente scorretta: testi
naturali ammettono spesso più continuazioni plausibili. La metrica è soprattutto
utile per confrontare build, quantizzazioni e ottimizzazioni sullo stesso corpus.

L'accuratezza globale deve essere calcolata come somma dei token corretti divisa
per somma dei token valutati. Non fare la media semplice delle percentuali dei
segmenti: darebbe troppo peso ai campioni corti. Il grafico per-segmento serve a
mostrare la variabilità lungo il corpus, mentre i KPI globali restano ponderati.

## Baseline della ristrutturazione

Verifica eseguita il 13 luglio 2026:

- build SwiftPM Debug riuscita;
- build Release di `DS4Demo` riuscita;
- build Xcode dell'app riuscita;
- 246 test scoperti/eseguiti dalla suite, 96 skip legati soprattutto
  all'ambiente Metal;
- due asserzioni note in `ProjectCacheTests.testSymlinksAreRefused`, già
  presenti nella baseline precedente alla ristrutturazione.

Questa fotografia non è una deroga permanente: quando cambia l'ambiente o il
test dei symlink viene corretto, aggiornare il documento.

## Checklist per una modifica

- [ ] Il codice compila in SwiftPM Debug.
- [ ] I test puri interessati passano.
- [ ] I test Metal sono stati realmente eseguiti su un Mac quando necessario.
- [ ] La demo Release compila.
- [ ] Il progetto Xcode è stato rigenerato se sono cambiati file o cartelle.
- [ ] Le opzioni lossless/lossy sono dichiarate.
- [ ] README locale e documento tematico sono aggiornati.
- [ ] Nessun file generato è stato modificato a mano.

Vedere anche [GUIDA-SVILUPPO.md](GUIDA-SVILUPPO.md) e
[BACKEND-METAL.md](BACKEND-METAL.md).
