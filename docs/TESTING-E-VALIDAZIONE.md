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

## Test prestazionali

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
