[English](README.md) | **Italiano**

# View model della Chat

`ChatStore` è il proprietario dello stato sul main actor per la feature Chat.
Il file principale dichiara lo stato osservabile condiviso e
l'inizializzazione; estensioni mirate separano le responsabilità:

- `+ModelLifecycle`: discovery locale filtrata dal catalogo, ripristino dei
  percorsi gestiti/manuali, caricamento e scaricamento del modello ed
  esposizione dell'engine condiviso.
- `+Generation`: invio del prompt, eventi di streaming, cancellazione ed
  errori.
- `+ToolLoop` e `+Agents`: esecuzione dei tool e coordinamento dei sub-agent.
- `+Sessions`: ciclo di vita delle chat salvate e ripristino della cronologia.
- `+Attachments`: selezione dei file e preparazione del contesto.
- `+PerformanceSettings`, `+Tuning` e `+Benchmark`: knob di runtime, metriche
  della cache e i brevi benchmark delle impostazioni di prefill. Il preset
  veloce misurato possiede qui la scelta persistita di `DS4MultiQuantCache` e
  la esporta come `DS4_MULTI_QUANT_CACHE` prima del successivo caricamento del
  modello.
- `+MachineAutoTune`: l'adapter GUI per il nucleo puro di auto-tune di
  `DS4Engine`. Congela i dati di utilizzo, misura una sola volta il root caldo
  caricato, poi esegue al massimo una misurazione di reload/warmup
  completamente attesa per ogni configurazione unica. Una cache locale alla run
  conserva l'osservazione di decode valida più alta completa; le visite
  ripetute sono cache hit. Gli slot della cache degli esperti usano una
  camminata sequenziale che privilegia la salita, bloccano la direzione dopo
  una vittoria e si fermano alla prima sconfitta invece di spazzare l'intera
  griglia. La promozione richiede un risultato di decode strettamente
  superiore, qualità esatta rispetto al root e i gate di prefill, stabilità,
  RAM e swap. La contabilità dello swap tiene l'init a freddo, il warmup e il
  primer scartato in una finestra di setup diagnostica; una barriera di
  assestamento fail-closed ancora la successiva finestra a regime misurata, il
  cui limite di 128 MiB alimenta da solo il gate di promozione. Un root già
  sotto il normale floor di RAM del 10% usa un unico envelope immutabile
  relativo alla baseline; solo geometrie residenti neutre o riducenti in
  memoria possono allora essere misurate. Le configurazioni candidate devono
  restare solo a livello di ambiente; persisti il finalista detentore del
  record solo dopo che il suo warmup con l'agente attivo e la sua sonda finale
  di swap a regime hanno avuto successo, mai una prova intermedia. L'adozione è
  una transazione durevole la cui recovery all'avvio riporta
  un'installazione/commit interrotta allo snapshot iniziale completo.

Le view chiamano questo livello; questo livello chiama `InferenceService`.
Mantieni tutte le mutazioni pubblicate sul main actor, evita una seconda
istanza dell'engine e metti il nuovo comportamento nell'estensione
corrispondente alla sua responsabilità invece di far crescere
`ChatStore.swift`.

Un modello gestito dall'app sotto `Application Support/DwarfStar/models` viene
ripristinato dal suo percorso persistito in chiaro; un modello esterno viene
ripristinato tramite il bookmark security-scoped di proprietà di `ModelPicker`.
Non lasciare che il ripristino del bookmark sovrascriva una selezione gestita
più recente e non spostare la logica di catalogo/download in questo view model.

## Proprietà asincrona e ordine dei risultati dei tool

Ogni turno dell'utente cattura `conversationEpoch`. Stop, Nuova Chat,
l'attivazione di una sessione e il successivo turno nuovo fanno avanzare
quell'epoch oltre a cancellare il task corrente. Gli eventi di stream, il
dispatch dei tool, l'avanzamento dei sub-agent, la pulizia al completamento e i
callback di contesto/profilo devono verificare l'epoch catturato prima di
leggere un vecchio indice di messaggio o mutare lo stato della UI. Questo
secondo controllo di proprietà è necessario perché la cancellazione può andare
in race con un `AsyncStream` o con l'await di un tool esterno.

Per un blocco assistant multi-tool, mantieni uno slot di risultato per ogni
chiamata emessa. I risultati automatici, i rifiuti per policy, gli errori di
duplicato/limite, i risultati dei sub-agent e i valori inseriti manualmente
riempiono quegli slot originali. Passa gli slot completati all'engine
nell'ordine delle chiamate; raggruppare i risultati automatici prima di quelli
manuali cambia l'associazione posizionale chiamata/risultato di DSML.
