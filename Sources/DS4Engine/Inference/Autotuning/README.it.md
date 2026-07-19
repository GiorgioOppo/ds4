[English](README.md) | **Italiano**

# Nucleo di auto-tuning della macchina

Questa directory contiene il livello decisionale puro e neutrale rispetto
all'architettura usato dall'auto-tuner della macchina nella GUI.
Deliberatamente non carica un GGUF, non crea un device Metal, non modifica
`UserDefaults` e non legge le variabili d'ambiente del processo. L'adapter di
DwarfStar è responsabile di questi effetti e alimenta questo livello con
misurazioni indipendenti.

`MachineAutoTune.swift` fornisce:

- una configurazione tipizzata e un manifest, consapevole della RAM, dei
  controlli a qualità esatta;
- firme di qualità compatte e fail-closed per tracce di logit a vocabolario
  completo limitate: un frame per token greedy, una dimensione di vocabolario
  stabile e diversa da zero e la concordanza top-1/token sono obbligatorie;
- gate single-shot high-water, di screening e A/B/B/A bilanciati nell'ordine;
- salvaguardie su prestazioni, metriche secondarie, stabilità, memoria libera e
  swap;
- un envelope low-RAM immutabile derivato dal root engine già caricato;
- una macchina a stati direzionale che privilegia la salita per ladder
  ordinate, più la selezione dei vicini e una mediana matematicamente corretta
  per la ricerca per coordinate.

`MachineAutoTuneSwapCounter.swift` converte il contatore cumulativo di byte di
swapout dell'host in due finestre esplicite. L'init del modello, il warmup e il
benchmark caldo scartato appartengono a una finestra di carico diagnostica; il
gate di promozione vede solo il successivo benchmark misurato a regime. Una
barriera di assestamento fail-closed dopo il primer scartato valida e ancora il
campione iniziale dello stato a regime; un contatore non disponibile o non
valido quindi interrompe l'osservazione invece di indebolire la protezione. Il
campione finale deve essere prelevato prima di quiesce/teardown. Il wrap del
contatore è gestito esplicitamente, mentre una regressione inspiegata del
contatore viene trattata come un reset e fallisce in modo chiuso invece di
essere riportata come swap zero.

La GUI usa `highWaterComparison`: il root caldo già caricato e ogni candidato
unico vengono misurati una volta, poi memorizzati in una cache locale alla run.
Viene conservata per intero l'osservazione idonea con la mediana di decode più
alta — le metriche di osservazioni diverse non vengono mai combinate. La
promozione richiede un risultato di decode strettamente superiore, al massimo
l'8% di regressione in prefill, una stabilità di almeno 0.75, il floor di RAM,
non più di 128 MiB di swapout a regime e qualità bit-exact rispetto al root
immutabile. Una configurazione ripetuta è un cache hit e non esegue alcun
reload. Questo scambia deliberatamente la stima del rumore ABBA con la
velocità; un outlier verso l'alto può rendere il record conservativamente
difficile da battere, ma non può indebolire un gate di sicurezza o di qualità.

Gli slot della cache degli esperti usano la macchina a stati direzionale invece
di uno sweep completo. Dopo che `20→22` vince, prova `24` senza provare `18`;
se `24` perde, quel knob si ferma e nessun valore più grande del manifest viene
misurato. Se la prima sonda verso l'alto perde, il vicino immediatamente
inferiore è l'unico fallback. Gli sweep completi restano riservati a griglie
hardware genuinamente non monotone.

`screening`, `repeatability` e `abba` restano disponibili per il tuner isolato
in un processo separato e per i test. La GUI non finge più che osservazioni
riutilizzate costituiscano nuove coppie ABBA.

Quando il root noto come caricabile lascia già meno del 10% di memoria libera,
`MachineAutoTuneMemoryEnvelope` fornisce una policy vincolata invece di rendere
impossibile il tuning. Richiede almeno 512 MiB mentre il root noto come
caricabile è ancora residente, ancora l'intera run al livello di RAM libera del
root (con un punto percentuale di tolleranza sui contatori VM) e permette solo
candidati il cui delta residente noto è zero o negativo. Il ripristino del root
e questi candidati limitati mantengono la stessa riserva known-loadable di
512 MiB dopo il teardown; i candidati sconosciuti o che aumentano la memoria
richiedono la più severa riserva transitoria del 12%/1.5 GiB. I normali gate di
velocità, qualità esatta, stabilità e swapout restano invariati. La GUI deve
applicare lo stesso envelope immutabile prima dell'init, dopo init/warmup e
durante l'installazione finale; non deve mai ribasare il floor dopo una
promozione intermedia.

Il manifest sicuro standard esclude deliberatamente `q8NSG`. Cambiare la
suddivisione della riduzione Q8 può cambiare l'ordine di accumulazione in
virgola mobile e quindi i bit grezzi dei logit. Il knob resta parte di
`MachineAutoTuneKnob` e `MachineAutoTuneConfiguration` per manifest
sperimentali espliciti, ma la ricerca predefinita a qualità esatta della GUI
non lo esplora automaticamente.

I test usano osservazioni e firme sintetiche, così il comportamento di gate e
ricerca resta deterministico e gira senza file di modello né accesso alla GPU.

`MachineAutoTuneTransactionStore.swift` è il confine di adozione durevole usato
dalla GUI dopo la validazione del detentore del record. Il vincitore viene
installato e riscaldato con l'agente selezionato prima che le preferenze
vengano committate, e il chat engine completamente pronto deve comunque
soddisfare il floor di RAM effettivo della run e il gate di 128 MiB nella sua
sonda a regime post-warmup. Il suo swap di setup a freddo viene riportato
separatamente. Un record atomico versionato consente al successivo avvio
dell'app di riportare un commit multi-chiave interrotto allo snapshot iniziale
completo. Le sue transizioni di recovery sono testate con defaults temporanei
isolati e URL di record dedicati.
