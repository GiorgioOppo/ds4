# Ciclo di vita dei modelli

## Download

La GUI legge `DeepSeekV4ModelCatalog`, sceglie un `ModelCatalogEntry` e scarica
i suoi `ModelTarget` con `ModelDownloader.acquire`. Il file viene scritto come
`.part`; una richiesta successiva usa HTTP Range per riprendere i byte già
presenti. Il risultato distingue esplicitamente un nuovo download da un file
finale regolare e non vuoto già presente.

La GUI usa come destinazione scrivibile
`~/Library/Application Support/DwarfStar/models/`. Prima di aprire la rete
cerca inoltre il filename esatto nelle directory modello di sviluppo e nella
directory del GGUF attivo: se trova un finale regolare e non vuoto lo riusa in
posizione. Un finale vuoto non è considerato valido.

La ripresa accetta `206` solo con `Content-Range` coerente. Se il server ignora
Range e risponde `200`, il `.part` viene troncato e riscritto; `416` vale come
completamento solo quando la dimensione remota coincide con quella locale.
Cancellazione ed errori di trasporto conservano il `.part`; preflight dello
spazio, validator remoto e gate per path evitano rispettivamente saturazione,
append su oggetti cambiati e doppio writer.

Prima della rinomina atomica i nuovi download vengono verificati con il digest
SHA-256 fissato nel catalogo. I tre Flash e PRO Q2 singolo sono eseguibili e
selezionabili; il package PRO Q4 resta `downloadOnly`. MTP è un accessorio
distinto e non è una voce del catalogo principale.

| Voce | Artefatti | Download | Selezione/esecuzione |
|---|---:|---:|---:|
| Flash Q2 imatrix | 1 | sì | sì |
| Flash mixed Q2/Q4 imatrix | 1 | sì | sì |
| Flash Q4 imatrix | 1 | sì | sì |
| Pro Q2 imatrix | 1 | sì | sì |
| Pro Q4 split | 2 shard | sì | no |

La scansione automatica propone i tre filename Flash e il Pro Q2 selezionabili.
**Browse** resta disponibile per file esterni, ma `InferenceService.inspectModel`
e `BackendSelector` ne validano architettura e profilo prima di cambiare il
modello attivo.

## Credenziali

La precedenza del token Hugging Face è: valore esplicito, `HF_TOKEN`, file
standard della cache Hugging Face. Nell'app il valore esplicito proviene dal
Keychain tramite `HFTokenStore`; non deve essere mostrato integralmente o
incluso nei messaggi di errore.

## Artefatti derivati

`ExpertBundleTool.ensure` apre il GGUF in mmap, deriva geometria e
quantizzazione degli esperti e chiama `ExpertBundle.openOrBuild`. Il sidecar e
la cache dense-Q4 sono ricostruibili: il GGUF resta sempre la fonte primaria.

## Uso locale e distribuito

Il servizio locale risolve i sidecar dalla directory configurata. In modalità
distribuita il coordinator include gli artefatti attivi nel manifest e il
worker scarica soltanto quelli non già verificati. La configurazione di
assegnazione decide se il worker deve usarli.

Il downloader è più generale del loader locale: il package PRO Q4 può essere
acquisito senza essere proposto come modello attivo. Nessun percorso corrente
carica il componente MTP separato; il self-speculative `DS4_SPEC_K` della demo
non usa quei pesi.

Vedi [`Download`](Download/README.md) e
[`Distributed/Files`](../Distributed/Files/README.md).
