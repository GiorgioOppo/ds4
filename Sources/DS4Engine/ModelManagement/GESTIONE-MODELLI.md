# Ciclo di vita dei modelli

## Download

La GUI seleziona un `ModelTarget` e chiama `ModelDownloader.download`. Il file
viene scritto come `.part`; una richiesta successiva usa HTTP Range per
riprendere i byte già presenti. Alla conclusione viene calcolato SHA-256 e, se
il target ha un digest noto, il file è accettato solo in caso di corrispondenza.

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

Il downloader è più generale del backend: Pro e MTP possono comparire nel
catalogo, ma al momento sono artefatti di sola acquisizione/ispezione. Sia il
motore locale sia quello distribuito rifiutano il profilo Pro; nessun percorso
carica il componente MTP separato. Il self-speculative `DS4_SPEC_K` della demo
non usa quei pesi.

Vedi [`Download`](Download/README.md) e
[`Distributed/Files`](../Distributed/Files/README.md).
