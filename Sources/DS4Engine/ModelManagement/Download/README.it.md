[English](README.md) | **Italiano**

# ModelManagement/Download

Implementa download nativi e riprendibili dei GGUF, senza `curl` o processi
esterni.

## Componenti

- `ModelDownloader.swift`: API `acquire`, controllo locale, percorso sicuro,
  preflight dello spazio, finalizzazione atomica ed esito tipizzato.
- `HTTPRangeFileTransfer.swift`: sessione HTTP senza cache, ricezione a blocchi,
  resume validato e scrittura diretta del `.part`.
- `ModelDownloadTypes.swift`: progresso, fasi ed esito
  `alreadyPresent`/`downloaded` consumati dalla GUI.
- `HFTokenStore.swift`: lettura/scrittura del token Hugging Face nel Keychain,
  forma mascherata e descrizione della fonte attiva.

## Flusso

Il downloader scrive in `<ggufDir>/<nome>.part`, riprende dall'offset presente e
rinomina solo a stream concluso. `URLSessionDataDelegate` consegna blocchi
`Data`: non esiste un'iterazione Swift per ciascun byte dei GGUF da centinaia di
GB. La sessione ephemeral non usa cache HTTP, cookie o credential storage e la
delegate queue è seriale: viene elaborato un solo blocco per volta. Le notifiche
di avanzamento sono limitate per non sovraccaricare la GUI.

I descrittori usati per scrivere il `.part` e per la verifica SHA dopo un resume
sono marcati `F_NOCACHE`: i byte sequenziali del download non devono riempire la
unified file cache di macOS ed espellere pesi o pagine utili del modello. La
verifica legge al massimo 8 MiB per volta e svuota l'autorelease pool a ogni
blocco. Anche il sidecar JSON di resume è accettato solo entro 64 KiB. Il picco
RAM del downloader resta quindi indipendente dalla dimensione totale del GGUF.

Una risposta `206` è accettata solo se `Content-Range` parte dalla dimensione
locale. Se il server risponde `200` a una richiesta Range, il `.part` viene
realmente troncato prima del riavvio; `416` è considerato completo solo quando
la dimensione remota coincide esattamente. ETag/Last-Modified sono conservati
in un piccolo sidecar e inviati come `If-Range` alla ripresa.

I nuovi download del catalogo hanno SHA-256 fissati e sono verificati prima
della rinomina. Un file finale regolare e non vuoto già presente produce subito
`alreadyPresent`: non viene riscaricato né riletto integralmente a ogni apertura.
Quando il target ha `expectedSizeBytes`, il byte count deve coincidere; questa
guardia permette di riusare i GGUF GLM da centinaia di GB senza hash completo a
ogni apertura e senza accettare un finale troncato. Un file finale vuoto non è
mai considerato un GGUF valido.

Il downloader esegue un preflight dello spazio, include un margine per il
filesystem e considera i byte del `.part` già presenti. Un gate actor impedisce
acquisizioni concorrenti dello stesso path. La GUI sceglie
`~/Library/Application Support/DwarfStar/models/` perché le Resources di una
app installata non sono scrivibili.

La risoluzione del token segue: esplicito → `HF_TOKEN` →
`~/.cache/huggingface/token`. Il Keychain è consultato dalla GUI, non dal metodo
generico `resolveToken`, così CLI e test non generano prompt inattesi.

## Dipendenze e sicurezza

Foundation gestisce URLSession/file, CryptoKit l'integrità e Security il
Keychain. La verifica del contenuto è indipendente dalla rotazione dei
certificati CDN. Il bearer token è rimosso quando Hugging Face redirige verso un
host CDN differente. Non inserire token in URL, log o stato non protetto.

## Estensione

Per aggiungere un target, usare ID e nome file stabili, sorgente Hugging Face
con revisione preferibilmente bloccata, dimensione esatta quando disponibile e
SHA-256 autorevole. `ModelDownloader` costruisce la URL dal `source` del target,
quindi cataloghi diversi non condividono più un repository globale. Conservare cancellazione, resume e callback
di avanzamento; evitare buffer proporzionali alla dimensione del modello.

Un target scaricabile non implica compatibilità con il decoder. Il catalogo
principale dichiara il supporto runtime per entry: i tre Flash e PRO Q2
singolo-file sono selezionabili, mentre il package PRO Q4 e i tre GLM 5.2
restano `downloadOnly`. MTP è un accessorio separato e non compare tra i modelli GUI.

Le impostazioni correlate sono nella
[Configuration Reference](../../../../README.it.md#riferimento-di-configurazione).
