[English](CRITTOGRAFIA.md) | **Italiano**

# Crittografia e conformità all'esportazione (export compliance) — DwarfStar

Questo documento registra come DwarfStar usa la crittografia, sia ai fini
della conformità alle norme di esportazione Apple (export compliance, App
Store / TestFlight) sia per chiarezza sulla sicurezza operativa. È un
inventario tecnico, non una consulenza legale. Le regole di esportazione e le
domande dell'App Store possono cambiare; ripetere il questionario di App Store
Connect per ogni modifica sostanziale a networking, storage, dipendenze o
codice crittografico.

Ultima revisione rispetto alla guida pubblica di Apple e alla guida del BIS
statunitense: 2026-07-13.

## Riepilogo di conformità

L'app è contrassegnata come:

```text
ITSAppUsesNonExemptEncryption = NO
```

In questo progetto il valore è impostato in `project.yml` come
`INFOPLIST_KEY_ITSAppUsesNonExemptEncryption: NO` e viene propagato
nell'`Info.plist` generato.

L'inventario attuale del progetto supporta `NO`: DwarfStar usa HTTPS/TLS e i
servizi Keychain del sistema operativo, oltre a funzioni di hashing, e non
implementa né incorpora un algoritmo di cifratura proprietario. Apple documenta
`NO` per le app che non usano crittografia o che usano solo forme esenti dal
caricamento dei documenti di esportazione in App Store.

Questa chiave da sola **non** determina ogni obbligo di esportazione
statunitense. La guida di Apple nota esplicitamente che alcune forme di
crittografia esente possono comunque richiedere un rapporto di
autoclassificazione BIS di fine anno, mentre il BIS lega quel rapporto alla
precisa classificazione License Exception ENC. Questo documento quindi non
afferma che un rapporto o un CCATS non possano mai essere richiesti; l'editore
deve confermare la classificazione e lo scenario di distribuzione.

## Crittografia per componente

| Componente | Crittografia usata | Valutazione tecnica |
|---|---|---|
| **Download dei modelli** (`ModelDownloader` -> `huggingface.co`) | **HTTPS/TLS** tramite `URLSession` di Foundation; **SHA-256** fissato nel catalogo (`CryptoKit.SHA256`) per ogni nuovo download del modello principale | TLS fornito dal sistema operativo più hashing; coerente con l'attuale inventario `NO`. |
| **KV cache su disco** (`KVCFile`, `DiskKVStore`) | **SHA-1** (`CryptoKit.Insecure.SHA1`) per assegnare i nomi ai file di checkpoint | Solo hashing; non fornisce riservatezza. |
| **Trasferimento file distribuito** (`DistFiles`, `DistWorker+Files`) | **SHA-256** per l'identità dei file e per i checkpoint concatenati riprendibili | Solo hashing di integrità; il trasporto stesso non è cifrato. |
| **Archiviazione del token Hugging Face** (`HFTokenStore`) | **Keychain** di macOS tramite il framework Security | Archiviazione sicura fornita dal sistema operativo; nessuna crittografia custom in DwarfStar. |
| **Server HTTP locale** (`LocalServer`) | Nessuna; HTTP in chiaro | Nessuna cifratura o riservatezza. |
| **Trasporto di inferenza distribuita** (`DistTransport`) | Nessuna; TCP in chiaro sulla LAN | Nessuna cifratura o riservatezza. |
| **Engine di inferenza** (`DS4Core`, `DS4Metal`, `DS4Engine`) | Nessuna | Non applicabile. |

Note importanti:

- **Nessuna crittografia custom.** `CryptoKit` è usato solo per gli hash: SHA-1
  per i nomi della KV cache e SHA-256 per l'integrità dei modelli e dei file
  distribuiti. Il framework Security è usato per il Keychain del sistema
  operativo.
- **L'hashing non è cifratura** ai fini dell'export compliance. Non nasconde i
  dati; li identifica o li verifica.
- **Perché SHA-256 sul contenuto invece del pinning TLS.** Gli URL Hugging Face
  `resolve/<revision>/...` (`main` per DeepSeek, un commit fissato per GLM)
  reindirizzano a una CDN LFS le cui chiavi pubbliche sono fuori dal controllo
  di questa app e possono ruotare. Il pinning delle chiavi pubbliche via ATS
  sarebbe fragile. L'hashing del contenuto finale protegge da corruzione o
  manomissione indipendentemente dalla rotazione delle chiavi della CDN.
- **Che cosa viene verificato.** Un artefatto di catalogo appena trasferito
  viene controllato rispetto al numero di byte della risposta e al suo SHA-256
  fissato in `ModelCatalogRegistry` prima che il file `.part` venga rinominato
  in `.gguf`. I trasferimenti ripresi vengono sottoposti a hash con un
  passaggio a memoria limitata sull'intero file, non solo sul suffisso
  aggiunto.
- **Politica sui file esistenti.** Un file finale regolare e non vuoto con il
  nome esatto del catalogo è trattato come contenuto installato di proprietà
  dell'utente e non viene riletto soltanto per calcolare l'hash di centinaia di
  gigabyte. Se `expectedSizeBytes` è fissato, come per GLM 5.2, anche la
  dimensione deve corrispondere. Questa è una politica esplicita di
  performance, non un'asserzione crittografica su quel file preesistente.
  Rimuoverlo o rinominarlo per forzare una nuova acquisizione verificata se la
  sua provenienza è incerta.
- **Redirect delle credenziali.** Il bearer token opzionale di Hugging Face
  viene inviato a `huggingface.co`; quando la richiesta viene reindirizzata
  verso un host LFS/CDN, l'header `Authorization` viene rimosso. La CDN usa il
  proprio URL di redirect firmato.
- **Il TLS è interamente fornito dal sistema operativo.** DwarfStar non
  implementa TLS; usa le API di rete di Apple.

## Avviso di sicurezza

Il server locale e il trasporto distribuito sono intenzionalmente semplici e
funzionano in chiaro:

- Il server HTTP è pensato per `127.0.0.1`. Se lo si associa a `0.0.0.0` o a un
  indirizzo LAN, i prompt e il testo generato non sono cifrati.
- L'inferenza distribuita scambia hidden state e dati dei token tra i Mac in
  chiaro sulla rete locale.

Usare queste funzionalità solo su reti fidate. Se si espone il server oltre il
loopback, collocarlo dietro TLS, ad esempio tramite Caddy, Nginx, WireGuard o
un tunnel SSH.

## Risposta in App Store Connect

Con l'attuale inventario tecnico, `ITSAppUsesNonExemptEncryption = NO` è
coerente con la descrizione di Apple e semplifica le domande in fase di invio.
L'Account Holder o l'App Manager devono comunque rispondere in App Store
Connect basandosi sulla build effettiva e sui paesi di distribuzione previsti.
Se una versione futura aggiunge crittografia non esente, la dichiarazione deve
essere modificata e la documentazione richiesta deve essere fornita.

## Riferimenti ufficiali

- [Apple: `ITSAppUsesNonExemptEncryption`](https://developer.apple.com/documentation/bundleresources/information-property-list/itsappusesnonexemptencryption)
- [Apple: Panoramica dell'export compliance](https://developer.apple.com/help/app-store-connect/manage-app-information/overview-of-export-compliance)
- [Apple: Conformità alle normative sull'esportazione della crittografia](https://developer.apple.com/documentation/security/complying-with-encryption-export-regulations)
- [BIS: Autoclassificazione annuale](https://www.bis.gov/learn-support/encryption-controls/annual-self-classification)
