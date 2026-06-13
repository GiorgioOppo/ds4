# Crittografia ed export compliance — DwarfStar

Documento di riferimento sull'uso della crittografia nell'app, ai fini della
**dichiarazione di export compliance** richiesta da Apple (App Store / TestFlight)
e per chiarezza di sicurezza.

## Dichiarazione (sintesi)

L'app è marcata **`ITSAppUsesNonExemptEncryption = NO`** (in `project.yml` →
`INFOPLIST_KEY_ITSAppUsesNonExemptEncryption: NO`, finisce nell'`Info.plist`
generato).

Significa: l'app usa **solo crittografia esente** — quella standard fornita dal
sistema operativo (HTTPS/TLS) e una funzione di hash. **Non** implementa né
incorpora algoritmi di cifratura proprietari o non esenti. Conseguenza pratica:
**nessun report annuale di autoclassificazione** e **nessun CCATS** richiesti.

## Cosa usa l'app, componente per componente

| Componente | Crittografia | Esente? |
|---|---|---|
| **Download dei modelli** (`ModelDownloader` → `huggingface.co`) | **HTTPS/TLS** tramite `URLSession`/Foundation: cifratura standard del sistema operativo | ✅ esente (cifratura standard OS) |
| **KV cache su disco** (`KVCFile`, `DiskKVStore`) | **SHA-1** (`CryptoKit.Insecure.SHA1`) per **nominare** i file di checkpoint | ✅ non è cifratura — è una funzione di hash (non controllata) |
| **Server HTTP locale** (`LocalServer`) | nessuna — **HTTP in chiaro** | ✅ nessuna cifratura |
| **Inferenza distribuita** (`DistTransport`) | nessuna — **TCP in chiaro** sulla LAN | ✅ nessuna cifratura |
| **Motore di inferenza** (DS4Core/DS4Metal/DS4Engine) | nessuna | — |

Note:
- **Niente crittografia proprietaria.** L'unico modulo crypto importato è
  `CryptoKit`, usato esclusivamente per l'**hash SHA-1** dei nomi file della
  cache KV (porting fedele del formato `ds4_kvstore.c`). L'hashing **non** è
  cifratura ai fini dell'export.
- L'HTTPS è gestito interamente dal sistema operativo: l'app non implementa TLS,
  si limita a usare le API di rete standard.

## Avvertenza di sicurezza (non export compliance, ma rilevante)

Il **server HTTP** e il **transport distribuito** viaggiano **in chiaro**:

- il server è pensato per `127.0.0.1` (locale); se lo esponi su `0.0.0.0` o sulla
  LAN, le richieste/risposte (prompt e testo generato) **non sono cifrate**;
- l'inferenza distribuita scambia stati nascosti e token tra i Mac **in chiaro**
  sulla rete locale.

Usali solo su **reti fidate**. Per esposizione oltre il loopback, mettili dietro
un reverse proxy con TLS (es. Caddy/Nginx) o un tunnel (WireGuard/SSH).

## Come rispondere in App Store Connect

Con `ITSAppUsesNonExemptEncryption = NO` già nell'`Info.plist`, App Store
Connect **non** porrà più la domanda sulla crittografia a ogni build, e non sarà
necessario allegare documentazione di export. Se in futuro venisse aggiunta
crittografia **non** esente (es. cifratura end-to-end proprietaria dei dati), la
dichiarazione andrà aggiornata a `YES` e fornita la documentazione richiesta.
