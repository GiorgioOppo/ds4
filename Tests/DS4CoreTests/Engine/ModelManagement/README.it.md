# Test dell'Engine di gestione modelli

- `HFTokenStoreTests.swift` verifica il comportamento di memorizzazione dei
  token senza esporre una credenziale reale.
- `ModelDownloaderTests.swift` copre il catalogo tipizzato multi-famiglia, la
  selezionabilità a runtime, la sorgente Hugging Face per target, i percorsi
  sicuri, lo stato locale mancante/vuoto/presente, l'esito skip-existing, il
  rifiuto per dimensione bloccata (pinned), i calcoli dello spazio libero, la
  costruzione degli URL, gli errori leggibili e i checksum.

I test del catalogo asseriscono che le tre voci Flash complete e il Pro Q2 a
file singolo sono selezionabili, che il Pro Q4 resta solo download con il suo
confine di pacchetto a due shard, e che MTP è assente dalle voci dei modelli
principali. I test sui file esistenti usano fixture regolari non vuote, così
non serve alcuna richiesta di rete. Le tre voci GLM 5.2 sono verificate
rispetto ai loro nomi di file bloccati, conteggi di byte, digest SHA-256 e
revisione, e devono restare non selezionabili.

Non leggere mai il token reale del Keychain dello sviluppatore né contattare
Hugging Face da uno unit test. Usa servizi isolati, destinazioni temporanee e
payload fissi.
