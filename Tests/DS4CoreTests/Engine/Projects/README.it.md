# Test dell'engine dei progetti

`ProjectCacheTests.swift` copre l'indicizzazione, le query, le modifiche, i
confini dei file e il rifiuto dei symlink non sicuri.

Le fixture devono risiedere sotto una root di progetto temporanea. Ogni
modifica alla gestione dei percorsi richiede test di traversal, di link finale
e di genitore linkato, più un caso positivo per la creazione di directory
annidate genuinamente mancanti. I test possono usare una directory temporanea
separata come bersaglio di evasione, ma devono asserire che i suoi contenuti
non siano stati letti né modificati.
