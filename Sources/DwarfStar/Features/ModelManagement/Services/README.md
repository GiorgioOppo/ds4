# Model Management Services

`DownloadRunner.swift` è l'adapter main-actor attorno a
`DS4Engine.ModelDownloader`. Deriva le entry dal catalogo Engine, cerca ogni
artifact nelle directory locali note e scarica in Application Support soltanto
quelli mancanti. Per i package multi-file esegue gli artifact in sequenza e li
considera installati solo quando sono presenti tutti.

Le callback molto frequenti vengono coalesciate con un `AsyncStream` bounded a
circa 8 aggiornamenti UI al secondo. La continuation viene chiusa con `defer`
anche su errore/cancellazione, evitando uno stato bloccato. Networking, SHA-256,
HTTP Range, preflight disco e token storage restano in `DS4Engine`; questo layer
non logga credenziali.

La barra assegna il 90% al trasferimento e il 10% alla verifica SHA: resta
monotona quando una ripresa termina il download e la fase di verifica ricomincia
a contare i byte da zero. Ogni snapshot porta con sé la fase corrente, così il
coalescing non può nascondere “Verifica integrità”.
