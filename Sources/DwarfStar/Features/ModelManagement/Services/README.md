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

Il buffer dell'`AsyncStream` conserva solo i quattro eventi più recenti. Insieme
alla delegate queue seriale e all'I/O `F_NOCACHE` dell'Engine, né la GUI né la
page cache crescono in proporzione ai gigabyte già scaricati.

La barra assegna il 90% al trasferimento e il 10% alla verifica SHA: resta
monotona quando una ripresa termina il download e la fase di verifica ricomincia
a contare i byte da zero. Ogni snapshot porta con sé la fase corrente, così il
coalescing non può nascondere “Verifica integrità”.

`active` contiene una struct dentro una proprietà `@Observable`: il consumer
non modifica mai un suo campo rileggendo contemporaneamente la proprietà. Ogni
evento viene applicato a una copia locale e pubblicato con una sola assegnazione,
evitando accessi `_modify` sovrapposti e riducendo le invalidazioni SwiftUI.
