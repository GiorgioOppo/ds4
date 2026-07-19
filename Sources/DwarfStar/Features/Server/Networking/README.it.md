[English](README.md) | **Italiano**

# Networking del server

Questa directory implementa i dettagli di trasporto HTTP indipendenti dal
protocollo:

- `HTTPRequest.swift` è il valore della richiesta interpretata.
- `LocalServer+Networking.swift` accetta le connessioni e legge richieste con
  limiti.
- `LocalServer+JSON.swift` fornisce gli helper di codifica JSON.
- `LocalServer+HTTPResponses.swift` scrive status line, header, body e frame
  di streaming.

Il networking passa le richieste interpretate agli adapter API. Preserva i
limiti di dimensione del body e di tempo di lettura, evita di acquisire il
gate di inferenza durante le letture di rete e mantieni tutte le risposte
HTTP valide anche in caso di errori di parsing o di cancellazione.
